"""Independent, bounded-memory output audit; never implements vectorization."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from statistics import mean, median


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(values):
    return all(math.isfinite(v) for v in values)


def audit(root: Path, minimum_duration: float, model: Path | None):
    root = root.resolve()
    assert read(root / "status.json")["status"] == "complete", "processing incomplete"
    project, timeline, report = (read(root / f"{name}.json") for name in ("project", "timeline", "report"))
    assert project["implementation"] == "rust"
    frames = timeline["frames"]
    assert len(frames) == report["frame_count"] and frames
    assert timeline["duration"] >= minimum_duration
    expected_hash = hashlib.sha256(model.read_bytes()).hexdigest() if model else None
    errors, warnings = [], []
    digest = hashlib.sha256()
    byte_count = svg_files = strokes = nodes = triangles = suppressed = 0
    harmonic_maximum = 0
    previous, previous_shot = {}, None
    width_deltas, motion_residuals, confidences = [], [], []
    events, cuts, matches, region_matches = {}, 0, 0, 0
    asset_paths, asset_buffers = {}, {}
    validated_assets = set()

    def local(name):
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file(), f"missing/invalid asset: {name}"
        return path

    for asset in project["assets"]:
        for field in ("svg", "fills_svg", "strokes_svg", "geometry", "mesh_buffer"):
            path = local(asset[field])
            hashed = hashlib.sha256()
            tail = b""
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    hashed.update(chunk)
                    if field.endswith("svg"):
                        text = (tail + chunk).lower()
                        assert b"<image" not in text and b"data:image" not in text, f"raster embedded: {path}"
                        assert not re.search(rb'[=,\s][+-]?(?:nan|inf)[,\s"/]', text), f"nonfinite SVG coordinate: {path}"
                        tail = text[-256:]
            digest.update(asset[field].encode())
            digest.update(hashed.digest())
            byte_count += path.stat().st_size
            svg_files += field.endswith("svg")
        asset_paths[asset["id"]] = asset["geometry"]
        asset_buffers[asset["id"]] = asset["mesh_buffer"]

    @lru_cache(maxsize=2)
    def shapes(asset_id):
        nonlocal nodes, strokes, triangles, suppressed, harmonic_maximum
        count_asset = asset_id not in validated_assets
        validated_assets.add(asset_id)
        geometry = read(local(asset_paths[asset_id]))
        recovery = geometry.get("fill_recovery") or {}
        if recovery and not recovery.get("harmonic_converged", True):
            errors.append(f"harmonic not converged: asset {asset_id}")
        harmonic_maximum = max(harmonic_maximum, recovery.get("harmonic_iterations", 0))
        if expected_hash:
            assert recovery["line_model"]["sha256"] == expected_hash
        shape_stats = {}
        for stroke in geometry["strokes"]:
            assert len(stroke["nodes"]) >= 2
            for node in stroke["nodes"]:
                assert finite([*node["center"], node["left"], node["right"], *node["color_linear"], node["confidence"]])
                assert node["left"] > 0 and node["right"] > 0
                assert all(0 <= c <= 1 for c in node["color_linear"])
            ns = stroke["nodes"]
            shape_stats[stroke["id"]] = (median(n["left"] + n["right"] for n in ns),
                [mean(n["center"][c] for n in ns) for c in (0, 1)])
            nodes += len(ns) if count_asset else 0
        strokes += len(geometry["strokes"]) if count_asset else 0
        mesh = geometry.get("mesh")
        if mesh:
            data = mesh["geometry"]
            assert all(finite(p) for p in data["points"])
            assert all(len(t) == 3 and len(set(t)) == 3 and min(t) >= 0 and max(t) < len(data["points"]) for t in data["triangles"])
            assert all(len(e) == 2 and min(e) >= 0 and max(e) < len(data["points"]) for e in data["constraints"])
            assert len(mesh["colors_linear"]) == len(data["triangles"])
            assert all(finite(color) and all(0 <= v <= 1 for v in color) for face in mesh["colors_linear"] for color in face)
            assert local(asset_buffers[asset_id]).stat().st_size == len(data["triangles"]) * 60
            triangles += len(data["triangles"]) if count_asset else 0
            suppressed += mesh.get("suppressed_palette_boundaries", 0) if count_asset else 0
            if mesh["vertex_budget_reached"] or mesh["unmet_pixels"]:
                warnings.append(f"mesh tolerance unmet: asset {asset_id}, pixels={mesh['unmet_pixels']}")
        return shape_stats

    end = 0.0
    source_time = -math.inf
    for i, frame in enumerate(frames):
        assert abs(frame["time"] - end) < 1e-6 and frame["duration"] > 0
        assert frame["source_time"] > source_time
        source_time = frame["source_time"]
        end += frame["duration"]
        assert frame["asset"] in asset_paths
        current_shapes = shapes(frame["asset"])
        tracking = read(local(frame["tracking"]))
        associations = tracking["associations"]
        assert len({a["track"] for a in associations}) == len(associations)
        assert len({a["stroke"] for a in associations}) == len(associations)
        current = {}
        affine = tracking["camera_affine"]
        assert finite(affine)
        cuts += bool(tracking["cut"])
        for event in tracking["events"]:
            events[event["kind"]] = events.get(event["kind"], 0) + 1
        for association in associations:
            confidence = association["confidence"]
            assert math.isfinite(confidence) and 0 <= confidence <= 1
            width, center = current_shapes[association["stroke"]]
            track = association["track"]
            current[track] = (width, center)
            if track in previous and previous_shot == tracking["shot"] and not tracking["cut"]:
                old_width, point = previous[track]
                predicted = [affine[0]*point[0]+affine[1]*point[1]+affine[2], affine[3]*point[0]+affine[4]*point[1]+affine[5]]
                width_deltas.append(abs(width-old_width))
                motion_residuals.append(math.dist(center, predicted))
                confidences.append(confidence)
                matches += 1
        region_matches += sum(a["confidence"] > 0 and not a["filtered"] for a in tracking["regions"])
        previous, previous_shot = current, tracking["shot"]
        if (i + 1) % 240 == 0:
            print(f"Audited {i+1}/{len(frames)} timeline frames", flush=True)
    assert abs(end-timeline["duration"]) < 1e-6
    assert validated_assets == set(asset_paths), "unreferenced assets were not audited"
    preview = None
    if report["preview_encoded"]:
        probe = read(local("preview-probe.json"))
        preview = {"duration": probe["format"].get("duration"), "streams": [s["codec_type"] for s in probe["streams"]]}
        assert "video" in preview["streams"]
        assert abs(float(preview["duration"])-timeline["duration"]) < 0.15
    result = {"passed": not errors, "errors": errors, "warnings": warnings,
        "frame_count": len(frames), "duration": end, "working_size": [project["width"], project["height"]],
        "unique_assets": len(project["assets"]), "pure_vector_svg_files": svg_files,
        "asset_bytes": byte_count, "asset_set_sha256": digest.hexdigest(),
        "stroke_count": strokes, "node_count": nodes, "fill_triangles": triangles,
        "suppressed_palette_boundaries": suppressed, "maximum_harmonic_iterations": harmonic_maximum,
        "cuts": cuts, "events": events, "matched_track_pairs": matches,
        "median_matched_width_delta_pixels": median(width_deltas) if width_deltas else None,
        "median_camera_compensated_center_residual_pixels": median(motion_residuals) if motion_residuals else None,
        "mean_match_confidence": mean(confidences) if confidences else None,
        "region_association_records": region_matches, "preview": preview,
        "metric_notes": "Track residuals describe the algorithm's own associations, not semantic accuracy or a temporal-stability ground-truth claim."}
    (root / "acceptance-audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert not errors, "audit failed; inspect acceptance-audit.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--minimum-duration", type=float, default=0)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    audit(args.project, args.minimum_duration, args.model)
