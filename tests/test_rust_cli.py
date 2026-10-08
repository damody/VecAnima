"""Python is only the fixture/verification harness; processing runs in Rust."""
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest

from vecanima.media import probe, run

ROOT = Path(__file__).resolve().parents[1]


def test_local_ai_lineart_matches_reference_and_model_changes_reject_resume(tmp_path, rust_cli):
    """Optional downloaded model; all production inference still runs in Rust."""
    import hashlib
    import shutil
    from PIL import Image, ImageEnhance
    original_model = ROOT / "third_party/anilines/basic.onnx"
    if not original_model.is_file():
        pytest.skip("Optional AniLines ONNX model has not been exported")
    image = np.full((65, 97, 3), 235, np.uint8)
    cv2.rectangle(image, (18, 18), (77, 52), (100, 170, 200), -1)
    cv2.line(image, (10, 36), (85, 36), (20, 15, 10), 2, cv2.LINE_AA)
    source = tmp_path / "reference.png"
    cv2.imwrite(str(source), image)
    model = tmp_path / "模型.onnx"
    shutil.copyfile(original_model, model)
    result_path = tmp_path / "lineart.png"
    result = rust_cli("lineart", source, "--model", model, "--out", result_path)
    assert result["model"]["sha256"] == hashlib.sha256(model.read_bytes()).hexdigest()
    sharpened = np.array(ImageEnhance.Sharpness(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))).enhance(6.0))
    padded = cv2.copyMakeBorder(sharpened, 0, (-65) % 16, 0, (-97) % 16, cv2.BORDER_REFLECT_101)
    net = cv2.dnn.readNetFromONNX(str(original_model))
    net.setInput(cv2.dnn.blobFromImage(padded, 1 / 255.))
    expected = np.clip(net.forward()[0, 0, :65, :97] * 255 + .5, 0, 255).astype(np.uint8)
    actual = cv2.imread(str(result_path), cv2.IMREAD_GRAYSCALE)
    assert actual.shape == (65, 97)
    assert np.max(np.abs(actual.astype(int) - expected.astype(int))) <= 2
    out = tmp_path / "ai-out"
    args = ("image", source, "--out", out, "--line-model", model, "--colors", 4)
    rust_cli(*args)
    frame = json.loads((out / "geometry/000001.json").read_text(encoding="utf-8"))
    assert frame["fill_recovery"]["restored_pixels"] > 0
    assert frame["fill_recovery"]["line_model"]["sha256"] == result["model"]["sha256"]
    assert rust_cli(*args, "--resume")["verified_vector_cache_hits"] == 1
    with model.open("ab") as stream:
        stream.write(b"changed")
    failed = rust_cli(*args, "--resume", check=False)
    assert failed.returncode != 0 and "fingerprint changed" in failed.stderr


def test_native_mesh_image_resume_and_corrupt_cache(tmp_path, rust_cli):
    import hashlib
    source = tmp_path / "gradient.png"
    # A smooth linear-light gradient exercises interpolation without any strokes.
    x = np.linspace(0, 1, 96)
    encoded = np.where(x <= .0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - .055)
    image = np.tile(np.rint(encoded * 255).astype(np.uint8)[None, :, None], (64, 1, 3))
    cv2.imwrite(str(source), image)
    out = tmp_path / "mesh"
    args = ("image", source, "--out", out, "--fill-model", "mesh", "--colors", 8)
    rust_cli(*args)
    geometry = json.loads((out / "geometry/000001.json").read_text(encoding="utf-8"))
    assert geometry["version"] == 3
    assert geometry["mesh"]["max_channel_error"] <= 8
    assert "linear-light-mesh" in (out / "frames/000001.svg").read_text()
    assert "<image" not in (out / "frames/000001.svg").read_text()
    names = ["geometry/000001.json", "frames/000001.svg", "raster/000001.png"]
    fingerprint = lambda: [hashlib.sha256((out / n).read_bytes()).hexdigest() for n in names]
    before = fingerprint()
    assert rust_cli(*args, "--resume")["verified_vector_cache_hits"] == 1
    assert fingerprint() == before
    next((out / "cache/vectorized").glob("*.json")).write_text("[", encoding="utf-8")
    assert rust_cli(*args, "--resume")["verified_vector_cache_hits"] == 0
    assert fingerprint() == before
    failed = rust_cli(*args, "--mesh-error", 10, "--resume", check=False)
    assert failed.returncode != 0 and "fingerprint changed" in failed.stderr


def test_sequence_vfr_tracking_cut_and_odd_preview(tmp_path, rust_cli):
    manifest = []
    for i, duration in enumerate([.04, .12, .08, .16]):
        image = np.full((49, 65, 3), 240 if i < 3 else 0, np.uint8)
        if i < 3:
            cv2.rectangle(image, (12 + i, 10), (40 + i, 35), (80, 100, 160), -1)
            cv2.line(image, (15 + i, 22), (38 + i, 22), (10, 10, 10), 3)
        name = f"{i}.png"
        cv2.imwrite(str(tmp_path / name), image)
        manifest.append({"file": name, "duration": duration})
    source = tmp_path / "sequence.json"
    source.write_text(json.dumps({"frames": manifest}), encoding="utf-8")
    out = tmp_path / "sequence-out"
    result = rust_cli("sequence", source, "--out", out, "--temporal", "--colors", 6)
    assert result["frame_count"] == 4
    assert pts(out / "preview.mp4") == pytest.approx([0, .04, .16, .24], abs=1e-5)
    info = probe(out / "preview.mp4")
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (66, 50)
    cut = json.loads((out / "tracking/000004.json").read_text())
    assert cut["cut"] and cut["shot"] == 1
    assert json.loads((out / "tracking/000002.json").read_text())["associations"]


def test_known_camera_translation_preserves_stroke_identity_without_cross_cut_links(tmp_path, rust_cli):
    """Known motion is an external oracle, rather than self-reported confidence."""
    source_image = np.full((96, 128, 3), 230, np.uint8)
    rng = np.random.default_rng(41)
    for x, y in rng.integers([8, 8], [120, 88], size=(60, 2)):
        cv2.circle(source_image, (int(x), int(y)), 1, (210, 210, 210), -1)
    for x in range(20, 109):
        a = (x, round(48 + np.sin(x / 19) * 8))
        b = (x + 1, round(48 + np.sin((x + 1) / 19) * 8))
        cv2.line(source_image, a, b, (20 + x // 2, 30, 50), 2, cv2.LINE_AA)
    manifest = []
    for i in range(4):
        image = cv2.warpAffine(source_image, np.float32([[1, 0, i * 2], [0, 1, i]]), (128, 96), borderValue=(230, 230, 230)) if i < 3 else np.zeros_like(source_image)
        name = f"motion-{i}.png"
        cv2.imwrite(str(tmp_path / name), image)
        manifest.append({"file": name, "duration": .04})
    source = tmp_path / "known-motion.json"
    source.write_text(json.dumps({"frames": manifest}), encoding="utf-8")
    out = tmp_path / "known-motion-out"
    rust_cli("sequence", source, "--out", out, "--temporal", "--colors", 8, "--no-preview")
    identities, widths, centers = [], [], []
    for i in range(1, 4):
        frame = json.loads((out / f"geometry/{i:06}.json").read_text())
        stroke = max(frame["strokes"], key=lambda s: len(s["nodes"]))
        tracking = json.loads((out / f"tracking/{i:06}.json").read_text())
        association = next(a for a in tracking["associations"] if a["stroke"] == stroke["id"])
        identities.append(association["track"])
        widths.append(np.median([n["left"] + n["right"] for n in stroke["nodes"]]))
        centers.append(np.mean([n["center"] for n in stroke["nodes"]], axis=0))
        assert not tracking["cut"]
    assert len(set(identities)) == 1
    assert np.max(np.abs(np.diff(centers, axis=0) - [2, 1])) < .75
    assert np.ptp(widths) < .3
    cut = json.loads((out / "tracking/000004.json").read_text())
    assert cut["cut"] and not set(identities).intersection(a["track"] for a in cut["associations"])


@pytest.fixture(scope="module")
def rust_cli():
    executable = ROOT / "target/release/vecanima.exe"
    libraries = list((ROOT / ".tools/opencv-sdk").glob("x64/*/bin/opencv_core4140.dll"))
    if not executable.exists() or not libraries:
        pytest.skip("Build the Rust Release CLI and OpenCV 4.14.0 SDK first")
    env = dict(os.environ)
    env["PATH"] = str(libraries[0].parent) + os.pathsep + env.get("PATH", "")

    def invoke(*args, check=True):
        result = subprocess.run([str(executable), *map(str, args)], capture_output=True,
                                text=True, encoding="utf-8", env=env)
        if check:
            assert result.returncode == 0, result.stderr
            return json.loads(result.stdout)
        return result
    return invoke


def pts(source):
    info = json.loads(run("ffprobe", ["-v", "error", "-select_streams", "v:0", "-show_frames",
                                    "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(source)]).stdout)
    return [float(f["best_effort_timestamp_time"]) for f in info["frames"]]


@pytest.mark.parametrize("with_audio,offset", [(False, 0), (True, 5)])
def test_rust_seek_audio_timeline_and_rendered_preview(tmp_path, rust_cli, with_audio, offset):
    source = tmp_path / "input.mkv"
    args = ["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1"]
    if with_audio:
        args += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1", "-c:a", "pcm_s16le"]
    args += ["-c:v", "ffv1", "-output_ts_offset", str(offset), str(source)]
    run("ffmpeg", args)
    # Spaces, apostrophes and Unicode exercise Windows concat path handling.
    out = tmp_path / "Rust's 輸出"
    result = rust_cli("vectorize", source, "--out", out, "--start", ".15", "--duration", ".5", "--width", 64, "--colors", 5)
    timeline = json.loads((out / "timeline.json").read_text())
    assert [f["source_time"] for f in timeline["frames"]] == pytest.approx([p for p in pts(source) if offset + .15 <= p < offset + .65])
    assert pts(out / "preview.mp4") == pytest.approx([f["time"] for f in timeline["frames"]], abs=1e-5)
    info = probe(out / "preview.mp4")
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == result["frame_count"]
    assert float(video["duration"]) == pytest.approx(timeline["duration"], abs=1e-5)
    assert any(s["codec_type"] == "audio" for s in info["streams"]) == with_audio
    if with_audio:
        audio = next(s for s in info["streams"] if s["codec_type"] == "audio")
        assert abs(float(audio["duration"]) - float(video["duration"])) < .025
        assert abs(float(audio["start_time"]) - float(video["start_time"])) < .025
    project = json.loads((out / "project.json").read_text())
    assert project["implementation"] == "rust"
    assert all("<image" not in (out / a["svg"]).read_text() for a in project["assets"])
    assert json.loads((out / "status.json").read_text())["status"] == "complete"


def test_rust_vfr_exact_reuse_and_black_frame(tmp_path, rust_cli):
    lines = ["ffconcat version 1.0"]
    for i, duration in enumerate([.04, .12, .08, .20, .04, .12]):
        # Equal consecutive images must reuse geometry without changing timing.
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((48, 64, 3), (i // 2) * 100, np.uint8))
        lines += [f"file '{i}.png'", "option framerate 1000", f"duration {duration}"]
    lines += ["file '5.png'", "option framerate 1000"]
    listing = tmp_path / "input.ffconcat"
    listing.write_text("\n".join(lines))
    source = tmp_path / "vfr.mkv"
    run("ffmpeg", ["-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-fps_mode", "vfr", "-c:v", "ffv1", str(source)])
    out = tmp_path / "out"
    result = rust_cli("vectorize", source, "--out", out, "--duration", ".6", "--width", 64, "--colors", 6)
    frames = json.loads((out / "timeline.json").read_text())["frames"]
    assert [f["duration"] for f in frames[:-1]] == pytest.approx([.04, .12, .08, .20, .04])
    assert pts(out / "preview.mp4") == pytest.approx([f["time"] for f in frames], abs=1e-5)
    assert result["unique_assets"] == 3
    assert frames[0]["asset"] == frames[1]["asset"]
    assert cv2.imread(str(out / "raster/000001.png")).max() == 0


def test_rust_no_preview_repeatability_hole_and_single_pixel_line(tmp_path, rust_cli):
    image = np.full((64, 64, 3), 240, np.uint8)
    image[8:56, 8:56] = (30, 120, 210)
    image[24:40, 24:40] = 240
    image[15, 10:54] = 10
    picture = tmp_path / "fixture.png"
    cv2.imwrite(str(picture), image)
    source = tmp_path / "fixture.mkv"
    run("ffmpeg", ["-v", "error", "-loop", "1", "-i", str(picture), "-t", "0.2", "-r", "10", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source)])
    texts = []
    for name in ("a", "b"):
        out = tmp_path / name
        rust_cli("vectorize", source, "--out", out, "--duration", ".2", "--width", 64, "--colors", 3, "--epsilon", ".1", "--no-preview")
        assert not (out / "preview.mp4").exists()
        assert (out / "index.html").exists()
        rendered = cv2.imread(str(out / "raster/000001.png"))
        assert rendered[30, 30].min() > 200
        assert rendered[15, 20].max() < 25
        assert rendered[45, 45, 2] > rendered[45, 45, 0] + 80
        texts.append((out / "frames/000001.svg").read_bytes())
    assert texts[0] == texts[1]
    # Refusal must preserve every existing output, including status.
    before = (tmp_path / "a/status.json").read_bytes()
    failure = rust_cli("vectorize", source, "--out", tmp_path / "a", "--width", 64, check=False)
    assert failure.returncode != 0 and "empty" in failure.stderr
    assert (tmp_path / "a/status.json").read_bytes() == before


def test_rust_invalid_settings_do_not_create_output(tmp_path, rust_cli):
    out = tmp_path / "out"
    result = rust_cli("vectorize", ROOT / "Isekai.mkv", "--out", out, "--colors", 0, check=False)
    assert result.returncode != 0 and "colors" in result.stderr
    assert not out.exists()
