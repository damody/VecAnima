from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from . import __version__
from .media import extract, preview, probe, run, write_json
from .vectorize import Settings, fit_palette, rasterize, svg, vectorize


def fresh_output(path: Path):
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def analyze(source: Path, out: Path, count=12):
    fresh_output(out)
    info = probe(source)
    write_json(out / "source.json", info)
    duration = float(info["format"]["duration"])
    stamps = np.linspace(min(1, duration / 10), duration * 0.95, count)
    rows = []
    samples = []
    for i, stamp in enumerate(stamps):
        result = out / f"sample-{i:02d}.png"
        run("ffmpeg", ["-hide_banner", "-v", "error", "-nostdin", "-ss", str(stamp),
                       "-i", str(source.resolve()), "-map", "0:v:0", "-frames:v", "1",
                       "-vf", "scale=320:180", str(result.resolve())])
        image = cv2.imread(str(result))
        if image is None:
            raise RuntimeError(f"Cannot read thumbnail at {stamp}")
        tile = cv2.copyMakeBorder(image, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(tile, f"{stamp:.2f}s", (10, 202), cv2.FONT_HERSHEY_SIMPLEX, .6, (230, 230, 230), 1)
        rows.append(tile)
        samples.append({"time": float(stamp), "file": result.name})
    while len(rows) % 4:
        rows.append(np.zeros_like(rows[0]))
    sheet = np.vstack([np.hstack(rows[i:i+4]) for i in range(0, len(rows), 4)])
    cv2.imwrite(str(out / "contact-sheet.jpg"), sheet)
    write_json(out / "samples.json", samples)
    return {"duration": duration, "samples": count, "contact_sheet": str(out / "contact-sheet.jpg")}


def build(source: Path, out: Path, start: float, duration: float, width: int, settings: Settings,
          encode_preview=True):
    fresh_output(out)
    state = {"version": __version__, "status": "incomplete", "source": str(source.resolve()),
             "start": start, "duration": duration, "width": width, "settings": asdict(settings)}
    write_json(out / "status.json", state)
    began = time.perf_counter()
    try:
        timeline = extract(source, out / "original", start, duration, width)
        write_json(out / "source.json", timeline.pop("probe"))
        frames = timeline["frames"]
        for name in ("frames", "raster", "geometry"):
            (out / name).mkdir()
        sample_indices = np.unique(np.linspace(0, len(frames)-1, min(12, len(frames)), dtype=int))
        palette = fit_palette([cv2.imread(str(out / "original" / frames[i]["file"]))
                               for i in sample_indices], settings)
        palette_list = palette.tolist()
        metrics = []
        assets = []
        cache = {}
        previous_original = previous_rendered = None
        for i, frame in enumerate(frames):
            tick = time.perf_counter()
            original = cv2.imread(str(out / "original" / frame["file"]))
            digest = hashlib.sha256(original.tobytes()).hexdigest()
            svg_file = f"frames/{i+1:06d}.svg"
            geometry_file = f"geometry/{i+1:06d}.json"
            reused = digest in cache
            if reused:
                asset = cache[digest]
                text = (out / asset["svg"]).read_text(encoding="utf-8")
                project = json.loads((out / asset["geometry"]).read_text(encoding="utf-8"))
            else:
                project = vectorize(original, palette, settings)
                text = svg(project)
                (out / svg_file).write_text(text, encoding="utf-8")
                write_json(out / geometry_file, project)
                asset = {"id": len(assets), "svg": svg_file, "geometry": geometry_file}
                assets.append(asset)
                cache[digest] = asset
            frame["asset"] = asset["id"]
            frame["svg"] = asset["svg"]
            png, rendered = rasterize(text)
            (out / "raster" / frame["file"]).write_bytes(png)
            diff = original.astype(np.float32) - rendered.astype(np.float32)
            mse = float(np.mean(diff * diff))
            row = {"index": i, "mae": float(np.mean(abs(diff))), "mse": mse,
                   "psnr_db": 10 * float(np.log10(255**2 / mse)) if mse else None,
                   "paths": len(project["layers"]),
                   "vertices": sum(len(r) for l in project["layers"] for r in l["rings"]),
                   "svg_bytes": len(text.encode()), "reused_exact_frame": reused,
                   "seconds": time.perf_counter() - tick}
            if previous_original is not None:
                stable = np.max(abs(original.astype(np.int16) - previous_original.astype(np.int16)), axis=2) <= 2
                delta = np.mean(abs(rendered.astype(np.float32) - previous_rendered.astype(np.float32)), axis=2)
                row["near_static_pixel_fraction"] = float(stable.mean())
                row["near_static_delta_mae"] = float(delta[stable].mean()) if stable.any() else None
            previous_original, previous_rendered = original, rendered
            metrics.append(row)
            if i % 24 == 0 or i == len(frames)-1:
                print(f"Vectorized {i+1}/{len(frames)} frames", flush=True)
        write_json(out / "timeline.json", timeline)
        write_json(out / "project.json", {"format": "vecanima", "format_version": 1,
                   "model": "flat-regions-and-dark-mask-baseline", "width": rendered.shape[1],
                   "height": rendered.shape[0], "timeline": "timeline.json", "assets": assets,
                   "palette_lab8": palette_list, "settings": asdict(settings),
                   "limitations": ["No fitted stroke centerlines", "No shared region boundaries",
                                    "No triangulated gradients", "No motion tracking or temporal filter"]})
        if encode_preview:
            encoded = preview(source, out, timeline)
            video = next(s for s in encoded["streams"] if s["codec_type"] == "video")
            if int(video.get("nb_frames", 0)) != len(frames):
                raise RuntimeError(f"Preview frame count mismatch: {video.get('nb_frames')} vs {len(frames)}")
            if abs(float(video.get("duration", 0)) - timeline["duration"]) > .002:
                raise RuntimeError("Preview duration mismatch")
            write_viewer(out, timeline)
        # A visual sample grid at the actual rendering resolution.
        selected = np.unique(np.linspace(0, len(frames)-1, min(4, len(frames)), dtype=int))
        pairs = []
        for i in selected:
            f = frames[i]
            a = cv2.imread(str(out / "original" / f["file"]))
            b = cv2.imread(str(out / "raster" / f["file"]))
            pairs.append(np.hstack([a, b]))
        cv2.imwrite(str(out / "comparison.jpg"), np.vstack(pairs))
        report = {"frame_count": len(frames), "unique_assets": len(assets),
                  "duration": timeline["duration"], "elapsed_seconds": time.perf_counter()-began,
                  "mean_mae": float(np.mean([r["mae"] for r in metrics])),
                  "mean_psnr_db": float(np.mean([r["psnr_db"] for r in metrics if r["psnr_db"] is not None]))
                    if any(r["psnr_db"] is not None for r in metrics) else None,
                  "unique_svg_bytes": sum((out/a["svg"]).stat().st_size for a in assets),
                  "metric_notes": "Near-static delta is an unwarped diagnostic, not a motion-compensated flicker score.",
                  "frames": metrics}
        write_json(out / "report.json", report)
        state["status"] = "complete"
        state["tools"] = {n: run(n, ["-version"]).stdout.splitlines()[0] for n in ("ffmpeg", "ffprobe")}
        write_json(out / "status.json", state)
        return {k: v for k, v in report.items() if k != "frames"}
    except Exception as exc:
        state["error"] = str(exc)
        write_json(out / "status.json", state)
        raise


def write_viewer(out, timeline):
    # Inline timeline permits file:// playback without fetching JSON.
    data = json.dumps([{"time": f["time"], "svg": f["svg"]} for f in timeline["frames"]])
    html = '''<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>VecAnima baseline preview</title>
<style>body{background:#111820;color:#e5edf4;font:16px system-ui;margin:30px}button,input{margin:8px}
.panels{display:flex;gap:16px;flex-wrap:wrap}.panel{flex:1;min-width:320px}video,img{width:100%;background:#000}
#frame{image-rendering:auto}#scrub{width:60%}.viewport{overflow:auto;max-height:65vh}</style>
<h1>VecAnima 實驗預覽</h1><p>左：向量結果的有聲 MP4 預覽；右：依相同時間戳切換的 SVG，可放大檢查。
此版為純色色塊與暗線遮罩基準，尚未加入線寬模型、網格漸層與跨幀追蹤。</p>
<div class="panels"><div class="panel"><video id="video" controls src="preview.mp4"></video></div>
<div class="panel"><div class="viewport"><img id="frame" alt="SVG 向量影格"></div></div></div>
<p><button id="prev">上一幀</button><button id="next">下一幀</button><span id="label"></span>
<input id="scrub" type="range" min="0" step="1"><label>縮放 <input id="zoom" type="range" min="1" max="4" step=".25" value="1"></label></p>
<script>const frames=TIMELINE;const v=document.getElementById('video'),im=document.getElementById('frame'),
s=document.getElementById('scrub'),label=document.getElementById('label');let current=0,manual=false;
s.max=frames.length-1;
function show(i){current=Math.max(0,Math.min(frames.length-1,i));im.src=frames[current].svg;
s.value=current;label.textContent=`${current+1}/${frames.length} · ${frames[current].time.toFixed(3)}s`;}
function seek(i){v.pause();manual=true;show(i);v.currentTime=frames[current].time+.0001;}
document.getElementById('prev').onclick=()=>seek(current-1);document.getElementById('next').onclick=()=>seek(current+1);
s.oninput=()=>seek(Number(s.value));document.getElementById('zoom').oninput=e=>im.style.width=(100*e.target.value)+'%';
function sync(){let lo=0,hi=frames.length;while(lo<hi){const m=(lo+hi)>>1;if(frames[m].time<=v.currentTime+.00001)lo=m+1;else hi=m;}
const i=Math.max(0,lo-1);if(i!==current)show(i);}
v.addEventListener('play',()=>manual=false);v.addEventListener('pointerdown',()=>manual=false);
v.addEventListener('seeked',()=>{if(!manual)sync();});
function update(){if(!v.paused&&!manual)sync();requestAnimationFrame(update);}show(0);update();</script></html>'''
    (out / "index.html").write_text(html.replace("TIMELINE", data), encoding="utf-8")
