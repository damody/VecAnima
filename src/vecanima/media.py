"""FFmpeg media IO. All subprocess arguments are passed without a shell."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def binary(name: str) -> str:
    override = os.environ.get(f"VECANIMA_{name.upper()}")
    local = ROOT / ".tools" / f"{name}.exe"
    found = override or (str(local) if local.exists() else shutil.which(name))
    if not found:
        raise RuntimeError(f"{name} unavailable; run scripts/setup.ps1")
    return found


def run(name: str, args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run([binary(name), *map(str, args)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"{name} failed ({result.returncode}):\n{result.stderr[-6000:]}")
    return result


def probe(path: Path) -> dict:
    return json.loads(run("ffprobe", ["-v", "error", "-show_format", "-show_streams",
                                       "-of", "json", str(path.resolve())]).stdout)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def extract(path: Path, directory: Path, start: float, duration: float, width: int) -> dict:
    """Accurate input seeking; showinfo records the decoded frames' relative PTS.

    FFmpeg normalizes input format start_time, then subtracts the requested seek.
    Source absolute PTS is therefore format_start + requested_start + showinfo PTS.
    """
    info = probe(path)
    videos = [s for s in info["streams"] if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError("No video stream")
    stream = videos[0]
    directory.mkdir(parents=True, exist_ok=False)
    filters = f"scale={width}:trunc(ow/dar/2)*2,setsar=1,showinfo"
    result = run("ffmpeg", ["-hide_banner", "-nostdin", "-ss", str(start), "-i", str(path.resolve()),
                            "-t", str(duration), "-map", f"0:{stream['index']}", "-an", "-sn",
                            "-vf", filters, "-fps_mode", "passthrough", "-enc_time_base", "1/1000000",
                            str(directory / "%06d.png")])
    (directory.parent / "decode.log").write_text(result.stderr, encoding="utf-8")
    records = []
    for line in result.stderr.splitlines():
        if "showinfo" not in line:
            continue
        match = re.search(r"\bn:\s*(\d+).*?\bpts:\s*(-?\d+)\s+pts_time:([\d.eE+-]+)", line)
        if match:
            d = re.search(r"\bduration_time:([\d.eE+-]+)", line)
            records.append({"decode_index": int(match[1]), "pts": int(match[2]),
                            "seek_relative_time": float(match[3]),
                            "decoded_duration": float(d[1]) if d else None})
    files = sorted(directory.glob("*.png"))
    if not files or len(records) < len(files):
        raise RuntimeError("Decoded images and PTS log do not match, or the requested range is empty")
    lookahead = records[len(files)] if len(records) > len(files) else None
    records = records[:len(files)]
    first = records[0]["seek_relative_time"]
    origin = float(info.get("format", {}).get("start_time", 0))
    for i, (frame, file) in enumerate(zip(records, files)):
        frame["file"] = file.name
        frame["source_time"] = origin + start + frame["seek_relative_time"]
        frame["time"] = frame["seek_relative_time"] - first
        if i + 1 < len(records):
            dt = records[i + 1]["seek_relative_time"] - frame["seek_relative_time"]
            frame["duration_estimated"] = False
        else:
            dt = (lookahead["seek_relative_time"] - frame["seek_relative_time"]) if lookahead else frame["decoded_duration"]
            frame["duration_estimated"] = not bool(dt and dt > 0)
            if not dt or dt <= 0:
                dt = records[-1]["time"] - records[-2]["time"] if i else duration
            dt = min(dt, duration - frame["seek_relative_time"])
        if dt <= 0:
            raise RuntimeError("Non-increasing video timestamps")
        frame["duration"] = dt
    return {"source": str(path.resolve()), "source_stream": stream["index"],
            "source_time_base": stream["time_base"], "source_format_start": origin,
            "requested_start": start, "requested_duration": duration,
            "first_frame_offset": first, "duration": records[-1]["time"] + records[-1]["duration"],
            "frames": records, "probe": info}


def preview(source: Path, directory: Path, timeline: dict) -> dict:
    frames = timeline["frames"]
    lines = ["ffconcat version 1.0"]
    for frame in frames:
        lines += [f"file 'raster/{frame['file']}'", "option framerate 1000000",
                  f"duration {frame['duration']:.9f}"]
    lines += [f"file 'raster/{frames[-1]['file']}'", "option framerate 1000000"]
    listing = directory / "preview.ffconcat"
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audio_start = timeline["requested_start"] + timeline["first_frame_offset"]
    run("ffmpeg", ["-hide_banner", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(listing.resolve()),
                   "-ss", str(audio_start), "-i", str(source.resolve()),
                   "-map", "0:v:0", "-map", "1:a:0?", "-sn", "-dn",
                   "-t", str(timeline["duration"]), "-c:v", "libx264", "-preset", "fast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-fps_mode", "vfr",
                   "-enc_time_base", "1/1000000", "-bf", "0", "-bsf:v",
                   f"setts=duration=if(eq(N\\,{len(frames)-1})\\,{frames[-1]['duration']:.9f}/TB\\,DURATION)",
                   "-video_track_timescale", "1000000", "-c:a", "aac", "-b:a", "160k",
                   "-movflags", "+faststart", str((directory / "preview.mp4").resolve())])
    result = probe(directory / "preview.mp4")
    write_json(directory / "preview-probe.json", result)
    return result
