import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from vecanima.media import binary, extract, probe, run
from vecanima.pipeline import build
from vecanima.vectorize import Settings


@pytest.fixture(scope="module", autouse=True)
def require_ffmpeg():
    try:
        binary("ffmpeg")
        binary("ffprobe")
    except RuntimeError:
        pytest.skip("Project FFmpeg not installed")


def source_pts(source):
    info = json.loads(run("ffprobe", ["-v","error","-select_streams","v:0", "-show_frames",
                                    "-show_entries","frame=best_effort_timestamp_time", "-of","json", str(source)]).stdout)
    return [float(f["best_effort_timestamp_time"]) for f in info["frames"]]


@pytest.mark.parametrize("with_audio,offset", [(False,0),(True,5)])
def test_nonzero_seek_and_audio_roundtrip(tmp_path, with_audio, offset):
    source = tmp_path / "source.mkv"
    args = ["-v","error","-f","lavfi","-i","testsrc2=size=64x48:rate=10:duration=1"]
    if with_audio:
        args += ["-f","lavfi","-i","sine=frequency=440:sample_rate=48000:duration=1", "-c:a","pcm_s16le"]
    args += ["-c:v","ffv1","-output_ts_offset", str(offset), str(source)]
    run("ffmpeg",args)
    out = tmp_path / "experiment"
    result = build(source,out,.15,.5,64,Settings(colors=5))
    timeline = json.loads((out/"timeline.json").read_text())
    actual = [f["source_time"] for f in timeline["frames"]]
    expected = [p for p in source_pts(source) if offset+.15 <= p < offset+.65]
    assert actual == pytest.approx(expected,abs=1e-6)
    assert result["frame_count"] == len(expected)
    encoded = probe(out/"preview.mp4")
    assert any(s["codec_type"] == "audio" for s in encoded["streams"]) == with_audio
    if with_audio:
        audio = next(s for s in encoded["streams"] if s["codec_type"] == "audio")
        video = next(s for s in encoded["streams"] if s["codec_type"] == "video")
        assert abs(float(audio["duration"])-float(video["duration"])) < .025
        assert abs(float(audio["start_time"])-float(video["start_time"])) < .025


def test_vfr_durations_are_not_replaced_with_average_fps(tmp_path):
    lines = ["ffconcat version 1.0"]
    for i, duration in enumerate([.04,.12,.08,.20,.04,.12]):
        image = np.full((48,64,3),i*35,np.uint8)
        cv2.imwrite(str(tmp_path/f"{i}.png"),image)
        lines += [f"file '{i}.png'", "option framerate 1000",f"duration {duration}"]
    lines += ["file '5.png'","option framerate 1000"]
    (tmp_path/"input.ffconcat").write_text("\n".join(lines))
    source = tmp_path/"vfr.mkv"
    run("ffmpeg",["-v","error","-f","concat","-safe","0","-i",str(tmp_path/"input.ffconcat"),
                  "-fps_mode","vfr","-c:v","ffv1",str(source)])
    out = tmp_path/"experiment"
    build(source,out,0,.6,64,Settings(colors=6))
    timeline = json.loads((out/"timeline.json").read_text())
    frames = timeline["frames"]
    assert [f["source_time"] for f in frames] == pytest.approx([p for p in source_pts(source) if p < .6])
    assert [f["duration"] for f in frames[:-1]] == pytest.approx([.04,.12,.08,.20,.04])
    rendered_pts = source_pts(out/"preview.mp4")
    assert rendered_pts == pytest.approx([f["time"] for f in frames],abs=.00001)
    video = next(s for s in probe(out/"preview.mp4")["streams"] if s["codec_type"] == "video")
    assert float(video["duration"]) == pytest.approx(timeline["duration"],abs=.00001)


def test_nonempty_output_is_preserved(tmp_path):
    out = tmp_path/"out"
    out.mkdir()
    (out/"keep.txt").write_text("keep")
    with pytest.raises(ValueError,match="empty"):
        build(Path("unused.mkv"),out,0,1,64,Settings())
    assert (out/"keep.txt").read_text() == "keep"
