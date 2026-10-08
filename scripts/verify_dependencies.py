"""Verify the local offline archive using only the Python standard library."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    directory = ROOT / "third_party"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if digest(ROOT / "uv.lock") != manifest["uv_lock_sha256"]:
        raise RuntimeError("uv.lock changed; refresh the offline archive first")
    for package in manifest["packages"]:
        if digest(directory / package["file"]) != package["sha256"]:
            raise RuntimeError(f"Invalid wheel: {package['name']}")
    ffmpeg = json.loads((ROOT / "scripts" / "ffmpeg-lock.json").read_text())
    if digest(ROOT / ".tools" / "ffmpeg-download.zip") != ffmpeg["sha256"]:
        raise RuntimeError("Invalid FFmpeg archive")
    if "uv" in manifest and digest(ROOT / manifest["uv"]["file"]) != manifest["uv"]["sha256"]:
        raise RuntimeError("Invalid local uv binary")
    opencv = json.loads((ROOT / "scripts/opencv-lock.json").read_text())
    if digest(ROOT / opencv["archive"]) != opencv["sha256"]:
        raise RuntimeError("Invalid OpenCV source archive")
    print(f"Verified {len(manifest['packages'])} wheels, FFmpeg archive, local uv and OpenCV {opencv['version']} source archive")


if __name__ == "__main__":
    main()
