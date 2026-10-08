"""Install a checksum-verified, project-local Windows FFmpeg build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    tools = ROOT / ".tools"
    tools.mkdir(exist_ok=True)
    manifest = tools / "ffmpeg.json"
    lock = json.loads((ROOT / "scripts" / "ffmpeg-lock.json").read_text())
    if manifest.exists():
        info = json.loads(manifest.read_text())
        if info["binaries"] == lock["binaries"] and all((tools / f"{name}.exe").exists() and hashlib.sha256(
            (tools / f"{name}.exe").read_bytes()).hexdigest() == lock["binaries"][name]
               for name in ("ffmpeg", "ffprobe")):
            print("Verified existing FFmpeg installation")
            return
    # A fixed archive and reviewed hash keep setup reproducible across releases.
    url, expected = lock["url"], lock["sha256"]
    archive = tools / "ffmpeg-download.zip"
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
        if args.offline:
            raise RuntimeError("Verified local FFmpeg archive is required for offline installation")
        print("Downloading FFmpeg essentials", flush=True)
        with urllib.request.urlopen(url, timeout=120) as source, archive.open("wb") as target:
            shutil.copyfileobj(source, target)
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: expected {expected}, received {actual}")
    hashes = {}
    with zipfile.ZipFile(archive) as z:
        for name in ("ffmpeg", "ffprobe"):
            member = next(n for n in z.namelist() if n.endswith(f"/bin/{name}.exe"))
            data = z.read(member)
            (tools / f"{name}.exe").write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
            if hashes[name] != lock["binaries"][name]:
                raise RuntimeError(f"Unexpected {name} binary hash")
        for member in z.namelist():
            if member.lower().endswith(("/license", "/license.txt", "/readme.txt")):
                (tools / Path(member).name).write_bytes(z.read(member))
    version = subprocess.check_output([str(tools / "ffmpeg.exe"), "-version"], text=True).splitlines()[0]
    if version != lock["version"]:
        raise RuntimeError("Unexpected FFmpeg version")
    manifest.write_text(json.dumps({"url": url, "sha256": actual, "version": version,
                                    "binaries": hashes}, indent=2), encoding="utf-8")
    print(version)


if __name__ == "__main__":
    main()
