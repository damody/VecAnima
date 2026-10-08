"""Archive compatible locked wheels in the project, including hashes and licenses.

Run with the installed project Python after setup. This downloads wheels only;
it never installs anything into the system Python environment.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
import zipfile

from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    dest = ROOT / "third_party"
    wheels = dest / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    ranks = {tag: i for i, tag in enumerate(sys_tags())}
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    records = []
    requirements = []
    for package in lock["package"]:
        if "registry" not in package["source"]:
            continue
        candidates = []
        for wheel in package.get("wheels", []):
            filename = unquote(Path(urlparse(wheel["url"]).path).name)
            tags = parse_wheel_filename(filename)[3]
            rank = min((ranks[t] for t in tags if t in ranks), default=None)
            if rank is not None:
                candidates.append((rank, filename, wheel))
        if not candidates:
            raise RuntimeError(f"No compatible wheel for {package['name']}; no source build attempted")
        _, filename, wheel = min(candidates, key=lambda value: (value[0], value[1]))
        target = wheels / filename
        expected = wheel["hash"].removeprefix("sha256:")
        if not target.exists() or sha256(target) != expected:
            print(f"Downloading {package['name']} {package['version']}", flush=True)
            partial = target.with_suffix(".whl.partial")
            curl = shutil.which("curl.exe")
            if curl:
                subprocess.run([curl, "--fail", "--location", "--silent", "--show-error",
                                "--connect-timeout", "20", "--max-time", "180", "--retry", "2",
                                "--output", str(partial), wheel["url"]], check=True)
            else:
                with urlopen(wheel["url"], timeout=60) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if sha256(partial) != expected:
                raise RuntimeError(f"SHA256 mismatch for {filename}")
            partial.replace(target)
        license_files = []
        with zipfile.ZipFile(target) as archive:
            for i, member in enumerate(archive.namelist()):
                name = Path(member).name
                if member.endswith("/") or not name.lower().startswith(("license", "licence", "copying", "notice")):
                    continue
                saved = dest / "licenses" / package["name"] / f"{i:04d}-{name}"
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(archive.read(member))
                license_files.append(saved.relative_to(dest).as_posix())
        meta = importlib.metadata.metadata(package["name"])
        records.append({"name": package["name"], "version": package["version"],
                        "file": target.relative_to(dest).as_posix(), "url": wheel["url"],
                        "sha256": expected, "bytes": target.stat().st_size,
                        "license": meta.get("License-Expression") or meta.get("License"),
                        "license_files": license_files})
        requirements.append(f"{package['name']}=={package['version']} --hash=sha256:{expected}")
    (dest / "requirements.txt").write_text("# Locked for this archived Python/platform wheel set.\n"
                                           + "\n".join(requirements) + "\n", encoding="utf-8")
    manifest = {"python": sys.version.split()[0], "platform": platform.platform(),
                "uv_lock_sha256": sha256(ROOT / "uv.lock"), "packages": records,
                "total_wheel_bytes": sum(p["bytes"] for p in records)}
    # Preserve the already available package manager in the project too.
    uv = shutil.which("uv")
    if uv:
        uv_target = ROOT / ".tools" / "uv.exe"
        uv_target.parent.mkdir(exist_ok=True)
        if Path(uv).resolve() != uv_target.resolve():
            shutil.copyfile(uv, uv_target)
        manifest["uv"] = {"file": ".tools/uv.exe", "sha256": sha256(uv_target),
                          "version": subprocess.check_output([str(uv_target), "--version"], text=True).strip()}
    (dest / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {len(records)} wheels, {manifest['total_wheel_bytes']:,} bytes", flush=True)


if __name__ == "__main__":
    main()
