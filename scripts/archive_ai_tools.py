"""Archive the installed local model-conversion toolchain from official indexes."""
import hashlib
from html.parser import HTMLParser
from importlib.metadata import version
import json
from pathlib import Path
import urllib.parse
import urllib.request
from urllib.error import HTTPError
import zipfile
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "third_party/wheels"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href")


def main():
    DEST.mkdir(exist_ok=True)
    ranks = {t: i for i, t in enumerate(sys_tags())}
    packages = ("torch", "onnx", "filelock", "fsspec", "jinja2", "markupsafe",
                "mpmath", "setuptools", "sympy", "typing-extensions", "ml-dtypes",
                "protobuf", "numpy", "networkx", "packaging")
    rows = []
    for name in packages:
        installed = version(name)
        if name == "torch":
            parser = Links()
            parser.feed(urllib.request.urlopen("https://download.pytorch.org/whl/cpu/torch/").read().decode())
            links = [u for u in parser.links if f"torch-{installed}-cp312-cp312-win_amd64.whl" in urllib.parse.unquote(u)]
            if len(links) != 1:
                raise RuntimeError(f"Expected one official Torch wheel, found {len(links)}")
            link = links[0]
            url, _, fragment = link.partition("#")
            filename = urllib.parse.unquote(url.rsplit("/", 1)[1])
            expected = urllib.parse.parse_qs(fragment)["sha256"][0]
        else:
            release = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{installed}/json"))
            choices = []
            for row in release["urls"]:
                if not row["filename"].endswith(".whl"):
                    continue
                tags = parse_wheel_filename(row["filename"])[3]
                rank = min((ranks[t] for t in tags if t in ranks), default=None)
                if rank is not None:
                    choices.append((rank, row))
            row = min(choices, key=lambda c: c[0])[1]
            filename, url, expected = row["filename"], row["url"], row["digests"]["sha256"]
        path = DEST / filename
        if not path.exists():
            temporary = path.with_suffix(".download")
            try:
                urllib.request.urlretrieve(url, temporary)
            except HTTPError as error:
                if error.code != 403 or not url.startswith("https://download-r2.pytorch.org/"):
                    raise
                # Same official artifact and mandatory published SHA256.
                url = url.replace("https://download-r2.pytorch.org/", "https://download.pytorch.org/", 1)
                urllib.request.urlretrieve(url, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch: {filename}")
            temporary.rename(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Archived checksum mismatch: {filename}")
        license_root = (ROOT / "third_party/licenses" / f"ai-{name}").resolve()
        with zipfile.ZipFile(path) as wheel:
            for member in wheel.namelist():
                if member.endswith("/") or not any(k in Path(member).name.upper() for k in ("LICENSE", "COPYING", "NOTICE")):
                    continue
                target = (license_root / member).resolve()
                if not target.is_relative_to(license_root):
                    raise RuntimeError("Unsafe license member")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(wheel.read(member))
        rows.append({"name": name, "version": installed, "filename": filename,
                     "url": url, "sha256": expected})
        print(filename, flush=True)
    (ROOT / "third_party/anilines/tools-manifest.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (ROOT / "third_party/anilines/tools-requirements.txt").write_text(
        "\n".join(f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}" for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
