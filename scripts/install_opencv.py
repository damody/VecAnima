"""Verify and build the user-supplied OpenCV source ZIP locally (no downloads)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    spec = json.loads((ROOT / "scripts/opencv-lock.json").read_text())
    archive = ROOT / spec["archive"]
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != spec["sha256"]:
            raise RuntimeError("OpenCV source ZIP hash mismatch")
    sources = ROOT / ".tools/opencv-src"
    sources.mkdir(parents=True, exist_ok=True)
    source = sources / f"opencv-{spec['version']}"
    marker = source / ".vecanima-extracted-sha256"
    if not marker.exists() or marker.read_text() != spec["sha256"]:
        print("Extracting verified OpenCV source ZIP", flush=True)
        with zipfile.ZipFile(archive) as zipped:
            for item in zipped.infolist():
                destination = (sources / item.filename).resolve()
                if not destination.is_relative_to(source.resolve()):
                    raise RuntimeError(f"Unsafe ZIP member: {item.filename}")
            zipped.extractall(sources)
        marker.write_text(spec["sha256"])
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("CMake must be installed and on PATH")
    build = ROOT / ".tools/opencv-build"
    install = ROOT / ".tools/opencv-sdk"
    command = [cmake, "-S", str(source), "-B", str(build),
               "-G", "Visual Studio 18 2026", "-A", "x64",
               f"-DCMAKE_INSTALL_PREFIX={install}", "-DBUILD_LIST=core,imgproc,imgcodecs,dnn",
               "-DBUILD_opencv_world=OFF", "-DBUILD_SHARED_LIBS=ON",
               "-DCPU_BASELINE=SSE2", "-DCPU_DISPATCH=", "-DOPENCV_DOWNLOAD_PATH=" + str(ROOT / ".tools/opencv-downloads")]
    for name in ("BUILD_TESTS", "BUILD_PERF_TESTS", "BUILD_EXAMPLES", "BUILD_opencv_apps",
                 "BUILD_JAVA", "BUILD_opencv_python2", "BUILD_opencv_python3", "WITH_IPP",
                 "WITH_ITT", "WITH_OPENCL", "WITH_FFMPEG", "WITH_MSMF", "WITH_DSHOW",
                 "WITH_CUDA", "WITH_OPENEXR", "WITH_WEBP", "WITH_TIFF", "WITH_JASPER",
                 "WITH_AVIF", "WITH_JPEGXL", "WITH_OPENJPEG", "WITH_VTK"):
        command.append(f"-D{name}=OFF")
    for name in ("BUILD_PNG", "BUILD_JPEG", "BUILD_ZLIB", "WITH_PROTOBUF", "BUILD_PROTOBUF"):
        command.append(f"-D{name}=ON")
    subprocess.run(command, check=True)
    subprocess.run([cmake, "--build", str(build), "--config", "Release", "--parallel", str(args.jobs)], check=True)
    subprocess.run([cmake, "--install", str(build), "--config", "Release"], check=True)
    (install / "vecanima-build.json").write_text(json.dumps({
        "source": spec, "generator": "Visual Studio 18 2026", "configuration": "Release",
        "modules": ["core", "imgproc", "imgcodecs", "dnn"], "configure_command": command,
    }, indent=2) + "\n")
    print(f"OpenCV {spec['version']} SDK ready: {install}", flush=True)


if __name__ == "__main__":
    main()
