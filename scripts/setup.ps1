param([switch]$Offline)
$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $taskUv = if (Test-Path '.tools/uv.exe') { (Resolve-Path '.tools/uv.exe').Path } else { 'uv' }
    if ($Offline) {
        if (-not (Test-Path '.venv/Scripts/python.exe')) {
            & $taskUv venv --python 3.12 --no-python-downloads .venv
            if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required for offline setup' }
        }
        & .venv/Scripts/python.exe scripts/verify_dependencies.py
        if ($LASTEXITCODE -ne 0) { throw 'Offline archive verification failed' }
        & $taskUv pip install --python .venv/Scripts/python.exe --offline --no-index --require-hashes --find-links third_party/wheels -r third_party/requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Offline dependency installation failed' }
        & $taskUv pip install --python .venv/Scripts/python.exe --offline --no-index --no-deps --no-build-isolation --editable .
        if ($LASTEXITCODE -ne 0) { throw 'Offline project installation failed' }
    } else {
        & $taskUv sync --frozen --extra dev
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
    }
    if ($Offline) {
        & .venv/Scripts/python.exe scripts/install_ffmpeg.py --offline
    } else {
        & .venv/Scripts/python.exe scripts/install_ffmpeg.py
    }
    if ($LASTEXITCODE -ne 0) { throw 'FFmpeg installation failed' }
    & .venv/Scripts/python.exe -m vecanima doctor
    if ($LASTEXITCODE -ne 0) { throw 'Environment check failed' }
} finally { Pop-Location }
