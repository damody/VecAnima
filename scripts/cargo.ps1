# Process-local OpenCV / LLVM setup. No persistent PATH or system settings changed.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CargoArguments)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path $PSScriptRoot -Parent
Push-Location $taskRoot
try {
    $taskOpenCv = Join-Path $taskRoot '.tools/opencv-sdk'
    if (-not (Test-Path "$taskOpenCv/include/opencv2/core/version.hpp")) { throw 'Run scripts/install_opencv.py first' }
    $taskVersionHeader = Get-Content "$taskOpenCv/include/opencv2/core/version.hpp" -Raw
    foreach ($taskVersionPart in @(@('MAJOR', 4), @('MINOR', 14), @('REVISION', 0))) {
        if ($taskVersionHeader -notmatch "#define\s+CV_VERSION_$($taskVersionPart[0])\s+$($taskVersionPart[1])\b") {
            throw 'This project requires the pinned OpenCV 4.14.0 SDK'
        }
    }
    $taskLib = Get-ChildItem "$taskOpenCv/x64" -Recurse -Filter 'opencv_core*.lib' |
        Where-Object { $_.BaseName -notmatch 'd$' } | Select-Object -First 1
    if (-not $taskLib) { throw 'OpenCV import library not found' }
    $taskSuffix = $taskLib.BaseName -replace '^opencv_core', ''
    foreach ($taskModule in @('imgproc', 'imgcodecs', 'dnn')) {
        if (-not (Test-Path (Join-Path $taskLib.DirectoryName "opencv_$taskModule$taskSuffix.lib"))) {
            throw "Missing OpenCV module library: $taskModule"
        }
    }
    $env:OPENCV_LINK_LIBS = "opencv_core$taskSuffix,opencv_imgproc$taskSuffix,opencv_imgcodecs$taskSuffix,opencv_dnn$taskSuffix"
    $env:OPENCV_LINK_PATHS = $taskLib.DirectoryName
    $env:OPENCV_INCLUDE_PATHS = "$taskOpenCv/include"
    $env:OPENCV_MSVC_CRT = 'dynamic'
    if (-not $env:LIBCLANG_PATH) {
        $taskClang = Get-Command clang -ErrorAction Stop
        $taskClangDir = Split-Path $taskClang.Source -Parent
        if (-not (Test-Path "$taskClangDir/libclang.dll")) { throw 'Set LIBCLANG_PATH to a directory with libclang.dll' }
        $env:LIBCLANG_PATH = $taskClangDir
    }
    $taskBinDir = Join-Path $taskLib.Directory.Parent.FullName 'bin'
    $env:PATH = "$taskBinDir;$env:LIBCLANG_PATH;$env:PATH"
    & cargo @CargoArguments
    if ($LASTEXITCODE -ne 0) { throw "cargo failed with exit code $LASTEXITCODE" }
} finally { Pop-Location }

