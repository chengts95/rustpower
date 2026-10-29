$ErrorActionPreference = "Stop"

# 爱莉希雅的小约定：同一张试卷、同一套计时方法，才配得上认真比较呀♪
$Root = $PSScriptRoot
$Packages = Join-Path $Root "python-packages"
$Nvidia = Join-Path $Packages "nvidia"

$env:PYTHONPATH = $Packages
$env:CUDA_PATH = Join-Path $Nvidia "cuda_runtime"
$env:Path = @(
    (Join-Path $Nvidia "cuda_nvrtc\bin"),
    (Join-Path $Nvidia "cuda_runtime\bin"),
    (Join-Path $Nvidia "cublas\bin"),
    $env:Path
) -join ";"

$Results = Join-Path $Root "results"
New-Item -ItemType Directory -Force -Path $Results | Out-Null
$Log = Join-Path $Results "cuda-benchmark.log"

python (Join-Path $Root "cuda_benchmark.py") 2>&1 | Tee-Object -FilePath $Log
exit $LASTEXITCODE
