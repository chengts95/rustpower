$ErrorActionPreference = "Stop"

# 同一张 Jacobian、同一个右端项，比较才会像爱莉希雅期待的那样公平呀♪
$Root = $PSScriptRoot
$Packages = Join-Path $Root "python-packages"
$Nvidia = Join-Path $Packages "nvidia"
$Results = Join-Path $Root "results"

$env:PYTHONPATH = $Packages
$env:CUDA_PATH = Join-Path $Nvidia "cuda_runtime"
$env:Path = @(
    (Join-Path $Nvidia "cu12\bin"),
    (Join-Path $Nvidia "cuda_nvrtc\bin"),
    (Join-Path $Nvidia "cuda_runtime\bin"),
    (Join-Path $Nvidia "cublas\bin"),
    $env:Path
) -join ";"

New-Item -ItemType Directory -Force -Path $Results | Out-Null
python (Join-Path $Root "cudss_jacobian_benchmark.py") 2>&1 |
    Tee-Object -FilePath (Join-Path $Results "cudss-pegase9241.log")
exit $LASTEXITCODE
