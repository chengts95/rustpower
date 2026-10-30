@echo off
setlocal
call "%~dp0configure.cmd"
if errorlevel 1 exit /b 1
set "OUT=%~1"
if not defined OUT set "OUT=%ROOT%\target\acpf-cuda-dll"
if not exist "%OUT%" mkdir "%OUT%"
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1
rem Keep NVCC's device frontend independent of the Windows console code page.
powershell -NoProfile -Command "$e = New-Object System.Text.UTF8Encoding($true); [IO.File]::WriteAllText('%OUT%\executor_utf8.cu', [IO.File]::ReadAllText('%ROOT%\native\acpf_cuda\src\executor.cu', [Text.Encoding]::UTF8), $e); [IO.File]::WriteAllText('%OUT%\newton_kernels_utf8.cu', [IO.File]::ReadAllText('%ROOT%\native\acpf_cuda\src\newton_kernels.cu', [Text.Encoding]::UTF8), $e); [IO.File]::WriteAllText('%OUT%\branch_kernels_utf8.cu', [IO.File]::ReadAllText('%ROOT%\native\acpf_cuda\src\branch_kernels.cu', [Text.Encoding]::UTF8), $e)"
if errorlevel 1 exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -shared -arch=%RUSTPOWER_CUDA_ARCH% -std=c++17 -O2 -Xcompiler /EHsc,/MD,/utf-8 ^
  -I"%ROOT%\native\acpf_cuda\include" -I"%CUDSS_DIR%\include" -L"%CUDSS_DIR%\lib" ^
  "%OUT%\executor_utf8.cu" "%OUT%\newton_kernels_utf8.cu" "%OUT%\branch_kernels_utf8.cu" ^
  -lcudss -Xlinker /IMPLIB:"%OUT%\acpf_cuda.lib" -o "%OUT%\acpf_cuda.dll"
if errorlevel 1 exit /b 1
rem Cargo copies these runtime dependencies beside the selected profile's binaries.
for %%F in (cudart64_12.dll cublas64_12.dll cublasLt64_12.dll nvrtc64_120_0.dll) do (
  if exist "%CUDA_PATH%\bin\%%F" copy /y "%CUDA_PATH%\bin\%%F" "%OUT%\%%F" >nul
)
for %%F in ("%CUDA_PATH%\bin\nvrtc-builtins64_*.dll" "%CUDA_PATH%\bin\nvJitLink_*.dll") do (
  if exist "%%F" copy /y "%%F" "%OUT%\" >nul
)
for %%F in (cudss64_1.dll cudss_mtlayer_vcomp14064_1.dll) do (
  if exist "%CUDSS_DIR%\bin\%%F" copy /y "%CUDSS_DIR%\bin\%%F" "%OUT%\%%F" >nul
)
exit /b 0
