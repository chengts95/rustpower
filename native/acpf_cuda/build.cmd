@echo off
setlocal
call "%~dp0configure.cmd"
if errorlevel 1 exit /b 1
set "OUT=%ROOT%\target\acpf-cuda"
if not exist "%OUT%" mkdir "%OUT%"
call "%VCVARS%" >nul
if errorlevel 1 exit /b 1
rem NVCC's device frontend needs a BOM to read UTF-8 comments under the local code page.
powershell -NoProfile -Command "$e = New-Object System.Text.UTF8Encoding($true); [IO.File]::WriteAllText('%OUT%\kernels_utf8.cu', [IO.File]::ReadAllText('%ROOT%\native\acpf_cuda\src\kernels.cu', [Text.Encoding]::UTF8), $e); [IO.File]::WriteAllText('%OUT%\newton_kernels_utf8.cu', [IO.File]::ReadAllText('%ROOT%\native\acpf_cuda\src\newton_kernels.cu', [Text.Encoding]::UTF8), $e)"
if errorlevel 1 exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -ptx -arch=%RUSTPOWER_CUDA_PTX% -std=c++17 -lineinfo "%OUT%\kernels_utf8.cu" -o "%OUT%\kernels.ptx"
if errorlevel 1 exit /b 1
"%CUDA_PATH%\bin\nvcc.exe" -ptx -arch=%RUSTPOWER_CUDA_PTX% -std=c++17 -lineinfo "%OUT%\newton_kernels_utf8.cu" -o "%OUT%\newton_kernels.ptx"
if errorlevel 1 exit /b 1
cl /nologo /std:c++17 /EHsc /O2 /W4 /permissive- /utf-8 ^
  /I"%CUDA_PATH%\include" /I"%CUDSS_DIR%\include" ^
  "%ROOT%\native\acpf_cuda\src\main.cpp" ^
  /Fo:"%OUT%\main.obj" /Fe:"%OUT%\acpf_batch_demo.exe"
exit /b %errorlevel%
