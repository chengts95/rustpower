@echo off
rem Shared local toolchain discovery. CUDA_PATH/CUDSS_DIR may be supplied explicitly.
set "ROOT=%~dp0..\.."
if not defined CUDA_PATH (
  for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_PATH=%%D"
)
if not exist "%CUDA_PATH%\bin\nvcc.exe" (
  echo CUDA toolkit not found. Set CUDA_PATH to its installation root. 1>&2
  exit /b 1
)
if not exist "%CUDSS_DIR%\lib\cudss.lib" (
  echo cuDSS not found. Set CUDSS_DIR to a directory containing include, lib and bin. 1>&2
  exit /b 1
)
if not defined RUSTPOWER_CUDA_ARCH set "RUSTPOWER_CUDA_ARCH=sm_75"
set "RUSTPOWER_CUDA_PTX=%RUSTPOWER_CUDA_ARCH:sm_=compute_%"
set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if defined ProgramFiles(x86) set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo Visual Studio vswhere.exe not found. 1>&2
  exit /b 1
)
set "VCVARS_FILE=%TEMP%\rustpower-vcvars-%RANDOM%-%RANDOM%.txt"
"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" > "%VCVARS_FILE%"
set /p VCVARS=<"%VCVARS_FILE%"
del /q "%VCVARS_FILE%" >nul 2>&1
if not exist "%VCVARS%" (
  echo Visual C++ x64 toolchain not found. 1>&2
  exit /b 1
)
exit /b 0
