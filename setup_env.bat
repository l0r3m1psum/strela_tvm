@echo off

if not exist venv (
	python -m venv venv || exit /b 1
)

REM We activate the virtual environment first so that we can restore the
REM previous environment when deactivating it.
call venv\Scripts\activate || exit /b 1

REM This is necessary both for reproducibility and to avoid that a Visual Studio
REM update changes the default version used.
FOR /F "tokens=*" %%a in ('"%PROGRAMFILES(X86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [17.0^,18.0^) -latest -property installationPath') do SET vspath=%%a
if %errorlevel% neq 0 (
	echo Can't find vswhere, probably Visual Studio is not installed. 1>&2
	exit /b 1
)
if not defined vspath (
	echo Can't find Visual Studio 2022 (Version 17^). 1>&2
	REM https://archive.org/details/vs_community__e8aae2bc1239469a8cb34a7eeb742747
	exit /b 1
)
call "%vspath%\VC\Auxiliary\Build\vcvarsall.bat" x64 || exit /b 1

set build_type=Debug

set CXX=clang++
REM LLVM on Windows by default for x86_64 targets Linux...
set TVM_WIN_TARGET=x86_64-pc-windows-msvc
set "PATH=%PATH%;%cd%\3rdparty\llvm-project\llvm\build\%build_type%\bin"
set "TVM_HOME=%CD%\3rdparty\tvm"
set "PYTHONPATH=%TVM_HOME%\python;%PYTHONPATH%"