@echo off

set "BUILD_CONFIG=RelWithDebInfo"

if /I "%~1"=="/h" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="/help" goto :usage
if /I "%~1"=="--help" goto :usage

if not "%~1"=="" (
	if /I "%~1"=="Release" (
		set "BUILD_CONFIG=Release"
	) else if /I "%~1"=="RelWithDebInfo" (
		set "BUILD_CONFIG=RelWithDebInfo"
	) else (
		echo [ERROR] Unsupported build config: %~1
		goto :usage
	)
)

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

set "REPO_ROOT=%SCRIPT_DIR%\.."
if "%REPO_ROOT:~-1%"=="\" set "REPO_ROOT=%REPO_ROOT:~0,-1%"

if not exist "%REPO_ROOT%\openvino" (
	set "REPO_ROOT=%SCRIPT_DIR%"
)

if not exist "%REPO_ROOT%\openvino" (
	echo [ERROR] Cannot find openvino directory under "%REPO_ROOT%"
	exit /b 1
)

if not exist "%REPO_ROOT%\openvino.genai" (
	echo [ERROR] Cannot find openvino.genai directory under "%REPO_ROOT%"
	exit /b 1
)

set OV_GENAI_USE_MODELING_API=1
set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
set OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
set OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym

set "PATH=%REPO_ROOT%\openvino\bin\intel64\%BUILD_CONFIG%;%REPO_ROOT%\openvino.genai\build\openvino_genai;%PATH%"

set "SAMPLES_DIR=%REPO_ROOT%\openvino.genai\build\bin\%BUILD_CONFIG%"
if not exist "%SAMPLES_DIR%" (
	set "SAMPLES_DIR=%REPO_ROOT%\openvino.genai\build\src\cpp\src\modeling\samples\%BUILD_CONFIG%"
)
if not exist "%SAMPLES_DIR%" (
	echo [ERROR] Sample directory not found for %BUILD_CONFIG%: "%SAMPLES_DIR%"
	echo [HINT] Build openvino.genai %BUILD_CONFIG% first, e.g. run build.bat %BUILD_CONFIG%
	exit /b 1
)

cd /d "%SAMPLES_DIR%"

@REM greedy_causal_lm.exe C:\data\models\Huggingface\Qwen3.5-35B-A3B-Base "ffmpeg is tool for " GPU 0 1 100
@REM modeling_qwen3_5.exe --model C:\data\models\Huggingface\Qwen3.5-35B-A3B-Base --mode text --prompt "write opencl gemm kernel and host code: " --output-tokens 30 --cache-model

goto :eof

:usage
echo Usage: %~nx0 [Release^|RelWithDebInfo]
echo   Release        Configure runtime for Release binaries
echo   RelWithDebInfo Configure runtime for RelWithDebInfo binaries (default)
exit /b 2
