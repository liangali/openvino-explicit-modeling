@REM @echo off
@REM setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for %%I in ("%SCRIPT_DIR%\..") do set "ROOT_DIR=%%~fI"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "OPENVINO_BIN=%ROOT_DIR%\openvino\bin\intel64\Release"
set "TBB_BIN=%ROOT_DIR%\openvino\temp\Windows_AMD64\tbb\bin"
set "GENAI_DLL_DIR=%ROOT_DIR%\openvino.genai\build\openvino_genai"
set "GENAI_BIN_DIR=%ROOT_DIR%\openvino.genai\build\bin"
set "MODELING_EXE=%GENAI_BIN_DIR%\modeling_qwen3_5.exe"
set "GREEDY_LM_EXE=%GENAI_BIN_DIR%\greedy_causal_lm.exe"

if not exist "%OPENVINO_BIN%" (
    echo [ERROR] OpenVINO bin directory not found: %OPENVINO_BIN%
    exit /b 1
)

if not exist "%GENAI_DLL_DIR%" (
    echo [ERROR] OpenVINO GenAI DLL directory not found: %GENAI_DLL_DIR%
    exit /b 1
)

if not exist "%GENAI_BIN_DIR%" (
    echo [ERROR] OpenVINO GenAI bin directory not found: %GENAI_BIN_DIR%
    exit /b 1
)

if not exist "%MODELING_EXE%" (
    echo [ERROR] Executable not found: %MODELING_EXE%
    exit /b 1
)

set OV_GENAI_SAVE_OV_MODEL=1
set OV_GENAI_USE_MODELING_API=1
set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
set OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
set OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym
set "PATH=%OPENVINO_BIN%;%TBB_BIN%;%GENAI_DLL_DIR%;%GENAI_BIN_DIR%;%PATH%"

cd %GENAI_BIN_DIR%

echo modeling_qwen3_5.exe --model C:\data\models\Huggingface\Qwen3.5-35B-A3B --cache-model --mode text --prompt "write opencl gemm kernel and host code" --output-tokens 300
echo greedy_causal_lm.exe C:\data\models\Huggingface\Qwen3.5-35B-A3B "write opencl gemm kernel and host code" GPU 1 3 300 int4_asym 128 int4_asym
