@echo off
REM ============================================================
REM Run VLMPipeline with Qwen3.5 VL model (e.g. Qwen3-VL-30B-A3B)
REM 重要: openvino .pyd 为 cp312，必须使用 Python 3.12
REM ============================================================
REM Usage:
REM   run_vlm_qwen3_5.bat [model_path] [image_path] [--device GPU] [--prompt "your question"]
REM   run_vlm_qwen3_5.bat [model_path] --test-image
REM Examples:
REM   run_vlm_qwen3_5.bat "D:\Data\models\Huggingface\Qwen3-VL-30B-A3B-Instruct" "test.jpg"
REM   run_vlm_qwen3_5.bat "D:\Data\models\Huggingface\Qwen3.5-35B-A3B" "image.png" --device GPU
REM   run_vlm_qwen3_5.bat "D:\Data\models\...\Qwen3-VL-30B-A3B" --test-image --prompt "What is in this image?"
REM ============================================================
REM Requires: pip install pillow
REM Model: Qwen3.5 / Qwen3.5-MoE VL (config.json: model_type qwen3_5/qwen3_5_moe + vision_config)
REM ============================================================

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
cd ..\..
set "OPENVINO_ROOT=%CD%"
cd /d "%SCRIPT_DIR%"

REM 1. 排除 site-packages，避免加载 pip 安装的 openvino（与自编译版本冲突）
set "PYTHONNOUSERSITE=1"

REM 2. Modeling API required for Qwen3.5 VLM
set "OV_GENAI_USE_MODELING_API=1"

REM 3. In-flight quantization (for large models)
if "%OV_GENAI_INFLIGHT_QUANT_MODE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym"
)
if "%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128"
)
if "%OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym"
)

echo [INFO] OV_GENAI_USE_MODELING_API=1
echo [INFO] OV_GENAI_INFLIGHT_QUANT_MODE=%OV_GENAI_INFLIGHT_QUANT_MODE%
echo [INFO] OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%
echo.

REM 4. Python paths
set "OV_PYTHON="
if exist "%OPENVINO_ROOT%openvino\bin\intel64\Release\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\bin\intel64\Release\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\lib\Release\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\lib\Release\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\lib\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\lib\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\bin\Release\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\bin\Release\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\install\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\install\python"
)

if "%OV_PYTHON%"=="" (
    echo [WARN] OpenVINO Python not found in build. Use system openvino if installed.
) else (
    set "PYTHONPATH=%OV_PYTHON%;%PYTHONPATH%"
)

set "PYTHONPATH=%OPENVINO_ROOT%openvino.genai\build;%PYTHONPATH%"

REM 5. DLL paths
set "OPENVINO_LIB_PATHS=%OPENVINO_ROOT%openvino\bin\intel64\Release"
if exist "%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin" (
    set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin"
)
if exist "%OPENVINO_ROOT%openvino\build\install\runtime\bin\intel64\Release" (
    set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%openvino\build\install\runtime\bin\intel64\Release"
)

set "PATH=%OPENVINO_ROOT%openvino.genai\build\openvino_genai;%PATH%"
set "PATH=%OPENVINO_ROOT%openvino\bin\intel64\Release;%PATH%"
if exist "%OPENVINO_ROOT%openvino.genai\build\bin\Release" (
    set "PATH=%OPENVINO_ROOT%openvino.genai\build\bin\Release;%PATH%"
)
if exist "%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin" (
    set "PATH=%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin;%PATH%"
)

REM 6. Run Python script
echo Running: python run_vlm_qwen3_5.py %*
echo.
python run_vlm_qwen3_5.py %*

endlocal
