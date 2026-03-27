@echo off
REM ============================================================
REM Run Youtu-LLM-2B with full environment setup + Python script
REM ============================================================
REM Usage:
REM   run_youtu_llm_python.bat [model_path] [--device GPU] [--prompt "your question"]
REM Examples:
REM   run_youtu_llm_python.bat
REM   run_youtu_llm_python.bat "C:\models\Youtu-LLM-2B"
REM   run_youtu_llm_python.bat "C:\models\Youtu-LLM-2B" --device GPU --max-new-tokens 256
REM ============================================================

setlocal
set "SCRIPT_DIR=%~dp0"
set "OPENVINO_ROOT=%SCRIPT_DIR%..\..\"
cd /d "%SCRIPT_DIR%"

REM 1. Setup OpenVINO environment (if setupvars exists)
@REM if exist "%OPENVINO_ROOT%openvino\scripts\setupvars\setupvars.bat" (
@REM     call "%OPENVINO_ROOT%openvino\scripts\setupvars\setupvars.bat"
@REM )

REM 2. Youtu-LLM requires modeling API
set "OV_GENAI_USE_MODELING_API=1"

REM 2.1 Enable in-flight quantization by default for large Qwen3.5 models.
REM      You can override from the shell before running this .bat.
if "%OV_GENAI_INFLIGHT_QUANT_MODE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym"
)
if "%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128"
)
if "%OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym"
)

echo [INFO] OV_GENAI_INFLIGHT_QUANT_MODE=%OV_GENAI_INFLIGHT_QUANT_MODE%
echo [INFO] OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%
echo [INFO] OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=%OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE%
echo.

REM 3. Setup Python paths for openvino and openvino_genai packages
REM    Try to find openvino Python package from current build
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
    echo [WARN] OpenVINO Python package not found in build. Please ensure you have built openvino Python bindings,
    echo        or run: pip install openvino
) else (
    set "PYTHONPATH=%OV_PYTHON%;%PYTHONPATH%"
)

REM Add openvino.genai to PYTHONPATH
set "PYTHONPATH=%OPENVINO_ROOT%openvino.genai\build;%PYTHONPATH%"

REM 4. Setup DLL paths (openvino package_utils will read OPENVINO_LIB_PATHS and add_dll_directory)
set "OPENVINO_LIB_PATHS=%OPENVINO_ROOT%openvino\bin\intel64\Release"

if exist "%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin" (
    set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin"
)
if exist "%OPENVINO_ROOT%openvino\build\install\runtime\bin\intel64\Release" (
    set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%openvino\build\install\runtime\bin\intel64\Release"
)

REM Add to PATH for immediate DLL loading
set "PATH=%OPENVINO_ROOT%openvino.genai\build\openvino_genai;%PATH%"
set "PATH=%OPENVINO_ROOT%openvino\bin\intel64\Release;%PATH%"

if exist "%OPENVINO_ROOT%openvino.genai\build\bin\Release" (
    set "PATH=%OPENVINO_ROOT%openvino.genai\build\bin\Release;%PATH%"
)
if exist "%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin" (
    set "PATH=%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin;%PATH%"
)

REM 5. Run Python script with all arguments
echo Running: python run_text_qwen3_5.py %*
echo.
python run_text_qwen3_5.py %*

endlocal
