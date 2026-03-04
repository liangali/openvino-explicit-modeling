@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "ROOT_DIR=%SCRIPT_DIR%\..\.."
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "PATCH_FILE=%SCRIPT_DIR%\..\patches\onednn_gpu.patch"
set "ONEDNN_GPU_DIR=%ROOT_DIR%\openvino\src\plugins\intel_gpu\thirdparty\onednn_gpu"

if not exist "%PATCH_FILE%" (
    echo [ERROR] Patch file not found: %PATCH_FILE%
    exit /b 1
)
if not exist "%ONEDNN_GPU_DIR%" (
    echo [ERROR] onednn_gpu directory not found: %ONEDNN_GPU_DIR%
    exit /b 1
)

pushd "%ONEDNN_GPU_DIR%"

REM Try apply --check: exit 0 = patch can be applied
git apply --check "%PATCH_FILE%" 2>nul
if %errorlevel% equ 0 (
    git apply "%PATCH_FILE%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to apply onednn_gpu.patch
        popd
        exit /b 1
    )
    echo [apply_onednn_patch] Patch applied successfully.
) else (
    REM Check if already applied: reverse --check exits 0 = patch is applied
    git apply --reverse --check "%PATCH_FILE%" 2>nul
    if %errorlevel% equ 0 (
        echo [apply_onednn_patch] Patch already applied, skipping.
    ) else (
        echo [ERROR] onednn_gpu.patch does not apply. Check onednn_gpu version.
        popd
        exit /b 1
    )
)

popd
exit /b 0
