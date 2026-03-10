@echo off
setlocal EnableExtensions

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for %%I in ("%SCRIPT_DIR%\..") do set "ROOT_DIR=%%~fI"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set "OPENVINO_DIR=%ROOT_DIR%\openvino"
set "GENAI_DIR=%ROOT_DIR%\openvino.genai"
set "OPENVINO_URL=https://github.com/liangali/openvino.git"
set "GENAI_URL=https://github.com/liangali/openvino.genai"
set "TARGET_BRANCH=explicit-modeling"
set "PATCH_SCRIPT=%SCRIPT_DIR%\scripts\apply_onednn_patch.bat"

echo [INFO] repo.bat path     : %SCRIPT_DIR%
echo [INFO] workspace root    : %ROOT_DIR%

call :ensure_git
if errorlevel 1 exit /b 1

call :ensure_repo "%OPENVINO_DIR%" "%OPENVINO_URL%" "%TARGET_BRANCH%" "openvino"
if errorlevel 1 exit /b 1

call :ensure_repo "%GENAI_DIR%" "%GENAI_URL%" "%TARGET_BRANCH%" "openvino.genai"
if errorlevel 1 exit /b 1

if not exist "%PATCH_SCRIPT%" (
    echo [ERROR] Patch helper not found: %PATCH_SCRIPT%
    exit /b 1
)

echo [PATCH] onednn_gpu
call "%PATCH_SCRIPT%"
if errorlevel 1 (
    echo [ERROR] Failed to apply onednn_gpu patch.
    exit /b 1
)

echo [OK] Repository setup finished.
echo      Ready to build from: %ROOT_DIR%
exit /b 0

:ensure_git
where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] git.exe not found in PATH.
    exit /b 1
)
exit /b 0

:ensure_repo
set "REPO_DIR=%~1"
set "REPO_URL=%~2"
set "REPO_BRANCH=%~3"
set "REPO_NAME=%~4"

if exist "%REPO_DIR%\.git" (
    echo [INFO] %REPO_NAME% already exists: %REPO_DIR%
) else (
    if exist "%REPO_DIR%" (
        echo [ERROR] %REPO_NAME% path exists but is not a git repo: %REPO_DIR%
        exit /b 1
    )

    echo [CLONE] %REPO_NAME%
    git clone "%REPO_URL%" "%REPO_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to clone %REPO_NAME%.
        exit /b 1
    )
)

pushd "%REPO_DIR%"

echo [CHECKOUT] %REPO_NAME% -^> %REPO_BRANCH%
git fetch --all --tags
if errorlevel 1 (
    echo [ERROR] Failed to fetch %REPO_NAME%.
    popd
    exit /b 1
)

git checkout "%REPO_BRANCH%"
if errorlevel 1 (
    echo [ERROR] Failed to checkout branch %REPO_BRANCH% for %REPO_NAME%.
    popd
    exit /b 1
)

echo [SUBMODULE] %REPO_NAME%
git submodule update --init --recursive
if errorlevel 1 (
    echo [ERROR] Failed to init submodules for %REPO_NAME%.
    popd
    exit /b 1
)

popd
exit /b 0
