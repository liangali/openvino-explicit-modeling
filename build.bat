@echo off
setlocal

set "BUILD_CONFIGS="

if /I "%~1"=="/h" goto :usage
if /I "%~1"=="-h" goto :usage
if /I "%~1"=="/help" goto :usage
if /I "%~1"=="--help" goto :usage

if "%~1"=="" (
	echo [ERROR] Build config is required.
	goto :usage
)

if not "%~1"=="" (
	if /I "%~1"=="Release" (
		set "BUILD_CONFIGS=Release"
	) else if /I "%~1"=="RelWithDebInfo" (
		set "BUILD_CONFIGS=RelWithDebInfo"
	) else (
		echo [ERROR] Unsupported build config: %~1
		goto :usage
	)
)

echo [build] Configs: %BUILD_CONFIGS%

set "OPENVINO_CMAKE_EXTRA_ARGS="
set "BUILD_PARALLEL=16"
echo %BUILD_CONFIGS% | findstr /I /C:"RelWithDebInfo" >nul
if not errorlevel 1 (
	set "OPENVINO_CMAKE_EXTRA_ARGS=-DENABLE_DEBUG_CAPS=ON -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON -DCMAKE_C_FLAGS_RELWITHDEBINFO=/FS -DCMAKE_CXX_FLAGS_RELWITHDEBINFO=/FS"
	set "BUILD_PARALLEL=8"
)
if defined OPENVINO_CMAKE_EXTRA_ARGS echo [build] RelWithDebInfo detected, enabling extra OpenVINO CMake args: %OPENVINO_CMAKE_EXTRA_ARGS%
echo [build] Parallel jobs: %BUILD_PARALLEL%

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "ROOT_DIR=%SCRIPT_DIR%\.."
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

@REM # Apply onednn_gpu patch (local only, not committed - for PR compliance)
call "%SCRIPT_DIR%\scripts\apply_onednn_patch.bat"
if %errorlevel% neq 0 exit /b %errorlevel%

@REM # build openvino
if not exist "%ROOT_DIR%\openvino\build" mkdir "%ROOT_DIR%\openvino\build"
pushd "%ROOT_DIR%\openvino\build"
cmake -DCMAKE_BUILD_TYPE=Release %OPENVINO_CMAKE_EXTRA_ARGS% ..
for %%C in (%BUILD_CONFIGS%) do (
	echo [build] openvino - %%C
	cmake --build . --config %%C --verbose -j%BUILD_PARALLEL%
	if errorlevel 1 exit /b 1
)
popd

@REM # build openvino.genai
if not exist "%ROOT_DIR%\openvino.genai\build" mkdir "%ROOT_DIR%\openvino.genai\build"
pushd "%ROOT_DIR%\openvino.genai\build"
cmake -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR="%ROOT_DIR%\openvino\build" ..
for %%C in (%BUILD_CONFIGS%) do (
	echo [build] openvino.genai - %%C
	cmake --build . --config %%C --verbose -j%BUILD_PARALLEL%
	if errorlevel 1 exit /b 1
)
popd

exit /b 0

:usage
echo Usage: %~nx0 [Release^|RelWithDebInfo]
echo   Release        Build Release only
echo   RelWithDebInfo Build RelWithDebInfo only
exit /b 2
