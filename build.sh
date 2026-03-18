#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BUILD_OPENVINO=0
BUILD_GENAI=0
INVALID_ARG=""
PARALLEL_JOBS=""

parse_arg() {
    local arg="$1"
    IFS=',' read -ra TARGETS <<< "$arg"
    for target in "${TARGETS[@]}"; do
        case "$target" in
            0)
                BUILD_OPENVINO=1
                ;;
            1)
                BUILD_GENAI=1
                ;;
            *)
                INVALID_ARG="$target"
                return 1
                ;;
        esac
    done
    return 0
}

usage() {
    echo "Usage:"
    echo "  ./build.sh            (build openvino + openvino.genai)"
    echo "  ./build.sh 0          (build openvino only)"
    echo "  ./build.sh 1          (build openvino.genai only)"
    echo "  ./build.sh 0,1        (build both)"
    echo "  ./build.sh 0 1        (build both)"
    echo ""
    echo "Options:"
    echo "  -j N                  Use N parallel jobs for building (default: auto)"
    echo "  --help                Show this help message"
}

ensure_build_tools() {
    local missing_tools=()
    
    # Check for C/C++ compiler
    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        missing_tools+=("gcc or clang")
    else
        echo "[INFO] C/C++ compiler found in PATH."
    fi
    
    # Check for cmake
    if ! command -v cmake &> /dev/null; then
        missing_tools+=("cmake")
    fi
    
    # Check for ninja
    if ! command -v ninja &> /dev/null; then
        missing_tools+=("ninja")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo "[ERROR] Missing required build tools: ${missing_tools[*]}"
        echo "        Please install them using your package manager:"
        echo "        Ubuntu/Debian: sudo apt-get install build-essential cmake ninja-build"
        echo "        Fedora/RHEL:   sudo dnf install gcc gcc-c++ cmake ninja-build"
        echo "        Arch:          sudo pacman -S base-devel cmake ninja"
        return 1
    fi
    
    echo "[INFO] All required build tools (compiler, cmake, ninja) found."
    return 0
}

# Parse arguments
if [ $# -eq 0 ]; then
    BUILD_OPENVINO=1
    BUILD_GENAI=1
else
    for arg in "$@"; do
        case "$arg" in
            --help|-h)
                usage
                exit 0
                ;;
            -j)
                shift
                PARALLEL_JOBS="$1"
                ;;
            -j*)
                PARALLEL_JOBS="${arg#-j}"
                ;;
            *)
                if ! parse_arg "$arg"; then
                    echo "[ERROR] Invalid build target: $INVALID_ARG"
                    usage
                    exit 1
                fi
                ;;
        esac
    done
fi

if [ "$BUILD_OPENVINO" -eq 0 ] && [ "$BUILD_GENAI" -eq 0 ]; then
    echo "[ERROR] No valid build target selected."
    usage
    exit 1
fi

# Ensure build tools are available
if ! ensure_build_tools; then
    exit 1
fi

# Determine parallel jobs
BUILD_ARGS=""
if [ -n "$PARALLEL_JOBS" ]; then
    BUILD_ARGS="-j $PARALLEL_JOBS"
    echo "[INFO] Using $PARALLEL_JOBS parallel jobs"
fi

# Build openvino
if [ "$BUILD_OPENVINO" -eq 1 ]; then
    echo "[BUILD] openvino"
    if ! cmake -S "$ROOT/openvino" -B "$ROOT/openvino/build" -G Ninja -DCMAKE_BUILD_TYPE=Release; then
        echo "[ERROR] CMake configure failed for openvino."
        exit 1
    fi
    
    if ! cmake --build "$ROOT/openvino/build" --config Release $BUILD_ARGS; then
        echo "[ERROR] Build failed for openvino."
        exit 1
    fi
    echo "[OK] OpenVINO build completed successfully"
fi

# Build openvino.genai
if [ "$BUILD_GENAI" -eq 1 ]; then
    echo "[BUILD] openvino.genai"
    
    # Check if openvino build exists
    if [ ! -d "$ROOT/openvino/build" ]; then
        echo "[ERROR] OpenVINO build directory not found at $ROOT/openvino/build"
        echo "        Please build OpenVINO first by running: ./build.sh 0"
        exit 1
    fi
    
    if ! cmake -S "$ROOT/openvino.genai" -B "$ROOT/openvino.genai/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR="$ROOT/openvino/build"; then
        echo "[ERROR] CMake configure failed for openvino.genai."
        echo "        Make sure openvino build directory exists and is valid."
        exit 1
    fi
    
    if ! cmake --build "$ROOT/openvino.genai/build" --config Release $BUILD_ARGS; then
        echo "[ERROR] Build failed for openvino.genai."
        echo ""
        echo "If you see compilation errors, the code may need patches."
        echo "Check the patches/ directory for available fixes."
        exit 1
    fi
    echo "[OK] OpenVINO GenAI build completed successfully"
fi

echo "[OK] Build finished."
exit 0
