#!/bin/bash
#
# Build script for faiss-rest-api
# Usage:
#   ./build.sh                  # normal Release build
#   ./build.sh --static         # attempt fully static single binary (experimental)
#   ./build.sh --debug          # Debug build
#   ./build.sh --no-clean       # don't delete build/ first
#

set -e

BUILD_TYPE="Release"
STATIC=0
CLEAN=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --static)
            STATIC=1
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --no-clean)
            CLEAN=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--static] [--debug] [--no-clean]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ $CLEAN -eq 1 ]; then
    echo "==> Cleaning build directory..."
    rm -rf build
fi

mkdir -p build
cd build

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
)

if [ $STATIC -eq 1 ]; then
    echo "==> Building for static single binary (experimental)..."
    CMAKE_ARGS+=(-DBUILD_STATIC=ON)
fi

echo "==> Configuring with CMake..."
cmake .. "${CMAKE_ARGS[@]}"

echo "==> Building..."
cmake --build . --parallel $(nproc)

echo ""
echo "==> Build complete!"
echo "    Binary: $(pwd)/faiss_rest_api"
echo ""

if [ $STATIC -eq 1 ]; then
    echo "Note: Static build attempted. Check with:"
    echo "    file $(pwd)/faiss_rest_api"
    echo "    ldd  $(pwd)/faiss_rest_api"
fi
