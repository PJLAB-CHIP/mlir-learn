set -e  # Exit on error

SOURCE_DIR="."
BUILD_DIR="./build"
PROJECT_NO=""
BUILD_TYPE="Release"

# Check if stdout is terminal -> enable/disable colored output
if [ -t 1 ]; then 
    STDOUT_IS_TERMINAL=ON; export GTEST_COLOR=yes
else
    STDOUT_IS_TERMINAL=OFF; export GTEST_COLOR=no
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -S|--source-dir)
            SOURCE_DIR=$2; shift ;;
        -B|--build-dir)
            BUILD_DIR=$2; shift ;;
        Release|Debug|RelWithDebInfo|RD)
            BUILD_TYPE=${1/RD/RelWithDebInfo} ;;
        -n|--project-no)
            PROJECT_NO=$2; shift ;;
        --rm-build-dir)
            rm -rf $BUILD_DIR ;;
        *)
            # [TODO] Add detailed help message
            echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

CC=clang CXX=clang++ \
cmake -S . -G Ninja -B build \
    -DCMAKE_PREFIX_PATH="${LLVM_PROJECT_DIR}/build/lib/cmake/mlir" \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_SHARED_LIBS=OFF \
    -DPROJECT_NO=${PROJECT_NO} \
    -DSTDOUT_IS_TERMINAL=${STDOUT_IS_TERMINAL}

cmake --build build -j $(nproc)
