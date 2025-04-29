TARGET_NO=$1

CC=clang CXX=clang++ \
cmake -S . -G Ninja -B build \
    -DCMAKE_PREFIX_PATH="${LLVM_PROJECT_DIR}/build/lib/cmake/mlir" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DTARGET_NO=${TARGET_NO}

cmake --build build -j $(nproc)
