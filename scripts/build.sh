CC=clang CXX=clang++ \
cmake -S . -G Ninja -B build \
    -DCMAKE_PREFIX_PATH="$ENV{LLVM_PROJECT_DIR}/lib/cmake/llvm;$ENV{MLIR_PROJECT_DIR}/lib/cmake/mlir" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j $(nproc)
