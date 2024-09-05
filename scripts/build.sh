CC=clang CXX=clang++ \
cmake -S . -G Ninja -B build \
    -DCMAKE_PREFIX_PATH="$ENV{MLIR_HOME}/lib/cmake/mlir;$ENV{LLVM_HOME}/lib/cmake/llvm" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j $(nproc)
