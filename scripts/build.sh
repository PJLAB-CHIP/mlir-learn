
CC=clang CXX=clang++ \
cmake -S . -G Ninja -B build \
    -DCMAKE_PREFIX_PATH="$ENV{MLIR_HOME}/lib/cmake/mlir;$ENV{LLVM_HOME}/lib/cmake/llvm"

cmake --build build -j $(nproc)
