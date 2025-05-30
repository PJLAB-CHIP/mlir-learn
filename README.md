# MLIR Learn

Tutorial: https://github.com/j2kun/mlir-tutorial

## 1. Quick Start

### 1.1. Build LLVM with MLIR Enabled

You should have clang, clang++, cmake and ninja installed on your system.

Clone LLVM to your local machine: 

```bash
git clone https://github.com/llvm/llvm-project && cd ./llvm-project
```

Build LLVM with MLIR enabled:

```bash
CC=clang CXX=clang++ \
cmake -S . -B ./build -G Ninja ./llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_COLOR_DIAGNOSTICS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_BUILD_EXAMPLES=OFF \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLVM_ENABLE_LLD=ON \
    -DBUILD_SHARED_LIBS=ON

cmake --build ./build --target check-mlir -j $(nproc)
```

Set the environment variable `LLVM_PROJECT_DIR` and `MLIR_PROJECT_DIR` which will be used when building **this** project:

```bash
export LLVM_PROJECT_DIR="/path/to/llvm-project"
export PATH="$LLVM_PROJECT_DIR/build/bin:$PATH"
```

### 1.2. Build This Project

```bash
bash scripts/build.sh <TARGET_NO>
```

### 1.3. Test with `llvm-lit`

```bash
llvm-lit -v <TARGET_DIR>/test
```

