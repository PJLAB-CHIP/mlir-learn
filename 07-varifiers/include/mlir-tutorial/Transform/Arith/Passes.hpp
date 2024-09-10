#pragma once

#include "mlir-tutorial/Transform/Arith/MulToAdd.hpp"
#include "llvm/Support/Compiler.h"

namespace mlir::tutorial
{

#define GEN_PASS_REGISTRATION
#include "mlir-tutorial/Transform/Arith/Passes.hpp.inc"

}  // namespace mlir::tutorial