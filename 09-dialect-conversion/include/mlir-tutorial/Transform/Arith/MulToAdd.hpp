#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial
{
#define GEN_PASS_DECL_MULTOADD
#include "mlir-tutorial/Transform/Arith/Passes.hpp.inc"

}  // namespace mlir::tutorial
