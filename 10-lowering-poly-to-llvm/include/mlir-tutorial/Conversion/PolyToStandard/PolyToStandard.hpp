#pragma once

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>

namespace mlir::tutorial::poly
{

#define GEN_PASS_DECL
#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp.inc"

#define GEN_PASS_REGISTRATION
#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp.inc"

}  // namespace mlir::tutorial::poly
