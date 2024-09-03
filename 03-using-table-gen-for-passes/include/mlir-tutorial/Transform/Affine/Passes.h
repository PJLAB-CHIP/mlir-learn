#pragma once

#include "mlir-tutorial/Transform/Affine/AffineFullUnroll.hpp"
#include "mlir-tutorial/Transform/Affine/AffineFullUnrollPatternRewrite.hpp"

namespace mlir::tutorial
{

#define GEN_PASS_REGISTRATION
#include "mlir-tutorial/Transform/Affine/Passes.h.inc"

}  // namespace mlir::tutorial
