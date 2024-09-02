#pragma once

#include "03-using-table-gen-for-passes/Transform/Affine/AffineFullUnroll.hpp"
#include "03-using-table-gen-for-passes/Transform/Affine/AffineFullUnrollPatternRewrite.hpp"

namespace mlir::tutorial
{

#define GEN_PASS_REGISTRATION
#include "03-using-table-gen-for-passes/Transform/Affine/Passes.h.inc"

}  // namespace mlir::tutorial
