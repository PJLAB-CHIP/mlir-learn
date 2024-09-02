#pragma once

#include <mlir/Pass/Pass.h>


namespace mlir::tutorial
{

#define GEN_PASS_DECL_AFFINEFULLUNROLLPATTERNREWRITE
#include "03-using-table-gen-for-passes/Transform/Affine/Passes.h.inc"

} // namespace mlir::tutorial
