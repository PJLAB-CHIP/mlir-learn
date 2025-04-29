#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mlir-tutorial/Transform/Affine/AffineFullUnrollPatternRewrite.hpp"

namespace mlir::tutorial
{

#define GEN_PASS_DEF_AFFINEFULLUNROLLPATTERNREWRITE
#include "mlir-tutorial/Transform/Affine/Passes.hpp.inc"

using ::mlir::affine::AffineForOp;
using ::mlir::affine::loopUnrollFull;

// A pattern that matches on AffineForOp and unrolls it.
struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp>
{
    AffineFullUnrollPattern(mlir::MLIRContext* context)
        : OpRewritePattern<AffineForOp>(context, /*benefit=*/1)
    {
    }

    LogicalResult matchAndRewrite(AffineForOp op,
                                  PatternRewriter& rewriter) const override
    {
        // This is technically not allowed, since in a RewritePattern all
        // modifications to the IR are supposed to go through the `rewriter`
        // arg, but it works for our limited test cases.
        return loopUnrollFull(op);
    }
};

// A pass that invokes the pattern rewrite engine.
struct AffineFullUnrollPatternRewrite
    : impl::AffineFullUnrollPatternRewriteBase<AffineFullUnrollPatternRewrite>
{
    using AffineFullUnrollPatternRewriteBase::
        AffineFullUnrollPatternRewriteBase;
    void runOnOperation() final
    {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<AffineFullUnrollPattern>(&getContext());
        // One could use GreedyRewriteConfig here to slightly tweak the behavior
        // of the pattern application.
        (void) applyPatternsGreedily(getOperation(), std::move(patterns),
                                     GreedyRewriteConfig().enableFolding());
    }
};

}  // namespace mlir::tutorial
