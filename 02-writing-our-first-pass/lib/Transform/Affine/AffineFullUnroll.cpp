#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mlir-tutorial/Transform/Affine/AffineFullUnroll.hpp"

namespace mlir::tutorial
{

using ::mlir::affine::AffineForOp;
using ::mlir::affine::loopUnrollFull;

void AffineFullUnrollPass::runOnOperation()
{
    getOperation()  // Returns the FuncOp this pass is run on
        .walk       // Traverse the AST of the FuncOp
        ([&](AffineForOp op) {
            if (failed(loopUnrollFull(op))) {
                op->emitError() << "Failed to unroll loop";
                signalPassFailure();
            }
        });
}

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
void AffineFullUnrollPassAsPatternRewrite::runOnOperation()
{
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AffineFullUnrollPattern>(&getContext());
    // One could use GreedyRewriteConfig here to slightly tweak the behavior of
    // the pattern application.
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace mlir::tutorial
