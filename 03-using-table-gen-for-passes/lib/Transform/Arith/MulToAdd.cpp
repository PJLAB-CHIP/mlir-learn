#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>

#include "03-using-table-gen-for-passes/Transform/Arith/MulToAdd.hpp"

namespace mlir::tutorial
{

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;

// Whether to use arith::ShLIOp to replace the constant multiplication.
constexpr bool USE_ARITH_SHLIOP = false;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2, otherwise do
// nothing.
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp>
{
    // By setting the benefit argument of PowerOfTwoExpand to be larger than
    // PeelFromMul, we tell the greedy rewrite engine to prefer PowerOfTwoExpand
    // whenever possible.
    PowerOfTwoExpand(mlir::MLIRContext* context)
        : OpRewritePattern<MulIOp>(context, /*benefit=*/2)
    {
    }

    LogicalResult matchAndRewrite(MulIOp op,
                                  PatternRewriter& rewriter) const override
    {
        // Value is a reference to the operands of the operation,
        // represented as a abstraction of the SSA value. It is not the
        // actual value of the operand.
        Value lhs = op.getOperand(0);

        // canonicalization patterns ensure the constant is on the right, if
        // there is a constant See
        // https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
        Value rhs = op.getOperand(1);

        // Get the defining operation of the right-hand side operand. This
        // is a operation that produces the concrete value of the operand.
        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
        if (!rhsDefiningOp) {
            return failure();
        }
        int64_t value = rhsDefiningOp.value();
        bool is_power_of_two = (value & (value - 1)) == 0;

        if (!is_power_of_two) {
            return failure();
        }

        if constexpr (!USE_ARITH_SHLIOP) {
            auto newConstant = rewriter.create<ConstantOp>(
                rhsDefiningOp.getLoc(),
                rewriter.getIntegerAttr(rhs.getType(), value / 2));
            auto newMul =
                rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
            auto newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, newMul);
            rewriter.replaceOp(op, newAdd);
            rewriter.eraseOp(rhsDefiningOp);
        } else {
            int64_t shift_amount = 0;
            while (value > 1) {
                value >>= 1;
                shift_amount++;
            }
            auto newConstant = rewriter.create<ConstantOp>(
                rhsDefiningOp.getLoc(),
                rewriter.getIntegerAttr(rhs.getType(), shift_amount));
            auto newShl =
                rewriter.create<arith::ShLIOp>(op.getLoc(), lhs, newConstant);
            rewriter.replaceOp(op, newShl);
            rewriter.eraseOp(rhsDefiningOp);
        }

        return success();
    }
};

// Replace y = 9*x with y = 8*x + x
struct PeelFromMul : public OpRewritePattern<MulIOp>
{
    PeelFromMul(mlir::MLIRContext* context)
        : OpRewritePattern<MulIOp>(context, /*benefit=*/1)
    {
    }

    LogicalResult matchAndRewrite(MulIOp op,
                                  PatternRewriter& rewriter) const override
    {
        Value lhs = op.getOperand(0);
        Value rhs = op.getOperand(1);

        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
        if (!rhsDefiningOp) {
            return failure();
        }

        // Here we have guraranteed that value is not a power of two
        // because PowerOfTwoExpand has a higher benefit than PeelFromMul
        // and will be applied first.
        int64_t value = rhsDefiningOp.value();

        if (value <= 1) {
            return failure();
        }

        auto newConstant = rewriter.create<ConstantOp>(
            rhsDefiningOp.getLoc(),
            rewriter.getIntegerAttr(rhs.getType(), value - 1));

        auto newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
        auto newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, lhs);

        rewriter.replaceOp(op, newAdd);

        return success();
    }
};

void MulToAddPass::runOnOperation()
{
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowerOfTwoExpand>(&getContext());
    patterns.add<PeelFromMul>(&getContext());
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

}  // namespace mlir::tutorial
