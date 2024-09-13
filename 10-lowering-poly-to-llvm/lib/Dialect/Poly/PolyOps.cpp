#include <iostream>

#include <llvm/IR/DerivedTypes.h>
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/PatternMatch.h>

#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"
// Required after PatternMatch.h
#include "mlir-tutorial/Dialect/Poly/PolyCanonicalize.cpp.inc"

namespace mlir::tutorial::poly
{

OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor)
{
    // For add/sub which are elementwise operations on the coefficients, we get
    // to use an existing upstream helper method, constFoldBinaryOp, which
    // through some template metaprogramming wizardry, allows us to specify only
    // the elementwise operation itself.

    return constFoldBinaryOp<IntegerAttr, APInt, void>(
        adaptor.getOperands(),
        [&](const APInt& a, const APInt& b) { return a + b; });
}

OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor)
{
    // For add/sub which are elementwise operations on the coefficients, we get
    // to use an existing upstream helper method, constFoldBinaryOp, which
    // through some template metaprogramming wizardry, allows us to specify only
    // the elementwise operation itself.

    return constFoldBinaryOp<IntegerAttr, APInt, void>(
        adaptor.getOperands(),
        [&](const APInt& a, const APInt& b) { return a - b; });
}

OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor)
{
    // For mul, we have to write out the multiplication routine manually. In
    // what’s below, I’m implementing the naive textbook polymul algorithm,
    // which could be optimized if one expects people to start compiling
    // programs with large, static polynomials in them.

    auto lhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[0]);
    auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[1]);

    if (!lhs || !rhs) {
        return nullptr;
    }

    auto degree =
        mlir::cast<PolynomialType>(getResult().getType()).getDegreeBound();
    auto maxIndex = lhs.size() + rhs.size() - 1;

    SmallVector<APInt, 8> result;

    result.reserve(maxIndex);
    for (int64_t i = 0; i < maxIndex; ++i) {
        result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
    }

    int64_t i = 0;
    for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
         ++lhsIt) {
        int64_t j = 0;
        for (auto rhsIt = rhs.value_begin<APInt>();
             rhsIt != rhs.value_end<APInt>(); ++rhsIt) {
            result[(i + j) % degree] += (*lhsIt) * (*rhsIt);
            ++j;
        }
        ++i;
    }

    return DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(result.size()),
                              mlir::IntegerType::get(getContext(), 32)),
        result);
}

OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor)
{
    // The from_tensor op is similar, but has an extra cast that acts as an
    // assertion, since the tensor might have been constructed with weird
    // types we don’t want as input. If the dyn_cast fails, the result is
    // nullptr, which is cast by MLIR to a failed OpFoldResult.

    // Returns null if the cast failed, which corresponds to a failed fold.
    return dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
}

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor)
{
    // For poly.constant the implementation is trivial: just return the
    // input attribute.
    return adaptor.getCoefficients();
}

LogicalResult EvalOp::verify()
{
    auto pointTy = getPoint().getType();
    bool isSignlessInteger = pointTy.isSignlessInteger(32);
    auto complexPt = llvm::dyn_cast<ComplexType>(pointTy);
    return isSignlessInteger || complexPt
               ? success()
               : emitOpError("argument point must be a 32-bit "
                             "integer, or a complex number");
}

/// @brief Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other
/// uses.
struct DifferenceOfSquares : public OpRewritePattern<SubOp>
{
    DifferenceOfSquares(mlir::MLIRContext* context)
        : OpRewritePattern<SubOp>(context, /*benefit=*/1)
    {
    }

    LogicalResult matchAndRewrite(SubOp op,
                                  PatternRewriter& rewriter) const override
    {
        // <lhsOp> subOp <rhsOp>

        Value lhs = op.getOperand(0);
        Value rhs = op.getOperand(1);

        // If either arg has another use, then this rewrite is probably less
        // efficient, because it cannot delete the mul ops.
        if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
            return failure();
        }

        auto rhsMul = rhs.getDefiningOp<MulOp>();
        auto lhsMul = lhs.getDefiningOp<MulOp>();
        if (!rhsMul || !lhsMul) {
            return failure();
        }
        auto rhsOpName = rhsMul->getName().getStringRef().str();

        std::cout << "rhsOpName: " << rhsOpName << std::endl;

        bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
        bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();

        if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
            return failure();
        }

        auto x = lhsMul.getLhs();
        auto y = rhsMul.getLhs();

        auto newAdd = rewriter.create<AddOp>(op.getLoc(), x, y);
        auto newSub = rewriter.create<SubOp>(op.getLoc(), x, y);
        auto newMul = rewriter.create<MulOp>(op.getLoc(), newAdd, newSub);

        rewriter.replaceOp(op, newMul);
        // We don't need to remove the original ops because MLIR already has
        // canonicalization patterns that remove unused ops.

        return success();
    }
};

void AddOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context)
{
}

void SubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context)
{
    // results.add<DifferenceOfSquares>(context);
}

void MulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context)
{
}

void EvalOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context)
{
    //// results.add<LiftConjThroughEval>(context);
    //// results.add<DifferenceOfSquares>(context);
    // Same as above:
    populateWithGenerated(results);
}

}  // namespace mlir::tutorial::poly