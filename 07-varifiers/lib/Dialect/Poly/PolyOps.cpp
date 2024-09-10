#include <mlir/Dialect/CommonFolders.h>

#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/IR/DerivedTypes.h"

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

    auto lhs = dyn_cast<DenseIntElementsAttr>(adaptor.getOperands()[0]);
    auto rhs = dyn_cast<DenseIntElementsAttr>(adaptor.getOperands()[1]);

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
    // assertion, since the tensor might have been constructed with weird types
    // we don’t want as input. If the dyn_cast fails, the result is nullptr,
    // which is cast by MLIR to a failed OpFoldResult.

    // Returns null if the cast failed, which corresponds to a failed fold.
    return dyn_cast<DenseIntElementsAttr>(adaptor.getInput());
}

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor)
{
    // For poly.constant the implementation is trivial: just return the input
    // attribute.
    return adaptor.getCoefficients();
}

LogicalResult EvalOp::verify() {
  return getPoint().getType().isSignlessInteger(32)
             ? success()
             : emitOpError("argument point must be a 32-bit integer");
}

}  // namespace mlir::tutorial::poly