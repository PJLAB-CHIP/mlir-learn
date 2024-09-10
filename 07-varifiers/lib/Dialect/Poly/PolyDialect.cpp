#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>

#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"

#include "mlir-tutorial/Dialect/Poly/PolyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir-tutorial/Dialect/Poly/PolyOps.cpp.inc"

namespace mlir::tutorial::poly
{

void PolyDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc"
        >();
    addOperations<
#define GET_OP_LIST
#include "mlir-tutorial/Dialect/Poly/PolyOps.cpp.inc"
        >();
}

Operation* PolyDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc)
{
    auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
    if (!coeffs) {
        return nullptr;
    }
    return builder.create<ConstantOp>(loc, type, coeffs);
}

}  // namespace mlir::tutorial::poly
