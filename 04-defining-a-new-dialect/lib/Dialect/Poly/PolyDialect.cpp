#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>

#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"

#include "mlir-tutorial/Dialect/Poly/PolyDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc"

namespace mlir::tutorial::poly
{

void PolyDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc"
        >();
}

}  // namespace mlir::tutorial::poly
