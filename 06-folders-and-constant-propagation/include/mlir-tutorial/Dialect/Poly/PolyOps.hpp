#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>

#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"

#define GET_OP_CLASSES
#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp.inc"
