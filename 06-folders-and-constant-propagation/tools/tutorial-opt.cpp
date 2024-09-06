#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"
#include "mlir-tutorial/Transform/Affine/Passes.hpp"
#include "mlir-tutorial/Transform/Arith/Passes.hpp"

int main(int argc, char** argv)
{
    // Register all built-in MLIR dialects
    mlir::DialectRegistry registry;
    registry.insert<mlir::tutorial::poly::PolyDialect>();
    mlir::registerAllDialects(registry);

    // Register all built-in MLIR passes
    mlir::registerAllPasses();

    // Register our pass
    mlir::tutorial::registerAffinePasses();
    mlir::tutorial::registerArithPasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}