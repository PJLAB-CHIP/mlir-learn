#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "03-using-table-gen-for-passes/Transform/Affine/Passes.h"


int main(int argc, char** argv)
{
    // Register all built-in MLIR dialects
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    // Register our pass
     mlir::tutorial::registerAffinePasses();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}