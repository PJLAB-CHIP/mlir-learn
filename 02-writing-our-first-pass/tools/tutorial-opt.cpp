#include <mlir/InitAllDialects.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <mlir-tutorial/Transform/Affine/AffineFullUnroll.hpp>
#include <mlir-tutorial/Transform/Arith/MulToAdd.hpp>

int main(int argc, char** argv)
{
    // Register all built-in MLIR dialects
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    // Register our pass
    mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();
    mlir::PassRegistration<
        mlir::tutorial::AffineFullUnrollPassAsPatternRewrite>();
    mlir::PassRegistration<mlir::tutorial::MulToAddPass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}