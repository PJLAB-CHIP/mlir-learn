#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Dialect/Bufferization/Pipelines/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>
#include <mlir/Transforms/Passes.h>

#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"
#include "mlir-tutorial/Transform/Affine/Passes.hpp"
#include "mlir-tutorial/Transform/Arith/Passes.hpp"

void polyToLLVMPipelineBuilder(mlir::OpPassManager& manager)
{
    // Poly
    manager.addPass(mlir::tutorial::poly::createPolyToStandard());
    manager.addPass(mlir::createCanonicalizerPass());

    manager.addPass(mlir::createConvertElementwiseToLinalgPass());
    manager.addPass(mlir::createConvertTensorToLinalgPass());

    // One-shot bufferize, from
    // https://mlir.llvm.org/docs/Bufferization/#ownership-based-buffer-deallocation
    mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    manager.addPass(
        mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
    mlir::bufferization::BufferDeallocationPipelineOptions deallocationOptions;
    mlir::bufferization::buildBufferDeallocationPipeline(manager,
                                                         deallocationOptions);

    manager.addPass(mlir::createConvertLinalgToLoopsPass());

    // Needed to lower memref.subview
    manager.addPass(mlir::memref::createExpandStridedMetadataPass());

    manager.addPass(mlir::createConvertSCFToCFPass());
    manager.addPass(mlir::createConvertControlFlowToLLVMPass());
    manager.addPass(mlir::createArithToLLVMConversionPass());
    manager.addPass(mlir::createConvertFuncToLLVMPass());
    manager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    manager.addPass(mlir::createReconcileUnrealizedCastsPass());

    // Cleanup
    manager.addPass(mlir::createCanonicalizerPass());
    manager.addPass(mlir::createSCCPPass());
    manager.addPass(mlir::createCSEPass());
    manager.addPass(mlir::createSymbolDCEPass());
}

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
    mlir::tutorial::poly::registerPolyToStandardPasses();

    mlir::PassPipelineRegistration<>(
        "poly-to-llvm", "Run passes to lower the poly dialect to LLVM",
        polyToLLVMPipelineBuilder);

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}