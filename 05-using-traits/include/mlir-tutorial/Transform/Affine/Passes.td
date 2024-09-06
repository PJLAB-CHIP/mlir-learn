#ifndef MLIR_TUTORIAL_TRANSFORM_AFFINE_PASSES_TD_
#define MLIR_TUTORIAL_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

/*******************************************************************************
class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         OperationPass<mlir::func::FuncOp>>
{
private:
    /**
     * @brief Manually unrolls all affine loops.
     */
    void runOnOperation() override;

    StringRef getArgument() const final
    {
        return "affine-full-unroll";
    }

    StringRef getDescription() const final
    {
        return "Fully unroll all affine loops";
    }
};
*******************************************************************************/
def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

/*******************************************************************************
class AffineFullUnrollPassAsPatternRewrite
    : public PassWrapper<AffineFullUnrollPassAsPatternRewrite,
                         OperationPass<mlir::func::FuncOp>>
{
private:
    /**
     * @brief Unrolls all affine loops using pattern rewrite engine.
     */
    void runOnOperation() override;

    StringRef getArgument() const final
    {
        return "affine-full-unroll-rewrite";
    }

    StringRef getDescription() const final
    {
        return "Fully unroll all affine loops using pattern rewrite engine";
    }
};
*******************************************************************************/
def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
  let summary = "Fully unroll all affine loops using the pattern rewrite engine";
  let description = [{
    Fully unroll all affine loops using the pattern rewrite engine.
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

#endif  // MLIR_TUTORIAL_TRANSFORM_AFFINE_PASSES_TD_