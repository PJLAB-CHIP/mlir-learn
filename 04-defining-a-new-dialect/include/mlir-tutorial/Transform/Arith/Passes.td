#ifndef MLIR_TUTORIAL_TRANSFORM_AFFINE_PASSES_TD_
#define MLIR_TUTORIAL_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

/*
class MulToAddPass
    : public PassWrapper<MulToAddPass, OperationPass<mlir::func::FuncOp>>
{
private:
    void runOnOperation() override;

    StringRef getArgument() const final
    {
        return "mul-to-add";
    }

    StringRef getDescription() const final
    {
        return "Convert multiplicatoins to repeated additions";
    }
};
*/

def MulToAdd : Pass<"mul-to-add"> {
    let summary = "Replace mul with add";
    let description = [{
        Replace mul with add.
    }];
    let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif  // MLIR_TUTORIAL_TRANSFORM_AFFINE_PASSES_TD_
