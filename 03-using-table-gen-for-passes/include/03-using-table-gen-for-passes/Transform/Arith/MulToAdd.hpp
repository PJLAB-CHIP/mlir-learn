#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"


namespace mlir::tutorial
{

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

} // namespace mlir::tutorial
