#pragma once

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::tutorial
{

class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         // OperationPass is anchored to the FuncOp
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

}  // namespace mlir::tutorial
