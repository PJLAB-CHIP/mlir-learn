#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tutorial::poly
{

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp.inc"

// Convert types from poly to other dialects.
class PolyToStandardTypeConverter : public TypeConverter
{
public:
    PolyToStandardTypeConverter(MLIRContext* ctx)
    {
        // We call addConversion for each specific type we need to convert, and
        // then the default addConversion([](Type type) { return type; });
        // signifies “no work needed, this type is legal.” By itself this type
        // converter does nothing, because it is only used in conjunction with a
        // specific kind of rewrite pattern, a subclass of OpConversionPattern.

        addConversion([](Type type) { return type; });
        addConversion([ctx](PolynomialType type) -> Type {
            int degreeBound = type.getDegreeBound();
            IntegerType elementTy = IntegerType::get(
                ctx, 32, IntegerType::SignednessSemantics::Signless);
            return RankedTensorType::get({degreeBound}, elementTy);
        });

        // We don't include any custom materialization hooks because this
        // lowering is all done in a single pass. The dialect conversion
        // framework works by resolving intermediate (mid-pass) type conflicts
        // by inserting unrealized_conversion_cast ops, and only converting
        // those to custom materializations if they persist at the end of the
        // pass. In our case, we'd only need to use custom materializations if
        // we split this lowering across multiple passes.

        // // Convert from a tensor type to a poly type: use from_tensor
        // addSourceMaterialization([](OpBuilder& builder, Type type,
        //                             ValueRange inputs, Location loc) -> Value {
        //     return {builder.create<poly::FromTensorOp>(loc, type, inputs[0])};
        // });

        // // Convert from a poly type to a tensor type: use to_tensor
        // addTargetMaterialization([](OpBuilder& builder, Type type,
        //                             ValueRange inputs, Location loc) -> Value {
        //     return {builder.create<poly::ToTensorOp>(loc, type, inputs[0])};
        // });
    }
};

struct ConvertAdd : public OpConversionPattern<AddOp>
{
    ConvertAdd(mlir::MLIRContext* context) : OpConversionPattern<AddOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AddOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {

        auto addOp = rewriter.create<arith::AddIOp>(
            op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op.getOperation(), addOp);
        return success();
    }
};

struct ConvertSub : public OpConversionPattern<SubOp>
{
    ConvertSub(mlir::MLIRContext* context) : OpConversionPattern<SubOp>(context)
    {
    }
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
        SubOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto subOp = rewriter.create<arith::SubIOp>(
            op.getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op.getOperation(), subOp);
        return success();
    }
};

struct ConvertMul : public OpConversionPattern<MulOp>
{
    ConvertMul(mlir::MLIRContext* context) : OpConversionPattern<MulOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MulOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto polymulTensorType =
            cast<RankedTensorType>(adaptor.getLhs().getType());
        auto numTerms = polymulTensorType.getShape()[0];
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        // Create an all-zeros tensor to store the result
        auto polymulResult = b.create<arith::ConstantOp>(
            polymulTensorType, DenseElementsAttr::get(polymulTensorType, 0));

        // Loop bounds and step.
        auto lowerBound =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(0));
        auto numTermsOp = b.create<arith::ConstantOp>(b.getIndexType(),
                                                      b.getIndexAttr(numTerms));
        auto step =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));

        auto p0 = adaptor.getLhs();
        auto p1 = adaptor.getRhs();

        // for i = 0, ..., N-1
        //   for j = 0, ..., N-1
        //     product[i+j (mod N)] += p0[i] * p1[j]
        auto outerLoop = b.create<scf::ForOp>(
            lowerBound, numTermsOp, step, ValueRange(polymulResult.getResult()),
            [&](OpBuilder& builder, Location loc, Value p0Index,
                ValueRange loopState) {
                ImplicitLocOpBuilder b(op.getLoc(), builder);
                auto innerLoop = b.create<scf::ForOp>(
                    lowerBound, numTermsOp, step, loopState,
                    [&](OpBuilder& builder, Location loc, Value p1Index,
                        ValueRange loopState) {
                        ImplicitLocOpBuilder b(op.getLoc(), builder);
                        auto accumTensor = loopState.front();
                        auto destIndex = b.create<arith::RemUIOp>(
                            b.create<arith::AddIOp>(p0Index, p1Index),
                            numTermsOp);
                        auto mulOp = b.create<arith::MulIOp>(
                            b.create<tensor::ExtractOp>(p0,
                                                        ValueRange(p0Index)),
                            b.create<tensor::ExtractOp>(p1,
                                                        ValueRange(p1Index)));
                        auto result = b.create<arith::AddIOp>(
                            mulOp, b.create<tensor::ExtractOp>(
                                       accumTensor, destIndex.getResult()));
                        auto stored = b.create<tensor::InsertOp>(
                            result, accumTensor, destIndex.getResult());
                        b.create<scf::YieldOp>(stored.getResult());
                    });

                b.create<scf::YieldOp>(innerLoop.getResults());
            });

        rewriter.replaceOp(op, outerLoop.getResult(0));
        return success();
    }
};

struct ConvertEval : public OpConversionPattern<EvalOp>
{
    ConvertEval(mlir::MLIRContext* context)
        : OpConversionPattern<EvalOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        EvalOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto polyTensorType =
            cast<RankedTensorType>(adaptor.getPolynomial().getType());
        auto numTerms = polyTensorType.getShape()[0];
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        auto lowerBound =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));
        auto numTermsOp = b.create<arith::ConstantOp>(
            b.getIndexType(), b.getIndexAttr(numTerms + 1));
        auto step = lowerBound;

        auto poly = adaptor.getPolynomial();
        auto point = adaptor.getPoint();

        // Horner's method:
        //
        // accum = 0
        // for i = 1, 2, ..., N
        //   accum = point * accum + coeff[N - i]
        auto accum =
            b.create<arith::ConstantOp>(b.getI32Type(), b.getI32IntegerAttr(0));
        auto loop = b.create<scf::ForOp>(
            lowerBound, numTermsOp, step, accum.getResult(),
            [&](OpBuilder& builder, Location loc, Value loopIndex,
                ValueRange loopState) {
                ImplicitLocOpBuilder b(op.getLoc(), builder);
                auto accum = loopState.front();
                auto coeffIndex =
                    b.create<arith::SubIOp>(numTermsOp, loopIndex);
                auto mulOp = b.create<arith::MulIOp>(point, accum);
                auto result = b.create<arith::AddIOp>(
                    mulOp,
                    b.create<tensor::ExtractOp>(poly, coeffIndex.getResult()));
                b.create<scf::YieldOp>(result.getResult());
            });

        rewriter.replaceOp(op, loop.getResult(0));
        return success();
    }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp>
{
    ConvertFromTensor(mlir::MLIRContext* context)
        : OpConversionPattern<FromTensorOp>(context)
    {
    }
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
        FromTensorOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto resultTensorTy = cast<RankedTensorType>(
            typeConverter->convertType(op->getResultTypes()[0]));
        auto resultShape = resultTensorTy.getShape()[0];
        auto resultEltTy = resultTensorTy.getElementType();
        auto inputTensorTy = op.getInput().getType();
        auto inputShape = inputTensorTy.getShape()[0];
        // Zero pad the tensor if the coefficients' size is less than the
        // polynomial degree.
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        auto coeffValue = adaptor.getInput();
        if (inputShape < resultShape) {
            SmallVector<OpFoldResult, 1> low, high;
            low.push_back(rewriter.getIndexAttr(0));
            high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
            coeffValue = b.create<tensor::PadOp>(
                resultTensorTy, coeffValue, low, high,
                b.create<arith::ConstantOp>(
                    rewriter.getIntegerAttr(resultEltTy, 0)),
                /*nofold=*/false);
        }
        rewriter.replaceOp(op, coeffValue);
        return success();
    }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp>
{
    ConvertToTensor(mlir::MLIRContext* context)
        : OpConversionPattern<ToTensorOp>(context)
    {
    }
    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
        ToTensorOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp>
{
    ConvertConstant(mlir::MLIRContext* context)
        : OpConversionPattern<ConstantOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ConstantOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        auto constOp = b.create<arith::ConstantOp>(adaptor.getCoefficients());
        auto fromTensorOp =
            b.create<FromTensorOp>(op.getResult().getType(), constOp);
        rewriter.replaceOp(op, fromTensorOp.getResult());
        return success();
    }
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard>
{
    using PolyToStandardBase::PolyToStandardBase;

    void runOnOperation() override
    {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        // It allows you to declare an entire dialect as “illegal” after the
        // lowering is complete.
        target.addLegalDialect<arith::ArithDialect>();
        target.addIllegalDialect<PolyDialect>();

        RewritePatternSet patterns(context);

        PolyToStandardTypeConverter typeConverter(context);
        patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertMul,
                     ConvertEval, ConvertFromTensor, ConvertToTensor>(
            typeConverter, context);

        populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
            patterns, typeConverter);
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody());
        });

        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::ReturnOp>(
            [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

        populateCallOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::CallOp>(
            [&](func::CallOp op) { return typeConverter.isLegal(op); });

        populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
        target.markUnknownOpDynamicallyLegal([&](Operation* op) {
            return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
                   isLegalForBranchOpInterfaceTypeConversionPattern(
                       op, typeConverter) ||
                   isLegalForReturnOpTypeConversionPattern(op, typeConverter);
        });

        if (failed(
                // We use applyPartialConversion instead of the more natural
                // sounding alternative applyFullConversion for a silly reason:
                // the error messages are better. If applyFullConversion fails,
                // even in debug mode, you don’t get a lot of information about
                // what went wrong. In partial mode, you can see the steps and
                // what type conflicts it wasn’t able to patch up at the end.
                applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};

}  // namespace mlir::tutorial::poly