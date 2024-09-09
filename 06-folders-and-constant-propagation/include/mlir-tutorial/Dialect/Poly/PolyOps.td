#ifndef MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_
#define MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir-tutorial/Dialect/Poly/PolyDialect.td"
include "mlir-tutorial/Dialect/Poly/PolyTypes.td"

def PolyOrContainer : TypeOrContainer<Polynomial, "poly-or-container">;

class Poly_BinOp<string mnemonic> : 
    Op<Poly_Dialect, mnemonic, [
        Pure, 
        ElementwiseMappable, 
        // SameOperandsAndResultElementType trait (a vectorized cousin of SameOperandsAndResultType)
        // asserts that the poly type in all the arguments (and elements of containers) are the same
        SameOperandsAndResultElementType
    ]> 
{
    let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
    let results = (outs PolyOrContainer:$output);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
    let hasFolder = 1;
}

def Poly_AddOp : Poly_BinOp<"add"> {
    let summary = "Addition operation between polynomials.";
}

def Poly_SubOp : Poly_BinOp<"sub"> {
    let summary = "Subtraction operation between polynomials.";
}

def Poly_MulOp : Poly_BinOp<"mul"> {
    let summary = "Multiplication operation between polynomials.";
}

def Poly_FromTensorOp : Op<Poly_Dialect, "from_tensor", [Pure]> {
    let summary = "Creates a Polynomial from integer coefficients stored in a tensor.";
    let arguments = (ins TensorOf<[AnyInteger]>:$input);
    let results = (outs Polynomial:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
    let hasFolder = 1;
}

def Poly_EvalOp : Op<Poly_Dialect, "eval"> {
    let summary = "Evaluates a Polynomial at a given input value.";
    let arguments = (ins Polynomial:$input, AnyInteger:$point);
    let results = (outs AnyInteger:$output);
    let assemblyFormat = "$input `,` $point attr-dict `:` `(` type($input) `,` type($point) `)` `->` type($output)";
}

// Having a dedicated constant operation has benefits explained in the MLIR documentation on folding. 
// What's relevant here is that fold can be used to signal to passes like sccp that the result of an 
// op is constant (statically known), or it can be used to say that the result of an op is equivalent 
// to a pre-existing value created by a different op. For the constant case, a materializeConstant 
// hook is also needed to tell MLIR how to take the constant result and turn it into a proper IR op.
def Poly_ConstantOp : Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
    let summary = "Define a constant polynomial via an attribute.";
    // This defines the input to the op as an attribute (statically defined data) rather than a 
    // previous SSA value. The AnyIntElementsAttr is itself an attribute constraint, allowing any 
    // attribute that is has the IntElementsAttrBase as a base class to be used (e.g., 32-bit or 64-
    // bit integer attributes). 
    let arguments = (ins AnyIntElementsAttr:$coefficients);
    let results = (outs Polynomial:$output);
    let assemblyFormat = "$coefficients attr-dict `:` type($output)";
    let hasFolder = 1;
}

#endif  // MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_