#ifndef MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_
#define MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_

include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir-tutorial/Dialect/Poly/PolyDialect.td"
include "mlir-tutorial/Dialect/Poly/PolyTypes.td"

def PolyOrContainer : TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;

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
}

def Poly_EvalOp : Op<Poly_Dialect, "eval"> {
  let summary = "Evaluates a Polynomial at a given input value.";
  let arguments = (ins Polynomial:$input, AnyInteger:$point);
  let results = (outs AnyInteger:$output);
  let assemblyFormat = "$input `,` $point attr-dict `:` `(` type($input) `,` type($point) `)` `->` type($output)";
}

#endif  // MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_