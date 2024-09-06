// RUN: 06-tutorial-opt %s --sccp > %t

// Note: 
//   --sccp does sparse conditional constant propagation, which is a forward dataflow analysis that propagates constant values through the program, but not deleting the dead code.
//   --canonicalize is used to clean up the IR after the transformation, including removing the dead code.

func.func @test_arith_sccp() -> i32 {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = arith.addi %0, %0 : i32
  %3 = arith.muli %0, %0 : i32
  %4 = arith.addi %2, %3 : i32
  return %2 : i32
}