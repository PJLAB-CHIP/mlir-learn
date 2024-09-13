// RUN: 10-tutorial-opt %s --affine-full-unroll-rewrite > %t
// RUN: FileCheck %s < %t

func.func @test_dynamic_loop(%buffer: memref<?xi32>, %size: i32) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  %size_index = arith.index_cast %size : i32 to index
  // CHECK-NOT: affine.for
  %sum = affine.for %i = 0 to %size_index iter_args(%sum_iter = %sum_0) -> i32 {
    %cond = arith.cmpi slt, %i, %size_index : index
    %t = scf.if %cond -> (i32) {
      %t_1 = affine.load %buffer[%i] : memref<?xi32>
      scf.yield %t_1 : i32
    } else {
      %zero = arith.constant 0 : i32
      scf.yield %zero : i32
    }
    %sum_next = arith.addi %sum_iter, %t : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
