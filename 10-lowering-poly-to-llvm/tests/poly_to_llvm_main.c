// cat 10-lowering-poly-to-llvm/tests/Output/poly_to_llvm.mlir.tmp | mlir-translate --mlir-to-llvmir | llc --filetype=obj > poly_to_llvm.o
// clang -c 10-lowering-poly-to-llvm/tests/poly_to_llvm_main.c && clang poly_to_llvm_main.o ./poly_to_llvm.o -o a.out
// ./a.out
#include <stdio.h>

// This is the function we want to call from LLVM
int test_fn(int x);

int main(int argc, char* argv[])
{
    int i = 1;
    int result = test_fn(i);
    printf("Result: %d\n", result);
    return 0;
}