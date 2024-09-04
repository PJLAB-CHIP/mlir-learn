set(TARGET_NAME "04-Poly-Dialect-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Dialect/Poly/PolyDialect.td)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyDialect.hpp.inc --gen-dialect-decls)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyDialect.cpp.inc --gen-dialect-defs)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "04-Poly-Types-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Dialect/Poly/PolyTypes.td)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyTypes.hpp.inc --gen-typedef-decls)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc --gen-typedef-defs)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "04-Transform-Affine-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Transform/Affine/Passes.td)
mlir_tablegen(mlir-tutorial/Transform/Affine/Passes.hpp.inc -gen-pass-decls -name=Affine)
mlir_tablegen(mlir-tutorial/Transform/Affine/AffinePasses.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "04-Transform-Arith-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Transform/Arith/Passes.td)
mlir_tablegen(mlir-tutorial/Transform/Arith/Passes.hpp.inc -gen-pass-decls -name=Arith)
mlir_tablegen(mlir-tutorial/Transform/Arith/ArithPasses.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

# Add the generated files to a global property, so they can be used in the library
set_property(
    GLOBAL PROPERTY 04_TABLEGEN_TARGETS
    04-Poly-Dialect-IncGen
    04-Poly-Types-IncGen
    04-Transform-Affine-Passes-IncGen
    04-Transform-Arith-Passes-IncGen
)
