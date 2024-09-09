####################################################################################################
# @file    "05-using-traits/include/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define tabelegen targets and generate injectable headers and
#          source files for the Poly dialect and its operations.
# --------------------------------------------------------------------------------------------------
# @note    ${PROJECT_TARGET_PREFIX} is set in the parent CMakeLists.txt file.
####################################################################################################

set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Dialect-PolyDialect-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Dialect/Poly/PolyDialect.td)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyDialect.hpp.inc --gen-dialect-decls)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyDialect.cpp.inc --gen-dialect-defs)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Dialect-PolyTypes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Dialect/Poly/PolyTypes.td)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyTypes.hpp.inc --gen-typedef-decls)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc --gen-typedef-defs)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Dialect-PolyOps-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Dialect/Poly/PolyOps.td)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyOps.hpp.inc --gen-op-decls)
mlir_tablegen(mlir-tutorial/Dialect/Poly/PolyOps.cpp.inc --gen-op-defs)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Transform/Affine/Passes.td)
mlir_tablegen(mlir-tutorial/Transform/Affine/Passes.hpp.inc -gen-pass-decls -name=Affine)
mlir_tablegen(mlir-tutorial/Transform/Affine/AffinePasses.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Transform-Arith-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Transform/Arith/Passes.td)
mlir_tablegen(mlir-tutorial/Transform/Arith/Passes.hpp.inc -gen-pass-decls -name=Arith)
mlir_tablegen(mlir-tutorial/Transform/Arith/ArithPasses.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

set(
    ALL_TABLEGEN_TARGETS
    ${PROJECT_TARGET_PREFIX}-Dialect-PolyDialect-IncGen
    ${PROJECT_TARGET_PREFIX}-Dialect-PolyTypes-IncGen
    ${PROJECT_TARGET_PREFIX}-Dialect-PolyOps-IncGen
    ${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen
    ${PROJECT_TARGET_PREFIX}-Transform-Arith-Passes-IncGen
)
mlir_learn_log_info("|- Target <tablegen>: ${ALL_TABLEGEN_TARGETS}")

# Add the generated files to a global property, so they can be used in the library
set_property(
    GLOBAL PROPERTY ${PROJECT_TARGET_PREFIX}-TABLEGEN-TARGETS
    ${ALL_TABLEGEN_TARGETS}
)

