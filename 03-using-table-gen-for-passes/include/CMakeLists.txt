####################################################################################################
# @file    "03-using-table-gen-for-passes/include/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define tabelegen targets and generate injectable headers and
#          source files for the Poly dialect and its operations.
# --------------------------------------------------------------------------------------------------
# @note    ${PROJECT_TARGET_PREFIX} is set in the parent CMakeLists.txt file.
####################################################################################################

set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Transform/Affine/Passes.td)
mlir_tablegen(mlir-tutorial/Transform/Affine/Passes.h.inc -gen-pass-decls -name=Affine)
mlir_tablegen(mlir-tutorial/Transform/Affine/AffinePasses.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Transform-Arith-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Transform/Arith/Passes.td)
mlir_tablegen(mlir-tutorial/Transform/Arith/Passes.hpp.inc -gen-pass-decls -name=Arith)
mlir_tablegen(mlir-tutorial/Transform/Arith/ArithPasses.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

set(
    ALL_TABLEGEN_TARGETS
    ${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen
    ${PROJECT_TARGET_PREFIX}-Transform-Arith-Passes-IncGen
)
log_info("|- Target <tablegen>: ${ALL_TABLEGEN_TARGETS}")

# Add the generated files to a global property, so they can be used in the library
set_property(
    GLOBAL PROPERTY ${PROJECT_TARGET_PREFIX}-TABLEGEN-TARGETS
    ${ALL_TABLEGEN_TARGETS}
)