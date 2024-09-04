set(TARGET_NAME "04-tutorial-lib")
log_info("Target: ${TARGET_NAME} Library")

get_property(04_TABLEGEN_TARGETS GLOBAL PROPERTY 04_TABLEGEN_TARGETS)

add_mlir_dialect_library(
    # Library Name:
    ${TARGET_NAME}
    # Source Files:
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnroll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnrollPatternRewrite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Arith/MulToAdd.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Poly/PolyDialect.cpp

    DEPENDS
    ${04_TABLEGEN_TARGETS}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface

)
# Add alias with ${GLOABL_NAMESPACE}
add_library(${GLOBAL_NAMESPACE}::${TARGET_NAME} ALIAS ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES 
    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
)