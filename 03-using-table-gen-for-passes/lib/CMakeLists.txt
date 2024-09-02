set(TARGET_NAME "03-Passes")

log_info("Additional Header Directories: ${PROJECT_SOURCE_DIR}/include/03-using-table-gen-for-passes")

add_mlir_dialect_library(
    # Library Name:
    ${TARGET_NAME}
    # Source Files:
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnroll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnrollPatternRewrite.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/03-using-table-gen-for-passes

    DEPENDS
    03-Passes-IncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface

)
# Add alias with ${GLOABL_NAMESPACE}
add_library(${GLOBAL_NAMESPACE}::${TARGET_NAME} ALIAS ${TARGET_NAME}
)

set_target_properties(${TARGET_NAME} PROPERTIES 
    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
)