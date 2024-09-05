####################################################################################################
# @file    "04-defining-a-new-dialect/lib/CMakeLists.txt"
# --------------------------------------------------------------------------------------------------
# @brief   This CMake file is used to define the ${PROJECT_TARGET_PREFIX}-tutorial-lib library 
#          target.
# --------------------------------------------------------------------------------------------------
# @note    ${PROJECT_TARGET_PREFIX} is set in the parent CMakeLists.txt file.
####################################################################################################

set(TARGET_NAME ${PROJECT_TARGET_PREFIX}-tutorial-lib)
log_info("Target: ${TARGET_NAME} Library")

# Get the tablegen targets from the global property;
# Check "../include/CMakeLists.txt" for its definition.
get_property(
    TABLEGEN_TARGETS GLOBAL PROPERTY 
    ${PROJECT_TARGET_PREFIX}-TABLEGEN-TARGETS
)

add_mlir_dialect_library(
    # Library Name:
    ${TARGET_NAME}
    # Source Files:
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnroll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnrollPatternRewrite.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Arith/MulToAdd.cpp

    DEPENDS
    ${TABLEGEN_TARGETS}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface
)

target_include_directories(
    ${TARGET_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_OUTPUT_DIR}/include
)

set_target_properties(${TARGET_NAME} PROPERTIES 
    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
)