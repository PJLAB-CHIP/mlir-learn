set(TARGET_NAME "02-AffineFullUnroll")

add_library(
    ${TARGET_NAME} STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnroll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Arith/MulToAdd.cpp
)
add_library(${GLOBAL_NAMESPACE}::${TARGET_NAME} ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR})

# Link against MLIR and LLVM
target_link_libraries(
    ${TARGET_NAME} PUBLIC 
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRFuncDialect
)

# Include directories
target_include_directories(${TARGET_NAME} PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(${TARGET_NAME} PUBLIC ${MLIR_INCLUDE_DIRS})