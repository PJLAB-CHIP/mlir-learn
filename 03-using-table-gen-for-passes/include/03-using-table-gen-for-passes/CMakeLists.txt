set(TARGET_NAME "03-Passes-IncGen")

set(LLVM_TARGET_DEFINITIONS Transform/Affine/Passes.td)
mlir_tablegen(Transform/Affine/Passes.h.inc -gen-pass-decls -name=Affine)
mlir_tablegen(Transform/Affine/AffinePasses.md -gen-pass-doc)

add_public_tablegen_target(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES 
    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
)