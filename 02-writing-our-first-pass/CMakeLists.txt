project(02-writing-our-first-pass)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)


add_subdirectory("lib")
add_subdirectory("tools")