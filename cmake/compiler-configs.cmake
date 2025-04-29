include(${CMAKE_CURRENT_LIST_DIR}/logging.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(STACK_SIZE 1048576)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -fopenmp -Wall -Wextra -Werror -fno-rtti")
    if (WIN32)
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-stack,${STACK_SIZE}")
    else()
        string(APPEND CMAKE_EXE_LINKER_FLAGS " -Wl,-zstack-size=${STACK_SIZE}")
    endif()
else()
    log_fatal("Unsupported compiler")
endif()