# cmake/embed_kernels.cmake
# Binary embedding commands — replaces BAKE_COMPILED_KERNEL logic from CMakeLists.txt
#
# Converts precompiled kernel binaries into C array headers that are
# compiled directly into the library, with optional Zstd compression.
#
# Expected variables from parent scope:
#   KERNEL_HIPRT_COMP              — path to HIPRT .hipfb
#   KERNEL_HIPRT_COMP_COMPRESSED   — path for compressed output
#   KERNEL_OROCHI_COMP             — path to Orochi .hipfb
#   KERNEL_OROCHI_COMP_COMPRESSED  — path for compressed output
#   COMPILED_COMPRESSION           — ON/OFF
#   HIPRT_NAME                     — target name
#   PROJECT_SOURCE_DIR

include_guard(GLOBAL)

if(NOT BAKE_COMPILED_KERNEL)
    return()
endif()

message(STATUS "Embedded compiled kernels enabled (BAKE_COMPILED_KERNEL=ON)")

set(_archive_script "${PROJECT_SOURCE_DIR}/contrib/Orochi/scripts/create_archive.cmake")
set(_convert_script "${PROJECT_SOURCE_DIR}/cmake/convert_binary_to_array.cmake")

# --- HIPRT compiled kernel -> bvh_build_array.h ---
set(KERNEL_HIPRT_H "${PROJECT_SOURCE_DIR}/hiprt/impl/bvh_build_array.h")

add_custom_command(
    OUTPUT ${KERNEL_HIPRT_H}
    # 1) Optional Zstd compression
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_FILE=${KERNEL_HIPRT_COMP}
        -DOUTPUT_FILE=${KERNEL_HIPRT_COMP_COMPRESSED}
        -DDO_COMPRESS=${COMPILED_COMPRESSION}
        -P ${_archive_script}
    # 2) Convert binary to C array header
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_FILE=${KERNEL_HIPRT_COMP}
        -DCOMPRESSED_FILE=${KERNEL_HIPRT_COMP_COMPRESSED}
        -DOUTPUT_HEADER=${KERNEL_HIPRT_H}
        -DCOMPRESSION_ACTIVATED=${COMPILED_COMPRESSION}
        -P ${_convert_script}
    DEPENDS ${KERNEL_HIPRT_COMP}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Embedding HIPRT compiled kernel into bvh_build_array.h"
    VERBATIM
)

# --- Orochi compiled kernel -> oro_compiled_kernels.h ---
set(KERNEL_OROCHI_H "${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/cache/oro_compiled_kernels.h")

add_custom_command(
    OUTPUT ${KERNEL_OROCHI_H}
    # 1) Optional Zstd compression
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_FILE=${KERNEL_OROCHI_COMP}
        -DOUTPUT_FILE=${KERNEL_OROCHI_COMP_COMPRESSED}
        -DDO_COMPRESS=${COMPILED_COMPRESSION}
        -P ${_archive_script}
    # 2) Convert binary to C array header
    COMMAND ${CMAKE_COMMAND}
        -DINPUT_FILE=${KERNEL_OROCHI_COMP}
        -DCOMPRESSED_FILE=${KERNEL_OROCHI_COMP_COMPRESSED}
        -DOUTPUT_HEADER=${KERNEL_OROCHI_H}
        -DCOMPRESSION_ACTIVATED=${COMPILED_COMPRESSION}
        -P ${_convert_script}
    DEPENDS ${KERNEL_OROCHI_COMP}
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Embedding Orochi compiled kernel into oro_compiled_kernels.h"
    VERBATIM
)

# Custom target for baked compiled kernels
add_custom_target(bake_compiled_kernels ALL
    DEPENDS ${KERNEL_HIPRT_H} ${KERNEL_OROCHI_H} precompile_kernels
)

add_dependencies(${HIPRT_NAME} precompile_kernels bake_compiled_kernels)

# --- Zstd embedded library ---
if(COMPILED_COMPRESSION)
    file(GLOB _zstd_srcs
        ${PROJECT_SOURCE_DIR}/contrib/zstd/lib/common/*.c
        ${PROJECT_SOURCE_DIR}/contrib/zstd/lib/decompress/*.c
    )

    add_library(zstd_embedded STATIC ${_zstd_srcs})
    target_include_directories(zstd_embedded PUBLIC ${PROJECT_SOURCE_DIR}/contrib/zstd/lib)
    set_target_properties(zstd_embedded PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(zstd_embedded PRIVATE ZSTD_DISABLE_ASM)

    target_link_libraries(${HIPRT_NAME} PRIVATE zstd_embedded)
    target_compile_definitions(${HIPRT_NAME} PRIVATE ORO_LINK_ZSTD)
endif()

# Compile definitions for BAKE_COMPILED_KERNEL mode
target_compile_definitions(${HIPRT_NAME} PRIVATE
    ORO_PP_LOAD_FROM_STRING
    HIPRT_BITCODE_LINKING
    ORO_PRECOMPILED
    HIPRT_BAKE_COMPILED_KERNEL
)
