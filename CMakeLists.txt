cmake_minimum_required(VERSION 3.20)  # 3.20 needed for $<CONFIG> in add_custom_command OUTPUT
project(hiprt)

# ============================================================================
# Options
# ============================================================================

# Only one of these 3 flags should be ON at a time, or none.
option(BAKE_KERNEL "Enable the use of baked kernels source code" OFF)
option(BAKE_COMPILED_KERNEL "Compile and embed kernels as a buffer inside the binary (requires PRECOMPILE=ON)" OFF)
option(BITCODE "Enable bitcode linking: expects precompiled kernels as external files at runtime" OFF)

option(PRECOMPILE "Precompile kernels" OFF)
option(HIPRTEW "Use hiprtew" OFF)
option(NO_UNITTEST "Don't build unit tests" OFF)
option(COMPILED_COMPRESSION "Enable compression of compiled kernels (only with BAKE_COMPILED_KERNEL)" ON)
option(FORCE_DISABLE_CUDA "Force-disable CUDA support even if a CUDA install is detected" OFF)

# GENERATE_BAKE_KERNEL auto-enables with BAKE_KERNEL or BITCODE
if(BAKE_KERNEL OR BITCODE)
    set(_gen_bake_default ON)
else()
    set(_gen_bake_default OFF)
endif()
option(GENERATE_BAKE_KERNEL "Generate the baked kernel headers" ${_gen_bake_default})
message(STATUS "GENERATE_BAKE_KERNEL = ${GENERATE_BAKE_KERNEL}")

# ============================================================================
# Global settings
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Top-level project guard (for FetchContent safety)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(PROJECT_IS_TOP_LEVEL TRUE)
else()
    set(PROJECT_IS_TOP_LEVEL FALSE)
endif()

# Python (needed for bake_kernels.py)
find_program(PYTHON_EXECUTABLE NAMES python3 python
    PATHS /usr/bin /usr/local/bin /opt/local/bin
)
message(STATUS "Python path = ${PYTHON_EXECUTABLE}")

# ============================================================================
# HIP SDK detection (via cmake/FindHIP.cmake)
# ============================================================================
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(HIP REQUIRED)

# Populate GPU architecture list
hip_get_gpu_architectures(HIP_GPU_ARCHITECTURES)
set(HIP_FINAL_PATH "${HIP_PATH}")

# ============================================================================
# CUDA / Orochi CUEW support
# ============================================================================
if(NOT FORCE_DISABLE_CUDA)
    include(${PROJECT_SOURCE_DIR}/contrib/Orochi/Orochi/enable_cuew.cmake)
else()
    message(STATUS "CUDA support is forced disabled.")
endif()

# ============================================================================
# Version info and library name
# ============================================================================
function(get_version file)
    file(STRINGS ${file} lines)
    list(GET lines 0 major)
    list(GET lines 1 minor)
    list(GET lines 2 patch)
    list(GET lines 3 hash)
    set(major ${major} PARENT_SCOPE)
    set(minor ${minor} PARENT_SCOPE)
    set(patch ${patch} PARENT_SCOPE)
    set(hash "0x${hash}" PARENT_SCOPE)
endfunction()

function(write_version_info in_file header_file version_file version_str_out)
    if(NOT EXISTS ${version_file})
        message(FATAL_ERROR "version.txt missing!")
    endif()
    if(NOT EXISTS ${in_file})
        message(FATAL_ERROR "${in_file} missing!")
    endif()

    get_version(${version_file})
    file(READ ${in_file} header_content)

    math(EXPR HIPRT_VERSION "${major} * 1000 + ${minor}")

    # Zero-pad to 5 digits
    string(LENGTH "${HIPRT_VERSION}" _len)
    if(_len LESS 5)
        math(EXPR _pad "5 - ${_len}")
        string(REPEAT "0" ${_pad} _zeros)
        set(version_str "${_zeros}${HIPRT_VERSION}")
    else()
        set(version_str "${HIPRT_VERSION}")
    endif()

    set(HIPRT_API_VERSION ${HIPRT_VERSION})

    string(REPLACE "@HIPRT_MAJOR_VERSION@" "${major}" header_content "${header_content}")
    string(REPLACE "@HIPRT_MINOR_VERSION@" "${minor}" header_content "${header_content}")
    string(REPLACE "@HIPRT_PATCH_VERSION@" "${patch}" header_content "${header_content}")
    string(REPLACE "@HIPRT_HASH_VERSION@" "${hash}" header_content "${header_content}")
    string(REPLACE "@HIPRT_VERSION_STR@" "\"${version_str}\"" header_content "${header_content}")
    string(REPLACE "@HIPRT_API_VERSION@" "${HIPRT_API_VERSION}" header_content "${header_content}")
    string(REPLACE "@HIP_VERSION_STR@" "\"${HIP_VERSION_STR}\"" header_content "${header_content}")

    file(WRITE ${header_file} "${header_content}")
    set(${version_str_out} ${version_str} PARENT_SCOPE)
endfunction()

set(version_str_ "UNDEF")
write_version_info(
    "${PROJECT_SOURCE_DIR}/hiprt/hiprt.h.in"
    "${PROJECT_SOURCE_DIR}/hiprt/hiprt.h"
    "${PROJECT_SOURCE_DIR}/version.txt"
    version_str_)
write_version_info(
    "${PROJECT_SOURCE_DIR}/hiprt/hiprtew.h.in"
    "${PROJECT_SOURCE_DIR}/hiprt/hiprtew.h"
    "${PROJECT_SOURCE_DIR}/version.txt"
    version_str_)

set(HIPRT_NAME "hiprt${version_str_}")

# ============================================================================
# Build configurations
# ============================================================================
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;DebugGpu" CACHE STRING "Configs" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGGPU "")
set(CMAKE_EXE_LINKER_FLAGS_DEBUGGPU "")

# Platform-specific flags
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4305 /wd4018 /wd4996 /Zc:__cplusplus")
elseif(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

# Base output directory - let's keep it for now, but should be in build.
set(BASE_OUTPUT_DIR ${PROJECT_SOURCE_DIR}/dist/bin)
file(MAKE_DIRECTORY "${BASE_OUTPUT_DIR}/Debug")
file(MAKE_DIRECTORY "${BASE_OUTPUT_DIR}/Release")

foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
endforeach()
if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
endif()

# Platform OS postfix for kernel filenames
if(WIN32)
    set(KERNEL_OS_POSTFIX "win")
else()
    set(KERNEL_OS_POSTFIX "linux")
endif()

# ============================================================================
# HIPRT shared library
# ============================================================================
add_library(${HIPRT_NAME} SHARED)
target_compile_definitions(${HIPRT_NAME} PRIVATE
    HIPRT_EXPORTS
    GTEST_HAS_TR1_TUPLE=0
    __USE_HIP__
    HIPRT_PUBLIC_REPO
)
if(WIN32)
    target_compile_definitions(${HIPRT_NAME} PRIVATE __WINDOWS__)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(${HIPRT_NAME} PROPERTIES OUTPUT_NAME "${HIPRT_NAME}64D")
else()
    set_target_properties(${HIPRT_NAME} PROPERTIES OUTPUT_NAME "${HIPRT_NAME}64")
endif()

if(BITCODE)
    target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_BITCODE_LINKING ORO_PRECOMPILED)
endif()

if(WIN32)
    target_link_libraries(${HIPRT_NAME} PRIVATE version)
endif()

target_include_directories(${HIPRT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/contrib/Orochi
)

file(GLOB_RECURSE hiprt_sources
    "${PROJECT_SOURCE_DIR}/hiprt/*.h"
    "${PROJECT_SOURCE_DIR}/hiprt/*.cpp"
    "${PROJECT_SOURCE_DIR}/hiprt/*.inl"
)
list(FILTER hiprt_sources EXCLUDE REGEX "hiprt/bitcodes/.*")

file(GLOB_RECURSE orochi_sources
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/Orochi/*.h"
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/Orochi/*.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.h"
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.h"
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.cpp"
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.h"
    "${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.cpp"
)
target_sources(${HIPRT_NAME} PRIVATE ${hiprt_sources} ${orochi_sources})

# ============================================================================
# Include cmake modules for kernel compilation, baking, and embedding
# ============================================================================
include(${PROJECT_SOURCE_DIR}/cmake/compile_kernels.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/bake_kernels.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/embed_kernels.cmake)

# ============================================================================
# Unit test executable
# ============================================================================
if(NOT NO_UNITTEST)
    add_executable(unittest)

    if(BITCODE OR BAKE_COMPILED_KERNEL)
        target_compile_definitions(unittest PRIVATE HIPRT_BITCODE_LINKING)
    endif()

    target_compile_definitions(unittest PRIVATE GTEST_HAS_TR1_TUPLE=0)

    if(WIN32)
        target_compile_definitions(unittest PRIVATE __WINDOWS__)
        target_compile_options(unittest PRIVATE /wd4244)
        target_link_libraries(unittest PRIVATE version)
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set_target_properties(unittest PROPERTIES OUTPUT_NAME "unittest64D")
    else()
        set_target_properties(unittest PROPERTIES OUTPUT_NAME "unittest64")
    endif()

    target_include_directories(unittest PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/contrib/Orochi
    )
    target_link_libraries(unittest PRIVATE ${HIPRT_NAME})

    if(UNIX)
        target_link_libraries(unittest PRIVATE pthread dl)
    endif()

    file(GLOB_RECURSE unittest_sources
        "${PROJECT_SOURCE_DIR}/test/hiprtT*.h"
        "${PROJECT_SOURCE_DIR}/test/hiprtT*.cpp"
        "${PROJECT_SOURCE_DIR}/test/shared.h"
        "${PROJECT_SOURCE_DIR}/test/main.cpp"
        "${PROJECT_SOURCE_DIR}/test/CornellBox.h"
        "${PROJECT_SOURCE_DIR}/test/kernels/*.h"
        "${PROJECT_SOURCE_DIR}/contrib/gtest-1.6.0/gtest-all.cc"
    )
    target_sources(unittest PRIVATE ${unittest_sources} ${orochi_sources})

    target_include_directories(unittest PRIVATE
        ${PROJECT_SOURCE_DIR}/contrib/gtest-1.6.0
        ${PROJECT_SOURCE_DIR}/contrib/embree/include
    )

    if(WIN32)
        target_link_directories(unittest PRIVATE
            ${PROJECT_SOURCE_DIR}/contrib/embree/win
            ${PROJECT_SOURCE_DIR}/contrib/bin/win64
        )
        target_link_libraries(unittest PRIVATE ${PROJECT_SOURCE_DIR}/contrib/embree/win/embree4.lib)

        # Copy DLLs to output directories (only for top-level builds)
        if(PROJECT_IS_TOP_LEVEL)
            file(GLOB _embree_dlls "${PROJECT_SOURCE_DIR}/contrib/embree/win/*.dll")
            file(GLOB _orochi_dlls "${PROJECT_SOURCE_DIR}/contrib/Orochi/contrib/bin/win64/*.dll")
            foreach(_dll IN LISTS _embree_dlls _orochi_dlls)
                file(COPY ${_dll} DESTINATION ${BASE_OUTPUT_DIR}/Release)
                file(COPY ${_dll} DESTINATION ${BASE_OUTPUT_DIR}/Debug)
            endforeach()
        endif()
    endif()

    if(UNIX)
        target_link_directories(unittest PRIVATE ${PROJECT_SOURCE_DIR}/contrib/embree/linux)
    endif()

    target_link_libraries(unittest PRIVATE embree4 tbb)
endif()

# ============================================================================
# HIPRTEW test executable
# ============================================================================
if(HIPRTEW)
    add_executable(hiprtewtest)
    target_compile_definitions(hiprtewtest PRIVATE HIPRT_EXPORTS USE_HIPRTEW GTEST_HAS_TR1_TUPLE=0)
    if(WIN32)
        target_compile_definitions(hiprtewtest PRIVATE __WINDOWS__)
        target_compile_options(hiprtewtest PRIVATE /wd4244)
        target_link_libraries(hiprtewtest PRIVATE version)
    endif()

    target_include_directories(hiprtewtest PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/contrib/Orochi
    )
    if(UNIX)
        target_link_libraries(hiprtewtest PRIVATE pthread dl)
    endif()

    file(GLOB_RECURSE hiprtewtest_sources
        "${PROJECT_SOURCE_DIR}/test/hiprtewTest.h"
        "${PROJECT_SOURCE_DIR}/test/hiprtewTest.cpp"
        "${PROJECT_SOURCE_DIR}/contrib/gtest-1.6.0/gtest-all.cc"
    )
    target_sources(hiprtewtest PRIVATE ${hiprtewtest_sources} ${orochi_sources})
    target_include_directories(hiprtewtest PRIVATE ${PROJECT_SOURCE_DIR}/contrib/gtest-1.6.0)
endif()

# ============================================================================
# Install rules
# ============================================================================
include(${PROJECT_SOURCE_DIR}/cmake/install_rules.cmake)
