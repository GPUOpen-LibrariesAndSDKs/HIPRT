# FindHIP.cmake — HIP SDK discovery, compiler detection, GPU architecture list.
#
# Provides:
#   HIP_FOUND                — TRUE if HIP was found
#   HIP_PATH                 — resolved HIP SDK root
#   HIP_VERSION_MAJOR        — e.g. 6
#   HIP_VERSION_MINOR        — e.g. 4
#   HIP_VERSION_STR          — "6.4"  (for embedding in filenames)
#   HIP_VERSION_NUM          — e.g. 64  (for numeric comparisons)
#   HIP_GPU_ARCHITECTURES    — list of gfx arch strings
#   HIP_HIPCC_EXECUTABLE     — path to hipcc
#   HIP_CLANG_EXECUTABLE     — path to amdclang++ / clang++
#   HIP_NVIDIA_AVAILABLE     — TRUE if CUDA toolkit was found
#   Targets: hip::include, hip::host, hip::hip
#
# Functions:
#   hip_get_gpu_architectures(<out_var>)  — populate list of GPU arch strings

# --------------------------------------------------------------------------
# HIP SDK path resolution
# --------------------------------------------------------------------------
if(NOT DEFINED HIP_PATH)
    if(DEFINED ENV{HIP_PATH})
        set(HIP_PATH "$ENV{HIP_PATH}" CACHE PATH "Path to HIP installation")
    elseif(WIN32)
        set(HIP_PATH "C:/opt/rocm/6.4.2" CACHE PATH "Path to HIP installation")
    else()
        set(HIP_PATH "/opt/rocm" CACHE PATH "Path to HIP installation")
    endif()
endif()

# Strip trailing slashes
string(REGEX REPLACE "[/\\\\]+$" "" HIP_PATH "${HIP_PATH}")

if(NOT EXISTS "${HIP_PATH}")
    message(FATAL_ERROR "HIP not found at ${HIP_PATH}. Please set -DHIP_PATH=...")
endif()

message(STATUS "Using HIP_PATH: ${HIP_PATH}")
set(HIP_INCLUDE_DIR "${HIP_PATH}/include")
set(HIP_LIB_DIR     "${HIP_PATH}/lib")
set(HIP_BIN_DIR     "${HIP_PATH}/bin")

# --------------------------------------------------------------------------
# HIP version detection
# --------------------------------------------------------------------------
# Prefer reading hip_version.h, fall back to running hipcc --version.
set(_hip_version_header "${HIP_INCLUDE_DIR}/hip/hip_version.h")
if(EXISTS "${_hip_version_header}")
    file(READ "${_hip_version_header}" _hip_ver_content)
    string(REGEX MATCH "#define HIP_VERSION_MAJOR ([0-9]+)" _ "${_hip_ver_content}")
    set(HIP_VERSION_MAJOR "${CMAKE_MATCH_1}")
    string(REGEX MATCH "#define HIP_VERSION_MINOR ([0-9]+)" _ "${_hip_ver_content}")
    set(HIP_VERSION_MINOR "${CMAKE_MATCH_1}")
else()
    # Fallback: hipcc --version
    find_program(_hipcc_tmp NAMES hipcc hipcc.bat PATHS "${HIP_BIN_DIR}" NO_DEFAULT_PATH)
    if(_hipcc_tmp)
        execute_process(
            COMMAND "${_hipcc_tmp}" --version
            OUTPUT_VARIABLE _hipcc_out
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _ "${_hipcc_out}")
        set(HIP_VERSION_MAJOR "${CMAKE_MATCH_1}")
        set(HIP_VERSION_MINOR "${CMAKE_MATCH_2}")
    endif()
endif()

if(NOT DEFINED HIP_VERSION_MAJOR OR HIP_VERSION_MAJOR STREQUAL "")
    message(FATAL_ERROR "Could not determine HIP version from ${HIP_PATH}")
endif()

# Human-readable version for filenames: "6.4"
set(HIP_VERSION_STR "${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}")
# Numeric version for arch checks: 64 for 6.4, 61 for 6.1
math(EXPR HIP_VERSION_NUM "10 * ${HIP_VERSION_MAJOR} + ${HIP_VERSION_MINOR}")

message(STATUS "Detected HIP version: ${HIP_VERSION_STR} (numeric: ${HIP_VERSION_NUM})")

# --------------------------------------------------------------------------
# Find libraries
# --------------------------------------------------------------------------
find_library(HIP_LIBRARY
    NAMES amdhip64
    PATHS "${HIP_LIB_DIR}"
    NO_DEFAULT_PATH
)

find_library(HIP_COMGR_LIBRARY
    NAMES amd_comgr amd_comgr_3 amd_comgr_2 libamd_comgr
    PATHS "${HIP_LIB_DIR}"
    NO_DEFAULT_PATH
)

# --------------------------------------------------------------------------
# Find compilers
# --------------------------------------------------------------------------
find_program(HIP_HIPCC_EXECUTABLE
    NAMES hipcc hipcc.bat
    PATHS "${HIP_BIN_DIR}"
    NO_DEFAULT_PATH
)

if(WIN32)
    find_program(HIP_CLANG_EXECUTABLE
        NAMES clang++
        PATHS "${HIP_BIN_DIR}"
        NO_DEFAULT_PATH
    )
else()
    find_program(HIP_CLANG_EXECUTABLE
        NAMES amdclang++ clang++
        PATHS "${HIP_BIN_DIR}"
        NO_DEFAULT_PATH
    )
endif()

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
if(NOT HIP_LIBRARY)
    message(FATAL_ERROR "HIP runtime library (amdhip64) not found in ${HIP_LIB_DIR}")
endif()
if(NOT HIP_COMGR_LIBRARY)
    message(WARNING "amd_comgr library not found in ${HIP_LIB_DIR} — online compilation may not be available")
    set(HIP_COMGR_LIBRARY "")
endif()
if(NOT HIP_HIPCC_EXECUTABLE)
    message(FATAL_ERROR "hipcc compiler not found in ${HIP_BIN_DIR}")
endif()
if(NOT HIP_CLANG_EXECUTABLE)
    message(WARNING "amdclang++/clang++ not found in ${HIP_BIN_DIR} — bitcode linking will not be available")
endif()

message(STATUS "HIP libraries: ${HIP_LIBRARY}, ${HIP_COMGR_LIBRARY}")
message(STATUS "HIP compilers: hipcc=${HIP_HIPCC_EXECUTABLE}, clang=${HIP_CLANG_EXECUTABLE}")

# --------------------------------------------------------------------------
# Imported targets
# --------------------------------------------------------------------------
if(NOT TARGET hip::include)
    add_library(hip::include INTERFACE IMPORTED)
    set_target_properties(hip::include PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}")
endif()

if(NOT TARGET hip::host)
    add_library(hip::host INTERFACE IMPORTED)
    set(_hip_host_libs "${HIP_LIBRARY}")
    if(HIP_COMGR_LIBRARY)
        list(APPEND _hip_host_libs "${HIP_COMGR_LIBRARY}")
    endif()
    set_target_properties(hip::host PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${HIP_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${_hip_host_libs}")
endif()

if(NOT TARGET hip::hip)
    add_library(hip::hip INTERFACE IMPORTED)
    set_target_properties(hip::hip PROPERTIES
        INTERFACE_LINK_LIBRARIES hip::host)
endif()

# --------------------------------------------------------------------------
# GPU architecture list (port of common_tools.getAMDGPUArchs)
# --------------------------------------------------------------------------
function(hip_get_gpu_architectures out_var)
    # Base architectures (always available)
    set(_archs
        # Navi3
        gfx1100 gfx1101 gfx1102 gfx1103
        # Navi2
        # gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036
        # Not supported on Windows
        # Navi1
        # gfx1010 gfx1011 gfx1012 gfx1013
        # Vega
        # gfx900 gfx902 gfx904 gfx906 gfx908 gfx909 gfx90a gfx90c gfx942
    )

    # HIP 6.1+ — Strix Point
    if(HIP_VERSION_NUM GREATER_EQUAL 61)
        list(APPEND _archs gfx1150 gfx1151)
    endif()

    # HIP 6.3+
    if(HIP_VERSION_NUM GREATER_EQUAL 63)
        list(APPEND _archs gfx1152)
    endif()

    # HIP 6.4+ — Navi4 and Krackan
    if(HIP_VERSION_NUM GREATER_EQUAL 64)
        list(APPEND _archs gfx1200 gfx1201 gfx1153)
    endif()

    set(${out_var} ${_archs} PARENT_SCOPE)
endfunction()

# --------------------------------------------------------------------------
# NVIDIA support (optional)
# --------------------------------------------------------------------------
set(HIP_NVIDIA_AVAILABLE FALSE)
if(NOT FORCE_DISABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(HIP_NVIDIA_AVAILABLE TRUE)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION} — NVIDIA kernel compilation enabled")
    endif()
endif()

# --------------------------------------------------------------------------
set(HIP_FOUND TRUE)
