# cmake/compile_kernels.cmake
# Kernel compilation custom commands — replaces scripts/bitcodes/compile.py and precompile_bitcode.py
#
# Expected variables from parent scope:
#   HIP_HIPCC_EXECUTABLE   — path to hipcc
#   HIP_CLANG_EXECUTABLE   — path to amdclang++ (or clang++ on Windows)
#   HIP_FINAL_PATH         — root HIP SDK path
#   HIP_VERSION_STR        — e.g. "6.4"
#   HIPRT_NAME             — e.g. "hiprt03001"
#   version_str_           — e.g. "03001"
#   KERNEL_OS_POSTFIX      — "win" or "linux"
#   BASE_OUTPUT_DIR        — e.g. ${CMAKE_SOURCE_DIR}/dist/bin
#   HIP_GPU_ARCHITECTURES  — list of gfx arch strings
#   PARALLEL_JOBS          — number of parallel compilation jobs (default 15)

include_guard(GLOBAL)

if(NOT DEFINED PARALLEL_JOBS)
    set(PARALLEL_JOBS 15)
endif()

# Build --offload-arch flags from HIP_GPU_ARCHITECTURES list
set(_offload_arch_flags "")
foreach(_arch IN LISTS HIP_GPU_ARCHITECTURES)
    list(APPEND _offload_arch_flags "--offload-arch=${_arch}")
endforeach()

# Platform define
if(WIN32)
    set(_os_define "-DHIPCC_OS_WINDOWS")
else()
    set(_os_define "-DHIPCC_OS_LINUX")
endif()

# --- Source file lists for dependency tracking ---
set(_bvh_sources
    ${PROJECT_SOURCE_DIR}/hiprt/hiprt_vec.h
    ${PROJECT_SOURCE_DIR}/hiprt/hiprt_math.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/Aabb.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/AabbList.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/BvhCommon.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/BvhNode.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/QrDecomposition.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/Quaternion.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/Transform.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/Instance.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/InstanceList.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/MortonCode.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/Header.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/TriangleMesh.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/Triangle.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/BvhBuilderUtil.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/SbvhCommon.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/BvhConfig.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/NodeList.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/MemoryArena.h
    ${PROJECT_SOURCE_DIR}/hiprt/impl/Obb.h
    ${PROJECT_SOURCE_DIR}/hiprt/hiprt_types.h
    ${PROJECT_SOURCE_DIR}/hiprt/hiprt_common.h
)

# ============================================================================
# AMD Kernel Compilation
# ============================================================================

# Output paths use $<CONFIG> for multi-config generator correctness
set(KERNEL_HIPRT_BC     "${BASE_OUTPUT_DIR}/$<CONFIG>/${HIPRT_NAME}_${HIP_VERSION_STR}_amd_lib_${KERNEL_OS_POSTFIX}.bc")
set(KERNEL_HIPRT_COMP   "${BASE_OUTPUT_DIR}/$<CONFIG>/${HIPRT_NAME}_${HIP_VERSION_STR}_amd.hipfb")
set(KERNEL_OROCHI_COMP  "${BASE_OUTPUT_DIR}/$<CONFIG>/oro_compiled_kernels.hipfb")
set(KERNEL_UNITTEST_COMP "${BASE_OUTPUT_DIR}/$<CONFIG>/${HIPRT_NAME}_${HIP_VERSION_STR}_precompiled_bitcode_${KERNEL_OS_POSTFIX}.hipfb")

# Temp files for compiled kernel compression
set(KERNEL_HIPRT_COMP_COMPRESSED  "${CMAKE_BINARY_DIR}/${HIPRT_NAME}_${HIP_VERSION_STR}_amd.zstd")
set(KERNEL_OROCHI_COMP_COMPRESSED "${CMAKE_BINARY_DIR}/oro_compiled_kernels.zstd")

if(PRECOMPILE)
    # --- 1. HIPRT library bitcode (.bc) ---
    add_custom_command(
        OUTPUT ${KERNEL_HIPRT_BC}
        COMMAND ${HIP_HIPCC_EXECUTABLE}
            -x hip
            ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels_bitcode.h
            -O3 -std=c++17
            ${_offload_arch_flags}
            -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm
            -I${PROJECT_SOURCE_DIR}/contrib/Orochi/
            -I${PROJECT_SOURCE_DIR}
            -DHIPRT_BITCODE_LINKING ${_os_define}
            -ffast-math
            -parallel-jobs=${PARALLEL_JOBS}
            -o ${KERNEL_HIPRT_BC}
        DEPENDS ${_bvh_sources}
                ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels_bitcode.h
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Compiling HIPRT library bitcode (.bc)"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )

    # --- 2. HIPRT code object (.hipfb) ---
    add_custom_command(
        OUTPUT ${KERNEL_HIPRT_COMP}
        COMMAND ${HIP_HIPCC_EXECUTABLE}
            -x hip
            ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels.h
            -O3 -std=c++17
            ${_offload_arch_flags}
            --genco
            -I${PROJECT_SOURCE_DIR}
            -DHIPRT_BITCODE_LINKING ${_os_define}
            -ffast-math
            -parallel-jobs=${PARALLEL_JOBS}
            -o ${KERNEL_HIPRT_COMP}
        DEPENDS ${_bvh_sources}
                ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels.h
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Compiling HIPRT code object (.hipfb)"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )

    # --- 3. Orochi parallel primitives code object (.hipfb) ---
    add_custom_command(
        OUTPUT ${KERNEL_OROCHI_COMP}
        COMMAND ${HIP_HIPCC_EXECUTABLE}
            -x hip
            ${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/RadixSortKernels.h
            -O3 -std=c++17
            ${_offload_arch_flags}
            --genco
            -I${PROJECT_SOURCE_DIR}/contrib/Orochi/
            -include hip/hip_runtime.h
            -DHIPRT_BITCODE_LINKING ${_os_define}
            -ffast-math
            -parallel-jobs=${PARALLEL_JOBS}
            -o ${KERNEL_OROCHI_COMP}
        DEPENDS ${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/RadixSortKernels.h
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMENT "Compiling Orochi parallel primitives code object (.hipfb)"
        VERBATIM
        COMMAND_EXPAND_LISTS
    )

    add_custom_target(precompile_kernels ALL
        DEPENDS ${KERNEL_HIPRT_BC} ${KERNEL_HIPRT_COMP} ${KERNEL_OROCHI_COMP}
    )

    # --- 4. Unit test bitcode compilation and linking ---
    if(NOT NO_UNITTEST AND HIP_CLANG_EXECUTABLE)
        set(_unittest_custom_func "${CMAKE_BINARY_DIR}/${HIPRT_NAME}_${HIP_VERSION_STR}_custom_func_table.bc")
        set(_unittest_unit_test   "${CMAKE_BINARY_DIR}/${HIPRT_NAME}_${HIP_VERSION_STR}_unit_test_${KERNEL_OS_POSTFIX}.bc")

        # Compile custom_func_table.cpp to bitcode
        add_custom_command(
            OUTPUT ${_unittest_custom_func}
            COMMAND ${HIP_HIPCC_EXECUTABLE}
                -O3 -std=c++17
                ${_offload_arch_flags}
                -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm
                -I${PROJECT_SOURCE_DIR}
                ${_os_define}
                -ffast-math
                ${PROJECT_SOURCE_DIR}/test/bitcodes/custom_func_table.cpp
                -parallel-jobs=${PARALLEL_JOBS}
                -o ${_unittest_custom_func}
            DEPENDS ${PROJECT_SOURCE_DIR}/test/bitcodes/custom_func_table.cpp
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Compiling custom_func_table.cpp to bitcode"
            VERBATIM
            COMMAND_EXPAND_LISTS
        )

        # Compile unit_test.cpp to bitcode
        add_custom_command(
            OUTPUT ${_unittest_unit_test}
            COMMAND ${HIP_HIPCC_EXECUTABLE}
                -O3 -std=c++17
                ${_offload_arch_flags}
                -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm
                -I${PROJECT_SOURCE_DIR}
                ${_os_define}
                -ffast-math
                -DBLOCK_SIZE=64 -DSHARED_STACK_SIZE=16
                ${PROJECT_SOURCE_DIR}/test/bitcodes/unit_test.cpp
                -parallel-jobs=${PARALLEL_JOBS}
                -o ${_unittest_unit_test}
            DEPENDS ${PROJECT_SOURCE_DIR}/test/bitcodes/unit_test.cpp
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Compiling unit_test.cpp to bitcode"
            VERBATIM
            COMMAND_EXPAND_LISTS
        )

        # Link unit test bitcodes with HIPRT library bitcode
        add_custom_command(
            OUTPUT ${KERNEL_UNITTEST_COMP}
            COMMAND ${HIP_CLANG_EXECUTABLE}
                -fgpu-rdc --hip-link --cuda-device-only
                ${_offload_arch_flags}
                ${_unittest_custom_func}
                ${_unittest_unit_test}
                ${KERNEL_HIPRT_BC}
                -o ${KERNEL_UNITTEST_COMP}
            DEPENDS ${_unittest_custom_func} ${_unittest_unit_test} ${KERNEL_HIPRT_BC}
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
            COMMENT "Linking unit test precompiled bitcode (.hipfb)"
            VERBATIM
            COMMAND_EXPAND_LISTS
        )

        add_custom_target(precompile_unittest_kernels ALL
            DEPENDS ${KERNEL_UNITTEST_COMP}
        )
        add_dependencies(precompile_unittest_kernels precompile_kernels)
    endif()

    # ====================================================================
    # NVIDIA Kernel Compilation (optional)
    # ====================================================================
    if(HIP_NVIDIA_AVAILABLE)
        set(KERNEL_HIPRT_NV_LIB  "${BASE_OUTPUT_DIR}/$<CONFIG>/${HIPRT_NAME}_nv_lib.fatbin")
        set(KERNEL_HIPRT_NV      "${BASE_OUTPUT_DIR}/$<CONFIG>/${HIPRT_NAME}_nv.fatbin")
        set(KERNEL_OROCHI_NV     "${BASE_OUTPUT_DIR}/$<CONFIG>/oro_compiled_kernels.fatbin")

        set(_nvcc_ccbin "")
        if(WIN32 AND CMAKE_CXX_COMPILER)
            get_filename_component(_msvc_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
            set(_nvcc_ccbin "-ccbin=${_msvc_dir}")
        endif()

        add_custom_command(
            OUTPUT ${KERNEL_HIPRT_NV_LIB}
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
                -x cu
                ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels_bitcode.h
                -O3 ${_nvcc_ccbin}
                -std=c++17 -fatbin -arch=all --device-c
                -I${PROJECT_SOURCE_DIR}/contrib/Orochi/
                -I${PROJECT_SOURCE_DIR}
                -DHIPRT_BITCODE_LINKING --use_fast_math --threads 0
                -o ${KERNEL_HIPRT_NV_LIB}
            DEPENDS ${_bvh_sources}
                    ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels_bitcode.h
            COMMENT "Compiling HIPRT NVIDIA library fatbin"
            VERBATIM
        )

        add_custom_command(
            OUTPUT ${KERNEL_HIPRT_NV}
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
                -x cu
                ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels.h
                -O3 ${_nvcc_ccbin}
                -std=c++17 -fatbin -arch=all
                -I${PROJECT_SOURCE_DIR}/contrib/Orochi/
                -I${PROJECT_SOURCE_DIR}
                -DHIPRT_BITCODE_LINKING --use_fast_math --threads 0
                -o ${KERNEL_HIPRT_NV}
            DEPENDS ${_bvh_sources}
                    ${PROJECT_SOURCE_DIR}/hiprt/impl/hiprt_kernels.h
            COMMENT "Compiling HIPRT NVIDIA code object fatbin"
            VERBATIM
        )

        add_custom_command(
            OUTPUT ${KERNEL_OROCHI_NV}
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
                -x cu
                ${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/RadixSortKernels.h
                -O3 ${_nvcc_ccbin}
                -std=c++17 -fatbin -arch=all
                -I${PROJECT_SOURCE_DIR}/contrib/Orochi/
                -include cuda_runtime.h
                -DHIPRT_BITCODE_LINKING --use_fast_math --threads 0
                -o ${KERNEL_OROCHI_NV}
            DEPENDS ${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/RadixSortKernels.h
            COMMENT "Compiling Orochi NVIDIA parallel primitives fatbin"
            VERBATIM
        )

        add_custom_target(precompile_nvidia_kernels ALL
            DEPENDS ${KERNEL_HIPRT_NV_LIB} ${KERNEL_HIPRT_NV} ${KERNEL_OROCHI_NV}
        )
    endif()
endif()
