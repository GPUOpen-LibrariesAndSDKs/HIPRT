# cmake/bake_kernels.cmake
# Invoke tools/bake_kernels.py to generate stringified kernel cache headers.
#
# Expected variables from parent scope:
#   PYTHON_EXECUTABLE  — path to Python 3 interpreter
#   PROJECT_SOURCE_DIR — root of the HIPRT source tree

include_guard(GLOBAL)

if(BAKE_KERNEL OR GENERATE_BAKE_KERNEL)
    message(STATUS "Baking kernel source strings via tools/bake_kernels.py")

    execute_process(
        COMMAND ${PYTHON_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tools/bake_kernels.py
            --source-dir ${PROJECT_SOURCE_DIR}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        RESULT_VARIABLE _bake_result
    )

    if(NOT _bake_result EQUAL 0)
        message(FATAL_ERROR "bake_kernels.py failed with exit code ${_bake_result}")
    endif()

    target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_BAKE_KERNEL_GENERATED)
endif()

if(BAKE_KERNEL)
    target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_LOAD_FROM_STRING ORO_PP_LOAD_FROM_STRING)
endif()
