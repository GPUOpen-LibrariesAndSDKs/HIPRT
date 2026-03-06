# cmake/install_rules.cmake
# Installation rules for HIPRT
#
# Layout:
#   include/hiprt/          — public headers
#   include/hiprt/impl/     — implementation headers (excluding generated bvh_build_array.h)
#   include/contrib/Orochi/ParallelPrimitives/ — Orochi parallel primitives headers
#   lib/                    — import libraries, .bc bitcode files
#   bin/                    — shared libraries, executables, .hipfb/.fatbin device code

include_guard(GLOBAL)
include(GNUInstallDirs)

# --- Library ---
install(TARGETS ${HIPRT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# --- Public headers ---
file(GLOB _hiprt_headers "${PROJECT_SOURCE_DIR}/hiprt/*.h")
install(FILES ${_hiprt_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hiprt
)

# Implementation headers (exclude generated bvh_build_array.h)
file(GLOB _hiprt_impl_headers "${PROJECT_SOURCE_DIR}/hiprt/impl/*.h")
list(REMOVE_ITEM _hiprt_impl_headers "${PROJECT_SOURCE_DIR}/hiprt/impl/bvh_build_array.h")
install(FILES ${_hiprt_impl_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hiprt/impl
)

# Orochi parallel primitive headers
file(GLOB _oro_pp_headers "${PROJECT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.h")
install(FILES ${_oro_pp_headers}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/contrib/Orochi/ParallelPrimitives
)

# --- Precompiled device code ---
#if(PRECOMPILE AND NOT BAKE_COMPILED_KERNEL)
if(PRECOMPILE) # always include bc for flexibility
    # .hipfb code objects to bin/
    install(FILES ${KERNEL_HIPRT_COMP} ${KERNEL_OROCHI_COMP}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # .bc bitcode files to lib/
    if(DEFINED KERNEL_HIPRT_BC)
        install(FILES ${KERNEL_HIPRT_BC}
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()

    # NVIDIA fatbins
    if(HIP_NVIDIA_AVAILABLE)
        install(FILES ${KERNEL_HIPRT_NV} ${KERNEL_HIPRT_NV_LIB} ${KERNEL_OROCHI_NV}
            DESTINATION ${CMAKE_INSTALL_BINDIR}
        )
    endif()
endif()

# --- Third-party DLLs (only for top-level builds) ---
if(PROJECT_IS_TOP_LEVEL AND WIN32 AND NOT NO_UNITTEST)
    file(GLOB _embree_dlls "${PROJECT_SOURCE_DIR}/contrib/embree/win/*.dll")
    file(GLOB _orochi_dlls "${PROJECT_SOURCE_DIR}/contrib/Orochi/contrib/bin/win64/*.dll")
    install(FILES ${_embree_dlls} ${_orochi_dlls}
        DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
