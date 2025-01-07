2.5.cfa5e2a
- More flexible vector types
- Unifying math into a single header
- Collapse crash fix
- Other minor fixes

2.4.c587aa7
- H-PLOC and improved wide BVH conversion
- CMake support
- Configurable HIPRT path via a env. variable
- New gfx architectures supported 
- hiprtBuildTraceKernel can return only the HIP module
- HIP module caching and unloading (fixing a memory leak)
- Fixing matrix inversion and identity check
- Fixing refit and other minor issues

2.3.7df94af
- Transformation query API changed/extended

2.2.0e68f54 (December 2023)
- Multi-level instancing
- Triangle pairing
- AS Compaction
- Optimized BVH build speed

2.1.c202dac (November 2023)
- HIPRT binaries compiled with ROCm 5.7
- A fix for caching trace kernels
- A fix for the custom function table compilation
- A fix for the fast and balanced builders with custom streams

2.1.6fc8ff0 (September 2023)
- Dynamic traversal stack assignment
- Batch BVH construction
- Transformation query functions
- Improved BVH construction speed
- Improved RT speed for transformed instances
- Fixed geometry IO API
- Optional trace kernel caching

2.0.3a134c7 (May 2023)
- BVH memory optimization
- SBVH speed optimization
- Fixing hiprtBuildTraceKernels
- Dynamic loading via HIPRTEW
- Traversal optimization

2.0.0 (February 2023)
- Bitcode and precompilation (527.41 or newer driver is necessary to run on NVIDIA® on Windows®)
- Performance improvement
- Navi3x support
- MI60 and MI200 support
- Traversal hints for better performance
- Concurrent build via streams
- Custom function table
- Intersection filter
- Transformation matrices support
- Multiple templated kernels
- Added ray t min
