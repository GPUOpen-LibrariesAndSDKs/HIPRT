# HIPRT

## About 
HIP RT is a ray tracing library for HIP, making it easy to write ray-tracing applications in HIP. The APIs and library are designed to be minimal, lower level, and simple to use and integrate into any existing HIP applications.

Although there are other ray tracing APIs which introduce many new things, we designed HIP RT in a slightly different way so you do not need to learn many new kernel types.

Released binaries can be found at [HIP RT page under GPUOpen](https://gpuopen.com/hiprt/).
HIP RT library is developed and maintained by ARR, [Advanced Rendering Research Group](https://gpuopen.com/advanced-rendering-research/). 

## Development

This is the main repository for the source code for HIPRT.

## Cloning and Building 

1. `git clone https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT.git`
2. `cd HIPRT`
3. `git submodule update --init --recursive`
4. `git lfs fetch` (To get resources for running performance tests)

### Building with CMake Presets (recommended)

```bash
# List all presets
cmake --list-presets

# Configure + build (Ninja, cross-platform)
cmake --preset plain
cmake --build --preset plain-release

# Configure + build using Visual Studio 2022
cmake --preset plain-vs2022
cmake --build --preset plain-vs2022-release

# Configure + build using Visual Studio 2019
cmake --preset plain-vs2019
cmake --build --preset plain-vs2019-release
```

Available configure presets:

| Preset | Description |
|--------|-------------|
| `plain` | Baseline library, no precompile, JIT mode |
| `bake-kernel` | Stringified kernel source embedded in binary |
| `precompile` | External `.hipfb` generated at build time |
| `bake-compiled` | Compiled kernel blobs embedded in binary |
| `bake-compiled-nozip` | Same without Zstd compression |
| `nvidia` | AMD + NVIDIA fatbin path |
| `nvidia-only` | CI fast-path for NVIDIA-only machines |
| `no-tests` | Library-only build (e.g. for FetchContent consumers) |
| `plain-vs2022` | Plain build with Visual Studio 17 2022 generator |
| `plain-vs2019` | Plain build with Visual Studio 16 2019 generator |

### Building with CMake manually

#### On Windows:  
5. `cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DHIP_PATH="C:\Program Files\AMD\ROCm\6.4"`  
6. `cmake --build build --config Release`  

#### On Linux:  
5. `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHIP_PATH=/opt/rocm`  
6. `cmake --build build -j`  

### Using Bitcode
Pass `-DBITCODE=ON` to cmake, or use the `precompile` preset to enable precompiled bitcode.

With `PRECOMPILE=ON`, kernel compilation happens at build time via CMake custom commands — no external scripts needed.


## Running Unit Tests

There are three types of tests. 
1. HiprtTests           - tests covering all basic features.
2. ObjTestCases         - tests with loading meshes and testing advanced features like shadow/ AO.
3. PerformanceTestCases - tests with complex mesh to test performance features.

Example: `..\dist\bin\Release\unittest64.exe --width=512 --height=512 --referencePath=.\references\ --gtest_filter=hiprt*:Obj*" `

## Developing HIPRT

### Compiling Bundled Bitcode and Fatbinary 
Use the `precompile` preset or pass `-DPRECOMPILE=ON` to cmake. Kernel compilation is handled by CMake custom commands invoking `hipcc`/`amdclang++`/`nvcc` directly — no external Python scripts needed.

### Coding Guidelines
- Resolve compiler warnings.
- Use lower camel case for variable names (e.g., `nodeCount`) and upper camel case for constants (e.g., `LogSize`).
- Separate functions by one line.
- Use prefix `m_` for non-static member variables.
- Do not use static local variables.
- Do not use `void` for functions without arguments (leave it blank).
- Do not use blocks without any reason.
- Use references instead of pointers if possible.
- Use bit-fields instead of explicit bit masking if possible.
- Use `nullptr` instead of `NULL` or zero.
- Use `using` instead of `typedef`.
- Use C++-style casts (e.g., `static_cast`) instead of C-style cast.
- Add `const` for references and pointers if they are not being changed.
- Add `constexpr` for variables and functions if they can be constant in compile time (do not use `#define` if possible).
- Use `if constsexpr` instead of `#ifdef` if possible.
- Throw `std::runtime_error` with an appropriate message in case of failure in the core and catch it in `hiprt.cpp`.

#### String
- Use `std::string` instead of C strings (i.e., `char*`) and avoid C string functions as much as possible.
- Use `std::cout` and `std::cerr` instead of `printf`.
- Do not assign `char8_t` (or `std::u8string`) to `char` (or `std::string`). They will not be compatible in C++20.

#### File
- Use `std::ifstream` and `std::ofstream` instead of `FILE`.
- Use `std::filesystem::path` for files and paths instead of `std::string`.

#### Class
- Use the in-class initializer instead of the default constructor.
- Use the keyword `override` instead of `virtual` (or nothing) when overriding a virtual function from the base class.
  - Reason: The `override` keyword can help prevent bugs by producing compilation errors when the intended override is not actually implemented as an override. For example, when the function type is not exactly identical to the base class function. This can be caused by mistakes or if the virtual functions in the base class are changed due to refactor.
- Use `std::optional` instead of pointers for optional parameters.
  - Reason: `std::optional` guarantees that no auxiliary memory allocation is needed. Meaning, it does not involve dynamic memory allocation & deallocation on the heap, which results in better performance and less memory overhead.
- A base class destructor should be either public and virtual, or protected and non-virtual
  - Reason: This is to prevent undefined behavior. If the destructor is public, then the calling code can attempt to destroy a derived class object/instance through a base class pointer, and the result is undefined if the base class’s destructor is non-virtual.
- Implement the customized {copy/move} {constructor/assignment operator} if an user-defined destructor of a class is needed, or remove them using `= delete`
  - Reason: [Rule of five](https://en.cppreference.com/w/cpp/language/rule_of_three)

### Versioning
- When we update the master branch, we need to update the version number of hiprt in `version.txt`.
- If there is a change in the API, you need to update minor version. 
- If the major and minor versions matches, the binaries are compatible. 
- Each commit in the master should have a unique patch version. 
