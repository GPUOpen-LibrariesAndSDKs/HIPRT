# HIPRT
This is the main repository for the source code for HIPRT.

## Cloning and Building 

1. `git clone https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT`
2. `cd HIPRT`
3. `git submodule update --init --recursive`

&nbsp;&nbsp;&nbsp;On Windows:  
&nbsp;&nbsp;&nbsp;4. `.\tools\premake5\win\premake5.exe vs2022`  
&nbsp;&nbsp;&nbsp;5. `Open build\hiprt.sln with Visual Studio 2022.`  

&nbsp;&nbsp;&nbsp;On Linux:  
&nbsp;&nbsp;&nbsp;4. `./tools/premake5/linux64/premake5 gmake`  
&nbsp;&nbsp;&nbsp;5. `make -C build -j config=release_x64`  


### Using Bitcode
Add the option `--bitcode` to enable precompiled bitcode. 

#### Generation of bitcode
- After premake, go to `scripts/bitcodes`, then run `python compile.py` which compiles kernels to bitcode and fatbinary.
- Or pass `--precompile` to premake. it executes the `compile.py` during premake. Note that you cannot do it in git bash on windows (because of hipcc...)


## Running Unit Tests

There are three types of tests. 
1. HiprtTests           - tests covering all basic features.
2. ObjTestCases         - tests with loading meshes and testing advanced features like shadow/ AO.
3. PerformanceTestCases - tests with complex mesh to test performance features.

Example: `..\dist\bin\Release\unittest64.exe --width=512 --height=512 --referencePath=.\references\ --gtest_filter=hiprt*:Obj*" `

## Developing HIPRT

### Compiling Bundled Bitcode and Fatbinary 
- Clone `hipSdk` repo to the root directory.
- Go to `scripts/bitcodes`, run `python compile.py` which uses `hipcc` from the `hipSdk` directory. (todo. make it more general, maybe search for `hipcc` from path, if it's not found, use the directory above or something like this)
	- Note use python version 3.*+.
	- Git bash shell is not supported for compile.py.

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
