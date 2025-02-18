cmake_minimum_required(VERSION 3.10)
project(hiprt)

#
# Options
#

# Only one of those 3 flags should be ON at a time, or none. Multiple flags ON are untested and may cause issues.
option(BAKE_KERNEL "Enable the use of encrypted and baked kernels source code" OFF)
option(BAKE_COMPILED_KERNEL "If enabled, the kernels will be compiled and embedded as a buffer inside the binary. if you do that, you should also enable the flag 'PRECOMPILE'" OFF)
option(BITCODE "Enable bitcode linking: when running the program, it's expected to find precompiled kernels as external files." OFF)

option(PRECOMPILE "Precompile kernels" OFF)
option(HIPRTEW "Use hiprtew" OFF)
option(NO_ENCRYPT "Don't encrypt kernel source and binaries" OFF)
option(NO_UNITTEST "Don't build unit tests" OFF)
option(HIPRT_PREFER_HIP_5 "Prefer HIP 5" OFF)

option(FORCE_DISABLE_CUDA "By default Cuda support is automatically added if a Cuda install is detected. Turn this flag to ON to force Cuda to be disabled." OFF)


find_program(PYTHON_EXECUTABLE
	NAMES python
	PATHS /usr/bin /usr/local/bin /opt/local/bin
)
message(STATUS "Python path = ${PYTHON_EXECUTABLE}")



# GENERATE_BAKE_KERNEL is enabled by default if we use the flags 'BAKE_KERNEL' or 'BITCODE'.
# It can be forced to OFF, but in this case, some building functions from the HIPRT API, like hiprtBuildTraceKernelsFromBitcode will fail.
if(BAKE_KERNEL OR BITCODE)
	set(GENERATE_BAKE_KERNEL___DEFAULT ON)
else()
	set(GENERATE_BAKE_KERNEL___DEFAULT OFF)
endif()
option(GENERATE_BAKE_KERNEL "generate the baked kernels" ${GENERATE_BAKE_KERNEL___DEFAULT})
message(STATUS "GENERATE_BAKE_KERNEL= ${GENERATE_BAKE_KERNEL}")


# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define GTEST_HAS_TR1_TUPLE=0 globally
add_definitions(-DGTEST_HAS_TR1_TUPLE=0)







# Functions
function(copy_dir src_dir dst_dir pattern)
	file(GLOB_RECURSE files "${src_dir}/${pattern}")
	foreach(file ${files})
		file(RELATIVE_PATH rel_file ${src_dir} ${file})
		set(target "${dst_dir}/${rel_file}")
		get_filename_component(target_dir ${target} DIRECTORY)
		file(MAKE_DIRECTORY ${target_dir})
		file(COPY ${file} DESTINATION ${target_dir})
	endforeach()
endfunction()



function(file_exists file)
	if(EXISTS ${file})
		return(0)
	else()
		return(1)
	endif()
endfunction()

function(read_file file)
	file(READ ${file} content)
	return(${content})
endfunction()



function(get_version file)
	file(STRINGS ${file} lines)
	list(GET lines 0 major)
	list(GET lines 1 minor)
	list(GET lines 2 patch)
	set(patch_hex "0x${patch}")
	set(major ${major} PARENT_SCOPE)
	set(minor ${minor} PARENT_SCOPE)
	set(patch ${patch_hex} PARENT_SCOPE)
endfunction()



# Remove the last character from the string if it's a / or \\  
function(remove_trailing_slash strInOut)
	string(LENGTH "${${strInOut}}" strLength)
	math(EXPR lastIndex "${strLength} - 1")
	string(SUBSTRING "${${strInOut}}" ${lastIndex} 1 lastChar)
	if("${lastChar}" STREQUAL "\\" OR "${lastChar}" STREQUAL "/")
		string(SUBSTRING "${${strInOut}}" 0 ${lastIndex} strInOut_modified)
		set(${strInOut} "${strInOut_modified}" PARENT_SCOPE)
	endif()
endfunction()




# 'result'      is the HIP version as string, for example: 6.2
# 'result_path' is the output of the path to HIP, for example:  C:\Program Files\AMD\ROCm\6.2
function(get_hip_sdk_version result result_path)
	if(WIN32)
		set(root ".\\")
	endif()

	set(exec_perl "")
	set(hipCommand "hipcc")
	set(PATH $ENV{PATH})
	set(useHipFromPATH OFF)


	# Check if HIP_PATH is defined as a CMake parameter
	if(DEFINED HIP_PATH)
		message(STATUS "HIP_PATH is defined as a CMake parameter: ${HIP_PATH}")

	# Check if HIP_PATH is defined as an environment variable
	elseif(DEFINED ENV{HIP_PATH})
		
		set(HIP_PATH $ENV{HIP_PATH})
		message(STATUS "HIP_PATH is defined as an environment variable: ${HIP_PATH}")
			
	# if HIP_PATH is not in cmake, and not in environment variable
	else()
		message(WARNING "WARNING: HIP_PATH is not defined as a CMake parameter or an environment variable - NOT RECOMMENDED")

		# TODO: improve that, but it's not recommanded to use the cmake script without defining HIP_PATH anyway...
		set(${result_path} "UNKONWN_PATH" PARENT_SCOPE)

		# Check if HIP is in the PATH environment variable
		string(REPLACE ";" "\n" PATH_LIST ${PATH})
		foreach(token ${PATH_LIST})
			if("${token}" MATCHES "hip")
				if(EXISTS "${token}/hipcc")
					set(useHipFromPATH ON)
				endif()
			endif()
		endforeach()


	endif()


	# clean/format HIP_PATH here.
	if ( HIP_PATH )
		remove_trailing_slash(HIP_PATH)
		# message(STATUS "HIP_PATH formatted: ${HIP_PATH}")
		set(${result_path} ${HIP_PATH} PARENT_SCOPE)
	endif()


	# build hip command for Windows
	if(WIN32)
		set(exec_perl "perl")

		if(NOT HIP_PATH)
			if(useHipFromPATH)
				set(hipCommand "hipcc")
			else()
				# try classic path used by HIPRT developers
				set(hipCommand "hipSdk\\bin\\hipcc")
			endif()
		else()
		
			# HIP_PATH is expected to look like: C:\Program Files\AMD\ROCm\5.7
			
			if(EXISTS "${HIP_PATH}\\bin\\hipcc.exe")
				# in newer version of HIP SDK (>= 6.3), we are using 'hipcc.exe --version' to check the version
				# message(STATUS "using hipcc.exe to get the version")
				set(exec_perl "")
				set(hipCommand "${HIP_PATH}\\bin\\hipcc.exe")
			else()
				# in older version of HIP SDK, we are using 'perl hipcc --version' to check the version
				# message(STATUS "using perl hipcc to get the version")
				set(hipCommand "${HIP_PATH}\\bin\\${hipCommand}")
			endif()


		endif()
	
	# build hip command for Linux
	else()
	
		# If not defined we try to take it from the PATH
		if(NOT HIP_PATH)
			set(hipCommand "hipcc")
			
		# otherwise, build the hipcc command with full path.
		else()
			set(hipCommand "${HIP_PATH}/bin/${hipCommand}")
		endif()
	endif()


	file(WRITE ${CMAKE_BINARY_DIR}/hip_version_tmp.txt "")

	# message(STATUS "hipCommand : ${hipCommand}")
	# message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

	execute_process(
		COMMAND ${exec_perl} "${hipCommand}" --version
		OUTPUT_FILE ${CMAKE_BINARY_DIR}/hip_version_tmp.txt
		# ERROR_QUIET
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		)

	file(READ ${CMAKE_BINARY_DIR}/hip_version_tmp.txt version_output)
	string(REGEX MATCH "[0-9]+\\.[0-9]+" version "${version_output}")

	file(REMOVE ${CMAKE_BINARY_DIR}/hip_version_tmp.txt)

	if(NOT version)
		set(version "HIP_SDK_NOT_FOUND")
	endif()

	message(STATUS "HIP VERSION from command : ${version}")
	set(${result} ${version} PARENT_SCOPE)
endfunction()


# fill the HIP_VERSION_STR variable, it will look like: "6.2"
# also get the path to hip that'll be used along all this cmake.
get_hip_sdk_version(HIP_VERSION_STR HIP_FINAL_PATH)
message(STATUS "HIP_VERSION_STR = ${HIP_VERSION_STR}")
message(STATUS "HIP_FINAL_PATH = ${HIP_FINAL_PATH}")

function(write_version_info in_file header_file version_file version_str_out)

	if(NOT EXISTS ${version_file})
		message(FATAL_ERROR "Version.txt file missing!")
	endif()
	if(NOT EXISTS ${in_file})
		message(FATAL_ERROR "${in_file} file is missing!")
	endif()

	# Read version file and extract version information
	get_version(${version_file})

	# set(version "${major}${minor}")
	# set(version_str "${version}_${patch}")

	# Read the content of the header template file
	file(READ ${in_file} header_content)

	# Calculate HIPRT_API_VERSION
	math(EXPR HIPRT_VERSION "${major} * 1000 + ${minor}")


	# Format version_str as a zero-padded 5-digit string
	string(LENGTH "${HIPRT_VERSION}" HIPRT_VERSION_LEN)
	if(${HIPRT_VERSION_LEN} LESS 5)
		math(EXPR HIPRT_VERSION_PAD "5 - ${HIPRT_VERSION_LEN}")
		string(REPEAT "0" ${HIPRT_VERSION_PAD} HIPRT_VERSION_PADDED)
		set(version_str "${HIPRT_VERSION_PADDED}${HIPRT_VERSION}" )
	else()
		set(version_str "${HIPRT_VERSION}" )
	endif()

	# message(STATUS "HIPRT_API_VERSION: ${version_str}_${patch}")

	set(HIPRT_API_VERSION ${HIPRT_VERSION})

	# Replace placeholders with actual version values
	string(REPLACE "@HIPRT_MAJOR_VERSION@" "${major}" header_content "${header_content}")
	string(REPLACE "@HIPRT_MINOR_VERSION@" "${minor}" header_content "${header_content}")
	string(REPLACE "@HIPRT_PATCH_VERSION@" "${patch}" header_content "${header_content}")
	string(REPLACE "@HIPRT_VERSION_STR@" "\"${version_str}\"" header_content "${header_content}")
	string(REPLACE "@HIPRT_API_VERSION@" "${HIPRT_API_VERSION}" header_content "${header_content}")

	# Get HIP SDK version and replace placeholder
	string(REPLACE "@HIP_VERSION_STR@" "\"${HIP_VERSION_STR}\"" header_content "${header_content}")

	# Write the modified content to the header file
	file(WRITE ${header_file} "${header_content}")

	set(${version_str_out} ${version_str} PARENT_SCOPE)
endfunction()


# Set up configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;DebugGpu" CACHE STRING "Configs" FORCE)

set(CMAKE_SHARED_LINKER_FLAGS_DEBUGGPU "")
set(CMAKE_EXE_LINKER_FLAGS_DEBUGGPU "")

# Define platform-specific flags and settings
if(WIN32)
	add_definitions(-D__WINDOWS__)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4305 /wd4018 /wd4996 /Zc:__cplusplus")
elseif(UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

add_definitions(-D__USE_HIP__)


# this define is to identify that we are on the public repository of HIPRT.
# it helps AMD to maintain both a public and a private repo for experimentation.
add_definitions(-DHIPRT_PUBLIC_REPO)


if (NOT FORCE_DISABLE_CUDA)
	# Enable CUDA if possible
	include(${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/Orochi/enable_cuew.cmake)
else()
	message(STATUS "CUDA support is forced to disabled.")
endif()


# Base output directory
set(BASE_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin)
file(MAKE_DIRECTORY "${BASE_OUTPUT_DIR}/Debug")
file(MAKE_DIRECTORY "${BASE_OUTPUT_DIR}/Release")

# Set output directories for each build configuration
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} Debug Release RelWithDebInfo MinSizeRel)
	string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG_UPPER)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG_UPPER} ${BASE_OUTPUT_DIR}/${OUTPUTCONFIG})
endforeach()

# Set output directories for single-configuration generators
if(NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE})
endif()




# Write version info
set(version_str_ "UNDEF")
write_version_info("${CMAKE_CURRENT_SOURCE_DIR}/hiprt/hiprt.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/hiprt.h" "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" version_str_)
write_version_info("${CMAKE_CURRENT_SOURCE_DIR}/hiprt/hiprtew.h.in" "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/hiprtew.h" "${CMAKE_CURRENT_SOURCE_DIR}/version.txt" version_str_)


set(HIPRT_NAME "hiprt${version_str_}")

# Project: HIPRT
add_library(${HIPRT_NAME} SHARED)

target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_EXPORTS)


if( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
	set_target_properties(${HIPRT_NAME} PROPERTIES OUTPUT_NAME "${HIPRT_NAME}64D")
else()
	set_target_properties(${HIPRT_NAME} PROPERTIES OUTPUT_NAME "${HIPRT_NAME}64")
endif()


if(BITCODE)
	target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_BITCODE_LINKING ORO_PRECOMPILED)
endif()

if(NOT NO_ENCRYPT)
	target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_ENCRYPT )
	set(ENV{HIPRT_NO_ENCRYPT} "OFF")
else()
	set(ENV{HIPRT_NO_ENCRYPT} "ON")
endif()

if(HIPRT_PREFER_HIP_5)
	add_definitions(-DHIPRT_PREFER_HIP_5)
endif()



# files generated by compile.py and precompile_bitcode.py
if(WIN32)
	set(KERNEL_OS_POSTFIX "win")
else()
	set(KERNEL_OS_POSTFIX "linux")
endif()
set(KERNEL_HIPRT_COMP    "${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/hiprt${version_str_}_${HIP_VERSION_STR}_amd.hipfb")                                        # example:  hiprt02005_6.2_amd.hipfb
set(KERNEL_UNITTEST_COMP "${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/hiprt${version_str_}_${HIP_VERSION_STR}_precompiled_bitcode_${KERNEL_OS_POSTFIX}.hipfb")   # example:  hiprt02005_6.2_precompiled_bitcode_win.hipfb
set(KERNEL_OROCHI_COMP "${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/oro_compiled_kernels.hipfb")


# precompile kernels:
if(PRECOMPILE)

	if(FORCE_DISABLE_CUDA)
		set(CUDA_OPTION "")
	else()
		set(CUDA_OPTION "--nvidia")
	endif()


	set(bvh_source
	${CMAKE_SOURCE_DIR}/hiprt/hiprt_vec.h
	${CMAKE_SOURCE_DIR}/hiprt/hiprt_math.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/Aabb.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/AabbList.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/BvhCommon.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/BvhNode.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/Geometry.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/QrDecomposition.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/Quaternion.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/Transform.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/Instance.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/InstanceList.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/MortonCode.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/Scene.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/TriangleMesh.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/Triangle.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/BvhBuilderUtil.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/SbvhCommon.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/ApiNodeList.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/BvhConfig.h
	${CMAKE_SOURCE_DIR}/hiprt/impl/MemoryArena.h
	${CMAKE_SOURCE_DIR}/hiprt/hiprt_types.h
	${CMAKE_SOURCE_DIR}/hiprt/hiprt_common.h
	)

	message(">> add_custom_command: ${PYTHON_EXECUTABLE} compile.py ${CUDA_OPTION} --hipSdkPath \"${HIP_FINAL_PATH}\"")
	add_custom_command(
			OUTPUT ${KERNEL_HIPRT_COMP} ${KERNEL_OROCHI_COMP}
			COMMAND ${PYTHON_EXECUTABLE} compile.py ${CUDA_OPTION} --hipSdkPath ${HIP_FINAL_PATH}
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bitcodes
			COMMENT "Precompiling kernels via compile.py"
			VERBATIM
			DEPENDS ${bvh_source}
		)
		
	# create the 'precompile_kernels' project
	add_custom_target(precompile_kernels ALL
		DEPENDS ${KERNEL_HIPRT_COMP} ${KERNEL_OROCHI_COMP}
		)
	
	if(NOT NO_UNITTEST)

		set(unittest_kernel_source
		${CMAKE_SOURCE_DIR}/test/bitcodes/custom_func_table.cpp
		${CMAKE_SOURCE_DIR}/test/bitcodes/unit_test.cpp
		)
		
		message(">> add_custom_command: ${PYTHON_EXECUTABLE} precompile_bitcode.py ${CUDA_OPTION} --hipSdkPath \"${HIP_FINAL_PATH}\"")
		add_custom_command(
			OUTPUT ${KERNEL_UNITTEST_COMP}
			COMMAND ${PYTHON_EXECUTABLE} precompile_bitcode.py ${CUDA_OPTION} --hipSdkPath ${HIP_FINAL_PATH}
			DEPENDS ${KERNEL_HIPRT_COMP}  # Ensure compile.py has already run.
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bitcodes
			COMMENT "Precompiling unit tests kernels via precompile_bitcode.py"
			VERBATIM
			DEPENDS ${unittest_kernel_source}
			)
			
		# create the 'precompile_unittest_kernels' project
		add_custom_target(precompile_unittest_kernels ALL
			DEPENDS ${KERNEL_UNITTEST_COMP}
			)
			
		add_dependencies(${HIPRT_NAME} precompile_unittest_kernels)
	endif()
	

endif()



# convert the binary to a buffer that will be embeded inside the binary
# it's expected the step 'PRECOMPILE' has been executed.
if ( BAKE_COMPILED_KERNEL )
	
	message(">> precompiled will be embedded.")
	
	set(PYTHON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/scripts/convert_binary_to_array.py")

	# HIPRT binary
	set(KERNEL_HIPRT_H "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/impl/bvh_build_array.h")
	add_custom_command(
		OUTPUT ${KERNEL_HIPRT_H}
		COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_FILE} ${KERNEL_HIPRT_COMP} ${KERNEL_HIPRT_H}
		DEPENDS ${KERNEL_HIPRT_COMP}  # Ensure compile.py has already run.
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Converting HIPRT compiled kernel to header"
		VERBATIM
	)

	# Orochi binary
	set(KERNEL_OROCHI_H "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/cache/oro_compiled_kernels.h")
	add_custom_command(
		OUTPUT ${KERNEL_OROCHI_H}
		COMMAND ${PYTHON_EXECUTABLE} ${PYTHON_FILE} ${KERNEL_OROCHI_COMP} ${KERNEL_OROCHI_H}
		DEPENDS ${KERNEL_OROCHI_COMP}  # Ensure compile.py has already run.
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
		COMMENT "Converting Orochi compiled kernel to header"
		VERBATIM
	)

	# Create the 'bake_compiled_kernels' project
	add_custom_target(bake_compiled_kernels ALL
		DEPENDS ${KERNEL_HIPRT_H} ${KERNEL_OROCHI_H} precompile_kernels
	)

	add_dependencies(${HIPRT_NAME} precompile_kernels bake_compiled_kernels)

endif()



if(BAKE_KERNEL OR GENERATE_BAKE_KERNEL)
	message(">> BakeKernel Executed")
	if(WIN32)
		execute_process(
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/bakeKernel.bat
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			)
	else()
		execute_process(
			COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/bakeKernel.sh
			WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
			)
	endif()

	target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_BAKE_KERNEL_GENERATED)

endif()



if ( BAKE_COMPILED_KERNEL )
	# enable the 'BAKE_COMPILED_KERNEL' on Orochi:  this mode is activated by adding those 2 defines.
	target_compile_definitions(${HIPRT_NAME} PRIVATE ORO_PP_LOAD_FROM_STRING ORO_PRECOMPILED)
	
	#enable the 'BAKE_COMPILED_KERNEL' on HIPRT: 
	target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_BAKE_COMPILED_KERNEL )
endif()


if(BAKE_KERNEL)
	target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_LOAD_FROM_STRING ORO_PP_LOAD_FROM_STRING)
endif()



if(WIN32)
	target_link_libraries(${HIPRT_NAME} version)
endif()


target_include_directories(${HIPRT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${HIPRT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi)

file(GLOB_RECURSE hiprt_sources "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.inl")
list(FILTER hiprt_sources EXCLUDE REGEX "hiprt/bitcodes/.*")
file(GLOB_RECURSE orochi_sources "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/Orochi/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/Orochi/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.cpp")
target_sources(${HIPRT_NAME} PRIVATE ${hiprt_sources} ${orochi_sources})




#
# install script
#

install(TARGETS ${HIPRT_NAME} DESTINATION bin)

# add header files
file(GLOB HIPRT_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/*.h")
install(FILES ${HIPRT_HEADERS}
		DESTINATION include/hiprt)

# add header files from implementation folder
file(GLOB HIPRT_IMPL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/hiprt/impl/*.h")
install(FILES ${HIPRT_IMPL_HEADERS}
		DESTINATION include/hiprt/impl)
		
# add some header files from Orochi
file(GLOB HIPRT_ORO_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.h")
install(FILES ${HIPRT_ORO_HEADERS}
		DESTINATION include/contrib/Orochi/ParallelPrimitives)

# add hipfb files
file(GLOB HIPRT_HIPFB_FILES "${BASE_OUTPUT_DIR}/${CMAKE_BUILD_TYPE}/*.hipfb")
install(FILES ${HIPRT_HIPFB_FILES}
		DESTINATION bin)




# Project: Unit Test
if(NOT NO_UNITTEST)

	add_executable(unittest)

	if(BITCODE)
		target_compile_definitions(unittest PRIVATE HIPRT_BITCODE_LINKING)
	endif()
	if(WIN32)
		target_compile_options(unittest PRIVATE /wd4244)
		target_link_libraries(unittest PRIVATE version)
	endif()

	if( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
		set_target_properties(unittest PROPERTIES OUTPUT_NAME "unittest64D")
	else()
		set_target_properties(unittest PROPERTIES OUTPUT_NAME "unittest64")
	endif()


	target_include_directories(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi)
	target_link_libraries(unittest PRIVATE ${HIPRT_NAME})

	if(UNIX)
		target_link_libraries(unittest PRIVATE pthread dl)
	endif()

	file(GLOB_RECURSE unittest_sources "${CMAKE_CURRENT_SOURCE_DIR}/test/hiprtT*.h" "${CMAKE_CURRENT_SOURCE_DIR}/test/hiprtT*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/test/shared.h" "${CMAKE_CURRENT_SOURCE_DIR}/test/main.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/test/CornellBox.h" "${CMAKE_CURRENT_SOURCE_DIR}/test/kernels/*.h" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/gtest-1.6.0/gtest-all.cc")


	target_sources(unittest PRIVATE ${unittest_sources} ${orochi_sources})

	target_include_directories(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/gtest-1.6.0 ${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/include)


	if(WIN32)

		# Use target_link_directories to specify additional library directories
		target_link_directories(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win)
		target_link_directories(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/bin/win64)

		copy_dir(${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin/Release "*.dll")
		copy_dir(${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin/Debug "*.dll")
		copy_dir(${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/contrib/bin/win64 ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin/Release "*.dll")
		copy_dir(${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi/contrib/bin/win64 ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin/Debug "*.dll")

		# Explicitly link libraries from contrib/embree/win and contrib/bin/win64
		target_link_libraries(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/win/embree4.lib)

	endif()


	if(UNIX)
		target_link_directories(unittest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/embree/linux)
	endif()

	target_link_libraries(unittest PRIVATE embree4 tbb)

endif()


# Project: HIPRTEW Test
if(HIPRTEW)
	add_executable(hiprtewtest)

	target_compile_definitions(hiprtewtest PRIVATE HIPRT_EXPORTS USE_HIPRTEW)
	if(WIN32)
		target_compile_options(hiprtewtest PRIVATE /wd4244)
		target_link_libraries(hiprtewtest PRIVATE version)
	endif()

	target_include_directories(hiprtewtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/contrib/Orochi)
	if(UNIX)
		target_link_libraries(hiprtewtest PRIVATE pthread dl)
	endif()

	file(GLOB_RECURSE hiprtewtest_sources "${CMAKE_CURRENT_SOURCE_DIR}/test/hiprtewTest.h" "${CMAKE_CURRENT_SOURCE_DIR}/test/hiprtewTest.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/contrib/gtest-1.6.0/gtest-all.cc")
	target_sources(hiprtewtest PRIVATE ${hiprtewtest_sources} ${orochi_sources})

	target_include_directories(hiprtewtest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/contrib/gtest-1.6.0)

	target_compile_definitions(hiprtewtest PRIVATE GTEST_HAS_TR1_TUPLE=0)
endif()
