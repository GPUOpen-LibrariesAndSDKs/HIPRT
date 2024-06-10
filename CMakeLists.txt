cmake_minimum_required(VERSION 3.10)
project(hiprt)

# Options
option(BAKE_KERNEL "Encrypt and bake kernels" OFF)
option(BITCODE "Enable bitcode linking" OFF)
option(PRECOMPILE "Precompile kernels" OFF)
option(HIPRTEW "Use hiprtew" OFF)

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


function(get_hip_sdk_version result)
	if(WIN32)
		set(root ".\\")
	endif()

	set(exec_perl "")
	set(hipCommand "hipcc")
	set(HIP_PATH $ENV{HIP_PATH})
	set(PATH $ENV{PATH})
	set(hipInPath OFF)

	# Check if HIP is in the PATH environment variable
	string(REPLACE ";" "\n" PATH_LIST ${PATH})
	foreach(token ${PATH_LIST})
		if("${token}" MATCHES "hip")
			if(EXISTS "${token}/hipcc")
				set(hipInPath ON)
			endif()
		endif()
	endforeach()

	if(WIN32)
		set(exec_perl "perl")
		if(hipInPath)
			set(hipCommand "hipcc")
		elseif(NOT HIP_PATH)
			set(hipCommand "hipSdk\\bin\\hipcc")
		else()
			string(SUBSTRING ${HIP_PATH} -1 1 HIP_PATH_LAST_CHAR)
			if(HIP_PATH_LAST_CHAR STREQUAL "\\" OR HIP_PATH_LAST_CHAR STREQUAL "/")
				string(SUBSTRING ${HIP_PATH} 0 -1 HIP_PATH)
			endif()
			# HIP_PATH is expected to look like: C:\Program Files\AMD\ROCm\5.7
			set(hipCommand "\"${HIP_PATH}\\bin\\${hipCommand}\"")
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

	set(${result} ${version} PARENT_SCOPE)
endfunction()



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
	get_hip_sdk_version(HIP_VERSION_STR)
	string(REPLACE "@HIP_VERSION_STR@" "\"${HIP_VERSION_STR}\"" header_content "${header_content}")

	# Write the modified content to the header file
	file(WRITE ${header_file} "${header_content}")

	set(${version_str_out} ${version_str} PARENT_SCOPE)
endfunction()


# Set up configurations
set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo;DebugGpu" CACHE STRING "Configs" FORCE)

# Define platform-specific flags and settings
if(WIN32)
	add_definitions(-D__WINDOWS__)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244 /wd4305 /wd4018 /wd4996 /Zc:__cplusplus")
elseif(UNIX)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
endif()

add_definitions(-D__USE_HIP__)


# Enable CUDA if possible
include(${CMAKE_SOURCE_DIR}/contrib/Orochi/Orochi/enable_cuew.cmake)



# Base output directory
set(BASE_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dist/bin)

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
write_version_info("${CMAKE_SOURCE_DIR}/hiprt/hiprt.h.in" "${CMAKE_SOURCE_DIR}/hiprt/hiprt.h" "${CMAKE_SOURCE_DIR}/version.txt" version_str_)
write_version_info("${CMAKE_SOURCE_DIR}/hiprt/hiprtew.h.in" "${CMAKE_SOURCE_DIR}/hiprt/hiprtew.h" "${CMAKE_SOURCE_DIR}/version.txt" version_str_)


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

if(PRECOMPILE)
	execute_process(COMMAND python ${CMAKE_SOURCE_DIR}/scripts/bitcodes/compile.py)
endif()

if(BAKE_KERNEL OR BITCODE)
	message(">> BakeKernel Executed")
	if(WIN32)
		execute_process(COMMAND mkdir hiprt/cache)
		execute_process(COMMAND ${CMAKE_SOURCE_DIR}/tools/bakeKernel.bat)
	else()
		execute_process(COMMAND mkdir hiprt/cache)
		execute_process(COMMAND ${CMAKE_SOURCE_DIR}/tools/bakeKernel.sh)
	endif()

	if(BAKE_KERNEL)
		target_compile_definitions(${HIPRT_NAME} PRIVATE HIPRT_LOAD_FROM_STRING ORO_PP_LOAD_FROM_STRING)
	endif()
endif()

if(WIN32)
	target_link_libraries(${HIPRT_NAME} version)
endif()


target_include_directories(${HIPRT_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(${HIPRT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/contrib/Orochi)

file(GLOB_RECURSE hiprt_sources "${CMAKE_SOURCE_DIR}/hiprt/*.h" "${CMAKE_SOURCE_DIR}/hiprt/*.cpp" "${CMAKE_SOURCE_DIR}/hiprt/*.inl")
list(FILTER hiprt_sources EXCLUDE REGEX "hiprt/bitcodes/.*")
file(GLOB_RECURSE orochi_sources "${CMAKE_SOURCE_DIR}/contrib/Orochi/Orochi/*.h" "${CMAKE_SOURCE_DIR}/contrib/Orochi/Orochi/*.cpp" "${CMAKE_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.h" "${CMAKE_SOURCE_DIR}/contrib/Orochi/contrib/cuew/*.cpp" "${CMAKE_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.h" "${CMAKE_SOURCE_DIR}/contrib/Orochi/contrib/hipew/*.cpp" "${CMAKE_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.h" "${CMAKE_SOURCE_DIR}/contrib/Orochi/ParallelPrimitives/*.cpp")
target_sources(${HIPRT_NAME} PRIVATE ${hiprt_sources} ${orochi_sources})

# Project: Unit Test
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


target_include_directories(unittest PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/contrib/Orochi)
target_link_libraries(unittest PRIVATE ${HIPRT_NAME})

if(UNIX)
	target_link_libraries(unittest PRIVATE pthread dl)
endif()

file(GLOB_RECURSE unittest_sources "${CMAKE_SOURCE_DIR}/test/hiprtT*.h" "${CMAKE_SOURCE_DIR}/test/hiprtT*.cpp" "${CMAKE_SOURCE_DIR}/test/shared.h" "${CMAKE_SOURCE_DIR}/test/main.cpp" "${CMAKE_SOURCE_DIR}/test/CornellBox.h" "${CMAKE_SOURCE_DIR}/test/kernels/*.h" "${CMAKE_SOURCE_DIR}/contrib/gtest-1.6.0/gtest-all.cc")


target_sources(unittest PRIVATE ${unittest_sources} ${orochi_sources})

target_include_directories(unittest PRIVATE ${CMAKE_SOURCE_DIR}/contrib/gtest-1.6.0 ${CMAKE_SOURCE_DIR}/contrib/embree/include)





if(WIN32)

	# Use target_link_directories to specify additional library directories
	target_link_directories(unittest PRIVATE ${CMAKE_SOURCE_DIR}/contrib/embree/win)
	target_link_directories(unittest PRIVATE ${CMAKE_SOURCE_DIR}/contrib/bin/win64)

	copy_dir(${CMAKE_SOURCE_DIR}/contrib/embree/win ${CMAKE_SOURCE_DIR}/dist/bin/Release "*.dll")
	copy_dir(${CMAKE_SOURCE_DIR}/contrib/embree/win ${CMAKE_SOURCE_DIR}/dist/bin/Debug "*.dll")
	copy_dir(${CMAKE_SOURCE_DIR}/contrib/Orochi/contrib/bin/win64 ${CMAKE_SOURCE_DIR}/dist/bin/Release "*.dll")
	copy_dir(${CMAKE_SOURCE_DIR}/contrib/Orochi/contrib/bin/win64 ${CMAKE_SOURCE_DIR}/dist/bin/Debug "*.dll")

	# Explicitly link libraries from contrib/embree/win and contrib/bin/win64
	target_link_libraries(unittest PRIVATE ${CMAKE_SOURCE_DIR}/contrib/embree/win/embree4.lib)

endif()


if(UNIX)
	target_link_directories(unittest PRIVATE ${CMAKE_SOURCE_DIR}/contrib/embree/linux)
endif()




target_link_libraries(unittest PRIVATE embree4 tbb)

# Project: HIPRTEW Test
if(HIPRTEW)
	add_executable(hiprtewtest)

	target_compile_definitions(hiprtewtest PRIVATE HIPRT_EXPORTS USE_HIPRTEW)
	if(WIN32)
		target_compile_options(hiprtewtest PRIVATE /wd4244)
		target_link_libraries(hiprtewtest PRIVATE version)
	endif()

	target_include_directories(hiprtewtest PRIVATE ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/contrib/Orochi)
	if(UNIX)
		target_link_libraries(hiprtewtest PRIVATE pthread dl)
	endif()

	file(GLOB_RECURSE hiprtewtest_sources "${CMAKE_SOURCE_DIR}/test/hiprtewTest.h" "${CMAKE_SOURCE_DIR}/test/hiprtewTest.cpp" "${CMAKE_SOURCE_DIR}/contrib/gtest-1.6.0/gtest-all.cc")
	target_sources(hiprtewtest PRIVATE ${hiprtewtest_sources} ${orochi_sources})

	target_include_directories(hiprtewtest PRIVATE ${CMAKE_SOURCE_DIR}/contrib/gtest-1.6.0)

	target_compile_definitions(hiprtewtest PRIVATE GTEST_HAS_TR1_TUPLE=0)
endif()
