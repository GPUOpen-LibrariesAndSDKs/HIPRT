//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//
// Define the library path that HIPRT will use.
// Order matters: the first library file of the list to exist will be the one loaded.
//

#ifdef _WIN32

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace hiprt
{
namespace detail
{
inline std::string getEnvVariable( const char* name )
{
	const char* value = std::getenv( name );
	return value ? std::string( value ) : std::string();
}

inline std::string joinPath( const std::string& base, const char* leaf )
{
	if ( base.empty() ) return std::string( leaf );
	const char last = base.back();
	if ( last == '\\' || last == '/' ) return base + leaf;
	return base + "\\" + leaf;
}

inline void appendUnique( std::vector<std::string>& paths, const std::string& path )
{
	if ( path.empty() || std::find( paths.begin(), paths.end(), path ) != paths.end() ) return;
	paths.push_back( path );
}

inline void appendRocmRoot( std::vector<std::string>& paths, const std::string& root )
{
	if ( root.empty() ) return;
	appendUnique( paths, root );
	appendUnique( paths, joinPath( root, "bin" ) );
}

inline std::vector<std::string> rocmBinPaths()
{
	std::vector<std::string> result;
	appendRocmRoot( result, getEnvVariable( "ROCM_HOME" ) );
	appendRocmRoot( result, getEnvVariable( "ROCM_PATH" ) );
	appendRocmRoot( result, getEnvVariable( "HIP_PATH" ) );

	const std::string virtualEnv = getEnvVariable( "VIRTUAL_ENV" );
	if ( !virtualEnv.empty() )
	{
		appendRocmRoot( result, joinPath( virtualEnv, "Lib\\site-packages\\_rocm_sdk_devel" ) );
		appendRocmRoot( result, joinPath( virtualEnv, "Lib\\site-packages\\rocm_sdk_devel" ) );
	}

	return result;
}

inline const char** makeLibraryPaths(
	const char* const*		  dllNames,
	std::vector<std::string>& storage,
	std::vector<const char*>& pointers )
{
	storage.clear();
	pointers.clear();

	const std::vector<std::string> rocmBins = rocmBinPaths();
	for ( size_t i = 0; dllNames[i] != nullptr; ++i )
	{
		appendUnique( storage, dllNames[i] );
		for ( const std::string& bin : rocmBins )
			appendUnique( storage, joinPath( bin, dllNames[i] ) );
	}

	for ( const std::string& path : storage )
		pointers.push_back( path.c_str() );
	pointers.push_back( nullptr );

	return pointers.data();
}

inline const char** getHipPaths()
{
	static const char*		  dllNames[] = { "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll", nullptr };
	static std::vector<std::string> storage;
	static std::vector<const char*> pointers;
	return makeLibraryPaths( dllNames, storage, pointers );
}

inline const char** getHiprtcPaths()
{
	static const char* dllNames[] = { "hiprtc07013.dll", "hiprtc07012.dll", "hiprtc07011.dll", "hiprtc07010.dll",
									  "hiprtc0709.dll",  "hiprtc0708.dll",	"hiprtc0707.dll",  "hiprtc0706.dll",
									  "hiprtc0705.dll",  "hiprtc0704.dll",	"hiprtc0703.dll",  "hiprtc0702.dll",
									  "hiprtc0701.dll",  "hiprtc0700.dll",	"hiprtc0605.dll",  "hiprtc0604.dll",
									  "hiprtc0603.dll",  "hiprtc0602.dll",	"hiprtc0601.dll",  "hiprtc0600.dll",
									  "hiprtc0507.dll",  "hiprtc0506.dll",	"hiprtc0505.dll",  "hiprtc0504.dll",
									  "hiprtc0503.dll",  nullptr };
	static std::vector<std::string> storage;
	static std::vector<const char*> pointers;
	return makeLibraryPaths( dllNames, storage, pointers );
}
} // namespace detail
} // namespace hiprt

static const char** g_hip_paths	= hiprt::detail::getHipPaths();
static const char** g_hiprtc_paths = hiprt::detail::getHiprtcPaths();
#elif defined( __APPLE__ )

const char** g_hip_paths	= nullptr;
const char** g_hiprtc_paths = nullptr;
#else

const char* g_hip_paths[] = {

	// first, we try with the generic symbolic link.
	"libamdhip64.so",
	"/opt/rocm/lib/libamdhip64.so",
	"/opt/rocm/hip/lib/libamdhip64.so",

	// .. if it doesn't exist, we try the specific versions
	"libamdhip64.so.7",
	"/opt/rocm/lib/libamdhip64.so.7",
	"/opt/rocm/hip/lib/libamdhip64.so.7",

	"libamdhip64.so.6",
	"/opt/rocm/lib/libamdhip64.so.6",
	"/opt/rocm/hip/lib/libamdhip64.so.6",

	"libamdhip64.so.5",
	"/opt/rocm/lib/libamdhip64.so.5",
	"/opt/rocm/hip/lib/libamdhip64.so.5",

	nullptr };

const char* g_hiprtc_paths[] = {

	// first, we try with the generic symbolic link.
	"/opt/rocm/hip/lib/libhiprtc.so",
	"/opt/rocm/lib/libhiprtc.so",
	"libhiprtc.so",

	// .. if it doesn't exist, we try the specific versions
	"/opt/rocm/hip/lib/libhiprtc.so.7",
	"/opt/rocm/lib/libhiprtc.so.7",
	"libhiprtc.so.7",

	"/opt/rocm/hip/lib/libhiprtc.so.6",
	"/opt/rocm/lib/libhiprtc.so.6",
	"libhiprtc.so.6",

	"/opt/rocm/hip/lib/libhiprtc.so.5",
	"/opt/rocm/lib/libhiprtc.so.5",
	"libhiprtc.so.5",

	nullptr };
#endif
