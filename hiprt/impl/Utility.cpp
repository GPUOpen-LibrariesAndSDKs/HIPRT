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

#include <hiprt/hiprt_common.h>
#include <hiprt/impl/Error.h>
#include <hiprt/impl/Utility.h>
#include <locale>
#include <codecvt>

#if defined( __GNUC__ )
#include <dlfcn.h>
#include <sys/stat.h>
#include <errno.h>
#endif

#if defined( _WIN32 )
#define NOMINMAX
#include <Windows.h>
#endif

namespace hiprt
{
#if !defined( __GNUC__ )
const HMODULE getCurrentModule()
{
	HMODULE hModule = NULL;
	// hModule is NULL if GetModuleHandleEx fails.
	GetModuleHandleEx(
		GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
		(LPCTSTR)getCurrentModule,
		&hModule );
	return hModule;
}
#else
void getCurrentModule() {}
#endif

std::filesystem::path Utility::getCurrentDir()
{
#if !defined( __GNUC__ )
	char buff[MAX_PATH];
	GetModuleFileName( getCurrentModule(), buff, MAX_PATH );
#else
	Dl_info info;
	dladdr( (const void*)getCurrentModule, &info );
	const char* buff = info.dli_fname;
#endif
	std::string::size_type position = std::string( buff ).find_last_of( "\\/" );
	return std::string( buff ).substr( 0, position ) + "/";
}

uint32_t Utility::hashString( const std::string& str )
{
	uint32_t hash = 0;

	const char* data = str.data();
	for ( uint32_t i = 0; i < str.length(); ++i )
	{
		hash += *data++;
		hash += ( hash << 10 );
		hash ^= ( hash >> 6 );
	}

	hash += ( hash << 3 );
	hash ^= ( hash >> 11 );
	hash += ( hash << 15 );

	return hash;
}
} // namespace hiprt
