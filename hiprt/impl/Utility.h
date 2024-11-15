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
#include <hiprt/hiprt_common.h>
#include <filesystem>
#include <memory>

namespace hiprt
{
class Context;

class Utility
{
  public:
	static std::filesystem::path getCurrentDir();

	static uint32_t hashString( const std::string& str );

	template <typename... Args>
	static std::string format( const std::string& format, Args... args )
	{
		int sizeS = std::snprintf( nullptr, 0, format.c_str(), args... ) + 1;
		if ( sizeS <= 0 )
		{
			throw std::runtime_error( "Error during formatting." );
		}
		auto					size = static_cast<size_t>( sizeS );
		std::unique_ptr<char[]> buf( new char[size] );
		std::snprintf( buf.get(), size, format.c_str(), args... );
		return std::string( buf.get(), buf.get() + size - 1 );
	}

	static std::string getEnvVariable( const std::string& key )
	{
#if defined( __WINDOWS__ )
		char*  buffer	   = nullptr;
		size_t bufferCount = 0;
		_dupenv_s( &buffer, &bufferCount, key.c_str() );
		const std::string val = buffer != nullptr ? buffer : "";
		delete[] buffer;
#else
		const char* const env = getenv( key.c_str() );
		const std::string val = ( env == nullptr ) ? std::string() : std::string( env );
#endif
		return val;
	}

	static std::filesystem::path getRootDir()
	{
		std::string val = getEnvVariable( "HIPRT_PATH" );
		if ( val.empty() ) val = "..";
		return std::filesystem::path( val );
	}
};
} // namespace hiprt
