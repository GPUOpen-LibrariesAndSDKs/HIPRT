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
#include <Orochi/Orochi.h>
#include <vector>
#include <fstream>
#include <contrib/cpp20/source_location.h>
#include <gtest/gtest.h>
#define _ENABLE_HIPRTEW
#include <hiprt/hiprtew.h>

#if defined( _MSC_VER )
#define ASSERT( cond )  \
	if ( !( cond ) )    \
	{                   \
		__debugbreak(); \
	}
#elif defined( __GNUC__ )
#include <signal.h>
#define ASSERT( cond )    \
	if ( !( cond ) )      \
	{                     \
		raise( SIGTRAP ); \
	}
#else
#define ASSERT( cond )
#endif

void checkOro( oroError res, const source_location& location = source_location::current() );
void checkOrortc( orortcResult res, const source_location& location = source_location::current() );
void checkHiprt( hiprtError res, const source_location& location = source_location::current() );

class hiprtewTest : public ::testing::Test
{
  public:
	void SetUp();
	void TearDown() { oroCtxDestroy( m_oroCtx ); }

	void waitForCompletion( oroStream stream = 0 )
	{
		auto e = oroStreamSynchronize( stream );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void malloc( T*& ptr, size_t n )
	{
		oroError e = oroMalloc( reinterpret_cast<oroDeviceptr*>( &ptr ), sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	void free( void* ptr )
	{
		oroError e = oroFree( reinterpret_cast<oroDeviceptr>( ptr ) );
		ASSERT( e == oroSuccess );
	}

	void memset( void* ptr, int val, size_t n )
	{
		oroError e = oroMemset( reinterpret_cast<oroDeviceptr>( ptr ), val, n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyHtoD( T* dst, const T* src, size_t n )
	{
		oroError e = oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( dst ), const_cast<T*>( src ), sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoH( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoH( dst, reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoD( T* dst, T* src, size_t n )
	{
		oroError e =
			oroMemcpyDtoD( reinterpret_cast<oroDeviceptr>( dst ), reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyHtoDAsync( T* dst, const T* src, size_t n, oroStream stream )
	{
		oroError e =
			oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( dst ), const_cast<T*>( src ), sizeof( T ) * n, stream );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoHAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoHAsync( dst, reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n, stream );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( dst ), reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n, stream );
		ASSERT( e == oroSuccess );
	}

  protected:
	hiprtContextCreationInput m_ctxtInput;
	oroCtx					  m_oroCtx;
	oroDevice				  m_oroDevice;
};
