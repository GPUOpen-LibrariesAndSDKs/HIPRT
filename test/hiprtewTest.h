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
		oroError e = oroMalloc( (oroDeviceptr*)&ptr, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	void free( void* ptr )
	{
		oroError e = oroFree( (oroDeviceptr)ptr );
		ASSERT( e == oroSuccess );
	}

	void memset( void* ptr, int val, size_t n )
	{
		oroError e = oroMemset( (oroDeviceptr)ptr, val, n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyHtoD( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyHtoD( (oroDeviceptr)dst, src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoH( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoH( dst, (oroDeviceptr)src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoD( T* dst, T* src, size_t n )
	{
		oroError e = oroMemcpyDtoD( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyHtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyHtoDAsync( (oroDeviceptr)dst, src, sizeof( T ) * n, stream );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoHAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoHAsync( dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		ASSERT( e == oroSuccess );
	}

	template <typename T>
	void copyDtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		oroError e = oroMemcpyDtoDAsync( (oroDeviceptr)dst, (oroDeviceptr)src, sizeof( T ) * n, stream );
		ASSERT( e == oroSuccess );
	}

  protected:
	hiprtContextCreationInput m_ctxtInput;
	oroCtx					  m_oroCtx;
	oroDevice				  m_oroDevice;
};
