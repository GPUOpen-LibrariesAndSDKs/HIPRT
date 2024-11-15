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

#if !defined( __KERNELCC__ )
namespace hiprt
{
template <typename T, uint32_t N>
struct Vector;

template <typename T>
struct alignas( 2 * sizeof( T ) ) Vector<T, 2>
{
	T x, y;
};

template <typename T>
struct Vector<T, 3>
{
	T x, y, z;
};

template <typename T>
struct alignas( 4 * sizeof( T ) ) Vector<T, 4>
{
	T x, y, z, w;
};
} // namespace hiprt

using hiprtInt2	  = hiprt::Vector<int, 2>;
using hiprtInt3	  = hiprt::Vector<int, 3>;
using hiprtInt4	  = hiprt::Vector<int, 4>;
using hiprtUint2  = hiprt::Vector<unsigned int, 2>;
using hiprtUint3  = hiprt::Vector<unsigned int, 3>;
using hiprtUint4  = hiprt::Vector<unsigned int, 4>;
using hiprtFloat2 = hiprt::Vector<float, 2>;
using hiprtFloat3 = hiprt::Vector<float, 3>;
using hiprtFloat4 = hiprt::Vector<float, 4>;
#if defined( HIPRT_EXPORTS )
using int2	 = hiprtInt2;
using int3	 = hiprtInt3;
using int4	 = hiprtInt4;
using uint2	 = hiprtUint2;
using uint3	 = hiprtUint3;
using uint4	 = hiprtUint4;
using float2 = hiprtFloat2;
using float3 = hiprtFloat3;
using float4 = hiprtFloat4;
#endif
#else
using hiprtInt2	  = int2;
using hiprtInt3	  = int3;
using hiprtInt4	  = int4;
using hiprtUint2  = uint2;
using hiprtUint3  = uint3;
using hiprtUint4  = uint4;
using hiprtFloat2 = float2;
using hiprtFloat3 = float3;
using hiprtFloat4 = float4;
#endif

HIPRT_STATIC_ASSERT( sizeof( hiprtInt2 ) == 8 );
HIPRT_STATIC_ASSERT( sizeof( hiprtInt3 ) == 12 );
HIPRT_STATIC_ASSERT( sizeof( hiprtInt4 ) == 16 );
HIPRT_STATIC_ASSERT( sizeof( hiprtUint2 ) == 8 );
HIPRT_STATIC_ASSERT( sizeof( hiprtUint3 ) == 12 );
HIPRT_STATIC_ASSERT( sizeof( hiprtUint4 ) == 16 );
HIPRT_STATIC_ASSERT( sizeof( hiprtFloat2 ) == 8 );
HIPRT_STATIC_ASSERT( sizeof( hiprtFloat3 ) == 12 );
HIPRT_STATIC_ASSERT( sizeof( hiprtFloat4 ) == 16 );
