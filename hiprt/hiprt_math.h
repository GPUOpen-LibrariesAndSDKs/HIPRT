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
#include <hiprt/hiprt_vec.h>

namespace hiprt
{
// int vectors
HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int c ) { return int2{ c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int c ) { return int3{ c, c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int c ) { return int4{ c, c, c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int3 a ) { return int2{ a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int4 a ) { return int2{ a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int4 a ) { return int3{ a.x, a.y, a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int2& a, const int c ) { return int3{ a.x, a.y, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int c, const int2& a ) { return int3{ c, a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int2& a, const int c0, const int c1 ) { return int4{ a.x, a.y, c0, c1 }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int c0, const int2& a, const int c1 ) { return int4{ c0, a.x, a.y, c1 }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int c0, const int c1, const int2& a ) { return int4{ c0, c1, a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int3& a, const int c ) { return int4{ a.x, a.y, a.z, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int c, const int3& a ) { return int4{ c, a.x, a.y, a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const uint2& a )
{
	return int2{ static_cast<int>( a.x ), static_cast<int>( a.y ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const uint3& a )
{
	return int3{ static_cast<int>( a.x ), static_cast<int>( a.y ), static_cast<int>( a.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const uint4& a )
{
	return int4{ static_cast<int>( a.x ), static_cast<int>( a.y ), static_cast<int>( a.z ), static_cast<int>( a.w ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const float2& a )
{
	return int2{ static_cast<int>( a.x ), static_cast<int>( a.y ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const float3& a )
{
	return int3{ static_cast<int>( a.x ), static_cast<int>( a.y ), static_cast<int>( a.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const float4& a )
{
	return int4{ static_cast<int>( a.x ), static_cast<int>( a.y ), static_cast<int>( a.z ), static_cast<int>( a.w ) };
}

// uint vectors
HIPRT_HOST_DEVICE HIPRT_INLINE uint2 make_uint2( const unsigned int c ) { return uint2{ c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 make_uint3( const unsigned int c ) { return uint3{ c, c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const unsigned int c ) { return uint4{ c, c, c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 make_uint2( const uint3 a ) { return uint2{ a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 make_uint2( const uint4 a ) { return uint2{ a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 make_uint3( const uint4 a ) { return uint3{ a.x, a.y, a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 make_uint3( const uint2& a, const unsigned int c ) { return uint3{ a.x, a.y, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 make_uint3( const unsigned int c, const uint2& a ) { return uint3{ c, a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const uint2& a, const unsigned int c0, const unsigned int c1 )
{
	return uint4{ a.x, a.y, c0, c1 };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const unsigned int c0, const uint2& a, const unsigned int c1 )
{
	return uint4{ c0, a.x, a.y, c1 };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const unsigned int c0, const unsigned int c1, const uint2& a )
{
	return uint4{ c0, c1, a.x, a.y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const uint3& a, const unsigned int c ) { return uint4{ a.x, a.y, a.z, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const unsigned int c, const uint3& a ) { return uint4{ c, a.x, a.y, a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 make_uint2( const int2& a )
{
	return uint2{ static_cast<unsigned int>( a.x ), static_cast<unsigned int>( a.y ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 make_uint3( const int3& a )
{
	return uint3{ static_cast<unsigned int>( a.x ), static_cast<unsigned int>( a.y ), static_cast<unsigned int>( a.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const int4& a )
{
	return uint4{
		static_cast<unsigned int>( a.x ),
		static_cast<unsigned int>( a.y ),
		static_cast<unsigned int>( a.z ),
		static_cast<unsigned int>( a.w ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 make_uint2( const float2& a )
{
	return uint2{ static_cast<unsigned int>( a.x ), static_cast<unsigned int>( a.y ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 make_uint3( const float3& a )
{
	return uint3{ static_cast<unsigned int>( a.x ), static_cast<unsigned int>( a.y ), static_cast<unsigned int>( a.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 make_uint4( const float4& a )
{
	return uint4{
		static_cast<unsigned int>( a.x ),
		static_cast<unsigned int>( a.y ),
		static_cast<unsigned int>( a.z ),
		static_cast<unsigned int>( a.w ) };
}

// float vectors
HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float c ) { return float2{ c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float c ) { return float3{ c, c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float c ) { return float4{ c, c, c, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float3 a ) { return float2{ a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float4 a ) { return float2{ a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float4 a ) { return float3{ a.x, a.y, a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float2& a, const float c ) { return float3{ a.x, a.y, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float c, const float2& a ) { return float3{ c, a.x, a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float2& a, const float c0, const float c1 )
{
	return float4{ a.x, a.y, c0, c1 };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float c0, const float2& a, const float c1 )
{
	return float4{ c0, a.x, a.y, c1 };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float c0, const float c1, const float2& a )
{
	return float4{ c0, c1, a.x, a.y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float3& a, const float c ) { return float4{ a.x, a.y, a.z, c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float c, const float3& a ) { return float4{ c, a.x, a.y, a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const int2& a )
{
	return float2{ static_cast<float>( a.x ), static_cast<float>( a.y ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const int3& a )
{
	return float3{ static_cast<float>( a.x ), static_cast<float>( a.y ), static_cast<float>( a.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const int4& a )
{
	return float4{ static_cast<float>( a.x ), static_cast<float>( a.y ), static_cast<float>( a.z ), static_cast<float>( a.w ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const uint2& a )
{
	return float2{ static_cast<float>( a.x ), static_cast<float>( a.y ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const uint3& a )
{
	return float3{ static_cast<float>( a.x ), static_cast<float>( a.y ), static_cast<float>( a.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const uint4& a )
{
	return float4{ static_cast<float>( a.x ), static_cast<float>( a.y ), static_cast<float>( a.z ), static_cast<float>( a.w ) };
}
} // namespace hiprt

#if !defined( __HIPCC__ )
// int vectors
HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int2& a, const int2& b ) { return int2{ a.x + b.x, a.y + b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a, const int2& b ) { return int2{ a.x - b.x, a.y - b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int2& a, const int2& b ) { return int2{ a.x * b.x, a.y * b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int2& a, const int2& b ) { return int2{ a.x / b.x, a.y / b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator+=( int2& a, const int2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator-=( int2& a, const int2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator*=( int2& a, const int2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator/=( int2& a, const int2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator+=( int2& a, const int c )
{
	a.x += c;
	a.y += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator-=( int2& a, const int c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator*=( int2& a, const int c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2& operator/=( int2& a, const int c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a ) { return int2{ -a.x, -a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int2& a, const int c ) { return int2{ a.x + c, a.y + c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int c, const int2& a ) { return int2{ c + a.x, c + a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a, const int c ) { return int2{ a.x - c, a.y - c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int c, const int2& a ) { return int2{ c - a.x, c - a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int2& a, const int c ) { return int2{ c * a.x, c * a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int c, const int2& a ) { return int2{ c * a.x, c * a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int2& a, const int c ) { return int2{ a.x / c, a.y / c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int c, const int2& a ) { return int2{ c / a.x, c / a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int3& a, const int3& b )
{
	return int3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a, const int3& b )
{
	return int3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int3& a, const int3& b )
{
	return int3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int3& a, const int3& b )
{
	return int3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator+=( int3& a, const int3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator-=( int3& a, const int3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator*=( int3& a, const int3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator/=( int3& a, const int3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator+=( int3& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator-=( int3& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator*=( int3& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3& operator/=( int3& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a ) { return int3{ -a.x, -a.y, -a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int3& a, const int c ) { return int3{ c + a.x, c + a.y, c + a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int c, const int3& a ) { return int3{ c + a.x, c + a.y, c + a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a, const int c ) { return int3{ a.x - c, a.y - c, a.z - c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int c, const int3& a ) { return int3{ c - a.x, c - a.y, c - a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int3& a, const int c ) { return int3{ c * a.x, c * a.y, c * a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int c, const int3& a ) { return int3{ c * a.x, c * a.y, c * a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int3& a, const int c ) { return int3{ a.x / c, a.y / c, a.z / c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int c, const int3& a ) { return int3{ c / a.x, c / a.y, c / a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int4& a, const int4& b )
{
	return int4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a, const int4& b )
{
	return int4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int4& a, const int4& b )
{
	return int4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int4& a, const int4& b )
{
	return int4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator+=( int4& a, const int4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator-=( int4& a, const int4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator*=( int4& a, const int4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator/=( int4& a, const int4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator+=( int4& a, const int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator-=( int4& a, const int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator*=( int4& a, const int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4& operator/=( int4& a, const int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a ) { return int4{ -a.x, -a.y, -a.z, -a.w }; }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int4& a, const int c )
{
	return int4{ c + a.x, c + a.y, c + a.z, c + a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int c, const int4& a )
{
	return int4{ c + a.x, c + a.y, c + a.z, c + a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a, const int c )
{
	return int4{ a.x - c, a.y - c, a.z - c, a.w - c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int c, const int4& a )
{
	return int4{ c - a.x, c - a.y, c - a.z, c - a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int4& a, const int c )
{
	return int4{ c * a.x, c * a.y, c * a.z, c * a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int c, const int4& a )
{
	return int4{ c * a.x, c * a.y, c * a.z, c * a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int4& a, const int c )
{
	return int4{ a.x / c, a.y / c, a.z / c, a.w / c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int c, const int4& a )
{
	return int4{ c / a.x, c / a.y, c / a.z, c / a.w };
}

// uint vectors
HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator+( const uint2& a, const uint2& b ) { return uint2{ a.x + b.x, a.y + b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator-( const uint2& a, const uint2& b ) { return uint2{ a.x - b.x, a.y - b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator*( const uint2& a, const uint2& b ) { return uint2{ a.x * b.x, a.y * b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator/( const uint2& a, const uint2& b ) { return uint2{ a.x / b.x, a.y / b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator+=( uint2& a, const uint2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator-=( uint2& a, const uint2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator*=( uint2& a, const uint2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator/=( uint2& a, const uint2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator+=( uint2& a, const unsigned int c )
{
	a.x += c;
	a.y += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator-=( uint2& a, const unsigned int c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator*=( uint2& a, const unsigned int c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2& operator/=( uint2& a, const unsigned int c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator+( const uint2& a, const unsigned int c ) { return uint2{ a.x + c, a.y + c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator+( const unsigned int c, const uint2& a ) { return uint2{ c + a.x, c + a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator-( const uint2& a, const unsigned int c ) { return uint2{ a.x - c, a.y - c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator-( const unsigned int c, const uint2& a ) { return uint2{ c - a.x, c - a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator*( const uint2& a, const unsigned int c ) { return uint2{ c * a.x, c * a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator*( const unsigned int c, const uint2& a ) { return uint2{ c * a.x, c * a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator/( const uint2& a, const unsigned int c ) { return uint2{ a.x / c, a.y / c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 operator/( const unsigned int c, const uint2& a ) { return uint2{ c / a.x, c / a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator+( const uint3& a, const uint3& b )
{
	return uint3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator-( const uint3& a, const uint3& b )
{
	return uint3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator*( const uint3& a, const uint3& b )
{
	return uint3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator/( const uint3& a, const uint3& b )
{
	return uint3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator+=( uint3& a, const uint3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator-=( uint3& a, const uint3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator*=( uint3& a, const uint3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator/=( uint3& a, const uint3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator+=( uint3& a, const unsigned int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator-=( uint3& a, const unsigned int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator*=( uint3& a, const unsigned int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3& operator/=( uint3& a, const unsigned int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator+( const uint3& a, const unsigned int c )
{
	return uint3{ c + a.x, c + a.y, c + a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator+( const unsigned int c, const uint3& a )
{
	return uint3{ c + a.x, c + a.y, c + a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator-( const uint3& a, const unsigned int c )
{
	return uint3{ a.x - c, a.y - c, a.z - c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator-( const unsigned int c, const uint3& a )
{
	return uint3{ c - a.x, c - a.y, c - a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator*( const uint3& a, const unsigned int c )
{
	return uint3{ c * a.x, c * a.y, c * a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator*( const unsigned int c, const uint3& a )
{
	return uint3{ c * a.x, c * a.y, c * a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator/( const uint3& a, const unsigned int c )
{
	return uint3{ a.x / c, a.y / c, a.z / c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 operator/( const unsigned int c, const uint3& a )
{
	return uint3{ c / a.x, c / a.y, c / a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator+( const uint4& a, const uint4& b )
{
	return uint4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator-( const uint4& a, const uint4& b )
{
	return uint4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator*( const uint4& a, const uint4& b )
{
	return uint4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator/( const uint4& a, const uint4& b )
{
	return uint4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator+=( uint4& a, const uint4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator-=( uint4& a, const uint4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator*=( uint4& a, const uint4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator/=( uint4& a, const uint4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator+=( uint4& a, const unsigned int c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator-=( uint4& a, const unsigned int c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator*=( uint4& a, const unsigned int c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4& operator/=( uint4& a, const unsigned int c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator+( const uint4& a, const unsigned int c )
{
	return uint4{ c + a.x, c + a.y, c + a.z, c + a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator+( const unsigned int c, const uint4& a )
{
	return uint4{ c + a.x, c + a.y, c + a.z, c + a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator-( const uint4& a, const unsigned int c )
{
	return uint4{ a.x - c, a.y - c, a.z - c, a.w - c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator-( const unsigned int c, const uint4& a )
{
	return uint4{ c - a.x, c - a.y, c - a.z, c - a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator*( const uint4& a, const unsigned int c )
{
	return uint4{ c * a.x, c * a.y, c * a.z, c * a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator*( const unsigned int c, const uint4& a )
{
	return uint4{ c * a.x, c * a.y, c * a.z, c * a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator/( const uint4& a, const unsigned int c )
{
	return uint4{ a.x / c, a.y / c, a.z / c, a.w / c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 operator/( const unsigned int c, const uint4& a )
{
	return uint4{ c / a.x, c / a.y, c / a.z, c / a.w };
}

// float vectors
HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float2& a, const float2& b ) { return float2{ a.x + b.x, a.y + b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a, const float2& b ) { return float2{ a.x - b.x, a.y - b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float2& a, const float2& b ) { return float2{ a.x * b.x, a.y * b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float2& a, const float2& b ) { return float2{ a.x / b.x, a.y / b.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator+=( float2& a, const float2& b )
{
	a.x += b.x;
	a.y += b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator-=( float2& a, const float2& b )
{
	a.x -= b.x;
	a.y -= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator*=( float2& a, const float2& b )
{
	a.x *= b.x;
	a.y *= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator/=( float2& a, const float2& b )
{
	a.x /= b.x;
	a.y /= b.y;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator+=( float2& a, const float c )
{
	a.x += c;
	a.y += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator-=( float2& a, const float c )
{
	a.x -= c;
	a.y -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator*=( float2& a, const float c )
{
	a.x *= c;
	a.y *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2& operator/=( float2& a, const float c )
{
	a.x /= c;
	a.y /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a ) { return float2{ -a.x, -a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float2& a, const float c ) { return float2{ a.x + c, a.y + c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float c, const float2& a ) { return float2{ c + a.x, c + a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a, const float c ) { return float2{ a.x - c, a.y - c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float c, const float2& a ) { return float2{ c - a.x, c - a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float2& a, const float c ) { return float2{ c * a.x, c * a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float c, const float2& a ) { return float2{ c * a.x, c * a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float2& a, const float c ) { return float2{ a.x / c, a.y / c }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float c, const float2& a ) { return float2{ c / a.x, c / a.y }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float3& a, const float3& b )
{
	return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a, const float3& b )
{
	return float3{ a.x - b.x, a.y - b.y, a.z - b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float3& a, const float3& b )
{
	return float3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float3& a, const float3& b )
{
	return float3{ a.x / b.x, a.y / b.y, a.z / b.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator+=( float3& a, const float3& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator-=( float3& a, const float3& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator*=( float3& a, const float3& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator/=( float3& a, const float3& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator+=( float3& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator-=( float3& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator*=( float3& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3& operator/=( float3& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a ) { return float3{ -a.x, -a.y, -a.z }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float3& a, const float c )
{
	return float3{ c + a.x, c + a.y, c + a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float c, const float3& a )
{
	return float3{ c + a.x, c + a.y, c + a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a, const float c )
{
	return float3{ a.x - c, a.y - c, a.z - c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float c, const float3& a )
{
	return float3{ c - a.x, c - a.y, c - a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float3& a, const float c )
{
	return float3{ c * a.x, c * a.y, c * a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float c, const float3& a )
{
	return float3{ c * a.x, c * a.y, c * a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float3& a, const float c )
{
	return float3{ a.x / c, a.y / c, a.z / c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float c, const float3& a )
{
	return float3{ c / a.x, c / a.y, c / a.z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float4& a, const float4& b )
{
	return float4{ a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a, const float4& b )
{
	return float4{ a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4& a, const float4& b )
{
	return float4{ a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float4& a, const float4& b )
{
	return float4{ a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator+=( float4& a, const float4& b )
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	a.w += b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator-=( float4& a, const float4& b )
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	a.w -= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator*=( float4& a, const float4& b )
{
	a.x *= b.x;
	a.y *= b.y;
	a.z *= b.z;
	a.w *= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator/=( float4& a, const float4& b )
{
	a.x /= b.x;
	a.y /= b.y;
	a.z /= b.z;
	a.w /= b.w;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator+=( float4& a, const float c )
{
	a.x += c;
	a.y += c;
	a.z += c;
	a.w += c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator-=( float4& a, const float c )
{
	a.x -= c;
	a.y -= c;
	a.z -= c;
	a.w -= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator*=( float4& a, const float c )
{
	a.x *= c;
	a.y *= c;
	a.z *= c;
	a.w *= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4& operator/=( float4& a, const float c )
{
	a.x /= c;
	a.y /= c;
	a.z /= c;
	a.w /= c;
	return a;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a ) { return float4{ -a.x, -a.y, -a.z, -a.w }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float4& a, const float c )
{
	return float4{ c + a.x, c + a.y, c + a.z, c + a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float c, const float4& a )
{
	return float4{ c + a.x, c + a.y, c + a.z, c + a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a, const float c )
{
	return float4{ a.x - c, a.y - c, a.z - c, a.w - c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float c, const float4& a )
{
	return float4{ c - a.x, c - a.y, c - a.z, c - a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4& a, const float c )
{
	return float4{ c * a.x, c * a.y, c * a.z, c * a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float c, const float4& a )
{
	return float4{ c * a.x, c * a.y, c * a.z, c * a.w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float4& a, const float c )
{
	return float4{ a.x / c, a.y / c, a.z / c, a.w / c };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float c, const float4& a )
{
	return float4{ c / a.x, c / a.y, c / a.z, c / a.w };
}
#endif

namespace hiprt
{
HIPRT_HOST_DEVICE HIPRT_INLINE float min( const float a, const float b ) { return fminf( a, b ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float max( const float a, const float b ) { return fmaxf( a, b ); }

// int vectors
HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int2& a, const int2& b )
{
	int x = max( a.x, b.x );
	int y = max( a.y, b.y );
	return int2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int2& a, const int c )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	return int2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int c, const int2& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int2& a, const int2& b )
{
	int x = min( a.x, b.x );
	int y = min( a.y, b.y );
	return int2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int2& a, const int c )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	return int2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int c, const int2& a ) { return min( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int3& a, const int3& b )
{
	int x = max( a.x, b.x );
	int y = max( a.y, b.y );
	int z = max( a.z, b.z );
	return int3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int3& a, const int c )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	int z = max( a.z, c );
	return int3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int c, const int3& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int3& a, const int3& b )
{
	int x = min( a.x, b.x );
	int y = min( a.y, b.y );
	int z = min( a.z, b.z );
	return int3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int3& a, const int c )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	int z = min( a.z, c );
	return int3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int c, const int3& a ) { return min( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int4& a, const int4& b )
{
	int x = max( a.x, b.x );
	int y = max( a.y, b.y );
	int z = max( a.z, b.z );
	int w = max( a.w, b.w );
	return int4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int4& a, const int c )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	int z = max( a.z, c );
	int w = max( a.w, c );
	return int4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int c, const int4& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int4& a, const int4& b )
{
	int x = min( a.x, b.x );
	int y = min( a.y, b.y );
	int z = min( a.z, b.z );
	int w = min( a.w, b.w );
	return int4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int4& a, const int c )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	int z = min( a.z, c );
	int w = min( a.w, c );
	return int4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int c, const int4& a ) { return min( a, c ); }

// uint vectors
HIPRT_HOST_DEVICE HIPRT_INLINE uint2 max( const uint2& a, const uint2& b )
{
	unsigned int x = max( a.x, b.x );
	unsigned int y = max( a.y, b.y );
	return uint2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 max( const uint2& a, const unsigned int c )
{
	unsigned int x = max( a.x, c );
	unsigned int y = max( a.y, c );
	return uint2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 max( const unsigned int c, const uint2& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 min( const uint2& a, const uint2& b )
{
	unsigned int x = min( a.x, b.x );
	unsigned int y = min( a.y, b.y );
	return uint2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 min( const uint2& a, const unsigned int c )
{
	unsigned int x = min( a.x, c );
	unsigned int y = min( a.y, c );
	return uint2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint2 min( const unsigned int c, const uint2& a ) { return min( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 max( const uint3& a, const uint3& b )
{
	unsigned int x = max( a.x, b.x );
	unsigned int y = max( a.y, b.y );
	unsigned int z = max( a.z, b.z );
	return uint3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 max( const uint3& a, const unsigned int c )
{
	unsigned int x = max( a.x, c );
	unsigned int y = max( a.y, c );
	unsigned int z = max( a.z, c );
	return uint3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 max( const unsigned int c, const uint3& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 min( const uint3& a, const uint3& b )
{
	unsigned int x = min( a.x, b.x );
	unsigned int y = min( a.y, b.y );
	unsigned int z = min( a.z, b.z );
	return uint3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 min( const uint3& a, const unsigned int c )
{
	unsigned int x = min( a.x, c );
	unsigned int y = min( a.y, c );
	unsigned int z = min( a.z, c );
	return uint3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 min( const unsigned int c, const uint3& a ) { return min( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 max( const uint4& a, const uint4& b )
{
	unsigned int x = max( a.x, b.x );
	unsigned int y = max( a.y, b.y );
	unsigned int z = max( a.z, b.z );
	unsigned int w = max( a.w, b.w );
	return uint4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 max( const uint4& a, const unsigned int c )
{
	unsigned int x = max( a.x, c );
	unsigned int y = max( a.y, c );
	unsigned int z = max( a.z, c );
	unsigned int w = max( a.w, c );
	return uint4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 max( const unsigned int c, const uint4& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 min( const uint4& a, const uint4& b )
{
	unsigned int x = min( a.x, b.x );
	unsigned int y = min( a.y, b.y );
	unsigned int z = min( a.z, b.z );
	unsigned int w = min( a.w, b.w );
	return uint4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 min( const uint4& a, const unsigned int c )
{
	unsigned int x = min( a.x, c );
	unsigned int y = min( a.y, c );
	unsigned int z = min( a.z, c );
	unsigned int w = min( a.w, c );
	return uint4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint4 min( const unsigned int c, const uint4& a ) { return min( a, c ); }

// float vectors
HIPRT_HOST_DEVICE HIPRT_INLINE float2 max( const float2& a, const float2& b )
{
	float x = fmaxf( a.x, b.x );
	float y = fmaxf( a.y, b.y );
	return float2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 max( const float2& a, const float c )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	return float2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 max( const float c, const float2& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 min( const float2& a, const float2& b )
{
	float x = fminf( a.x, b.x );
	float y = fminf( a.y, b.y );
	return float2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 min( const float2& a, const float c )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	return float2{ x, y };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 min( const float c, const float2& a ) { return min( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 max( const float3& a, const float3& b )
{
	float x = fmaxf( a.x, b.x );
	float y = fmaxf( a.y, b.y );
	float z = fmaxf( a.z, b.z );
	return float3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 max( const float3& a, const float c )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	float z = fmaxf( a.z, c );
	return float3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 max( const float c, const float3& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 min( const float3& a, const float3& b )
{
	float x = fminf( a.x, b.x );
	float y = fminf( a.y, b.y );
	float z = fminf( a.z, b.z );
	return float3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 min( const float3& a, const float c )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	float z = fminf( a.z, c );
	return float3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 min( const float c, const float3& a ) { return min( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 max( const float4& a, const float4& b )
{
	float x = fmaxf( a.x, b.x );
	float y = fmaxf( a.y, b.y );
	float z = fmaxf( a.z, b.z );
	float w = fmaxf( a.w, b.w );
	return float4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 max( const float4& a, const float c )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	float z = fmaxf( a.z, c );
	float w = fmaxf( a.w, c );
	return float4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 max( const float c, const float4& a ) { return max( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 min( const float4& a, const float4& b )
{
	float x = fminf( a.x, b.x );
	float y = fminf( a.y, b.y );
	float z = fminf( a.z, b.z );
	float w = fminf( a.w, b.w );
	return float4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 min( const float4& a, const float c )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	float z = fminf( a.z, c );
	float w = fminf( a.w, c );
	return float4{ x, y, z, w };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 min( const float c, const float4& a ) { return min( a, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 fma( const float3& a, const float3& b, const float3& c )
{
	float x = fmaf( a.x, b.x, c.x );
	float y = fmaf( a.y, b.y, c.y );
	float z = fmaf( a.z, b.z, c.z );
	return float3{ x, y, z };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float sumOfProducts( const float a, const float b, const float c, const float d )
{
	const float cd = c * d;
	return fmaf( a, b, cd ) + fmaf( c, d, -cd );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float differenceOfProducts( const float a, const float b, const float c, const float d )
{
	const float cd = c * d;
	return fmaf( a, b, -cd ) - fmaf( c, d, -cd );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 cross( const float3& a, const float3& b )
{
	return {
		differenceOfProducts( a.y, b.z, a.z, b.y ),
		differenceOfProducts( a.z, b.x, a.x, b.z ),
		differenceOfProducts( a.x, b.y, a.y, b.x ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float dot( const float3& a, const float3& b )
{
	return fmaf( a.x, b.x, sumOfProducts( a.y, b.y, a.z, b.z ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 normalize( const float3& a ) { return a / sqrtf( dot( a, a ) ); }

template <typename V>
HIPRT_HOST_DEVICE HIPRT_INLINE auto* ptr( V& a )
{
	return &a.x;
}

template <typename T, typename V>
HIPRT_HOST_DEVICE HIPRT_INLINE V mix( const V& lo, const V& hi, const T& t )
{
	return lo * ( static_cast<T>( 1 ) - t ) + hi * t;
}

template <typename T, typename V>
HIPRT_HOST_DEVICE HIPRT_INLINE V clamp( const V& v, const T& lo, const T& hi )
{
	return max( min( v, hi ), lo );
}

template <typename T>
HIPRT_HOST_DEVICE HIPRT_INLINE T sign( T val )
{
	return val < T( 0 ) ? T( -1 ) : ( val == T( 0 ) ? T( 0 ) : T( 1 ) );
}

#if defined( __KERNELCC__ )
HIPRT_DEVICE HIPRT_INLINE float atomicMinFloat( float* addr, float value )
{
	float old;
	old = ( __float_as_int( value ) >= 0 )
			  ? __int_as_float( atomicMin( reinterpret_cast<int*>( addr ), __float_as_int( value ) ) )
			  : __uint_as_float( atomicMax( reinterpret_cast<unsigned int*>( addr ), __float_as_uint( value ) ) );
	return old;
}

HIPRT_DEVICE HIPRT_INLINE float atomicMaxFloat( float* addr, float value )
{
	float old;
	old = ( __float_as_int( value ) >= 0 )
			  ? __int_as_float( atomicMax( reinterpret_cast<int*>( addr ), __float_as_int( value ) ) )
			  : __uint_as_float( atomicMin( reinterpret_cast<unsigned int*>( addr ), __float_as_uint( value ) ) );
	return old;
}
#endif
} // namespace hiprt
