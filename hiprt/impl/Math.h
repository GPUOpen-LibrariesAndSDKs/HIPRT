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

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const float2 a )
{
	return make_int2( static_cast<int>( a.x ), static_cast<int>( a.y ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int3& a ) { return make_int2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int4& a ) { return make_int2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 make_int2( const int c ) { return make_int2( c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const float3& a )
{
	return make_int3( static_cast<int>( a.x ), static_cast<int>( a.y ), static_cast<int>( a.z ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int4& a ) { return make_int3( a.x, a.y, a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int2& a, const int c ) { return make_int3( a.x, a.y, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 make_int3( const int c ) { return make_int3( c, c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const float4& a )
{
	return make_int4( static_cast<int>( a.x ), static_cast<int>( a.y ), static_cast<int>( a.z ), static_cast<int>( a.w ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int2& a, const int c0, const int c1 )
{
	return make_int4( a.x, a.y, c0, c1 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int3& a, const int c ) { return make_int4( a.x, a.y, a.z, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 make_int4( const int c ) { return make_int4( c, c, c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const int2& a )
{
	return make_float2( static_cast<float>( a.x ), static_cast<float>( a.y ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float3& a ) { return make_float2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float4& a ) { return make_float2( a.x, a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 make_float2( const float c ) { return make_float2( c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const int3& a )
{
	return make_float3( static_cast<float>( a.x ), static_cast<float>( a.y ), static_cast<float>( a.z ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float4& a ) { return make_float3( a.x, a.y, a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float2& a, const float c ) { return make_float3( a.x, a.y, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 make_float3( const float c ) { return make_float3( c, c, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const int4& a )
{
	return make_float4(
		static_cast<float>( a.x ), static_cast<float>( a.y ), static_cast<float>( a.z ), static_cast<float>( a.w ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float2& a, const float c0, const float c1 )
{
	return make_float4( a.x, a.y, c0, c1 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float3& a, const float c ) { return make_float4( a.x, a.y, a.z, c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 make_float4( const float c ) { return make_float4( c, c, c, c ); }

#if !defined( __HIPCC__ )
HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int2& a, const int2& b ) { return make_int2( a.x + b.x, a.y + b.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a, const int2& b ) { return make_int2( a.x - b.x, a.y - b.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int2& a, const int2& b ) { return make_int2( a.x * b.x, a.y * b.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int2& a, const int2& b ) { return make_int2( a.x / b.x, a.y / b.y ); }

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

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a ) { return make_int2( -a.x, -a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int2& a, const int c ) { return make_int2( a.x + c, a.y + c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator+( const int c, const int2& a ) { return make_int2( c + a.x, c + a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int2& a, const int c ) { return make_int2( a.x - c, a.y - c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator-( const int c, const int2& a ) { return make_int2( c - a.x, c - a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int2& a, const int c ) { return make_int2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator*( const int c, const int2& a ) { return make_int2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int2& a, const int c ) { return make_int2( a.x / c, a.y / c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 operator/( const int c, const int2& a ) { return make_int2( c / a.x, c / a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int3& a, const int3& b )
{
	return make_int3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a, const int3& b )
{
	return make_int3( a.x - b.x, a.y - b.y, a.z - b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int3& a, const int3& b )
{
	return make_int3( a.x * b.x, a.y * b.y, a.z * b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int3& a, const int3& b )
{
	return make_int3( a.x / b.x, a.y / b.y, a.z / b.z );
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

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a ) { return make_int3( -a.x, -a.y, -a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int3& a, const int c ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator+( const int c, const int3& a ) { return make_int3( c + a.x, c + a.y, c + a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int3& a, const int c ) { return make_int3( a.x - c, a.y - c, a.z - c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator-( const int c, const int3& a ) { return make_int3( c - a.x, c - a.y, c - a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int3& a, const int c ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator*( const int c, const int3& a ) { return make_int3( c * a.x, c * a.y, c * a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int3& a, const int c ) { return make_int3( a.x / c, a.y / c, a.z / c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int3 operator/( const int c, const int3& a ) { return make_int3( c / a.x, c / a.y, c / a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int4& a, const int4& b )
{
	return make_int4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a, const int4& b )
{
	return make_int4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int4& a, const int4& b )
{
	return make_int4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int4& a, const int4& b )
{
	return make_int4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
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

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a ) { return make_int4( -a.x, -a.y, -a.z, -a.w ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int4& a, const int c )
{
	return make_int4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator+( const int c, const int4& a )
{
	return make_int4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int4& a, const int c )
{
	return make_int4( a.x - c, a.y - c, a.z - c, a.w - c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator-( const int c, const int4& a )
{
	return make_int4( c - a.x, c - a.y, c - a.z, c - a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int4& a, const int c )
{
	return make_int4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator*( const int c, const int4& a )
{
	return make_int4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int4& a, const int c )
{
	return make_int4( a.x / c, a.y / c, a.z / c, a.w / c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 operator/( const int c, const int4& a )
{
	return make_int4( c / a.x, c / a.y, c / a.z, c / a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float2& a, const float2& b )
{
	return make_float2( a.x + b.x, a.y + b.y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a, const float2& b )
{
	return make_float2( a.x - b.x, a.y - b.y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float2& a, const float2& b )
{
	return make_float2( a.x * b.x, a.y * b.y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float2& a, const float2& b )
{
	return make_float2( a.x / b.x, a.y / b.y );
}

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

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a ) { return make_float2( -a.x, -a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float2& a, const float c ) { return make_float2( a.x + c, a.y + c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator+( const float c, const float2& a ) { return make_float2( c + a.x, c + a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float2& a, const float c ) { return make_float2( a.x - c, a.y - c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator-( const float c, const float2& a ) { return make_float2( c - a.x, c - a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float2& a, const float c ) { return make_float2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator*( const float c, const float2& a ) { return make_float2( c * a.x, c * a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float2& a, const float c ) { return make_float2( a.x / c, a.y / c ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float2 operator/( const float c, const float2& a ) { return make_float2( c / a.x, c / a.y ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float3& a, const float3& b )
{
	return make_float3( a.x + b.x, a.y + b.y, a.z + b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a, const float3& b )
{
	return make_float3( a.x - b.x, a.y - b.y, a.z - b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float3& a, const float3& b )
{
	return make_float3( a.x * b.x, a.y * b.y, a.z * b.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float3& a, const float3& b )
{
	return make_float3( a.x / b.x, a.y / b.y, a.z / b.z );
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

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a ) { return make_float3( -a.x, -a.y, -a.z ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float3& a, const float c )
{
	return make_float3( c + a.x, c + a.y, c + a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator+( const float c, const float3& a )
{
	return make_float3( c + a.x, c + a.y, c + a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float3& a, const float c )
{
	return make_float3( a.x - c, a.y - c, a.z - c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator-( const float c, const float3& a )
{
	return make_float3( c - a.x, c - a.y, c - a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float3& a, const float c )
{
	return make_float3( c * a.x, c * a.y, c * a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator*( const float c, const float3& a )
{
	return make_float3( c * a.x, c * a.y, c * a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float3& a, const float c )
{
	return make_float3( a.x / c, a.y / c, a.z / c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 operator/( const float c, const float3& a )
{
	return make_float3( c / a.x, c / a.y, c / a.z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float4& a, const float4& b )
{
	return make_float4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a, const float4& b )
{
	return make_float4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4& a, const float4& b )
{
	return make_float4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float4& a, const float4& b )
{
	return make_float4( a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w );
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

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a ) { return make_float4( -a.x, -a.y, -a.z, -a.w ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float4& a, const float c )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator+( const float c, const float4& a )
{
	return make_float4( c + a.x, c + a.y, c + a.z, c + a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float4& a, const float c )
{
	return make_float4( a.x - c, a.y - c, a.z - c, a.w - c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator-( const float c, const float4& a )
{
	return make_float4( c - a.x, c - a.y, c - a.z, c - a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float4& a, const float c )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator*( const float c, const float4& a )
{
	return make_float4( c * a.x, c * a.y, c * a.z, c * a.w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float4& a, const float c )
{
	return make_float4( a.x / c, a.y / c, a.z / c, a.w / c );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 operator/( const float c, const float4& a )
{
	return make_float4( c / a.x, c / a.y, c / a.z, c / a.w );
}
#endif

namespace hiprt
{
HIPRT_HOST_DEVICE HIPRT_INLINE float min( const float a, const float b ) { return fminf( a, b ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float max( const float a, const float b ) { return fmaxf( a, b ); }

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int2& a, const int2& b )
{
	int x = max( a.x, b.x );
	int y = max( a.y, b.y );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int2& a, const int c )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 max( const int c, const int2& a )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int2& a, const int2& b )
{
	int x = min( a.x, b.x );
	int y = min( a.y, b.y );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int2& a, const int c )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int2 min( const int c, const int2& a )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	return make_int2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int3& a, const int3& b )
{
	int x = max( a.x, b.x );
	int y = max( a.y, b.y );
	int z = max( a.z, b.z );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int3& a, const int c )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	int z = max( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 max( const int c, const int3& a )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	int z = max( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int3& a, const int3& b )
{
	int x = min( a.x, b.x );
	int y = min( a.y, b.y );
	int z = min( a.z, b.z );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int3& a, const int c )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	int z = min( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int3 min( const int c, const int3& a )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	int z = min( a.z, c );
	return make_int3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int4& a, const int4& b )
{
	int x = max( a.x, b.x );
	int y = max( a.y, b.y );
	int z = max( a.z, b.z );
	int w = max( a.w, b.w );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int4& a, const int c )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	int z = max( a.z, c );
	int w = max( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 max( const int c, const int4& a )
{
	int x = max( a.x, c );
	int y = max( a.y, c );
	int z = max( a.z, c );
	int w = max( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int4& a, const int4& b )
{
	int x = min( a.x, b.x );
	int y = min( a.y, b.y );
	int z = min( a.z, b.z );
	int w = min( a.w, b.w );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int4& a, const int c )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	int z = min( a.z, c );
	int w = min( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE int4 min( const int c, const int4& a )
{
	int x = min( a.x, c );
	int y = min( a.y, c );
	int z = min( a.z, c );
	int w = min( a.w, c );
	return make_int4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 max( const float2& a, const float2& b )
{
	float x = fmaxf( a.x, b.x );
	float y = fmaxf( a.y, b.y );
	return make_float2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 max( const float2& a, const float c )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	return make_float2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 max( const float c, const float2& a )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	return make_float2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 min( const float2& a, const float2& b )
{
	float x = fminf( a.x, b.x );
	float y = fminf( a.y, b.y );
	return make_float2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 min( const float2& a, const float c )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	return make_float2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float2 min( const float c, const float2& a )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	return make_float2( x, y );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 max( const float3& a, const float3& b )
{
	float x = fmaxf( a.x, b.x );
	float y = fmaxf( a.y, b.y );
	float z = fmaxf( a.z, b.z );
	return make_float3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 max( const float3& a, const float c )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	float z = fmaxf( a.z, c );
	return make_float3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 max( const float c, const float3& a )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	float z = fmaxf( a.z, c );
	return make_float3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 min( const float3& a, const float3& b )
{
	float x = fminf( a.x, b.x );
	float y = fminf( a.y, b.y );
	float z = fminf( a.z, b.z );
	return make_float3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 min( const float3& a, const float c )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	float z = fminf( a.z, c );
	return make_float3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 min( const float c, const float3& a )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	float z = fminf( a.z, c );
	return make_float3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 max( const float4& a, const float4& b )
{
	float x = fmaxf( a.x, b.x );
	float y = fmaxf( a.y, b.y );
	float z = fmaxf( a.z, b.z );
	float w = fmaxf( a.w, b.w );
	return make_float4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 max( const float4& a, const float c )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	float z = fmaxf( a.z, c );
	float w = fmaxf( a.w, c );
	return make_float4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 max( const float c, const float4& a )
{
	float x = fmaxf( a.x, c );
	float y = fmaxf( a.y, c );
	float z = fmaxf( a.z, c );
	float w = fmaxf( a.w, c );
	return make_float4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 min( const float4& a, const float4& b )
{
	float x = fminf( a.x, b.x );
	float y = fminf( a.y, b.y );
	float z = fminf( a.z, b.z );
	float w = fminf( a.w, b.w );
	return make_float4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 min( const float4& a, const float c )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	float z = fminf( a.z, c );
	float w = fminf( a.w, c );
	return make_float4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 min( const float c, const float4& a )
{
	float x = fminf( a.x, c );
	float y = fminf( a.y, c );
	float z = fminf( a.z, c );
	float w = fminf( a.w, c );
	return make_float4( x, y, z, w );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 fma( const float3& a, const float3& b, const float3& c )
{
	float x = fmaf( a.x, b.x, c.x );
	float y = fmaf( a.y, b.y, c.y );
	float z = fmaf( a.z, b.z, c.z );
	return make_float3( x, y, z );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 cross( const float3& a, const float3& b )
{
	return make_float3( a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float dot( const float3& a, const float3& b ) { return a.x * b.x + a.y * b.y + a.z * b.z; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 normalize( const float3& a ) { return a / sqrtf( dot( a, a ) ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 safeInv( float3 d )
{
	float  x	 = d.x;
	float  y	 = d.y;
	float  z	 = d.z;
	float  ooeps = 1e-5f;
	float3 invd;
	invd.x = 1.0 / ( abs( x ) > ooeps ? x : copysign( ooeps, x ) );
	invd.y = 1.0 / ( abs( y ) > ooeps ? y : copysign( ooeps, y ) );
	invd.z = 1.0 / ( abs( z ) > ooeps ? z : copysign( ooeps, z ) );
	return invd;
}

HIPRT_HOST_DEVICE HIPRT_INLINE int* ptr( int2& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE int* ptr( int3& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE int* ptr( int4& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE float* ptr( float2& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE float* ptr( float3& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE float* ptr( float4& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE const int* ptr( const int2& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE const int* ptr( const int3& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE const int* ptr( const int4& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE const float* ptr( const float2& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE const float* ptr( const float3& a ) { return &a.x; }

HIPRT_HOST_DEVICE HIPRT_INLINE const float* ptr( const float4& a ) { return &a.x; }

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

template <typename T, typename U>
HIPRT_HOST_DEVICE T roundUp( T value, U factor )
{
	return ( value + factor - 1 ) / factor * factor;
}

template <typename T, typename U>
HIPRT_HOST_DEVICE T divideRoundUp( T value, U factor )
{
	return ( value + factor - 1 ) / factor;
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
