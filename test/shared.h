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

#if defined( __KERNELCC__ )
#include <hiprt/hiprt_device.h>
#else
#include <hiprt/hiprt_vec.h>
#endif

static constexpr bool UseFilter		  = false;
static constexpr bool UseDynamicStack = false;

#if defined( __KERNELCC__ )
typedef typename hiprt::conditional<UseDynamicStack, hiprtDynamicStack, hiprtGlobalStack>::type Stack;
typedef hiprtEmptyInstanceStack																	InstanceStack;
#endif

#define int2 hiprtInt2
#define int3 hiprtInt3
#define int4 hiprtInt4
#define uint2 hiprtUint2

#define float2 hiprtFloat2
#define float3 hiprtFloat3
#define float4 hiprtFloat4

#define make_int2 make_hiprtInt2
#define make_int3 make_hiprtInt3
#define make_int4 make_hiprtInt4
#define make_uint2 make_hiprtUint2

#define make_float2 make_hiprtFloat2
#define make_float3 make_hiprtFloat3
#define make_float4 make_hiprtFloat4

#if !defined( __KERNELCC__ ) || defined( HIPRT_BITCODE_LINKING )
#include <test/Math.h>
#endif

enum
{
	VisualizeColor,
	VisualizeUv,
	VisualizeId,
	VisualizeHitDist,
	VisualizeNormal,
	VisualizeAo
};

struct Material
{
	float3 m_diffuse;
	float3 m_emission;

	HIPRT_HOST_DEVICE HIPRT_INLINE bool light() { return m_emission.x + m_emission.y + m_emission.z > 0.0f; }
};

struct Light
{
	float3 m_le;
	float3 m_lv0;
	float3 m_lv1;
	float3 m_lv2;
	float3 pad;
};

struct Camera
{
	float4 m_rotation;
	float3 m_translation;
	float  m_fov;
};

HIPRT_HOST_DEVICE HIPRT_INLINE float3 gammaCorrect( float3 a )
{
	float g = 1.0f / 2.2f;
	return { pow( a.x, g ), pow( a.y, g ), pow( a.z, g ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t lcg( uint32_t& seed )
{
	const uint32_t LcgA = 1103515245u;
	const uint32_t LcgC = 12345u;
	const uint32_t LcgM = 0x00FFFFFFu;
	seed				= ( LcgA * seed + LcgC );
	return seed & LcgM;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float randf( uint32_t& seed )
{
	return ( static_cast<float>( lcg( seed ) ) / static_cast<float>( 0x01000000 ) );
}

template <uint32_t N>
HIPRT_HOST_DEVICE HIPRT_INLINE uint2 tea( uint32_t val0, uint32_t val1 )
{
	uint32_t v0 = val0;
	uint32_t v1 = val1;
	uint32_t s0 = 0;

	for ( uint32_t n = 0; n < N; n++ )
	{
		s0 += 0x9e3779b9;
		v0 += ( ( v1 << 4 ) + 0xa341316c ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + 0xc8013ea4 );
		v1 += ( ( v0 << 4 ) + 0xad90777d ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + 0x7e95761e );
	}

	return make_uint2( v0, v1 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 sampleHemisphereCosine( float3 n, uint32_t& seed )
{
	float phi		  = hiprt::TwoPi * randf( seed );
	float sinThetaSqr = randf( seed );
	float sinTheta	  = sqrt( sinThetaSqr );

	float3 axis = fabs( n.x ) > 0.001f ? make_float3( 0.0f, 1.0f, 0.0f ) : make_float3( 1.0f, 0.0f, 0.0f );
	float3 t	= hiprt::cross( axis, n );
	t			= hiprt::normalize( t );
	float3 s	= hiprt::cross( n, t );

	return hiprt::normalize( s * cos( phi ) * sinTheta + t * sin( phi ) * sinTheta + n * sqrt( 1.0f - sinThetaSqr ) );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 rotate( const float4& rotation, const float3& p )
{
	float3 a = sinf( rotation.w / 2.0f ) * hiprt::normalize( make_float3( rotation ) );
	float  c = cosf( rotation.w / 2.0f );
	return 2.0f * hiprt::dot( a, p ) * a + ( c * c - hiprt::dot( a, a ) ) * p + 2.0f * c * hiprt::cross( a, p );
}

HIPRT_HOST_DEVICE HIPRT_INLINE hiprtRay
generateRay( float x, float y, int2 res, const Camera& camera, uint32_t& seed, bool isMultiSamples )
{
	const float	 offset		= ( isMultiSamples ) ? randf( seed ) : 0.5f;
	const float2 sensorSize = make_float2( 0.024f * ( res.x / static_cast<float>( res.y ) ), 0.024f );
	const float2 xy			= make_float2( ( x + offset ) / res.x, ( y + offset ) / res.y ) - make_float2( 0.5f, 0.5f );
	const float3 dir =
		make_float3( xy.x * sensorSize.x, xy.y * sensorSize.y, sensorSize.y / ( 2.0f * tan( camera.m_fov / 2.0f ) ) );

	const float3 holDir	 = rotate( camera.m_rotation, make_float3( 1.0f, 0.0f, 0.0f ) );
	const float3 upDir	 = rotate( camera.m_rotation, make_float3( 0.0f, -1.0f, 0.0f ) );
	const float3 viewDir = rotate( camera.m_rotation, make_float3( 0.0f, 0.0f, -1.0f ) );

	hiprtRay ray;
	ray.origin	  = camera.m_translation;
	ray.direction = hiprt::normalize( dir.x * holDir + dir.y * upDir + dir.z * viewDir );
	return ray;
}
