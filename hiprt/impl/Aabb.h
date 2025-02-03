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
#include <hiprt/hiprt_math.h>

namespace hiprt
{
class Aabb
{
  public:
	HIPRT_HOST_DEVICE Aabb() { reset(); }

	HIPRT_HOST_DEVICE Aabb( const float3& p ) : m_min( p ), m_max( p ) {}

	HIPRT_HOST_DEVICE Aabb( const float3& mi, const float3& ma ) : m_min( mi ), m_max( ma ) {}

	HIPRT_HOST_DEVICE Aabb( const Aabb& rhs, const Aabb& lhs )
	{
		m_min = min( lhs.m_min, rhs.m_min );
		m_max = max( lhs.m_max, rhs.m_max );
	}

	HIPRT_HOST_DEVICE Aabb( const Aabb& rhs ) : m_min( rhs.m_min ), m_max( rhs.m_max ) {}

	HIPRT_HOST_DEVICE void reset( void )
	{
		m_min = make_float3( FltMax );
		m_max = make_float3( -FltMax );
	}

	HIPRT_HOST_DEVICE Aabb& grow( const Aabb& rhs )
	{
		m_min = min( m_min, rhs.m_min );
		m_max = max( m_max, rhs.m_max );
		return *this;
	}

	HIPRT_HOST_DEVICE Aabb& grow( const float3& p )
	{
		m_min = min( m_min, p );
		m_max = max( m_max, p );
		return *this;
	}

	HIPRT_HOST_DEVICE float3 center() const { return ( m_max + m_min ) * 0.5f; }

	HIPRT_HOST_DEVICE float3 extent() const { return m_max - m_min; }

	HIPRT_HOST_DEVICE float area() const
	{
		float3 ext = extent();
		return 2 * ( ext.x * ext.y + ext.x * ext.z + ext.y * ext.z );
	}

	HIPRT_HOST_DEVICE bool valid( void ) { return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z; }

	HIPRT_HOST_DEVICE void intersect( const Aabb& box )
	{
		m_min = max( m_min, box.m_min );
		m_max = min( m_max, box.m_max );
	}

	HIPRT_HOST_DEVICE float2 intersect( const float3& origin, const float3& invDirection, float maxT ) const
	{
		float3 f	= ( m_max - origin ) * invDirection;
		float3 n	= ( m_min - origin ) * invDirection;
		float3 tmax = max( f, n );
		float3 tmin = min( f, n );
		float  t1	= fminf( fminf( fminf( tmax.x, tmax.y ), tmax.z ), maxT );
		float  t0	= fmaxf( fmaxf( fmaxf( tmin.x, tmin.y ), tmin.z ), 0.0f );
		return float2{ t0, t1 };
	}

#if defined( __KERNELCC__ )
	HIPRT_DEVICE void atomicGrow( const Aabb& aabb )
	{
		atomicMinFloat( &m_min.x, aabb.m_min.x );
		atomicMinFloat( &m_min.y, aabb.m_min.y );
		atomicMinFloat( &m_min.z, aabb.m_min.z );
		atomicMaxFloat( &m_max.x, aabb.m_max.x );
		atomicMaxFloat( &m_max.y, aabb.m_max.y );
		atomicMaxFloat( &m_max.z, aabb.m_max.z );
	}

	HIPRT_DEVICE void atomicGrow( const float3& p )
	{
		atomicMinFloat( &m_min.x, p.x );
		atomicMinFloat( &m_min.y, p.y );
		atomicMinFloat( &m_min.z, p.z );
		atomicMaxFloat( &m_max.x, p.x );
		atomicMaxFloat( &m_max.y, p.y );
		atomicMaxFloat( &m_max.z, p.z );
	}

	HIPRT_DEVICE Aabb shuffle( uint32_t index )
	{
		Aabb aabb;
		aabb.m_min.x = __shfl( aabb.m_min.x, index );
		aabb.m_min.y = __shfl( aabb.m_min.x, index );
		aabb.m_min.z = __shfl( aabb.m_min.x, index );
		aabb.m_max.x = __shfl( aabb.m_max.x, index );
		aabb.m_max.y = __shfl( aabb.m_max.y, index );
		aabb.m_max.z = __shfl( aabb.m_max.z, index );
		return aabb;
	}
#endif

  public:
	float3 m_min;
	float3 m_max;
};
} // namespace hiprt
