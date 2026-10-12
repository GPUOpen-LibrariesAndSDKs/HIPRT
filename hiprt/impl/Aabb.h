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
	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb() { reset(); }

	HIPRT_HOST_DEVICE HIPRT_INLINE explicit Aabb( const float3& point ) : m_min( point ), m_max( point ) {}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb( const float3& mn, const float3& mx ) : m_min( mn ), m_max( mx ) {}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb( const Aabb& aabb0, const Aabb& aabb1 )
	{
		m_min = min( aabb0.m_min, aabb1.m_min );
		m_max = max( aabb0.m_max, aabb1.m_max );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb( const Aabb& aabb ) : m_min( aabb.m_min ), m_max( aabb.m_max ) {}

	HIPRT_HOST_DEVICE HIPRT_INLINE void reset( void )
	{
		m_min = make_float3( FltMax );
		m_max = make_float3( -FltMax );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb& grow( const Aabb& aabb )
	{
		m_min = min( m_min, aabb.m_min );
		m_max = max( m_max, aabb.m_max );
		return *this;
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb& grow( const float3& point )
	{
		m_min = min( m_min, point );
		m_max = max( m_max, point );
		return *this;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float3 center() const { return ( m_max + m_min ) * 0.5f; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float3 extent() const { return m_max - m_min; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float area() const
	{
		const float3 ext = extent();
		return 2 * ( ext.x * ext.y + ext.x * ext.z + ext.y * ext.z );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE bool valid( void ) const
	{
		return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE bool contains( const float3& point ) const
	{
		return m_min.x <= point.x && m_min.y <= point.y && m_min.z <= point.z && point.x <= m_max.x && point.y <= m_max.y &&
		       point.z <= m_max.z;
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE Aabb& intersect( const Aabb& aabb )
	{
		m_min = max( m_min, aabb.m_min );
		m_max = min( m_max, aabb.m_max );
		return *this;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float2
	intersect( const float3& origin, const float3& invDirection, float maxT ) const
	{
		const float3 f    = ( m_max - origin ) * invDirection;
		const float3 n    = ( m_min - origin ) * invDirection;
		const float3 tmax = max( f, n );
		const float3 tmin = min( f, n );
		const float  t1   = fminf( fminf( fminf( tmax.x, tmax.y ), tmax.z ), maxT );
		const float  t0   = fmaxf( fmaxf( fmaxf( tmin.x, tmin.y ), tmin.z ), 0.0f );
		return float2{ t0, t1 };
	}

#if defined( __KERNELCC__ )
	HIPRT_DEVICE HIPRT_INLINE void atomicGrow( const Aabb& aabb )
	{
		atomicMinFloat( &m_min.x, aabb.m_min.x );
		atomicMinFloat( &m_min.y, aabb.m_min.y );
		atomicMinFloat( &m_min.z, aabb.m_min.z );
		atomicMaxFloat( &m_max.x, aabb.m_max.x );
		atomicMaxFloat( &m_max.y, aabb.m_max.y );
		atomicMaxFloat( &m_max.z, aabb.m_max.z );
	}

	HIPRT_DEVICE HIPRT_INLINE void atomicGrow( const float3& p )
	{
		atomicMinFloat( &m_min.x, p.x );
		atomicMinFloat( &m_min.y, p.y );
		atomicMinFloat( &m_min.z, p.z );
		atomicMaxFloat( &m_max.x, p.x );
		atomicMaxFloat( &m_max.y, p.y );
		atomicMaxFloat( &m_max.z, p.z );
	}

	[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE Aabb shuffle( uint32_t index )
	{
		Aabb aabb;
		aabb.m_min.x = shfl( m_min.x, index );
		aabb.m_min.y = shfl( m_min.x, index );
		aabb.m_min.z = shfl( m_min.x, index );
		aabb.m_max.x = shfl( m_max.x, index );
		aabb.m_max.y = shfl( m_max.y, index );
		aabb.m_max.z = shfl( m_max.z, index );
		return aabb;
	}
#endif

  public:
	float3 m_min{};
	float3 m_max{};
};
} // namespace hiprt
