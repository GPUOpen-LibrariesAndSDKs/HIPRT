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
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/Aabb.h>

namespace hiprt
{
class alignas( alignof( float3 ) ) Triangle
{
  public:
	Triangle() = default;

	HIPRT_HOST_DEVICE Triangle( const float3& v0, const float3& v1, const float3& v2 ) : m_v0( v0 ), m_v1( v1 ), m_v2( v2 ) {}

	HIPRT_HOST_DEVICE Aabb aabb() const
	{
		Aabb aabb;
		return aabb.grow( m_v0 ).grow( m_v1 ).grow( m_v2 );
	}

	HIPRT_HOST_DEVICE float3 normal( uint32_t flags = 0u ) const
	{
		return ( ( flags >> 5 ) & 1 ) ? cross( m_v2 - m_v0, m_v1 - m_v0 ) : cross( m_v1 - m_v0, m_v2 - m_v0 );
	}

	HIPRT_HOST_DEVICE
	bool intersect( const hiprtRay& ray, float2& uv, float& t, uint32_t flags ) const
	{
		float3 e1 = m_v1 - m_v0;
		float3 e2 = m_v2 - m_v0;
		float3 s1 = cross( ray.direction, e2 );

		float denom = dot( s1, e1 );

		if ( denom == 0.f ) return false;

		float  invDemom = 1.0f / denom;
		float3 d		= ray.origin - m_v0;
		float3 b;
		b.y = dot( d, s1 ) * invDemom;

		float3 s2 = cross( d, e1 );
		b.z		  = dot( ray.direction, s2 ) * invDemom;

		float t0 = dot( e2, s2 ) * invDemom;

		if ( ( b.y < 0.0f ) || ( b.y > 1.0f ) || ( b.z < 0.0f ) || ( b.y + b.z > 1.0f ) || ( t0 < ray.minT ) ||
			 ( t0 > ray.maxT ) )
		{
			return false;
		}
		else
		{
			b.x	 = 1.0f - b.y - b.z;
			uv.x = ptr( b )[( flags >> 0 ) & 3];
			uv.y = ptr( b )[( flags >> 2 ) & 3];
			t	 = t0;
			return true;
		}
	}

	HIPRT_HOST_DEVICE void split( uint32_t axis, float position, const Aabb& box, Aabb& leftBox, Aabb& rightBox ) const
	{
		leftBox = rightBox = Aabb();

		const float3* vertices = &m_v0;
		const float3* v1	   = &vertices[2];

		for ( uint32_t i = 0; i < 3; i++ )
		{
			const float3* v0 = v1;
			v1				 = &vertices[i];
			float v0p		 = ( &v0->x )[axis];
			float v1p		 = ( &v1->x )[axis];

			if ( v0p <= position ) leftBox.grow( *v0 );
			if ( v0p >= position ) rightBox.grow( *v0 );

			if ( ( v0p < position && v1p > position ) || ( v0p > position && v1p < position ) )
			{
				float3 t = mix( *v0, *v1, clamp( ( position - v0p ) / ( v1p - v0p ), 0.0f, 1.0f ) );
				leftBox.grow( t );
				rightBox.grow( t );
			}
		}

		( &leftBox.m_max.x )[axis]	= position;
		( &rightBox.m_min.x )[axis] = position;
		leftBox.intersect( box );
		rightBox.intersect( box );
	}

  public:
	float3 m_v0;
	float3 m_v1;
	float3 m_v2;
};

class alignas( alignof( float3 ) ) TrianglePair
{
  public:
	TrianglePair() = default;

	HIPRT_HOST_DEVICE TrianglePair( const float3& v0, const float3& v1, const float3& v2, const float3& v3 )
		: m_v0( v0 ), m_v1( v1 ), m_v2( v2 ), m_v3( v3 )
	{
	}

	HIPRT_HOST_DEVICE Triangle fetchTriangle( uint32_t index ) const
	{
		if ( index > 0 ) return Triangle( m_v1, m_v3, m_v2 );
		return Triangle( m_v0, m_v1, m_v2 );
	}

	HIPRT_HOST_DEVICE Aabb aabb() const
	{
		Aabb aabb;
		return aabb.grow( m_v0 ).grow( m_v1 ).grow( m_v2 ).grow( m_v3 );
	}

	HIPRT_HOST_DEVICE void split( uint32_t axis, float position, const Aabb& box, Aabb& leftBox, Aabb& rightBox ) const
	{
		Aabb leftBox0, rightBox0;
		fetchTriangle( 0 ).split( axis, position, box, leftBox0, rightBox0 );
		Aabb leftBox1, rightBox1;
		fetchTriangle( 1 ).split( axis, position, box, leftBox1, rightBox1 );
		leftBox	 = Aabb( leftBox0, leftBox1 );
		rightBox = Aabb( rightBox0, rightBox1 );
	}

  public:
	float3 m_v0;
	float3 m_v1;
	float3 m_v2;
	float3 m_v3;
};
} // namespace hiprt
