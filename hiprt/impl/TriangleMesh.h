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
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/BvhNode.h>

namespace hiprt
{
HIPRT_HOST_DEVICE HIPRT_INLINE uint3 tryPairTriangles( const uint3& a, const uint3& b )
{
	uint3 lb = uint3{ 3, 3, 3 };

	lb.x = ( b.x == a.x ) ? 0 : lb.x;
	lb.y = ( b.y == a.x ) ? 0 : lb.y;
	lb.z = ( b.z == a.x ) ? 0 : lb.z;

	lb.x = ( b.x == a.y ) ? 1 : lb.x;
	lb.y = ( b.y == a.y ) ? 1 : lb.y;
	lb.z = ( b.z == a.y ) ? 1 : lb.z;

	lb.x = ( b.x == a.z ) ? 2 : lb.x;
	lb.y = ( b.y == a.z ) ? 2 : lb.y;
	lb.z = ( b.z == a.z ) ? 2 : lb.z;

	if ( ( lb.x == 3 ) + ( lb.y == 3 ) + ( lb.z == 3 ) <= 1 ) return lb;
	return uint3{ InvalidValue, InvalidValue, InvalidValue };
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 shiftLeft( const uint3& a ) { return uint3{ a.y, a.z, a.x }; }

HIPRT_HOST_DEVICE HIPRT_INLINE uint3 shiftRight( const uint3& a ) { return uint3{ a.z, a.x, a.y }; }

class TriangleMesh
{
  public:
	HIPRT_HOST_DEVICE TriangleMesh( const hiprtTriangleMeshPrimitive& mesh )
		: m_vertexCount( mesh.vertexCount ), m_vertexStride( mesh.vertexStride ), m_triangleCount( mesh.triangleCount ),
		  m_triangleStride( mesh.triangleStride ), m_pairCount( mesh.trianglePairCount )
	{
		m_vertices		  = reinterpret_cast<const uint8_t*>( mesh.vertices );
		m_triangleIndices = reinterpret_cast<const uint8_t*>( mesh.triangleIndices );
		m_pairIndices	  = reinterpret_cast<const uint2*>( mesh.trianglePairIndices );
		if ( m_triangleCount == 0 || m_triangleIndices == nullptr ) m_triangleCount = m_vertexCount / 3;
	}

	HIPRT_HOST_DEVICE uint3 fetchTriangleIndices( uint32_t index ) const
	{
		if ( m_triangleIndices == nullptr ) return uint3{ 3 * index + 0, 3 * index + 1, 3 * index + 2 };
		const uint32_t* trianglePtr = reinterpret_cast<const uint32_t*>( m_triangleIndices + index * m_triangleStride );
		return uint3{ trianglePtr[0], trianglePtr[1], trianglePtr[2] };
	}

	HIPRT_HOST_DEVICE TriangleNode fetchTriangleNode( uint2 pairIndices ) const
	{
		uint3 indices0 = fetchTriangleIndices( pairIndices.x );
		uint3 indices1;

		uint32_t flags = DefaultTriangleFlags;
		if ( pairIndices.x != pairIndices.y )
		{
			indices1			= fetchTriangleIndices( pairIndices.y );
			uint3 vertexMapping = tryPairTriangles( indices1, indices0 );

			// align the first triangle to [1,2]
			uint3 flags0 = { 1, 2, 0 };
			if ( vertexMapping.y == 3 )
			{
				vertexMapping = shiftLeft( vertexMapping );
				indices0	  = shiftLeft( indices0 );
				flags0		  = shiftRight( flags0 );
			}
			else if ( vertexMapping.z == 3 )
			{
				vertexMapping = shiftRight( vertexMapping );
				indices0	  = shiftRight( indices0 );
				flags0		  = shiftLeft( flags0 );
			}
			// vertexMapping.x == 3

			// [2 1 0] -- L --> [0 2 1] -- flip --> [0 1 2]
			// [0 2 1] -- R --> [1 0 2] -- flip --> [0 1 2]
			// [1 0 2] -- flip --> [0 1 2]
			bool flip =
				!( ( vertexMapping.y == 0 && vertexMapping.z == 2 ) || ( vertexMapping.y == 2 && vertexMapping.z == 1 ) ||
				   ( vertexMapping.y == 1 && vertexMapping.z == 0 ) );

			// align the second triangle to [1,2]
			uint3 flags1 = flip ? uint3{ 2, 1, 0 } : uint3{ 0, 1, 2 };
			if ( ( vertexMapping.y == 2 && vertexMapping.z == 1 ) || ( vertexMapping.y == 1 && vertexMapping.z == 2 ) )
			{
				// [2 0 1] -- L --> [0 1 2]
				// [2 1 0] -- L --> [1 0 2] -- flip --> [0 1 2]
				indices1 = shiftLeft( indices1 );
				flags1	 = shiftRight( flags1 );
			}
			else if ( ( vertexMapping.y == 2 && vertexMapping.z == 0 ) || ( vertexMapping.y == 0 && vertexMapping.z == 2 ) )
			{
				// [1 2 0] -- R --> [0 1 2]
				// [0 2 1] -- R --> [1 0 2] -- flip --> [0 1 2]
				indices1 = shiftRight( indices1 );
				flags1	 = shiftLeft( flags1 );
			}

			// triangle flags
			flags = ( flip << 13 ) | ( flags1.y << 10 ) | ( flags1.x << 8 ) | ( flags0.y << 2 ) | ( flags0.x << 0 );
		}

		TriangleNode triNode;
		triNode.m_flags		 = flags;
		triNode.m_primIndex0 = pairIndices.x;
		triNode.m_primIndex1 = pairIndices.y;

		const float* vertexPtr0 = reinterpret_cast<const float*>( m_vertices + indices0.x * m_vertexStride );
		const float* vertexPtr1 = reinterpret_cast<const float*>( m_vertices + indices0.y * m_vertexStride );
		const float* vertexPtr2 = reinterpret_cast<const float*>( m_vertices + indices0.z * m_vertexStride );

		triNode.m_triPair.m_v0 = { vertexPtr0[0], vertexPtr0[1], vertexPtr0[2] };
		triNode.m_triPair.m_v1 = { vertexPtr1[0], vertexPtr1[1], vertexPtr1[2] };
		triNode.m_triPair.m_v2 = { vertexPtr2[0], vertexPtr2[1], vertexPtr2[2] };
		triNode.m_triPair.m_v3 = triNode.m_triPair.m_v2;

		if ( pairIndices.x != pairIndices.y )
		{
			const float* vertexPtr3 = reinterpret_cast<const float*>( m_vertices + indices1.z * m_vertexStride );
			triNode.m_triPair.m_v3	= { vertexPtr3[0], vertexPtr3[1], vertexPtr3[2] };
		}

		return triNode;
	}

	HIPRT_HOST_DEVICE TriangleNode fetchTriangleNode( uint32_t index ) const
	{
		uint2 pairIndices = make_uint2( index );
		if ( m_pairCount > 0 ) pairIndices = m_pairIndices[index];
		return fetchTriangleNode( pairIndices );
	}

	HIPRT_HOST_DEVICE Aabb fetchAabb( uint32_t index ) const { return fetchTriangleNode( index ).aabb(); }

	HIPRT_HOST_DEVICE float3 fetchCenter( uint32_t index ) const { return fetchAabb( index ).center(); }

	HIPRT_HOST_DEVICE uint32_t getCount() const { return m_pairCount > 0 ? m_pairCount : m_triangleCount; }

	HIPRT_HOST_DEVICE void setPairs( uint32_t pairCount, const uint2* pairIndices )
	{
		m_pairCount	  = pairCount;
		m_pairIndices = pairIndices;
	}

	HIPRT_HOST_DEVICE bool pairable() { return m_triangleIndices != nullptr && m_triangleCount > 2 && m_pairCount == 0; }

  private:
	const uint8_t* m_vertices;
	uint32_t	   m_vertexCount;
	uint32_t	   m_vertexStride;
	const uint8_t* m_triangleIndices;
	uint32_t	   m_triangleCount;
	uint32_t	   m_triangleStride;
	const uint2*   m_pairIndices = nullptr;
	uint32_t	   m_pairCount	 = 0u;
};
} // namespace hiprt
