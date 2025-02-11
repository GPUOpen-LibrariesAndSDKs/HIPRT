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
#include <hiprt/impl/Geometry.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/Triangle.h>

namespace hiprt
{
static constexpr uint32_t DefaultTriangleFlags = ( 2 << 2 ) | ( 1 << 0 );

static constexpr float Ci = 1.0f;
static constexpr float Ct = 1.0f;

enum
{
	TriangleType  = 0,
	TriangleType0 = 1,
	TriangleType1 = 2,
	BoxType		  = 5,
	InstanceType  = 6,
	CustomType	  = 7
};

enum
{
	RootIndex = BoxType
};

HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t getNodeType( uint32_t nodeIndex ) { return ( nodeIndex & 7 ); }

HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t getNodeAddr( uint32_t nodeIndex )
{
	return nodeIndex >> ( getNodeType( nodeIndex ) == BoxType ? 4 : 3 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t encodeNodeIndex( uint32_t nodeAddr, uint32_t nodeType )
{
	if ( nodeType == BoxType ) nodeAddr *= 2;
	return ( nodeAddr << 3 ) | nodeType;
}

HIPRT_HOST_DEVICE HIPRT_INLINE static uint64_t encodeBaseAddr( const void* baseAddr, uint32_t nodeIndex )
{
	uint64_t baseIndex = reinterpret_cast<uint64_t>( baseAddr ) >> 3ull;
	return baseIndex + nodeIndex;
}

HIPRT_HOST_DEVICE HIPRT_INLINE static bool isLeafNode( uint32_t nodeIndex )
{
	return getNodeType( nodeIndex ) != BoxType && nodeIndex != InvalidValue;
}

HIPRT_HOST_DEVICE HIPRT_INLINE static bool isInternalNode( uint32_t nodeIndex ) { return getNodeType( nodeIndex ) == BoxType; }

// 32B
struct alignas( 32 ) ScratchNode
{
	uint32_t m_childIndex0;
	uint32_t m_childIndex1;
	Aabb	 m_box;

	HIPRT_HOST_DEVICE Aabb aabb() const { return m_box; };

	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	HIPRT_HOST_DEVICE uint32_t getChildType( uint32_t i ) const { return getNodeType( ( &m_childIndex0 )[i] ); }

	HIPRT_HOST_DEVICE uint32_t getChildIndex( uint32_t i ) const { return ( &m_childIndex0 )[i]; }

	HIPRT_HOST_DEVICE uint32_t getChildAddr( uint32_t i ) const { return getNodeAddr( ( &m_childIndex0 )[i] ); }

	HIPRT_HOST_DEVICE void encodeChildIndex( uint32_t i, uint32_t childAddr, uint32_t nodeType )
	{
		( &m_childIndex0 )[i] = encodeNodeIndex( childAddr, nodeType );
	}
};
HIPRT_STATIC_ASSERT( sizeof( ScratchNode ) == 32 );

// 128B
struct alignas( DefaultAlignment ) BoxNode
{
	HIPRT_HOST_DEVICE Aabb aabb() const
	{
		Aabb aabb;
		if ( m_childIndex0 != InvalidValue ) aabb.grow( m_box0 );
		if ( m_childIndex1 != InvalidValue ) aabb.grow( m_box1 );
		if ( m_childIndex2 != InvalidValue ) aabb.grow( m_box2 );
		if ( m_childIndex3 != InvalidValue ) aabb.grow( m_box3 );
		return aabb;
	}

	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	HIPRT_HOST_DEVICE uint32_t getChildType( uint32_t i ) const { return getNodeType( ( &m_childIndex0 )[i] ); }

	HIPRT_HOST_DEVICE uint32_t getChildAddr( uint32_t i ) const { return getNodeAddr( ( &m_childIndex0 )[i] ); }

	HIPRT_HOST_DEVICE void encodeChildIndex( uint32_t i, uint32_t childAddr, uint32_t nodeType )
	{
		( &m_childIndex0 )[i] = encodeNodeIndex( childAddr, nodeType );
	}

	HIPRT_HOST_DEVICE Aabb getChildBox( uint32_t i ) { return ( &m_box0 )[i]; }

	HIPRT_HOST_DEVICE void setChildBox( uint32_t i, const Aabb& box ) { ( &m_box0 )[i] = box; }

	uint32_t m_childIndex0 = InvalidValue;
	uint32_t m_childIndex1 = InvalidValue;
	uint32_t m_childIndex2 = InvalidValue;
	uint32_t m_childIndex3 = InvalidValue;
	Aabb	 m_box0;
	Aabb	 m_box1;
	Aabb	 m_box2;
	Aabb	 m_box3;
	uint32_t m_parentAddr	 = InvalidValue;
	uint32_t m_updateCounter = 0;
	uint32_t m_childCount	 = 2;
};
HIPRT_STATIC_ASSERT( sizeof( BoxNode ) == 128 );

// 64B
struct alignas( DefaultAlignment ) TriangleNode
{
	HIPRT_HOST_DEVICE Aabb	aabb() const { return m_triPair.aabb(); }
	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	TrianglePair m_triPair;
	uint32_t	 padding;
	uint32_t	 m_primIndex0 = InvalidValue;
	uint32_t	 m_primIndex1 = InvalidValue;
	uint32_t	 m_flags;
};
HIPRT_STATIC_ASSERT( sizeof( TriangleNode ) == 64 );

// 8B
struct alignas( 4 ) CustomNode
{
	uint32_t m_primIndex = InvalidValue;
};
HIPRT_STATIC_ASSERT( sizeof( CustomNode ) == 4 );

// 64B
struct alignas( 64 ) InstanceNode
{
	union
	{
		float				 m_matrix[3][4];
		hiprtTransformHeader m_transform;
	};

	union
	{
		GeomHeader*	 m_geometry;
		SceneHeader* m_scene;
	};
	uint32_t m_mask = InvalidValue;
	uint32_t m_primIndex : 21;
	uint32_t m_type : 1;
	uint32_t m_static : 1;
	uint32_t m_identity : 1;
	uint32_t : 8;

	HIPRT_HOST_DEVICE const BoxNode* getBoxNodes() const
	{
		if ( m_type == hiprtInstanceTypeGeometry )
			return m_geometry->m_boxNodes;
		else
			return m_scene->m_boxNodes;
	}

	HIPRT_HOST_DEVICE hiprtRay transformRay( const hiprtRay& ray ) const
	{
		hiprtRay	 outRay;
		const float3 o	   = ray.origin;
		const float3 d	   = ray.direction;
		outRay.origin.x	   = m_matrix[0][0] * o.x + m_matrix[0][1] * o.y + m_matrix[0][2] * o.z + m_matrix[0][3];
		outRay.origin.y	   = m_matrix[1][0] * o.x + m_matrix[1][1] * o.y + m_matrix[1][2] * o.z + m_matrix[1][3];
		outRay.origin.z	   = m_matrix[2][0] * o.x + m_matrix[2][1] * o.y + m_matrix[2][2] * o.z + m_matrix[2][3];
		outRay.direction.x = m_matrix[0][0] * d.x + m_matrix[0][1] * d.y + m_matrix[0][2] * d.z;
		outRay.direction.y = m_matrix[1][0] * d.x + m_matrix[1][1] * d.y + m_matrix[1][2] * d.z;
		outRay.direction.z = m_matrix[2][0] * d.x + m_matrix[2][1] * d.y + m_matrix[2][2] * d.z;
		outRay.minT		   = ray.minT;
		outRay.maxT		   = ray.maxT;
		return outRay;
	}
};
HIPRT_STATIC_ASSERT( sizeof( InstanceNode ) == 64 );

// 32B
struct alignas( 32 ) ReferenceNode
{
	ReferenceNode() = default;
	HIPRT_HOST_DEVICE	   ReferenceNode( uint32_t primIndex ) : m_primIndex( primIndex ) {}
	HIPRT_HOST_DEVICE	   ReferenceNode( uint32_t primIndex, const Aabb& box ) : m_primIndex( primIndex ), m_box( box ) {}
	HIPRT_HOST_DEVICE Aabb aabb() const { return m_box; };

	Aabb	 m_box;
	uint32_t m_primIndex = InvalidValue;
};
HIPRT_STATIC_ASSERT( sizeof( ReferenceNode ) == 32 );

// 64B
struct alignas( 64 ) ApiNode
{
	HIPRT_HOST_DEVICE Aabb getChildBox( uint32_t i ) { return Aabb( m_childBoxesMin[i], m_childBoxesMax[i] ); }

	uint32_t m_childIndices[BranchingFactor];
	uint32_t m_leafFlags[BranchingFactor];
	float3	 m_childBoxesMin[BranchingFactor];
	float3	 m_childBoxesMax[BranchingFactor];
};
HIPRT_STATIC_ASSERT( sizeof( ApiNode ) == 128 );
HIPRT_STATIC_ASSERT( sizeof( ApiNode ) == sizeof( hiprtBvhNode ) );
HIPRT_STATIC_ASSERT( alignof( ApiNode ) == alignof( hiprtBvhNode ) );
} // namespace hiprt
