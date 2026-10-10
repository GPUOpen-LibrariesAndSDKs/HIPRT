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
#include <hiprt/impl/Header.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/Triangle.h>

namespace hiprt
{
enum
{
	TriangleType = 0,
	BoxType      = 5,
	InstanceType = 6,
	CustomType   = 7,
	InvalidType  = 15
};

static constexpr uint32_t RootIndex            = BoxType;
static constexpr uint32_t DefaultTriangleFlags = ( 2 << 2 ) | ( 1 << 0 );

static constexpr uint32_t MaxFatLeafSize     = 4;
static constexpr uint32_t MaxBranchingFactor = 16;

static constexpr uint32_t FatLeafBit    = 1u << 31u;
static constexpr uint32_t RangeEndBit   = 1u << 31u;
static constexpr uint32_t RangeStartBit = 1u << 30u;

static constexpr float Ci = 1.0f;
static constexpr float Ct = 1.0f;

struct GeomHeader;
struct SceneHeader;

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t getNodeType( const uint32_t nodeIndex, const uint32_t rtip )
{
	if ( rtip >= 31 )
		return nodeIndex & 15;
	else
		return nodeIndex & 7;
}

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE static uint32_t getNodeType( const uint32_t nodeIndex )
{
#ifndef __KERNELCC__
	throw std::runtime_error( "Function 'getNodeType()' is not supposed to run on the host." );
#else
	return getNodeType( nodeIndex, Rtip );
#endif
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t getNodeAddr( const uint32_t nodeIndex, const uint32_t rtip )
{
	const uint32_t nodeAddr = nodeIndex & ~FatLeafBit;
	if ( rtip >= 31 )
	{
		return nodeAddr >> 4;
	}
	else
	{
		const uint32_t nodeType = getNodeType( nodeAddr, rtip );
		return nodeAddr >> ( nodeType == BoxType || nodeType == InstanceType ? 4 : 3 );
	}
}

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE static uint32_t getNodeAddr( const uint32_t nodeIndex )
{
#ifndef __KERNELCC__
	throw std::runtime_error( "Function 'getNodeAddr()' is not supposed to run on the host." );
#else
	return getNodeAddr( nodeIndex, Rtip );
#endif
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t
encodeNodeIndex( const uint32_t nodeAddr, const uint32_t nodeType, const uint32_t rtip )
{
	if ( rtip >= 31 )
	{
		return ( nodeAddr << 4 ) | ( nodeType & 15 );
	}
	else
	{
		uint32_t nodeShift = 3;
		if ( nodeType == BoxType || nodeType == InstanceType ) nodeShift = 4;
		return ( nodeAddr << nodeShift ) | ( nodeType & 7 );
	}
}

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE static uint32_t encodeNodeIndex( const uint32_t nodeAddr, const uint32_t nodeType )
{
#ifndef __KERNELCC__
	throw std::runtime_error( "Function 'encodeNodeIndex()' is not supposed to run on the host." );
#else
	return encodeNodeIndex( nodeAddr, nodeType, Rtip );
#endif
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t triPairIndexToType( const uint32_t triPairIndex )
{
	return ( triPairIndex & 3 ) + ( ( triPairIndex & 4 ) << 1 );
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t typeToTriPairIndex( const uint32_t nodeType )
{
	return ( nodeType & 3 ) + ( ( nodeType & 8 ) >> 1 );
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static uint64_t
encodeBaseAddr( const void* baseAddr, const uint32_t nodeIndex = 0 )
{
	const uint64_t baseIndex = reinterpret_cast<uint64_t>( baseAddr ) >> 3ull;
	return baseIndex + nodeIndex;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static bool isFatLeafNode( const uint32_t nodeIndex )
{
	return nodeIndex & FatLeafBit;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static bool isLeafNode( uint32_t nodeIndex )
{
	return getNodeType( nodeIndex ) != BoxType && nodeIndex != InvalidValue;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static bool isInternalNode( const uint32_t nodeIndex )
{
	return getNodeType( nodeIndex ) == BoxType;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float ulp( const float x ) { return fabs( x - as_float( as_uint( x ) ^ 1 ) ); }

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float3 subDown( const float3 a, const float3 b )
{
	const float3 d = ( a - b );
	return d - float3{ ulp( d.x ), ulp( d.y ), ulp( d.z ) };
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float3 subUp( const float3 a, const float3 b )
{
	const float3 d = ( a - b );
	return d + float3{ ulp( d.x ), ulp( d.y ), ulp( d.z ) };
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float bf16LowerBound( const float f )
{
	const uint32_t msbs = as_uint( f ) & 0xffff0000;
	if ( f >= 0 ) return as_float( msbs );
	const uint32_t lsbs = as_uint( f ) & 0x0000ffff;
	return as_float( msbs + ( lsbs ? 0x10000 : 0 ) );
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float bf16UpperBound( const float f )
{
	const uint32_t msbs = as_uint( f ) & 0xffff0000;
	if ( f <= 0 ) return as_float( msbs );
	const uint32_t lsbs = as_uint( f ) & 0x0000ffff;
	return as_float( msbs + ( lsbs ? 0x10000 : 0 ) );
}

// 4B
struct alignas( 4 ) CustomNode
{
	static constexpr uint32_t MaxFatLeafSize         = 1;
	static constexpr uint32_t LanesPerLeafPacketTask = 1;

	uint32_t m_primIndex = InvalidValue;
};
HIPRT_STATIC_ASSERT( sizeof( CustomNode ) == 4 );

// 32B
struct alignas( 32 ) ScratchNode
{
	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb aabb() const { return m_box; };

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float area() const { return aabb().area(); }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildType( const uint32_t i ) const
	{
		return getNodeType( m_childIndices[i] & ~FatLeafBit );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildIndex( const uint32_t i ) const
	{
		return m_childIndices[i] & ~FatLeafBit;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildAddr( const uint32_t i ) const
	{
		return getNodeAddr( m_childIndices[i] & ~FatLeafBit );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void setChildFatLeafFlag( const uint32_t i ) { m_childIndices[i] |= FatLeafBit; }

	[[nodiscard]] HIPRT_HOST_DEVICE uint32_t operator[]( const uint32_t i ) const { return m_childIndices[i]; }

	Aabb     m_box;
	uint32_t m_childIndices[2];
};
HIPRT_STATIC_ASSERT( sizeof( ScratchNode ) == 32 );

// 64B
struct alignas( 64 ) ApiNode
{
	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb aabb() const { return m_box; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float area() const { return aabb().area(); }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildType( const uint32_t i ) const
	{
		return m_childTypes[i] & ~FatLeafBit;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildIndex( const uint32_t i ) const
	{
		return encodeNodeIndex( m_childAddrs[i], m_childTypes[i] & ~FatLeafBit );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildAddr( const uint32_t i ) const { return m_childAddrs[i]; }

	HIPRT_HOST_DEVICE HIPRT_INLINE void setChildFatLeafFlag( const uint32_t i ) { m_childTypes[i] |= FatLeafBit; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t operator[]( const uint32_t i ) const
	{
		return getChildIndex( i ) | ( m_childTypes[i] & FatLeafBit );
	}

	Aabb     m_box;
	uint32_t m_childAddrs[2];
	uint32_t m_childTypes[2];
};
HIPRT_STATIC_ASSERT( sizeof( ApiNode ) == sizeof( hiprtInternalNode ) );
HIPRT_STATIC_ASSERT( alignof( ApiNode ) == alignof( hiprtInternalNode ) );

// 32B
struct alignas( 32 ) ReferenceNode
{
	HIPRT_INLINE                   ReferenceNode() = default;
	HIPRT_HOST_DEVICE              HIPRT_INLINE explicit ReferenceNode( const uint32_t primIndex ) : m_primIndex( primIndex ) {}
	HIPRT_HOST_DEVICE HIPRT_INLINE ReferenceNode( const uint32_t primIndex, const Aabb& box )
		: m_box( box ), m_primIndex( primIndex )
	{
	}
	[[nodiscard]] HIPRT_HOST_DEVICE Aabb aabb() const { return m_box; };

	Aabb     m_box;
	uint32_t m_primIndex = InvalidValue;
};
HIPRT_STATIC_ASSERT( sizeof( ReferenceNode ) == 32 );
HIPRT_STATIC_ASSERT( sizeof( ReferenceNode ) == sizeof( hiprtLeafNode ) );
HIPRT_STATIC_ASSERT( alignof( ReferenceNode ) == alignof( hiprtLeafNode ) );

// 12B
struct ConversionTask
{
	HIPRT_INLINE ConversionTask() = default;

	HIPRT_HOST_DEVICE HIPRT_INLINE
	ConversionTask( const uint32_t nodeIndex, const uint32_t nodeAddr, const uint32_t parentAddr )
		: m_nodeIndex( nodeIndex ), m_nodeAddr( nodeAddr ), m_parentAddr( parentAddr )
	{
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE bool valid() const
	{
		return m_nodeIndex != InvalidValue && m_nodeAddr != InvalidValue && m_parentAddr != InvalidValue;
	}

#if defined( __KERNELCC__ )
	[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE ConversionTask atomic_load() const
	{
		ConversionTask task{};
#if HIPRT_RTIP >= 31
		task.m_nodeAddr   = __hip_atomic_load( &m_nodeAddr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT );
		task.m_nodeIndex  = __hip_atomic_load( &m_nodeIndex, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT );
		task.m_parentAddr = __hip_atomic_load( &m_parentAddr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT );
#else
		task  = *this;
#endif
		return task;
	}

	HIPRT_DEVICE HIPRT_INLINE void atomic_store( const ConversionTask& task )
	{
#if HIPRT_RTIP >= 31
		__hip_atomic_store( &m_nodeAddr, task.m_nodeAddr, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT );
		__hip_atomic_store( &m_nodeIndex, task.m_nodeIndex, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT );
		__hip_atomic_store( &m_parentAddr, task.m_parentAddr, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT );
#else
		*this = task;
#endif
	}
#endif

	union
	{
		uint32_t m_rangeOffset = InvalidValue;
		uint32_t m_nodeIndex;
	};

	uint32_t m_nodeAddr = InvalidValue;

	union
	{
		uint32_t m_rangeSize = InvalidValue;
		uint32_t m_parentAddr;
	};
};
HIPRT_STATIC_ASSERT( sizeof( ConversionTask ) >= sizeof( uint3 ) );

// triangle packet helper structs
namespace rtip11
{
// 128B
struct alignas( DefaultAlignment ) BoxNode
{
	static constexpr uint32_t BranchingFactor = 4;

	HIPRT_HOST_DEVICE HIPRT_INLINE void initBox(
		const uint32_t                  i,
		const uint32_t                  childCount,
		const uint32_t                  childIndex,
		const Aabb                      childBox,
		[[maybe_unused]] const Aabb     nodeBox,
		[[maybe_unused]] const uint32_t childRanges = InvalidValue,
		[[maybe_unused]] const uint32_t matrixId    = NoRotationIndex )
	{
		if ( i < childCount )
		{
			m_childIndices[i] = childIndex;
			m_childBoxes[i]   = childBox;
		}
		else
		{
			m_childIndices[i] = InvalidValue;
			m_childBoxes[i]   = Aabb();
		}
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void initBoxes(
		const uint32_t                  childCount,
		const uint32_t*                 childIndices,
		const Aabb*                     childBoxes,
		[[maybe_unused]] const uint32_t childRanges = InvalidValue,
		[[maybe_unused]] const uint32_t matrixId    = NoRotationIndex )
	{
		// at least one child is valid
		m_childIndices[0] = childIndices[0];
		m_childBoxes[0]   = childBoxes[0];

		for ( uint32_t i = 1; i < BranchingFactor; ++i )
		{
			if ( i < childCount )
			{
				m_childIndices[i] = childIndices[i];
				m_childBoxes[i]   = childBoxes[i];
			}
			else
			{
				m_childIndices[i] = InvalidValue;
				m_childBoxes[i]   = Aabb();
			}
		}
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void init(
		const uint32_t                  i,
		const uint32_t                  parentAddr,
		const uint32_t                  childCount,
		[[maybe_unused]] const uint32_t leafType,
		[[maybe_unused]] const uint32_t leafMask,
		[[maybe_unused]] const uint32_t boxNodeBase,
		[[maybe_unused]] const uint32_t primNodeBase,
		const uint32_t                  childIndex,
		const Aabb                      childBox,
		[[maybe_unused]] const Aabb     nodeBox,
		const uint32_t                  childRange = InvalidValue,
		const uint32_t                  matrixId   = NoRotationIndex )
	{
		if ( i == 0 )
		{
			m_parentAddr    = parentAddr;
			m_updateCounter = 0;
			m_childCount    = childCount;
		}

		initBox( i, childCount, childIndex, childBox, nodeBox, childRange, matrixId );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void init(
		const uint32_t                  parentAddr,
		const uint32_t                  childCount,
		[[maybe_unused]] const uint32_t boxNodeBase,
		[[maybe_unused]] const uint32_t primNodeBase,
		const uint32_t*                 childIndices,
		const Aabb*                     childBoxes,
		const uint32_t                  childRanges = InvalidValue,
		const uint32_t                  matrixId    = NoRotationIndex )
	{
		m_parentAddr    = parentAddr;
		m_updateCounter = 0;
		m_childCount    = childCount;
		initBoxes( childCount, childIndices, childBoxes, childRanges, matrixId );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void setPrimNodeBase( [[maybe_unused]] const uint32_t primNodeBase ) {}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb aabb() const
	{
		Aabb aabb;
		if ( m_childIndices[0] != InvalidValue ) aabb.grow( m_childBoxes[0] );
		if ( m_childIndices[1] != InvalidValue ) aabb.grow( m_childBoxes[1] );
		if ( m_childIndices[2] != InvalidValue ) aabb.grow( m_childBoxes[2] );
		if ( m_childIndices[3] != InvalidValue ) aabb.grow( m_childBoxes[3] );
		return aabb;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float area() const { return aabb().area(); }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getParentAddr() const { return m_parentAddr; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildCount() const { return m_childCount; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildIndex( const uint32_t i ) const { return m_childIndices[i]; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildType( const uint32_t i ) const
	{
		return getNodeType( getChildIndex( i ) );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildAddr( const uint32_t i ) const
	{
		return getNodeAddr( getChildIndex( i ) );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildRange( const uint32_t i ) const { return 1; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb getChildBox( const uint32_t i ) const { return m_childBoxes[i]; }

	HIPRT_HOST_DEVICE HIPRT_INLINE void patchParent( const uint32_t primNodeBase )
	{
		uint32_t primNodeOffset = primNodeBase;
		for ( uint32_t i = 0; i < getChildCount(); ++i )
		{
			const uint32_t childIndex = getChildIndex( i );
			if ( isLeafNode( childIndex ) ) m_childIndices[i] = encodeNodeIndex( primNodeOffset++, getNodeType( childIndex ) );
		}
	}

	uint32_t m_childIndices[BranchingFactor] = { InvalidValue, InvalidValue, InvalidValue, InvalidValue };
	Aabb     m_childBoxes[BranchingFactor];
	uint32_t m_parentAddr = InvalidValue;
	uint32_t m_updateCounter;
	uint32_t m_childCount;
};
HIPRT_STATIC_ASSERT( sizeof( BoxNode ) == 128 );
HIPRT_STATIC_ASSERT( BoxNode::BranchingFactor <= MaxBranchingFactor );

// 64B
struct alignas( 64 ) InstanceNode
{
	// We use this value to limit the number of children in root nodes.
	// There are no boxes in instance nodes in RTIP 1.1, and thus we set this value to 4.
	static constexpr uint32_t BranchingFactor        = 4;
	static constexpr uint32_t MaxFatLeafSize         = 1;
	static constexpr uint32_t LanesPerLeafPacketTask = 1;

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE hiprtRay transformRay( const hiprtRay& ray ) const
	{
		return hiprt::transformRay( ray, *this );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void init(
		const uint32_t                  primIndex,
		const uint32_t                  mask,
		const Frame&                    frame,
		const hiprtInstance&            instance,
		const hiprtTransformHeader&     transform,
		[[maybe_unused]] const BoxNode& root )
	{
		m_primIndex = primIndex;
		m_mask      = mask;
		m_type      = instance.type;
		m_static    = transform.frameCount == 1 ? 1 : 0;

		if ( instance.type == hiprtInstanceTypeScene )
			m_childPointer = reinterpret_cast<uint64_t>( instance.scene );
		else
			m_childPointer = reinterpret_cast<uint64_t>( instance.geometry );

		if ( transform.frameCount == 1 )
		{
			m_identity = computeInvTransformMatrix( frame, m_matrix ) ? 1 : 0;
		}
		else
		{
			m_identity  = 0;
			m_transform = transform;
		}
	}

	template <typename Header>
	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE const Header* getHeader() const
	{
		return reinterpret_cast<const Header*>( m_childPointer );
	}

	union
	{
		float                m_matrix[3][4];
		hiprtTransformHeader m_transform;
	};

	uint64_t m_childPointer;
	uint32_t m_mask = FullRayMask;
	uint32_t m_primIndex : InstanceIDBits;
	uint32_t m_type : 1;
	uint32_t m_static : 1;
	uint32_t m_identity : 1;
	uint32_t : 5;
};
HIPRT_STATIC_ASSERT( sizeof( InstanceNode ) == 64 );
HIPRT_STATIC_ASSERT( InstanceNode::MaxFatLeafSize <= MaxFatLeafSize );

// 64B
struct alignas( DefaultAlignment ) TrianglePairNode
{
	static constexpr uint32_t MaxFatLeafSize         = 1;
	static constexpr uint32_t LanesPerLeafPacketTask = 1;

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb  aabb() const { return m_triPair.aabb(); }
	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float area() const { return aabb().area(); }

	[[nodiscard]] HIPRT_HOST_DEVICE uint32_t getPrimIndex( const uint32_t triangleIndex ) const
	{
		return triangleIndex > 0 ? m_primIndex1 : m_primIndex0;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float3 getNormal( const uint32_t triangleIndex ) const
	{
		return m_triPair.fetchTriangle( triangleIndex ).normal( m_flags >> ( triangleIndex * 8 ) );
	}

	TrianglePair m_triPair;
	uint32_t     padding;
	uint32_t     m_primIndex0 = InvalidValue;
	uint32_t     m_primIndex1 = InvalidValue;
	uint32_t     m_flags;
};
HIPRT_STATIC_ASSERT( sizeof( TrianglePairNode ) == 64 );
HIPRT_STATIC_ASSERT( TrianglePairNode::MaxFatLeafSize <= MaxFatLeafSize );
} // namespace rtip11

namespace rtip31
{
struct TrianglePairOffsets
{
	HIPRT_INLINE TrianglePairOffsets() = default;

	HIPRT_HOST_DEVICE HIPRT_INLINE TrianglePairOffsets( const uint32_t pairOffset, const uint32_t packetOffset )
	{
		m_pairOffset   = pairOffset & 7;
		m_packetOffset = packetOffset & 31;
	}

	uint8_t m_pairOffset : 3;
	uint8_t m_packetOffset : 5;
};

struct TrianglePairData
{
	HIPRT_INLINE TrianglePairData() = default;

	HIPRT_HOST_DEVICE HIPRT_INLINE
	TrianglePairData( const uint2& pairIndices, const uint3& triIndices0, const uint3& triIndices1, const bool rangeEnd )
		: m_pairIndices( pairIndices )
	{
		m_descriptor = 0;
		m_descriptor |= rangeEnd ? 1 : 0;
		m_descriptor |= ( triIndices0.x + ( triIndices0.y << 4 ) + ( triIndices0.z << 8 ) ) << 17;
		m_descriptor |= ( triIndices1.x + ( triIndices1.y << 4 ) + ( triIndices1.z << 8 ) ) << 3;
	}

	uint2    m_pairIndices;
	uint32_t m_descriptor;
};

struct ChildInfo
{
	template <typename Node>
	HIPRT_HOST_DEVICE HIPRT_INLINE static void init(
		const uint32_t i,
		const uint32_t childCount,
		const uint32_t childIndex,
		const Aabb     childBox,
		const Aabb     nodeBox,
		const uint32_t childRange,
		Node&          node )
	{
		const float3 extent = subUp( nodeBox.m_max, nodeBox.m_min );
		const float3 origin = nodeBox.m_min;

		uint3 exponent;
		exponent.x = ( as_uint( extent.x ) + 0x7fffff ) >> 23;
		exponent.y = ( as_uint( extent.y ) + 0x7fffff ) >> 23;
		exponent.z = ( as_uint( extent.z ) + 0x7fffff ) >> 23;
		exponent.x = exponent.x == 0 ? 0 : max( 13u, exponent.x );
		exponent.y = exponent.y == 0 ? 0 : max( 13u, exponent.y );
		exponent.z = exponent.z == 0 ? 0 : max( 13u, exponent.z );

		if ( i == 0 )
		{
			node.m_origin    = origin;
			node.m_exponentX = exponent.x;
			node.m_exponentY = exponent.y;
			node.m_exponentZ = exponent.z;
		}

		float3 rcp_exponent;
		rcp_exponent.x = as_float( ( 254 - exponent.x + 12 ) << 23 );
		rcp_exponent.y = as_float( ( 254 - exponent.y + 12 ) << 23 );
		rcp_exponent.z = as_float( ( 254 - exponent.z + 12 ) << 23 );

		ChildInfo childInfo{};
		if ( i < childCount && childBox.valid() )
		{
			float3 qmin = subDown( childBox.m_min, origin ) * rcp_exponent;
			qmin.x      = floor( qmin.x );
			qmin.y      = floor( qmin.y );
			qmin.z      = floor( qmin.z );
			qmin        = clamp( qmin, 0.0f, 4095.0f );

			float3 qmax = subUp( childBox.m_max, origin ) * rcp_exponent;
			qmax.x      = ceil( qmax.x );
			qmax.y      = ceil( qmax.y );
			qmax.z      = ceil( qmax.z );
			qmax        = clamp( qmax, 1.0f, 4096.0f ) - make_float3( 1.0f );

			childInfo.m_minX = static_cast<uint32_t>( qmin.x );
			childInfo.m_minY = static_cast<uint32_t>( qmin.y );
			childInfo.m_minZ = static_cast<uint32_t>( qmin.z );
			childInfo.m_maxX = static_cast<uint32_t>( qmax.x );
			childInfo.m_maxY = static_cast<uint32_t>( qmax.y );
			childInfo.m_maxZ = static_cast<uint32_t>( qmax.z );

			childInfo.m_nodeType     = getNodeType( childIndex );
			childInfo.m_nodeRange    = childRange == InvalidValue ? 1 : childRange & 15;
			childInfo.m_instanceMask = 0xff;
		}
		else
		{
			childInfo.m_minX = 0xfff;
			childInfo.m_minY = 0xfff;
			childInfo.m_minZ = 0xfff;
			childInfo.m_maxX = 0xfff;
			childInfo.m_maxY = 0xfff;
			childInfo.m_maxZ = 0xfff;
		}

		node.m_childInfos[i] = childInfo;
	}

	template <typename Node>
	HIPRT_HOST_DEVICE HIPRT_INLINE static void init(
		const uint32_t  childCount,
		const uint32_t* childIndices,
		const Aabb*     childBoxes,
		const uint32_t  childRanges,
		Node&           node )
	{
		Aabb nodeBox;
		for ( uint32_t i = 0; i < Node::BranchingFactor; ++i )
			nodeBox.grow( childBoxes[i] );

		const float3 extent = subUp( nodeBox.m_max, nodeBox.m_min );
		const float3 origin = nodeBox.m_min;

		uint3 exponent;
		exponent.x = ( as_uint( extent.x ) + 0x7fffff ) >> 23;
		exponent.y = ( as_uint( extent.y ) + 0x7fffff ) >> 23;
		exponent.z = ( as_uint( extent.z ) + 0x7fffff ) >> 23;
		exponent.x = exponent.x == 0 ? 0 : max( 13u, exponent.x );
		exponent.y = exponent.y == 0 ? 0 : max( 13u, exponent.y );
		exponent.z = exponent.z == 0 ? 0 : max( 13u, exponent.z );

		node.m_origin    = origin;
		node.m_exponentX = exponent.x;
		node.m_exponentY = exponent.y;
		node.m_exponentZ = exponent.z;

		float3 rcp_exponent;
		rcp_exponent.x = as_float( ( 254 - exponent.x + 12 ) << 23 );
		rcp_exponent.y = as_float( ( 254 - exponent.y + 12 ) << 23 );
		rcp_exponent.z = as_float( ( 254 - exponent.z + 12 ) << 23 );

		for ( uint32_t i = 0; i < Node::BranchingFactor; ++i )
		{
			const Aabb& childBox = childBoxes[i];

			ChildInfo childInfo{};
			if ( i < childCount && childBox.valid() )
			{
				float3 qmin = subDown( childBox.m_min, origin ) * rcp_exponent;
				qmin.x      = floor( qmin.x );
				qmin.y      = floor( qmin.y );
				qmin.z      = floor( qmin.z );
				qmin        = clamp( qmin, 0.0f, 4095.0f );

				float3 qmax = subUp( childBox.m_max, origin ) * rcp_exponent;
				qmax.x      = ceil( qmax.x );
				qmax.y      = ceil( qmax.y );
				qmax.z      = ceil( qmax.z );
				qmax        = clamp( qmax, 1.0f, 4096.0f ) - make_float3( 1.0f );

				childInfo.m_minX = static_cast<uint32_t>( qmin.x );
				childInfo.m_minY = static_cast<uint32_t>( qmin.y );
				childInfo.m_minZ = static_cast<uint32_t>( qmin.z );
				childInfo.m_maxX = static_cast<uint32_t>( qmax.x );
				childInfo.m_maxY = static_cast<uint32_t>( qmax.y );
				childInfo.m_maxZ = static_cast<uint32_t>( qmax.z );

				childInfo.m_nodeType     = getNodeType( childIndices[i] );
				childInfo.m_nodeRange    = childRanges == InvalidValue ? 1 : ( childRanges >> ( 4 * i ) ) & 15;
				childInfo.m_instanceMask = 0xff;
			}
			else
			{
				childInfo.m_minX = 0xfff;
				childInfo.m_minY = 0xfff;
				childInfo.m_minZ = 0xfff;
				childInfo.m_maxX = 0xfff;
				childInfo.m_maxY = 0xfff;
				childInfo.m_maxZ = 0xfff;
			}

			node.m_childInfos[i] = childInfo;
		}
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE bool valid() const
	{
		return m_minX != 0xfff || m_minY != 0xfff || m_minZ != 0xfff || m_maxX != 0xfff || m_maxY != 0xfff || m_maxZ != 0xfff;
	}

	uint32_t m_minX : 12;
	uint32_t m_minY : 12;
	uint32_t : 8;

	uint32_t m_minZ : 12;
	uint32_t m_maxX : 12;
	uint32_t m_instanceMask : 8;

	uint32_t m_maxY : 12;
	uint32_t m_maxZ : 12;
	uint32_t m_nodeType : 4;
	uint32_t m_nodeRange : 4;
};
HIPRT_STATIC_ASSERT( sizeof( ChildInfo ) == 12 );

struct alignas( DefaultAlignment ) BoxNode
{
	static constexpr uint32_t BranchingFactor = 8;

	HIPRT_HOST_DEVICE HIPRT_INLINE void initBox(
		const uint32_t i,
		const uint32_t childCount,
		const uint32_t childIndex,
		const Aabb     childBox,
		const Aabb     nodeBox,
		const uint32_t childRange = InvalidValue,
		const uint32_t matrixId   = NoRotationIndex )
	{
		if ( i == 0 ) m_matrixId = matrixId;
		ChildInfo::init( i, childCount, childIndex, childBox, nodeBox, childRange, *this );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void initBoxes(
		const uint32_t  childCount,
		const uint32_t* childIndices,
		const Aabb*     childBoxes,
		const uint32_t  childRanges = InvalidValue,
		const uint32_t  matrixId    = NoRotationIndex )
	{
		m_matrixId = matrixId;
		ChildInfo::init( childCount, childIndices, childBoxes, childRanges, *this );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void init(
		const uint32_t                  i,
		const uint32_t                  parentAddr,
		const uint32_t                  childCount,
		[[maybe_unused]] const uint32_t leafType,
		[[maybe_unused]] const uint32_t leafMask,
		const uint32_t                  boxNodeBase,
		const uint32_t                  primNodeBase,
		const uint32_t                  childIndex,
		const Aabb                      childBox,
		const Aabb                      nodeBox,
		const uint32_t                  childRange = InvalidValue,
		const uint32_t                  matrixId   = NoRotationIndex )
	{
		if ( i == 0 )
		{
			m_boxNodeBase        = boxNodeBase << 4;
			m_primNodeBase       = primNodeBase << 4;
			m_parentAddr         = parentAddr;
			m_childCountMinusOne = childCount - 1;
		}

		initBox( i, childCount, childIndex, childBox, nodeBox, childRange, matrixId );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void init(
		const uint32_t  parentAddr,
		const uint32_t  childCount,
		const uint32_t  boxNodeBase,
		const uint32_t  primNodeBase,
		const uint32_t* childIndices,
		const Aabb*     childBoxes,
		const uint32_t  childRanges = InvalidValue,
		const uint32_t  matrixId    = NoRotationIndex )
	{
		m_boxNodeBase        = boxNodeBase << 4;
		m_primNodeBase       = primNodeBase << 4;
		m_parentAddr         = parentAddr;
		m_childCountMinusOne = childCount - 1;

		initBoxes( childCount, childIndices, childBoxes, childRanges, matrixId );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void setPrimNodeBase( const uint32_t primNodeBase ) { m_primNodeBase = primNodeBase << 4; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb aabb() const
	{
		Aabb box;
		for ( uint32_t i = 0; i < getChildCount(); ++i )
			box.grow( getChildBox( i ) );
		return box;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float area() const { return aabb().area(); }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getParentAddr() const { return m_parentAddr; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildCount() const { return m_childCountMinusOne + 1; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildIndex( const uint32_t i ) const
	{
		if ( i >= getChildCount() ) return InvalidValue;
		return encodeNodeIndex( getChildAddr( i ), getChildType( i ) );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildType( const uint32_t i ) const
	{
		return m_childInfos[i].m_nodeType;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildAddr( const uint32_t i ) const
	{
		uint32_t childType = getChildType( i );
		uint32_t childAddr = childType == BoxType ? m_boxNodeBase >> 4 : m_primNodeBase >> 4;
		for ( uint32_t j = 0; j < i; ++j )
			if ( ( getChildType( j ) == BoxType ) == ( childType == BoxType ) ) childAddr += m_childInfos[j].m_nodeRange;
		return childAddr;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb getChildBox( const uint32_t i ) const
	{
		if ( i >= getChildCount() ) return Aabb();

		float3 rcp_exponent;
		rcp_exponent.x = as_float( ( 254 - m_exponentX + 12 ) << 23 );
		rcp_exponent.y = as_float( ( 254 - m_exponentY + 12 ) << 23 );
		rcp_exponent.z = as_float( ( 254 - m_exponentZ + 12 ) << 23 );

		Aabb childBox;
		childBox.m_min.x = m_origin.x + m_childInfos[i].m_minX / rcp_exponent.x;
		childBox.m_min.y = m_origin.y + m_childInfos[i].m_minY / rcp_exponent.y;
		childBox.m_min.z = m_origin.z + m_childInfos[i].m_minZ / rcp_exponent.z;
		childBox.m_max.x = m_origin.x + ( m_exponentX != 0 ? ( m_childInfos[i].m_maxX + 1 ) / rcp_exponent.x : 0.0f );
		childBox.m_max.y = m_origin.y + ( m_exponentY != 0 ? ( m_childInfos[i].m_maxY + 1 ) / rcp_exponent.y : 0.0f );
		childBox.m_max.z = m_origin.z + ( m_exponentZ != 0 ? ( m_childInfos[i].m_maxZ + 1 ) / rcp_exponent.z : 0.0f );

		return childBox;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildRange( const uint32_t i ) const
	{
		return m_childInfos[i].m_nodeRange;
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void patchParent( const uint32_t primNodeBase )
	{
		setPrimNodeBase( primNodeBase );
		for ( uint32_t i = 0; i < getChildCount(); ++i )
		{
			const uint32_t childIndex = getChildIndex( i );
			if ( isLeafNode( childIndex ) )
			{
				m_childInfos[i].m_nodeType  = getNodeType( childIndex );
				m_childInfos[i].m_nodeRange = 1;
			}
		}
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void
	patchParent( const uint32_t primNodeBase, const uint32_t triPairCount, const TrianglePairOffsets* childOffsets )
	{
		setPrimNodeBase( primNodeBase );
		uint32_t leafIndex = 0;
		for ( uint32_t i = 0; i < getChildCount(); ++i )
		{
			const uint32_t childIndex = getChildIndex( i );
			if ( isLeafNode( childIndex ) )
			{
				const uint32_t childType    = triPairIndexToType( childOffsets[leafIndex].m_pairOffset );
				const uint32_t childOffset0 = childOffsets[leafIndex].m_packetOffset;
				const uint32_t childOffset1 = childOffsets[clamp( leafIndex + 1u, 0u, triPairCount - 1u )].m_packetOffset;
				const uint32_t childRange   = childOffset1 - childOffset0;
				m_childInfos[i].m_nodeType  = childType;
				m_childInfos[i].m_nodeRange = childRange;
				leafIndex++;
			}
		}
	}

	uint32_t m_boxNodeBase;
	uint32_t m_primNodeBase;
	uint32_t m_parentAddr;
	union
	{
		float3   m_origin;
		uint32_t m_updateCounter;
	};
	uint8_t m_exponentX;
	uint8_t m_exponentY;
	uint8_t m_exponentZ;
	uint8_t : 4;
	uint8_t  m_childCountMinusOne : 4;
	uint32_t m_matrixId : 7;
	uint32_t : 25;
	ChildInfo m_childInfos[8] = {};
};
HIPRT_STATIC_ASSERT( sizeof( BoxNode ) == 128 );
HIPRT_STATIC_ASSERT( BoxNode::BranchingFactor <= MaxBranchingFactor );

// 128B
struct alignas( 128 ) InstanceNode
{
	static constexpr uint32_t BranchingFactor        = 4;
	static constexpr uint32_t MaxFatLeafSize         = 1;
	static constexpr uint32_t LanesPerLeafPacketTask = 1;

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE hiprtRay transformRay( const hiprtRay& ray ) const
	{
		return hiprt::transformRay( ray, *this );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void
	initBoxes( const uint32_t childCount, const uint32_t* childIndices, const Aabb* childBoxes )
	{
		ChildInfo::init( childCount, childIndices, childBoxes, InvalidValue, *this );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void init(
		const uint32_t              primIndex,
		const uint32_t              mask,
		const Frame&                frame,
		const hiprtInstance&        instance,
		const hiprtTransformHeader& transform,
		const BoxNode&              root )
	{
		m_primIndex      = primIndex;
		m_mask           = mask;
		m_hwMask         = 0xff;
		m_type           = instance.type;
		m_static         = transform.frameCount == 1 ? 1 : 0;
		m_disableBoxSort = 0;

		if ( instance.type == hiprtInstanceTypeScene )
			m_childPointer = reinterpret_cast<uint64_t>( instance.scene );
		else
			m_childPointer = reinterpret_cast<uint64_t>( instance.geometry );

		if ( transform.frameCount == 1 )
		{
			m_identity = computeInvTransformMatrix( frame, m_matrix ) ? 1 : 0;
		}
		else
		{
			m_identity  = 0;
			m_transform = transform;
		}

		Aabb           childBoxes[InstanceNode::BranchingFactor];
		uint32_t       childIndices[InstanceNode::BranchingFactor];
		const uint32_t childCount = root.getChildCount();
		for ( uint32_t i = 0; i < childCount; ++i )
		{
			childBoxes[i]   = root.getChildBox( i );
			childIndices[i] = root.getChildIndex( i );
		}

		m_childCountMinusOne = childCount - 1;
		initBoxes( childCount, childIndices, childBoxes );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getChildCount() const { return m_childCountMinusOne + 1; }

	template <typename Header>
	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE const Header* getHeader() const
	{
		return reinterpret_cast<const Header*>( m_childPointer );
	}

	union
	{
		float                m_matrix[3][4];
		hiprtTransformHeader m_transform;
	};

	uint64_t m_childPointer;
	uint32_t m_mask = FullRayMask;
	uint32_t m_primIndex : InstanceIDBits;
	uint32_t m_hwMask : 8;

	float3  m_origin;
	uint8_t m_exponentX;
	uint8_t m_exponentY;
	uint8_t m_exponentZ;

	uint8_t m_disableBoxSort : 1;
	uint8_t m_type : 1;
	uint8_t m_static : 1;
	uint8_t m_identity : 1;
	uint8_t m_childCountMinusOne : 2;
	uint8_t : 2;

	ChildInfo m_childInfos[BranchingFactor];
};
HIPRT_STATIC_ASSERT( sizeof( InstanceNode ) == 128 );
HIPRT_STATIC_ASSERT( InstanceNode::MaxFatLeafSize <= MaxFatLeafSize );
HIPRT_STATIC_ASSERT( InstanceNode::BranchingFactor <= MaxBranchingFactor );

struct TrianglePacketHeader
{
	uint32_t m_primIndexAnchorSize;
	uint32_t m_primIndexPayloadSize;
	uint32_t m_vertCount;
	uint32_t m_triPairCount;
	uint32_t m_payloadSizeX;
	uint32_t m_payloadSizeY;
	uint32_t m_payloadSizeZ;
	uint32_t m_vertexTzBits;
	uint32_t m_indexSectionMidpoint;
};

// 128B
struct alignas( DefaultAlignment ) TrianglePacketNode
{
	static constexpr uint32_t MaxFatLeafSize         = 4;
	static constexpr uint32_t MaxVertices            = 16;
	static constexpr uint32_t MinTrianglePairs       = 2;
	static constexpr uint32_t MaxTrianglePairs       = 8;
	static constexpr uint32_t HeaderSize             = 52;
	static constexpr uint32_t DescriptorSize         = 29;
	static constexpr uint32_t LanesPerLeafPacketTask = 4;

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE HIPRT_INLINE void
	writeUnalignedBits( const uint32_t position, const uint32_t length, const uint32_t data )
	{
		if ( length )
		{
			const uint32_t hiOfs  = ( position + length ) % 32;
			const uint32_t loWord = position / 32;
			const uint32_t hiWord = ( position + length ) / 32;

			[[maybe_unused]] const uint32_t loMask = ( length == 32 ? ~0u : ( 1u << length ) - 1 ) << ( position % 32 );
			const uint32_t                  hiMask = ( 1u << hiOfs ) - 1;

			const uint32_t loBits = data << ( position % 32 );
#if defined( __KERNELCC__ )
			if constexpr ( Atomic )
			{
				if constexpr ( Clear )
				{
					__threadfence();
					atomicAnd( &m_data[loWord], ~loMask );
					__threadfence();
					atomicOr( &m_data[loWord], loBits );
					__threadfence();
				}
				else
				{
					atomicOr( &m_data[loWord], loBits );
				}
			}
			else
			{
				if constexpr ( Clear ) m_data[loWord] &= ~loMask;
				m_data[loWord] |= loBits;
			}
#else
			if constexpr ( Clear ) m_data[loWord] &= ~loMask;
			m_data[loWord] |= loBits;
#endif

			if ( hiWord < 32 && hiWord != loWord && hiMask > 0 )
			{
				const uint32_t hiBits = data >> ( length - hiOfs );
#if defined( __KERNELCC__ )
				if constexpr ( Atomic )
				{
					if constexpr ( Clear )
					{
						__threadfence();
						atomicAnd( &m_data[hiWord], ~hiMask );
						__threadfence();
						atomicOr( &m_data[hiWord], hiBits );
						__threadfence();
					}
					else
					{
						atomicOr( &m_data[hiWord], hiBits );
					}
				}
				else
				{
					if constexpr ( Clear ) m_data[hiWord] &= ~hiMask;
					m_data[hiWord] |= hiBits;
				}
#else
				if constexpr ( Clear ) m_data[hiWord] &= ~hiMask;
				m_data[hiWord] |= hiBits;
#endif
			}
		}
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t
	readUnalignedBits( const uint32_t position, const uint32_t length ) const
	{
		uint32_t data = 0;
		if ( length )
		{
			const uint32_t hiOfs  = ( position + length ) % 32;
			const uint32_t loWord = position / 32;
			const uint32_t hiWord = ( position + length ) / 32;
			const uint32_t loMask = ( length == 32 ? ~0u : ( 1u << length ) - 1 ) << ( position % 32 );
			const uint32_t hiMask = ( 1u << hiOfs ) - 1;
			const uint32_t loBits = ( m_data[loWord] & loMask ) >> ( position % 32 );

			data = loBits;
			if ( hiWord < 32 && hiWord != loWord && hiMask > 0 )
			{
				const uint32_t hiBits = ( m_data[hiWord] & hiMask ) << ( length - hiOfs );
				data |= hiBits;
			}
		}
		return data;
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE HIPRT_INLINE void writeHeader( const TrianglePacketHeader& hdr )
	{
		uint32_t dw0 = hdr.m_payloadSizeX - 1;
		dw0 += ( hdr.m_payloadSizeY - 1 ) << 5;
		dw0 += ( hdr.m_payloadSizeZ - 1 ) << 10;
		dw0 += hdr.m_vertexTzBits << 15;
		dw0 += ( hdr.m_vertCount - 1 ) << 20;
		dw0 += ( hdr.m_triPairCount - 1 ) << 28;
		m_data[0] = dw0;

		uint32_t dw1 = hdr.m_primIndexAnchorSize;
		dw1 += hdr.m_primIndexPayloadSize << 5;
		dw1 += hdr.m_indexSectionMidpoint << 10;

		[[maybe_unused]] const uint32_t mask = ( 1u << ( HeaderSize - 32 ) ) - 1;
#if defined( __KERNELCC__ )
		if constexpr ( Atomic )
		{
			if constexpr ( Clear )
			{
				__threadfence();
				atomicAnd( &m_data[1], ~mask );
				__threadfence();
				atomicOr( &m_data[1], dw1 );
				__threadfence();
			}
			else
			{
				atomicOr( &m_data[1], dw1 );
			}
		}
		else
		{
			if constexpr ( Clear ) m_data[1] &= ~mask;
			m_data[1] |= dw1;
		}
#else
		if constexpr ( Clear ) m_data[1] &= ~mask;
		m_data[1] |= dw1;
#endif
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE TrianglePacketHeader readHeader()
	{
		TrianglePacketHeader hdr{};

		hdr.m_payloadSizeX = readUnalignedBits( 0, 5 ) + 1;
		hdr.m_payloadSizeY = readUnalignedBits( 5, 5 ) + 1;
		hdr.m_payloadSizeZ = readUnalignedBits( 10, 5 ) + 1;
		hdr.m_vertexTzBits = readUnalignedBits( 15, 5 );

		hdr.m_vertCount    = getVertexCount();
		hdr.m_triPairCount = getTrianglePairCount();

		hdr.m_primIndexAnchorSize  = getPrimIndexAnchorSize();
		hdr.m_primIndexPayloadSize = getPrimIndexPayloadSize();
		hdr.m_indexSectionMidpoint = getIndexSectionMidpoint();

		return hdr;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getVertexCount() const { return readUnalignedBits( 20, 4 ) + 1; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getTrianglePairCount() const
	{
		return readUnalignedBits( 28, 3 ) + 1;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getIndexSectionMidpoint() const
	{
		return readUnalignedBits( 32 + 10, 10 );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getPrimIndexAnchorSize() const
	{
		return readUnalignedBits( 32 + 0, 5 );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getPrimIndexPayloadSize() const
	{
		return readUnalignedBits( 32 + 5, 5 );
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE HIPRT_INLINE void writePrimIndex(
		const uint32_t pairIndex, const uint32_t triangleIndex, const TrianglePacketHeader& hdr, const uint32_t primIndex )
	{
		const uint32_t flatTriIndex = 2 * pairIndex + triangleIndex;

		const uint32_t primIndexPayloadSize = hdr.m_primIndexPayloadSize;
		const uint32_t primIndexAnchorSize  = hdr.m_primIndexAnchorSize;
		const uint32_t primIndexAnchorPos   = hdr.m_indexSectionMidpoint;
		const uint32_t primIndexPayloadPos =
			primIndexAnchorPos + primIndexAnchorSize + ( flatTriIndex - 1 ) * primIndexPayloadSize;

		const uint32_t primIndexPos  = ( flatTriIndex == 0 ) ? primIndexAnchorPos : primIndexPayloadPos;
		const uint32_t primIndexSize = ( flatTriIndex == 0 ) ? primIndexAnchorSize : primIndexPayloadSize;
		const uint32_t primIndexMask = ( 1 << primIndexSize ) - 1;

		writeUnalignedBits<Atomic, Clear>( primIndexPos, primIndexSize, primIndex & primIndexMask );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t
	readPrimIndex( const uint32_t pairIndex, const uint32_t triangleIndex ) const
	{
		const uint32_t flatTriIndex = 2 * pairIndex + triangleIndex;

		const uint32_t primIndexPayloadSize = getPrimIndexPayloadSize();
		const uint32_t primIndexAnchorSize  = getPrimIndexAnchorSize();
		const uint32_t primIndexAnchorPos   = getIndexSectionMidpoint();
		const uint32_t primIndexPayloadPos =
			primIndexAnchorPos + primIndexAnchorSize + ( flatTriIndex - 1 ) * primIndexPayloadSize;

		const uint32_t primIndexAnchor = readUnalignedBits( primIndexAnchorPos, primIndexAnchorSize );
		if ( flatTriIndex == 0 ) return primIndexAnchor;

		const uint32_t primIndex     = readUnalignedBits( primIndexPayloadPos, primIndexPayloadSize );
		const uint32_t primIndexMask = ( 1 << primIndexPayloadSize ) - 1;

		if ( primIndexPayloadSize >= primIndexAnchorSize )
			return primIndex;
		else
			return primIndex | ( primIndexAnchor & ~primIndexMask );
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE HIPRT_INLINE void writeDescriptor( const uint32_t pairIndex, const uint32_t descriptor )
	{
		const uint32_t position = 1024 - ( pairIndex + 1 ) * DescriptorSize;
		writeUnalignedBits<Atomic, Clear>( position, DescriptorSize, descriptor );
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE HIPRT_INLINE void
	writeDescriptor( const uint32_t pairIndex, const uint3& triIndices0, const uint3& triIndices1, const bool rangeEnd )
	{
		uint32_t descriptor = 0;
		descriptor |= rangeEnd ? 1 : 0;
		descriptor |= ( triIndices0.x + ( triIndices0.y << 4 ) + ( triIndices0.z << 8 ) ) << 17;
		descriptor |= ( triIndices1.x + ( triIndices1.y << 4 ) + ( triIndices1.z << 8 ) ) << 3;
		writeDescriptor<Atomic, Clear>( pairIndex, descriptor );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint4
	readDescriptor( const uint32_t pairIndex, const uint32_t triangleIndex ) const
	{
		const uint32_t position   = 1024 - ( pairIndex + 1 ) * DescriptorSize;
		const uint32_t descriptor = readUnalignedBits( position, DescriptorSize );
		const uint32_t triIndices = descriptor >> ( triangleIndex > 0 ? 3 : 17 );
		return uint4{ triIndices & 15, ( triIndices >> 4 ) & 15, ( triIndices >> 8 ) & 15, descriptor & 1 };
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE HIPRT_INLINE void writeVertex( const uint32_t vertexIndex, const float3& vertex )
	{
		const uint32_t position = HeaderSize + 96 * vertexIndex;
		writeUnalignedBits<Atomic, Clear>( position + 0 * 32, 32, as_uint( vertex.x ) );
		writeUnalignedBits<Atomic, Clear>( position + 1 * 32, 32, as_uint( vertex.y ) );
		writeUnalignedBits<Atomic, Clear>( position + 2 * 32, 32, as_uint( vertex.z ) );
	}

	// Funnel shift, i.e. the low 32 bits of ( hi:lo ) >> shift. Maps to a single v_alignbit_b32.
	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t
	funnelShiftRight( const uint32_t lo, const uint32_t hi, const uint32_t shift )
	{
		return shift == 0u ? lo : ( lo >> shift ) | ( hi << ( 32u - shift ) );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float3 readVertex( const uint32_t vertexIndex ) const
	{
		const uint32_t position = HeaderSize + 96 * vertexIndex;

		// A vertex is 96 contiguous bits, so it spans at most four dwords. Reading those once and
		// funnel shifting is cheaper than three readUnalignedBits() calls: HeaderSize is not a
		// multiple of 32 and 96 is, so every vertex sits at the same non-zero bit offset and each
		// component would otherwise take a two-word straddle with a redundant mask on each half.
		const uint32_t word  = position >> 5;
		const uint32_t shift = position & 31u;

		const uint32_t w0 = m_data[word];
		const uint32_t w1 = m_data[word + 1];
		const uint32_t w2 = m_data[word + 2];
		// Matches readUnalignedBits(), which skips the upper half once it would leave the node.
		const uint32_t w3 = ( shift != 0u && word + 3 < 32 ) ? m_data[word + 3] : 0u;

		float3 vertex;
		vertex.x = as_float( funnelShiftRight( w0, w1, shift ) );
		vertex.y = as_float( funnelShiftRight( w1, w2, shift ) );
		vertex.z = as_float( funnelShiftRight( w2, w3, shift ) );
		return vertex;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Triangle
	fetchTriangle( const uint32_t pairIndex, const uint32_t triangleIndex ) const
	{
		const uint3 triIndices = make_uint3( readDescriptor( pairIndex, triangleIndex ) );

		Triangle triangle;
		triangle.m_vertices[0] = readVertex( triIndices.x );
		triangle.m_vertices[1] = readVertex( triIndices.y );
		triangle.m_vertices[2] = readVertex( triIndices.z );

		return triangle;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t
	getPrimIndex( const uint32_t pairIndex, const uint32_t triangleIndex ) const
	{
		return readPrimIndex( pairIndex, triangleIndex );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float3
	getNormal( const uint32_t pairIndex, const uint32_t triangleIndex ) const
	{
		return fetchTriangle( pairIndex, triangleIndex ).normal();
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE bool isRangeEnd( const uint32_t pairIndex ) const
	{
		return readDescriptor( pairIndex, 0 ).w != 0;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Obb
	obb( const uint32_t pairIndex, const uint32_t matrixIndex, const Aabb& box ) const
	{
		Obb obb( matrixIndex );

		uint32_t                  triPairIndex  = pairIndex;
		const TrianglePacketNode* triPacketNode = this;

		while ( true )
		{
			const Triangle& tri0 = triPacketNode->fetchTriangle( triPairIndex, 0 );
			obb.m_box.grow( tri0.obb( box, matrixIndex ).m_box );

			if ( triPacketNode->getPrimIndex( triPairIndex, 0 ) != triPacketNode->getPrimIndex( triPairIndex, 1 ) )
			{
				const Triangle& tri1 = triPacketNode->fetchTriangle( triPairIndex, 1 );
				obb.m_box.grow( tri1.obb( box, matrixIndex ).m_box );
			}

			bool nodeEnd  = triPairIndex + 1 == triPacketNode->getTrianglePairCount();
			bool rangeEnd = triPacketNode->isRangeEnd( triPairIndex );

			if ( rangeEnd ) break;

			triPairIndex++;
			if ( nodeEnd )
			{
				triPairIndex = 0;
				triPacketNode++;
			}
		}

		return obb;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Aabb aabb( const uint32_t pairIndex ) const
	{
		Aabb box;

		uint32_t                  triPairIndex  = pairIndex;
		const TrianglePacketNode* triPacketNode = this;

		while ( true )
		{
			const Triangle& tri0 = triPacketNode->fetchTriangle( triPairIndex, 0 );
			box.grow( tri0.m_vertices[0] );
			box.grow( tri0.m_vertices[1] );
			box.grow( tri0.m_vertices[2] );

			if ( triPacketNode->getPrimIndex( triPairIndex, 0 ) != triPacketNode->getPrimIndex( triPairIndex, 1 ) )
			{
				const Triangle& tri1 = triPacketNode->fetchTriangle( triPairIndex, 1 );
				box.grow( tri1.m_vertices[0] );
				box.grow( tri1.m_vertices[1] );
				box.grow( tri1.m_vertices[2] );
			}

			bool nodeEnd  = triPairIndex + 1 == triPacketNode->getTrianglePairCount();
			bool rangeEnd = triPacketNode->isRangeEnd( triPairIndex );

			if ( rangeEnd ) break;

			triPairIndex++;
			if ( nodeEnd )
			{
				triPairIndex = 0;
				triPacketNode++;
			}
		}

		return box;
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE float area( const uint32_t pairIndex ) const
	{
		return aabb( pairIndex ).area();
	}

	uint32_t m_data[32];
};

HIPRT_STATIC_ASSERT( sizeof( TrianglePacketNode ) == 128 );
HIPRT_STATIC_ASSERT( TrianglePacketNode::MaxFatLeafSize <= MaxFatLeafSize );

struct TrianglePacketData
{
	HIPRT_INLINE TrianglePacketData() = default;

	HIPRT_HOST_DEVICE HIPRT_INLINE
	TrianglePacketData( const uint32_t primIndex0, const uint32_t primIndex1, const uint32_t vertCount )
	{
		tryAddTrianglePair( primIndex0, primIndex1, vertCount );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getVertexCount() const { return m_vertCount; }

	HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getTrianglePairCount() const { return m_triPairCount; }

	HIPRT_HOST_DEVICE HIPRT_INLINE bool
	tryAddTrianglePair( const uint32_t primIndex0, const uint32_t primIndex1, const uint32_t newVertCount )
	{
		if ( m_triPairCount == 0 )
		{
			m_primIndexAnchor = primIndex0;
			m_primIndexDiff   = primIndex0 ^ primIndex1;
			m_triPairCount    = 1;
			m_vertCount       = newVertCount;
			return true;
		}
		else
		{
			if ( m_triPairCount + 1 > TrianglePacketNode::MaxTrianglePairs ||
			     m_vertCount + newVertCount > TrianglePacketNode::MaxVertices )
				return false;

			uint32_t newPrimIndexDiff =
				m_primIndexDiff | ( m_primIndexAnchor ^ primIndex0 ) | ( m_primIndexAnchor ^ primIndex1 );
			uint32_t primAnchorSize  = 32 - clz( m_primIndexAnchor );
			uint32_t primPayloadSize = 32 - clz( newPrimIndexDiff );

			uint32_t headerBits     = TrianglePacketNode::HeaderSize;
			uint32_t primIndexBits  = primAnchorSize + ( 2 * ( m_triPairCount + 1 ) - 1 ) * primPayloadSize;
			uint32_t vertexBits     = 96 * ( m_vertCount + newVertCount );
			uint32_t descriptorBits = TrianglePacketNode::DescriptorSize * ( m_triPairCount + 1 );
			if ( headerBits + vertexBits + descriptorBits + primIndexBits > 1024 ) return false;

			m_primIndexDiff = newPrimIndexDiff;
			m_vertCount += newVertCount;
			m_triPairCount++;

			return true;
		}
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE TrianglePacketHeader buildHeader() const
	{
		TrianglePacketHeader hdr;
		hdr.m_triPairCount         = m_triPairCount;
		hdr.m_vertCount            = m_vertCount;
		hdr.m_primIndexAnchorSize  = 32 - clz( m_primIndexAnchor );
		hdr.m_primIndexPayloadSize = 32 - clz( m_primIndexDiff );

		const uint32_t pairDescSize         = hdr.m_triPairCount * TrianglePacketNode::DescriptorSize;
		const uint32_t primIndexPayloadSize = ( ( 2 * hdr.m_triPairCount ) - 1 ) * hdr.m_primIndexPayloadSize;
		hdr.m_indexSectionMidpoint          = 1024 - pairDescSize - primIndexPayloadSize - hdr.m_primIndexAnchorSize;

		hdr.m_payloadSizeX = 32;
		hdr.m_payloadSizeY = 32;
		hdr.m_payloadSizeZ = 32;
		hdr.m_vertexTzBits = 0;
		return hdr;
	}

#if defined( __KERNELCC__ )
	[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE TrianglePacketData shuffle( const uint32_t index )
	{
		TrianglePacketData data{};
		data.m_primIndexAnchor = shfl( m_primIndexAnchor, index );
		data.m_primIndexDiff   = shfl( m_primIndexDiff, index );
		data.m_triPairCount    = shfl( m_triPairCount, index );
		data.m_vertCount       = shfl( m_vertCount, index );
		return data;
	}
#endif

	uint32_t m_primIndexAnchor;
	uint32_t m_primIndexDiff;
	uint32_t m_triPairCount = 0;
	uint32_t m_vertCount    = 0;
};

struct alignas( DefaultAlignment ) TrianglePacketCache
{
	HIPRT_INLINE     TrianglePacketCache() = default;
	TrianglePairData m_triPairData[TrianglePacketNode::MaxTrianglePairs];
	uint32_t         m_vertexIndices[TrianglePacketNode::MaxVertices];
};
HIPRT_STATIC_ASSERT( sizeof( TrianglePacketCache ) >= sizeof( TrianglePacketNode ) );
} // namespace rtip31

#if defined( __KERNELCC__ )
#if HIPRT_RTIP >= 31
using TriangleNode = rtip31::TrianglePacketNode;
using BoxNode      = rtip31::BoxNode;
using InstanceNode = rtip31::InstanceNode;
#else
using TriangleNode = rtip11::TrianglePairNode;
using BoxNode = rtip11::BoxNode;
using InstanceNode = rtip11::InstanceNode;
#endif
#endif
} // namespace hiprt
