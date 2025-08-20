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
#include <hiprt/impl/Triangle.h>
#include <hiprt/impl/Transform.h>

namespace hiprt
{
enum
{
	TriangleType = 0,
	BoxType		 = 5,
	InstanceType = 6,
	CustomType	 = 7
};

static constexpr uint32_t RootIndex							= BoxType;
static constexpr uint32_t DefaultTriangleFlags				= ( 2 << 2 ) | ( 1 << 0 );
static constexpr uint32_t TrianglePairDescriptorSize		= 29;
static constexpr uint32_t TriangleStructHeaderSize			= 52;
static constexpr uint32_t MaxVerticesPerTrianglePacket		= 16;
static constexpr uint32_t MaxTrianglePairsPerTrianglePacket = 8;
static constexpr uint32_t MinTrianglePairsPerPacket			= 2u;

static constexpr uint32_t FatLeafBit	= 1u << 31u;
static constexpr uint32_t RangeEndBit	= 1u << 31u;
static constexpr uint32_t RangeStartBit = 1u << 30u;

static constexpr float Ci = 1.0f;
static constexpr float Ct = 1.0f;

struct GeomHeader;
struct SceneHeader;

HIPRT_DEVICE HIPRT_INLINE static uint32_t getNodeType( uint32_t nodeIndex )
{
#ifndef __KERNELCC__
	throw std::runtime_error( "Function 'getNodeType()' is not supposed to run on the host." );
#else
	if constexpr ( Rtip >= 31 )
		return nodeIndex & 15;
	else
		return nodeIndex & 7;
#endif
}

HIPRT_DEVICE HIPRT_INLINE static uint32_t getNodeAddr( uint32_t nodeIndex )
{
#ifndef __KERNELCC__
	throw std::runtime_error( "Function 'getNodeAddr()' is not supposed to run on the host." );
#else
	nodeIndex &= ~FatLeafBit;
	if constexpr ( Rtip >= 31 )
	{
		return nodeIndex >> 4;
	}
	else
	{
		const uint32_t nodeType = getNodeType( nodeIndex );
		return nodeIndex >> ( nodeType == BoxType || nodeType == InstanceType ? 4 : 3 );
	}
#endif
}

HIPRT_DEVICE HIPRT_INLINE static uint32_t encodeNodeIndex( uint32_t nodeAddr, uint32_t nodeType )
{
#ifndef __KERNELCC__
	throw std::runtime_error( "Function 'encodeNodeIndex()' is not supposed to run on the host." );
#else
	if constexpr ( Rtip >= 31 )
	{
		return ( nodeAddr << 4 ) | nodeType;
	}
	else
	{
		if ( nodeType == BoxType || nodeType == InstanceType ) nodeAddr <<= 1;
		return ( nodeAddr << 3 ) | nodeType;
	}
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t triPairIndexToType( uint32_t triPairIndex )
{
	return ( triPairIndex & 3 ) + ( ( triPairIndex & 4 ) << 1 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE static uint32_t typeToTriPairIndex( uint32_t nodeType )
{
	return ( nodeType & 3 ) + ( ( nodeType & 8 ) >> 1 );
}

HIPRT_HOST_DEVICE HIPRT_INLINE static uint64_t encodeBaseAddr( const void* baseAddr, uint32_t nodeIndex = 0 )
{
	uint64_t baseIndex = reinterpret_cast<uint64_t>( baseAddr ) >> 3ull;
	return baseIndex + nodeIndex;
}

HIPRT_HOST_DEVICE HIPRT_INLINE static bool isFatLeafNode( uint32_t nodeIndex ) { return nodeIndex & FatLeafBit; }

HIPRT_HOST_DEVICE HIPRT_INLINE static bool isLeafNode( uint32_t nodeIndex )
{
	return getNodeType( nodeIndex ) != BoxType && nodeIndex != InvalidValue;
}

HIPRT_HOST_DEVICE HIPRT_INLINE static bool isInternalNode( uint32_t nodeIndex ) { return getNodeType( nodeIndex ) == BoxType; }

HIPRT_HOST_DEVICE HIPRT_INLINE float ulp( float x ) { return fabs( x - as_float( as_uint( x ) ^ 1 ) ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 subDown( float3 a, float3 b )
{
	float3 d = ( a - b );
	return d - float3{ ulp( d.x ), ulp( d.y ), ulp( d.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 subUp( float3 a, float3 b )
{
	float3 d = ( a - b );
	return d + float3{ ulp( d.x ), ulp( d.y ), ulp( d.z ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE bool copyInvTransformMatrix( const Frame& frame, float ( &matrix )[3][4] )
{
	const MatrixFrame invMatrixFrame = MatrixFrame::getMatrixFrameInv( frame );
	memcpy( &matrix[0][0], &invMatrixFrame.m_matrix[0][0], sizeof( float ) * 12 );
	return frame.identity();
}

// 128B
struct alignas( DefaultAlignment ) Box4Node
{
	static constexpr uint32_t BranchingFactor = 4;

	HIPRT_HOST_DEVICE void initBox(
		const uint32_t					i,
		const uint32_t					childCount,
		const uint32_t					childIndex,
		const Aabb						childBox,
		[[maybe_unused]] const Aabb		nodeBox,
		[[maybe_unused]] const uint32_t childRanges = InvalidValue,
		[[maybe_unused]] const uint32_t matrixId	= NoRotationIndex )
	{
		if ( i < childCount )
		{
			( &m_childIndex0 )[i] = childIndex;
			( &m_box0 )[i]		  = childBox;
		}
		else
		{
			( &m_childIndex0 )[i] = InvalidValue;
			( &m_box0 )[i]		  = Aabb();
		}
	}

	HIPRT_HOST_DEVICE void initBoxes(
		const uint32_t*					childIndices,
		const Aabb*						childBoxes,
		[[maybe_unused]] const uint32_t childRanges = InvalidValue,
		[[maybe_unused]] const uint32_t matrixId	= NoRotationIndex )
	{
		// at least one child is valid
		m_childIndex0 = childIndices[0];
		m_box0		  = childBoxes[0];

		for ( uint32_t i = 1; i < BranchingFactor; ++i )
		{
			if ( i < getChildCount() )
			{
				( &m_childIndex0 )[i] = childIndices[i];
				( &m_box0 )[i]		  = childBoxes[i];
			}
			else
			{
				( &m_childIndex0 )[i] = InvalidValue;
				( &m_box0 )[i]		  = Aabb();
			}
		}
	}

	HIPRT_HOST_DEVICE void init(
		const uint32_t					i,
		const uint32_t					parentAddr,
		const uint32_t					childCount,
		[[maybe_unused]] const uint32_t boxNodeBase,
		[[maybe_unused]] const uint32_t primNodeBase,
		const uint32_t					childIndex,
		const Aabb						childBox,
		[[maybe_unused]] const Aabb		nodeBox,
		const uint32_t					childRange = InvalidValue,
		const uint32_t					matrixId   = NoRotationIndex )
	{
		m_parentAddr	= parentAddr;
		m_updateCounter = 0;
		m_childCount	= childCount;
		initBox( i, childCount, childIndex, childBox, nodeBox, childRange, matrixId );
	}

	HIPRT_HOST_DEVICE void init(
		const uint32_t					parentAddr,
		const uint32_t					childCount,
		[[maybe_unused]] const uint32_t boxNodeBase,
		[[maybe_unused]] const uint32_t primNodeBase,
		const uint32_t*					childIndices,
		const Aabb*						childBoxes,
		const uint32_t					childRanges = InvalidValue,
		const uint32_t					matrixId	= NoRotationIndex )
	{
		m_parentAddr	= parentAddr;
		m_updateCounter = 0;
		m_childCount	= childCount;
		initBoxes( childIndices, childBoxes, childRanges, matrixId );
	}

	HIPRT_HOST_DEVICE void setBoxNodeBase( [[maybe_unused]] const uint32_t boxNodeBase ) {}

	HIPRT_HOST_DEVICE void setPrimNodeBase( [[maybe_unused]] const uint32_t primNodeBase ) {}

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

	HIPRT_HOST_DEVICE uint32_t getParentAddr() const { return m_parentAddr; }

	HIPRT_HOST_DEVICE uint32_t getChildCount() const { return m_childCount; }

	HIPRT_HOST_DEVICE uint32_t getChildIndex( uint32_t i ) const { return ( &m_childIndex0 )[i]; }

	HIPRT_HOST_DEVICE uint32_t getChildType( uint32_t i ) const { return getNodeType( getChildIndex( i ) ); }

	HIPRT_HOST_DEVICE uint32_t getChildAddr( uint32_t i ) const { return getNodeAddr( getChildIndex( i ) ); }

	HIPRT_HOST_DEVICE uint32_t getChildRange( uint32_t i ) const { return 1; }

	HIPRT_HOST_DEVICE Aabb getChildBox( uint32_t i ) const { return ( &m_box0 )[i]; }

	HIPRT_HOST_DEVICE void patchChild( uint32_t i, uint32_t childIndex, [[maybe_unused]] uint32_t childRange )
	{
		( &m_childIndex0 )[i] = childIndex;
	}

	uint32_t m_childIndex0 = InvalidValue;
	uint32_t m_childIndex1 = InvalidValue;
	uint32_t m_childIndex2 = InvalidValue;
	uint32_t m_childIndex3 = InvalidValue;
	Aabb	 m_box0;
	Aabb	 m_box1;
	Aabb	 m_box2;
	Aabb	 m_box3;
	uint32_t m_parentAddr = InvalidValue;
	uint32_t m_updateCounter;
	uint32_t m_childCount;
};
HIPRT_STATIC_ASSERT( sizeof( Box4Node ) == 128 );

struct ChildInfo
{
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

template <typename Node>
HIPRT_HOST_DEVICE void initChildInfo(
	const uint32_t i,
	const uint32_t childCount,
	const uint32_t childIndex,
	const Aabb	   childBox,
	const Aabb	   nodeBox,
	const uint32_t childRange,
	Node&		   node )
{
	float3 extent = subUp( nodeBox.m_max, nodeBox.m_min );
	float3 origin = nodeBox.m_min;

	uint3 exponent;
	exponent.x = ( as_uint( extent.x ) + 0x7fffff ) >> 23;
	exponent.y = ( as_uint( extent.y ) + 0x7fffff ) >> 23;
	exponent.z = ( as_uint( extent.z ) + 0x7fffff ) >> 23;
	exponent.x = exponent.x == 0 ? 0 : max( 13u, exponent.x );
	exponent.y = exponent.y == 0 ? 0 : max( 13u, exponent.y );
	exponent.z = exponent.z == 0 ? 0 : max( 13u, exponent.z );

	node.m_origin	 = origin;
	node.m_xExponent = exponent.x;
	node.m_yExponent = exponent.y;
	node.m_zExponent = exponent.z;

	float3 rcp_exponent;
	rcp_exponent.x = as_float( ( 254 - exponent.x + 12 ) << 23 );
	rcp_exponent.y = as_float( ( 254 - exponent.y + 12 ) << 23 );
	rcp_exponent.z = as_float( ( 254 - exponent.z + 12 ) << 23 );

	ChildInfo childInfo{};
	if ( i < childCount && childBox.valid() )
	{
		float3 qmin = subDown( childBox.m_min, origin ) * rcp_exponent;
		qmin.x		= floor( qmin.x );
		qmin.y		= floor( qmin.y );
		qmin.z		= floor( qmin.z );
		qmin		= clamp( qmin, 0.0f, 4095.0f );

		float3 qmax = subUp( childBox.m_max, origin ) * rcp_exponent;
		qmax.x		= ceil( qmax.x );
		qmax.y		= ceil( qmax.y );
		qmax.z		= ceil( qmax.z );
		qmax		= clamp( qmax, 1.0f, 4096.0f ) - make_float3( 1.0f );

		childInfo.m_minX = static_cast<uint32_t>( qmin.x );
		childInfo.m_minY = static_cast<uint32_t>( qmin.y );
		childInfo.m_minZ = static_cast<uint32_t>( qmin.z );
		childInfo.m_maxX = static_cast<uint32_t>( qmax.x );
		childInfo.m_maxY = static_cast<uint32_t>( qmax.y );
		childInfo.m_maxZ = static_cast<uint32_t>( qmax.z );

		childInfo.m_nodeType	 = getNodeType( childIndex );
		childInfo.m_nodeRange	 = childRange == InvalidValue ? 1 : childRange & 15;
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
HIPRT_HOST_DEVICE void
initChildInfos( const uint32_t* childIndices, const Aabb* childBoxes, const uint32_t childRanges, Node& node )
{
	Aabb nodeBox;
	for ( uint32_t i = 0; i < Node::BranchingFactor; ++i )
		nodeBox.grow( childBoxes[i] );

	float3 extent = subUp( nodeBox.m_max, nodeBox.m_min );
	float3 origin = nodeBox.m_min;

	uint3 exponent;
	exponent.x = ( as_uint( extent.x ) + 0x7fffff ) >> 23;
	exponent.y = ( as_uint( extent.y ) + 0x7fffff ) >> 23;
	exponent.z = ( as_uint( extent.z ) + 0x7fffff ) >> 23;
	exponent.x = exponent.x == 0 ? 0 : max( 13u, exponent.x );
	exponent.y = exponent.y == 0 ? 0 : max( 13u, exponent.y );
	exponent.z = exponent.z == 0 ? 0 : max( 13u, exponent.z );

	node.m_origin	 = origin;
	node.m_xExponent = exponent.x;
	node.m_yExponent = exponent.y;
	node.m_zExponent = exponent.z;

	float3 rcp_exponent;
	rcp_exponent.x = as_float( ( 254 - exponent.x + 12 ) << 23 );
	rcp_exponent.y = as_float( ( 254 - exponent.y + 12 ) << 23 );
	rcp_exponent.z = as_float( ( 254 - exponent.z + 12 ) << 23 );

	for ( uint32_t i = 0; i < Node::BranchingFactor; ++i )
	{
		const Aabb& childBox = childBoxes[i];

		ChildInfo childInfo{};
		if ( i < node.getChildCount() && childBox.valid() )
		{
			float3 qmin = subDown( childBox.m_min, origin ) * rcp_exponent;
			qmin.x		= floor( qmin.x );
			qmin.y		= floor( qmin.y );
			qmin.z		= floor( qmin.z );
			qmin		= clamp( qmin, 0.0f, 4095.0f );

			float3 qmax = subUp( childBox.m_max, origin ) * rcp_exponent;
			qmax.x		= ceil( qmax.x );
			qmax.y		= ceil( qmax.y );
			qmax.z		= ceil( qmax.z );
			qmax		= clamp( qmax, 1.0f, 4096.0f ) - make_float3( 1.0f );

			childInfo.m_minX = static_cast<uint32_t>( qmin.x );
			childInfo.m_minY = static_cast<uint32_t>( qmin.y );
			childInfo.m_minZ = static_cast<uint32_t>( qmin.z );
			childInfo.m_maxX = static_cast<uint32_t>( qmax.x );
			childInfo.m_maxY = static_cast<uint32_t>( qmax.y );
			childInfo.m_maxZ = static_cast<uint32_t>( qmax.z );

			childInfo.m_nodeType	 = getNodeType( childIndices[i] );
			childInfo.m_nodeRange	 = childRanges == InvalidValue ? 1 : ( childRanges >> ( 4 * i ) ) & 15;
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

struct alignas( DefaultAlignment ) Box8Node
{
	static constexpr uint32_t BranchingFactor = 8;

	HIPRT_HOST_DEVICE void initBox(
		const uint32_t i,
		const uint32_t childCount,
		const uint32_t childIndex,
		const Aabb	   childBox,
		const Aabb	   nodeBox,
		const uint32_t childRange = InvalidValue,
		const uint32_t matrixId	  = NoRotationIndex )
	{
		m_matrixId = matrixId;
		initChildInfo( i, childCount, childIndex, childBox, nodeBox, childRange, *this );
	}

	HIPRT_HOST_DEVICE void initBoxes(
		const uint32_t* childIndices,
		const Aabb*		childBoxes,
		const uint32_t	childRanges = InvalidValue,
		const uint32_t	matrixId	= NoRotationIndex )
	{
		m_matrixId = matrixId;
		initChildInfos( childIndices, childBoxes, childRanges, *this );
	}

	HIPRT_HOST_DEVICE void init(
		const uint32_t i,
		const uint32_t parentAddr,
		const uint32_t childCount,
		const uint32_t boxNodeBase,
		const uint32_t primNodeBase,
		const uint32_t childIndex,
		const Aabb	   childBox,
		const Aabb	   nodeBox,
		const uint32_t childRange = InvalidValue,
		const uint32_t matrixId	  = NoRotationIndex )
	{
		m_boxNodeBase		 = boxNodeBase << 4;
		m_primNodeBase		 = primNodeBase << 4;
		m_parentAddr		 = parentAddr;
		m_childCountMinusOne = childCount - 1;

		initBox( i, childCount, childIndex, childBox, nodeBox, childRange, matrixId );
	}

	HIPRT_HOST_DEVICE void init(
		const uint32_t	parentAddr,
		const uint32_t	childCount,
		const uint32_t	boxNodeBase,
		const uint32_t	primNodeBase,
		const uint32_t* childIndices,
		const Aabb*		childBoxes,
		const uint32_t	childRanges = InvalidValue,
		const uint32_t	matrixId	= NoRotationIndex )
	{
		m_boxNodeBase		 = boxNodeBase << 4;
		m_primNodeBase		 = primNodeBase << 4;
		m_parentAddr		 = parentAddr;
		m_childCountMinusOne = childCount - 1;

		initBoxes( childIndices, childBoxes, childRanges, matrixId );
	}

	HIPRT_HOST_DEVICE void setBoxNodeBase( const uint32_t boxNodeBase ) { m_boxNodeBase = boxNodeBase << 4; }

	HIPRT_HOST_DEVICE void setPrimNodeBase( const uint32_t primNodeBase ) { m_primNodeBase = primNodeBase << 4; }

	HIPRT_HOST_DEVICE Aabb aabb() const
	{
		Aabb box;
		for ( uint32_t i = 0; i < getChildCount(); ++i )
			box.grow( getChildBox( i ) );
		return box;
	}

	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	HIPRT_HOST_DEVICE uint32_t getParentAddr() const { return m_parentAddr; }

	HIPRT_HOST_DEVICE uint32_t getChildCount() const { return m_childCountMinusOne + 1; }

	HIPRT_HOST_DEVICE uint32_t getChildIndex( uint32_t i ) const
	{
		if ( i >= getChildCount() ) return InvalidValue;
		return encodeNodeIndex( getChildAddr( i ), getChildType( i ) );
	}

	HIPRT_HOST_DEVICE uint32_t getChildType( uint32_t i ) const { return m_childInfos[i].m_nodeType; }

	HIPRT_HOST_DEVICE uint32_t getChildAddr( uint32_t i ) const
	{
		uint32_t childType = getChildType( i );
		uint32_t childAddr = childType == BoxType ? m_boxNodeBase >> 4 : m_primNodeBase >> 4;
		for ( uint32_t j = 0; j < i; ++j )
			if ( ( getChildType( j ) == BoxType ) == ( childType == BoxType ) ) childAddr += m_childInfos[j].m_nodeRange;
		return childAddr;
	}

	HIPRT_HOST_DEVICE Aabb getChildBox( uint32_t i ) const
	{
		if ( i >= getChildCount() ) return Aabb();

		float3 rcp_exponent;
		rcp_exponent.x = as_float( ( 254 - m_xExponent + 12 ) << 23 );
		rcp_exponent.y = as_float( ( 254 - m_yExponent + 12 ) << 23 );
		rcp_exponent.z = as_float( ( 254 - m_zExponent + 12 ) << 23 );

		Aabb childBox;
		childBox.m_min.x = m_origin.x + m_childInfos[i].m_minX / rcp_exponent.x;
		childBox.m_min.y = m_origin.y + m_childInfos[i].m_minY / rcp_exponent.y;
		childBox.m_min.z = m_origin.z + m_childInfos[i].m_minZ / rcp_exponent.z;
		childBox.m_max.x = m_origin.x + ( m_xExponent != 0 ? ( m_childInfos[i].m_maxX + 1 ) / rcp_exponent.x : 0.0f );
		childBox.m_max.y = m_origin.y + ( m_yExponent != 0 ? ( m_childInfos[i].m_maxY + 1 ) / rcp_exponent.y : 0.0f );
		childBox.m_max.z = m_origin.z + ( m_zExponent != 0 ? ( m_childInfos[i].m_maxZ + 1 ) / rcp_exponent.z : 0.0f );

		return childBox;
	}

	HIPRT_HOST_DEVICE uint32_t getChildRange( uint32_t i ) const { return m_childInfos[i].m_nodeRange; }

	HIPRT_HOST_DEVICE void patchChild( uint32_t i, uint32_t childIndex, [[maybe_unused]] uint32_t childRange )
	{
		// TODO: do not use bit fields
		m_childInfos[i].m_nodeType	= getNodeType( childIndex );
		m_childInfos[i].m_nodeRange = childRange;
	}

	uint32_t m_boxNodeBase;
	uint32_t m_primNodeBase;
	uint32_t m_parentAddr;
	union
	{
		float3	 m_origin;
		uint32_t m_updateCounter;
	};
	uint8_t m_xExponent;
	uint8_t m_yExponent;
	uint8_t m_zExponent;
	uint8_t : 4;
	uint8_t	 m_childCountMinusOne : 4;
	uint32_t m_matrixId : 7;
	uint32_t : 25;
	ChildInfo m_childInfos[8] = {};
};
HIPRT_STATIC_ASSERT( sizeof( Box8Node ) == 128 );

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

struct TrianglePacketData
{
	TrianglePacketData() = default;

	HIPRT_HOST_DEVICE TrianglePacketData( const uint32_t primIndex0, const uint32_t primIndex1, const uint32_t vertCount )
	{
		tryAddTrianglePair( primIndex0, primIndex1, vertCount );
	}

	HIPRT_HOST_DEVICE bool
	tryAddTrianglePair( const uint32_t primIndex0, const uint32_t primIndex1, const uint32_t newVertCount )
	{
		if ( m_triPairCount == 0 )
		{
			m_primIndexAnchor = primIndex0;
			m_primIndexDiff	  = primIndex0 ^ primIndex1;
			m_triPairCount	  = 1;
			m_vertCount		  = newVertCount;
			return true;
		}
		else
		{
			if ( m_triPairCount + 1 >= MaxTrianglePairsPerTrianglePacket ||
				 m_vertCount + newVertCount >= MaxVerticesPerTrianglePacket )
				return false;

			uint32_t newPrimIndexDiff =
				m_primIndexDiff | ( m_primIndexAnchor ^ primIndex0 ) | ( m_primIndexAnchor ^ primIndex1 );
			uint32_t primAnchorSize	 = 32 - clz( m_primIndexAnchor );
			uint32_t primPayloadSize = 32 - clz( newPrimIndexDiff );

			uint32_t headerBits		= TriangleStructHeaderSize;
			uint32_t primIndexBits	= primAnchorSize + ( 2 * ( m_triPairCount + 1 ) - 1 ) * primPayloadSize;
			uint32_t vertexBits		= 96 * ( m_vertCount + newVertCount );
			uint32_t descriptorBits = TrianglePairDescriptorSize * ( m_triPairCount + 1 );
			if ( headerBits + vertexBits + descriptorBits + primIndexBits > 1024 ) return false;

			m_primIndexDiff = newPrimIndexDiff;
			m_vertCount += newVertCount;
			m_triPairCount++;

			return true;
		}
	}

	HIPRT_HOST_DEVICE TrianglePacketHeader buildHeader() const
	{
		TrianglePacketHeader hdr;
		hdr.m_triPairCount		   = m_triPairCount;
		hdr.m_vertCount			   = m_vertCount;
		hdr.m_primIndexAnchorSize  = 32 - clz( m_primIndexAnchor );
		hdr.m_primIndexPayloadSize = 32 - clz( m_primIndexDiff );

		const uint32_t pairDescSize			= hdr.m_triPairCount * TrianglePairDescriptorSize;
		const uint32_t primIndexPayloadSize = ( ( 2 * hdr.m_triPairCount ) - 1 ) * hdr.m_primIndexPayloadSize;
		hdr.m_indexSectionMidpoint			= 1024 - pairDescSize - primIndexPayloadSize - hdr.m_primIndexAnchorSize;

		hdr.m_payloadSizeX = 32;
		hdr.m_payloadSizeY = 32;
		hdr.m_payloadSizeZ = 32;
		hdr.m_vertexTzBits = 0;
		return hdr;
	}

#if defined( __KERNELCC__ )
	HIPRT_DEVICE TrianglePacketData shuffle( uint32_t index )
	{
		TrianglePacketData data{};
		data.m_primIndexAnchor = shfl( m_primIndexAnchor, index );
		data.m_primIndexDiff   = shfl( m_primIndexDiff, index );
		data.m_triPairCount	   = shfl( m_triPairCount, index );
		data.m_vertCount	   = shfl( m_vertCount, index );
		return data;
	}
#endif

	uint32_t m_primIndexAnchor;
	uint32_t m_primIndexDiff;
	uint32_t m_triPairCount = 0;
	uint32_t m_vertCount	= 0;
};

// 64B
struct alignas( DefaultAlignment ) TrianglePairNode
{
	HIPRT_HOST_DEVICE Aabb	aabb() const { return m_triPair.aabb(); }
	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	HIPRT_HOST_DEVICE uint32_t getPrimIndex( uint32_t triangleIndex ) const
	{
		return triangleIndex > 0 ? m_primIndex1 : m_primIndex0;
	}

	HIPRT_HOST_DEVICE float3 getNormal( uint32_t triangleIndex ) const
	{
		return m_triPair.fetchTriangle( triangleIndex ).normal( m_flags >> ( triangleIndex * 8 ) );
	}

	TrianglePair m_triPair;
	uint32_t	 padding;
	uint32_t	 m_primIndex0 = InvalidValue;
	uint32_t	 m_primIndex1 = InvalidValue;
	uint32_t	 m_flags;
};
HIPRT_STATIC_ASSERT( sizeof( TrianglePairNode ) == 64 );

// 128B
struct alignas( DefaultAlignment ) TrianglePacketNode
{
	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE void writeUnalignedBits( uint32_t position, uint32_t length, uint32_t data )
	{
		if ( length )
		{
			const uint32_t hiOfs  = ( position + length ) % 32;
			const uint32_t loWord = position / 32;
			const uint32_t hiWord = ( position + length ) / 32;

			[[maybe_unused]] const uint32_t loMask = ( length == 32 ? ~0u : ( 1u << length ) - 1 ) << ( position % 32 );
			const uint32_t					hiMask = ( 1u << hiOfs ) - 1;

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

	HIPRT_HOST_DEVICE uint32_t readUnalignedBits( uint32_t position, uint32_t length ) const
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
	HIPRT_HOST_DEVICE void writeHeader( const TrianglePacketHeader& hdr )
	{
		uint32_t dw0 = ( hdr.m_payloadSizeX - 1 );
		dw0 += ( ( hdr.m_payloadSizeY - 1 ) << 5 );
		dw0 += ( ( hdr.m_payloadSizeZ - 1 ) << 10 );
		dw0 += ( hdr.m_vertexTzBits << 15 );
		dw0 += ( ( hdr.m_vertCount - 1 ) << 20 );
		dw0 += ( hdr.m_triPairCount - 1 ) << 28;
		m_data[0] = dw0;

		uint32_t dw1 = hdr.m_primIndexAnchorSize;
		dw1 += hdr.m_primIndexPayloadSize << 5;
		dw1 += hdr.m_indexSectionMidpoint << 10;

		[[maybe_unused]] const uint32_t mask = ( 1u << ( TriangleStructHeaderSize - 32 ) ) - 1;
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

	HIPRT_HOST_DEVICE TrianglePacketHeader readHeader()
	{
		TrianglePacketHeader hdr{};

		hdr.m_payloadSizeX = readUnalignedBits( 0, 5 ) + 1;
		hdr.m_payloadSizeY = readUnalignedBits( 5, 5 ) + 1;
		hdr.m_payloadSizeZ = readUnalignedBits( 10, 5 ) + 1;
		hdr.m_vertexTzBits = readUnalignedBits( 15, 5 );

		hdr.m_vertCount	   = getVertexCount();
		hdr.m_triPairCount = getTrianglePairCount();

		hdr.m_primIndexAnchorSize  = getPrimIndexAnchorSize();
		hdr.m_primIndexPayloadSize = getPrimIndexPayloadSize();
		hdr.m_indexSectionMidpoint = getIndexSectionMidpoint();

		return hdr;
	}

	HIPRT_HOST_DEVICE uint32_t getVertexCount() const { return readUnalignedBits( 20, 4 ) + 1; }

	HIPRT_HOST_DEVICE uint32_t getTrianglePairCount() const { return readUnalignedBits( 28, 3 ) + 1; }

	HIPRT_HOST_DEVICE uint32_t getIndexSectionMidpoint() const { return readUnalignedBits( 32 + 10, 10 ); }

	HIPRT_HOST_DEVICE uint32_t getPrimIndexAnchorSize() const { return readUnalignedBits( 32 + 0, 5 ); }

	HIPRT_HOST_DEVICE uint32_t getPrimIndexPayloadSize() const { return readUnalignedBits( 32 + 5, 5 ); }

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE void
	writePrimIndex( uint32_t pairIndex, uint32_t triangleIndex, const TrianglePacketHeader& hdr, uint32_t primIndex )
	{
		const uint32_t flatTriIndex = 2 * pairIndex + triangleIndex;

		const uint32_t primIndexPayloadSize = hdr.m_primIndexPayloadSize;
		const uint32_t primIndexAnchorSize	= hdr.m_primIndexAnchorSize;
		const uint32_t primIndexAnchorPos	= hdr.m_indexSectionMidpoint;
		const uint32_t primIndexPayloadPos =
			primIndexAnchorPos + primIndexAnchorSize + ( flatTriIndex - 1 ) * primIndexPayloadSize;

		const uint32_t primIndexPos	 = ( flatTriIndex == 0 ) ? primIndexAnchorPos : primIndexPayloadPos;
		const uint32_t primIndexSize = ( flatTriIndex == 0 ) ? primIndexAnchorSize : primIndexPayloadSize;
		const uint32_t primIndexMask = ( 1 << primIndexSize ) - 1;

		writeUnalignedBits<Atomic, Clear>( primIndexPos, primIndexSize, primIndex & primIndexMask );
	}

	HIPRT_HOST_DEVICE uint32_t readPrimIndex( uint32_t pairIndex, uint32_t triangleIndex ) const
	{
		const uint32_t flatTriIndex = 2 * pairIndex + triangleIndex;

		const uint32_t primIndexPayloadSize = getPrimIndexPayloadSize();
		const uint32_t primIndexAnchorSize	= getPrimIndexAnchorSize();
		const uint32_t primIndexAnchorPos	= getIndexSectionMidpoint();
		const uint32_t primIndexPayloadPos =
			primIndexAnchorPos + primIndexAnchorSize + ( flatTriIndex - 1 ) * primIndexPayloadSize;

		const uint32_t primIndexAnchor = readUnalignedBits( primIndexAnchorPos, primIndexAnchorSize );
		if ( flatTriIndex == 0 ) return primIndexAnchor;

		const uint32_t primIndex	 = readUnalignedBits( primIndexPayloadPos, primIndexPayloadSize );
		const uint32_t primIndexMask = ( 1 << primIndexPayloadSize ) - 1;

		if ( primIndexPayloadSize >= primIndexAnchorSize )
			return primIndex;
		else
			return primIndex | ( primIndexAnchor & ~primIndexMask );
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE void writeDescriptor( uint32_t pairIndex, uint32_t descriptor )
	{
		const uint32_t position = 1024 - ( pairIndex + 1 ) * TrianglePairDescriptorSize;
		writeUnalignedBits<Atomic, Clear>( position, TrianglePairDescriptorSize, descriptor );
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE void
	writeDescriptor( uint32_t pairIndex, const uint3& triIndices0, const uint3& triIndices1, bool rangeEnd )
	{
		uint32_t descriptor = 0;
		descriptor |= rangeEnd ? 1 : 0;
		descriptor |= ( triIndices0.x + ( triIndices0.y << 4 ) + ( triIndices0.z << 8 ) ) << 17;
		descriptor |= ( triIndices1.x + ( triIndices1.y << 4 ) + ( triIndices1.z << 8 ) ) << 3;
		writeDescriptor<Atomic, Clear>( pairIndex, descriptor );
	}

	HIPRT_HOST_DEVICE uint4 readDescriptor( uint32_t pairIndex, uint32_t triangleIndex ) const
	{
		const uint32_t position	  = 1024 - ( pairIndex + 1 ) * TrianglePairDescriptorSize;
		const uint32_t descriptor = readUnalignedBits( position, TrianglePairDescriptorSize );
		const uint32_t triIndices = descriptor >> ( triangleIndex > 0 ? 3 : 17 );
		return uint4{ triIndices & 15, ( triIndices >> 4 ) & 15, ( triIndices >> 8 ) & 15, descriptor & 1 };
	}

	template <bool Atomic = false, bool Clear = false>
	HIPRT_HOST_DEVICE void writeVertex( uint32_t vertexIndex, const float3& vertex )
	{
		const uint32_t position = TriangleStructHeaderSize + 96 * vertexIndex;
		writeUnalignedBits<Atomic, Clear>( position + 0 * 32, 32, as_uint( vertex.x ) );
		writeUnalignedBits<Atomic, Clear>( position + 1 * 32, 32, as_uint( vertex.y ) );
		writeUnalignedBits<Atomic, Clear>( position + 2 * 32, 32, as_uint( vertex.z ) );
	}

	HIPRT_HOST_DEVICE float3 readVertex( uint32_t vertexIndex ) const
	{
		const uint32_t position = TriangleStructHeaderSize + 96 * vertexIndex;

		float3 vertex;
		vertex.x = as_float( readUnalignedBits( position + 0 * 32, 32 ) );
		vertex.y = as_float( readUnalignedBits( position + 1 * 32, 32 ) );
		vertex.z = as_float( readUnalignedBits( position + 2 * 32, 32 ) );
		return vertex;
	}

	HIPRT_HOST_DEVICE Triangle fetchTriangle( uint32_t pairIndex, uint32_t triangleIndex ) const
	{
		const uint3 triIndices = make_uint3( readDescriptor( pairIndex, triangleIndex ) );

		Triangle triangle;
		triangle.m_v0 = readVertex( triIndices.x );
		triangle.m_v1 = readVertex( triIndices.y );
		triangle.m_v2 = readVertex( triIndices.z );

		return triangle;
	}

	HIPRT_HOST_DEVICE uint32_t getPrimIndex( uint32_t pairIndex, uint32_t triangleIndex ) const
	{
		return readPrimIndex( pairIndex, triangleIndex );
	}

	HIPRT_HOST_DEVICE float3 getNormal( uint32_t pairIndex, uint32_t triangleIndex ) const
	{
		return fetchTriangle( pairIndex, triangleIndex ).normal();
	}

	HIPRT_HOST_DEVICE bool isRangeEnd( uint32_t pairIndex ) const { return readDescriptor( pairIndex, 0 ).w != 0; }

	HIPRT_HOST_DEVICE Obb obb( const uint32_t pairIndex, const uint32_t matrixIndex, const Aabb& box ) const
	{
		Obb obb( matrixIndex );

		uint32_t				  triPairIndex	= pairIndex;
		const TrianglePacketNode* triPacketNode = this;

		while ( true )
		{
			const Triangle& tri0 = triPacketNode->fetchTriangle( triPairIndex, 0 );
			tri0.crop( 0, box.m_min.x, box, obb );
			tri0.crop( 0, box.m_max.x, box, obb );
			tri0.crop( 1, box.m_min.y, box, obb );
			tri0.crop( 1, box.m_max.y, box, obb );
			tri0.crop( 2, box.m_min.z, box, obb );
			tri0.crop( 2, box.m_max.z, box, obb );

			if ( triPacketNode->getPrimIndex( triPairIndex, 0 ) != triPacketNode->getPrimIndex( triPairIndex, 1 ) )
			{
				const Triangle& tri1 = triPacketNode->fetchTriangle( triPairIndex, 1 );
				tri1.crop( 0, box.m_min.x, box, obb );
				tri1.crop( 0, box.m_max.x, box, obb );
				tri1.crop( 1, box.m_min.y, box, obb );
				tri1.crop( 1, box.m_max.y, box, obb );
				tri1.crop( 2, box.m_min.z, box, obb );
				tri1.crop( 2, box.m_max.z, box, obb );
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

	HIPRT_HOST_DEVICE Aabb aabb( const uint32_t pairIndex ) const
	{
		Aabb box;

		uint32_t				  triPairIndex	= pairIndex;
		const TrianglePacketNode* triPacketNode = this;

		while ( true )
		{
			const Triangle& tri0 = triPacketNode->fetchTriangle( triPairIndex, 0 );
			box.grow( tri0.m_v0 );
			box.grow( tri0.m_v1 );
			box.grow( tri0.m_v2 );

			if ( triPacketNode->getPrimIndex( triPairIndex, 0 ) != triPacketNode->getPrimIndex( triPairIndex, 1 ) )
			{
				const Triangle& tri1 = triPacketNode->fetchTriangle( triPairIndex, 1 );
				box.grow( tri1.m_v0 );
				box.grow( tri1.m_v1 );
				box.grow( tri1.m_v2 );
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

	HIPRT_HOST_DEVICE float area( const uint32_t pairIndex ) const { return aabb( pairIndex ).area(); }

	uint32_t m_data[32];
};
HIPRT_STATIC_ASSERT( sizeof( TrianglePacketNode ) == 128 );

struct TrianglePairData
{
	HIPRT_HOST_DEVICE TrianglePairData() {}

	HIPRT_HOST_DEVICE
	TrianglePairData( const uint2& pairIndices, const uint3& triIndices0, const uint3& triIndices1, const bool rangeEnd )
		: m_pairIndices( pairIndices )
	{
		m_descriptor = 0;
		m_descriptor |= rangeEnd ? 1 : 0;
		m_descriptor |= ( triIndices0.x + ( triIndices0.y << 4 ) + ( triIndices0.z << 8 ) ) << 17;
		m_descriptor |= ( triIndices1.x + ( triIndices1.y << 4 ) + ( triIndices1.z << 8 ) ) << 3;
	}

	uint2	 m_pairIndices;
	uint32_t m_descriptor;
};

struct TrianglePairOffsets
{
	HIPRT_HOST_DEVICE TrianglePairOffsets() {}

	HIPRT_HOST_DEVICE TrianglePairOffsets( const uint32_t pairOffset, const uint32_t packetOffset )
	{
		m_pairOffset   = pairOffset & 7;
		m_packetOffset = packetOffset & 31;
	}

	uint8_t m_pairOffset : 3;
	uint8_t m_packetOffset : 5;
};

struct alignas( DefaultAlignment ) TrianglePacketCache
{
	HIPRT_HOST_DEVICE TrianglePacketCache() {}
	TrianglePairData  m_triPairData[MaxTrianglePairsPerTrianglePacket];
	uint32_t		  m_vertexIndices[MaxVerticesPerTrianglePacket];
};
HIPRT_STATIC_ASSERT( sizeof( TrianglePacketCache ) >= sizeof( TrianglePacketNode ) );

// 8B
struct alignas( 4 ) CustomNode
{
	uint32_t m_primIndex = InvalidValue;
};
HIPRT_STATIC_ASSERT( sizeof( CustomNode ) == 4 );

struct InstanceNodeBase
{
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
};

// 64B
struct alignas( 64 ) UserInstanceNode : public InstanceNodeBase
{
	HIPRT_HOST_DEVICE void init(
		const uint32_t					 primIndex,
		const uint32_t					 mask,
		const Frame&					 frame,
		const hiprtInstance&			 instance,
		const hiprtTransformHeader&		 transform,
		[[maybe_unused]] const uint32_t	 childCount,
		[[maybe_unused]] const uint32_t* childIndices,
		[[maybe_unused]] const Aabb*	 childBoxes )
	{
		m_primIndex = primIndex;
		m_mask		= mask;
		m_type		= instance.type;
		m_static	= transform.frameCount == 1 ? 1 : 0;

		if ( instance.type == hiprtInstanceTypeScene )
			m_scene = reinterpret_cast<SceneHeader*>( instance.scene );
		else
			m_geometry = reinterpret_cast<GeomHeader*>( instance.geometry );

		if ( transform.frameCount == 1 )
		{
			m_identity = copyInvTransformMatrix( frame, m_matrix ) ? 1 : 0;
		}
		else
		{
			m_identity	= 0;
			m_transform = transform;
		}
	}

	uint32_t m_mask = FullRayMask;
	uint32_t m_primIndex : InstanceIDBits;
	uint32_t m_type : 1;
	uint32_t m_static : 1;
	uint32_t m_identity : 1;
	uint32_t : 5;
};
HIPRT_STATIC_ASSERT( sizeof( UserInstanceNode ) == 64 );

// 128B
struct alignas( 128 ) HwInstanceNode : public InstanceNodeBase
{
	static constexpr uint32_t BranchingFactor = 4;

	HIPRT_HOST_DEVICE void initBoxes( const uint32_t* childIndices, const Aabb* childBoxes )
	{
		initChildInfos( childIndices, childBoxes, InvalidValue, *this );
	}

	HIPRT_HOST_DEVICE void init(
		const uint32_t				primIndex,
		const uint32_t				mask,
		const Frame&				frame,
		const hiprtInstance&		instance,
		const hiprtTransformHeader& transform,
		const uint32_t				childCount,
		const uint32_t*				childIndices,
		const Aabb*					childBoxes,
		const uint32_t				childRanges = InvalidValue )
	{
		m_primIndex = primIndex;
		m_mask		= mask;
		m_hwMask	= 0xff;
		m_type		= instance.type;
		m_static	= transform.frameCount == 1 ? 1 : 0;

		if ( instance.type == hiprtInstanceTypeScene )
			m_scene = reinterpret_cast<SceneHeader*>( instance.scene );
		else
			m_geometry = reinterpret_cast<GeomHeader*>( instance.geometry );

		if ( transform.frameCount == 1 )
		{
			m_identity = copyInvTransformMatrix( frame, m_matrix ) ? 1 : 0;
		}
		else
		{
			m_identity	= 0;
			m_transform = transform;
		}

		m_disableBoxSort	 = 0;
		m_childCountMinusOne = childCount - 1;
		initBoxes( childIndices, childBoxes );
	}

	HIPRT_HOST_DEVICE uint32_t getChildCount() const { return m_childCountMinusOne + 1; }

	uint32_t m_mask = FullRayMask;
	uint32_t m_primIndex : InstanceIDBits;
	uint32_t m_hwMask : 8;

	float3	m_origin;
	uint8_t m_xExponent;
	uint8_t m_yExponent;
	uint8_t m_zExponent;

	uint8_t m_disableBoxSort : 1;
	uint8_t m_type : 1;
	uint8_t m_static : 1;
	uint8_t m_identity : 1;
	uint8_t m_childCountMinusOne : 2;
	uint8_t : 2;

	ChildInfo m_childInfos[BranchingFactor];
};
HIPRT_STATIC_ASSERT( sizeof( HwInstanceNode ) == 128 );

// 32B
struct alignas( 32 ) ScratchNode
{
	HIPRT_HOST_DEVICE Aabb aabb() const { return m_box; };

	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	HIPRT_HOST_DEVICE uint32_t getChildType( uint32_t i ) const { return getNodeType( ( &m_childIndex0 )[i] & ~FatLeafBit ); }

	HIPRT_HOST_DEVICE uint32_t getChildIndex( uint32_t i ) const { return ( &m_childIndex0 )[i] & ~FatLeafBit; }

	HIPRT_HOST_DEVICE uint32_t getChildAddr( uint32_t i ) const { return getNodeAddr( ( &m_childIndex0 )[i] & ~FatLeafBit ); }

	HIPRT_HOST_DEVICE void setChildFatLeafFlag( uint32_t i ) { ( &m_childIndex0 )[i] |= FatLeafBit; }

	HIPRT_HOST_DEVICE uint32_t operator[]( uint32_t i ) { return ( &m_childIndex0 )[i]; }

	Aabb	 m_box;
	uint32_t m_childIndex0;
	uint32_t m_childIndex1;
};
HIPRT_STATIC_ASSERT( sizeof( ScratchNode ) == 32 );

// 64B
struct alignas( 64 ) ApiNode
{
	HIPRT_HOST_DEVICE Aabb aabb() const { return m_box; }

	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	HIPRT_HOST_DEVICE uint32_t getChildType( uint32_t i ) const { return m_childTypes[i] & ~FatLeafBit; }

	HIPRT_HOST_DEVICE uint32_t getChildIndex( uint32_t i ) const
	{
		return encodeNodeIndex( m_childAddrs[i], m_childTypes[i] & ~FatLeafBit );
	}

	HIPRT_HOST_DEVICE uint32_t getChildAddr( uint32_t i ) const { return m_childAddrs[i]; }

	HIPRT_HOST_DEVICE void setChildFatLeafFlag( uint32_t i ) { m_childTypes[i] |= FatLeafBit; }

	HIPRT_HOST_DEVICE uint32_t operator[]( uint32_t i ) { return getChildIndex( i ) | ( m_childTypes[i] & FatLeafBit ); }

	Aabb	 m_box;
	uint32_t m_childAddrs[2];
	uint32_t m_childTypes[2];
};
HIPRT_STATIC_ASSERT( sizeof( ApiNode ) == sizeof( hiprtInternalNode ) );
HIPRT_STATIC_ASSERT( alignof( ApiNode ) == alignof( hiprtInternalNode ) );

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
HIPRT_STATIC_ASSERT( sizeof( ReferenceNode ) == sizeof( hiprtLeafNode ) );
HIPRT_STATIC_ASSERT( alignof( ReferenceNode ) == alignof( hiprtLeafNode ) );

#if defined( __KERNELCC__ )
#if HIPRT_RTIP >= 31
using TriangleNode = TrianglePacketNode;
using BoxNode	   = Box8Node;
using InstanceNode = HwInstanceNode;
#else
using TriangleNode = TrianglePairNode;
using BoxNode	   = Box4Node;
using InstanceNode = UserInstanceNode;
#endif
#endif
} // namespace hiprt
