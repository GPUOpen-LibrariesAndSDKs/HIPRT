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
#include <hiprt/impl/BvhConfig.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/Instance.h>

namespace hiprt
{
template <typename PrimitiveNode>
HIPRT_INLINE HIPRT_HOST_DEVICE constexpr bool isTrianglePairNode()
{
	return is_same_v<PrimitiveNode, rtip11::TrianglePairNode>;
}

template <typename PrimitiveNode>
HIPRT_INLINE HIPRT_HOST_DEVICE constexpr bool isTrianglePacketNode()
{
	return is_same_v<PrimitiveNode, rtip31::TrianglePacketNode>;
}

template <typename PrimitiveNode>
HIPRT_INLINE HIPRT_HOST_DEVICE constexpr bool isTriangleNode()
{
	return isTrianglePacketNode<PrimitiveNode>() || isTrianglePairNode<PrimitiveNode>();
}

template <typename PrimitiveNode>
HIPRT_INLINE HIPRT_HOST_DEVICE constexpr bool isInstanceNode()
{
	return is_same_v<PrimitiveNode, rtip11::InstanceNode> || is_same_v<PrimitiveNode, rtip31::InstanceNode>;
}

HIPRT_INLINE HIPRT_HOST_DEVICE uint32_t getBranchingFactor( const uint32_t rtip )
{
	if ( rtip >= 31 )
		return rtip31::BoxNode::BranchingFactor;
	else
		return rtip11::BoxNode::BranchingFactor;
}

HIPRT_INLINE HIPRT_HOST_DEVICE constexpr uint32_t getMaxBoxNodeSize()
{
	return sizeof( rtip31::BoxNode ) > sizeof( rtip11::BoxNode ) ? sizeof( rtip31::BoxNode ) : sizeof( rtip11::BoxNode );
}

HIPRT_INLINE HIPRT_HOST_DEVICE constexpr uint32_t getMaxBoxNodeAlignment()
{
	return alignof( rtip31::BoxNode ) > alignof( rtip11::BoxNode ) ? alignof( rtip31::BoxNode ) : alignof( rtip11::BoxNode );
}

HIPRT_INLINE HIPRT_HOST_DEVICE constexpr uint32_t getMaxFatLeafSize( const uint32_t rtip )
{
	if ( rtip >= 31 ) return rtip31::TrianglePacketNode::MaxFatLeafSize;
	return 1;
}

HIPRT_INLINE HIPRT_HOST_DEVICE Aabb getBoxNodeAabb( const uint32_t rtip, const void* boxNode )
{
	if ( rtip >= 31 )
		return reinterpret_cast<const rtip31::BoxNode*>( boxNode )->aabb();
	else
		return reinterpret_cast<const rtip11::BoxNode*>( boxNode )->aabb();
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t getMaxTrianglePacketNodeCount( const uint32_t rtip, const size_t count )
{
	if ( rtip >= 31 ) return 2 * DivideRoundUp( count, rtip31::TrianglePacketNode::MinTrianglePairs + 1 );
	return count;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t
getMaxBoxNodeCount( const size_t count, const uint32_t branchingFactor, const uint32_t maxFatLeafSize )
{
	const size_t maxLeafNodes     = DivideRoundUp( count, maxFatLeafSize + 1 );
	const size_t maxInternalNodes = 1 + DivideRoundUp( maxLeafNodes, branchingFactor - 1 );
	// We add 1 to compensate for the fact that the root node can have a limited branching factor for instances than other
	// nodes.
	return maxLeafNodes + maxInternalNodes + 1;
}

template <typename PrimitiveNode>
[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t getMaxPrimNodeCount( const uint32_t rtip, const size_t count )
{
	size_t primNodeCount = count;
	if constexpr ( is_same_v<PrimitiveNode, rtip31::TrianglePacketNode> )
		primNodeCount = getMaxTrianglePacketNodeCount( rtip, count );
	return primNodeCount;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t
getMaxPrimNodeCount( const hiprtGeometryBuildInput& buildInput, const uint32_t rtip, const size_t count )
{
	size_t primNodeCount = count;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh && rtip >= 31 )
		primNodeCount = getMaxTrianglePacketNodeCount( rtip, count );
	return primNodeCount;
}

template <typename BoxNode, typename PrimitiveNode>
[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t getMaxBoxNodeCount( const size_t count )
{
	const uint32_t maxFatLeafSize = is_same_v<PrimitiveNode, rtip31::TrianglePacketNode> ? PrimitiveNode::MaxFatLeafSize : 1;
	return getMaxBoxNodeCount( count, BoxNode::BranchingFactor, maxFatLeafSize );
}

template <typename BuildInput>
[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t
getMaxBoxNodeCount( const BuildInput& buildInput, const uint32_t rtip, const uint32_t branchingFactor, const size_t count )
{
	uint32_t maxFatLeafSize = 1;
	if constexpr ( is_same_v<BuildInput, hiprtGeometryBuildInput> )
		if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh && rtip >= 31 ) maxFatLeafSize = getMaxFatLeafSize( rtip );
	return getMaxBoxNodeCount( count, branchingFactor, maxFatLeafSize );
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t getPrimCount( const hiprtGeometryBuildInput& buildInput )
{
	size_t primCount{};
	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		primCount = buildInput.primitive.triangleMesh.triangleCount;
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		primCount = buildInput.primitive.aabbList.aabbCount;
		break;
	}
#if !defined( __KERNELCC__ )
	default:
		throw std::runtime_error( "Not supported" );
#endif
	}
	return primCount;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t
getPrimNodeSize( const hiprtGeometryBuildInput& buildInput, const size_t triangleNodeSize )
{
	size_t nodeSize{};
	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		nodeSize = triangleNodeSize;
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		nodeSize = sizeof( CustomNode );
		break;
	}
#if !defined( __KERNELCC__ )
	default:
		throw std::runtime_error( "Not supported" );
#endif
	}
	return nodeSize;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t getGeometryStorageBufferSize(
	const size_t primNodeCount, const size_t boxNodeCount, const size_t primNodeSize, const size_t boxNodeSize )
{
	return RoundUp( sizeof( GeomHeader ), DefaultAlignment ) + RoundUp( primNodeCount * primNodeSize, DefaultAlignment ) +
	       RoundUp( boxNodeCount * boxNodeSize, DefaultAlignment );
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE size_t getSceneStorageBufferSize(
	const size_t primCount,
	const size_t primNodeCount,
	const size_t boxNodeCount,
	const size_t primNodeSize,
	const size_t boxNodeSize,
	const size_t frameCount )
{
	return RoundUp( sizeof( SceneHeader ), DefaultAlignment ) + RoundUp( boxNodeCount * boxNodeSize, DefaultAlignment ) +
	       RoundUp( primNodeCount * primNodeSize, DefaultAlignment ) +
	       RoundUp( primCount * sizeof( Instance ), DefaultAlignment ) +
	       RoundUp( frameCount * sizeof( Frame ), DefaultAlignment );
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE bool
batchBuild( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return getPrimCount( buildInput ) <= buildOptions.batchBuildMaxPrimCount &&
	       ( buildOptions.buildFlags & 7 ) != hiprtBuildFlagBitCustomBvhImport;
}

[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE bool
batchBuild( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return buildInput.instanceCount <= buildOptions.batchBuildMaxPrimCount &&
	       ( buildOptions.buildFlags & 7 ) != hiprtBuildFlagBitCustomBvhImport;
}
} // namespace hiprt
