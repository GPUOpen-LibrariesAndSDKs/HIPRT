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
HIPRT_INLINE HIPRT_HOST_DEVICE size_t getMaxTrianglePacketNodeCount( const size_t count )
{
	return 2 * DivideRoundUp( count, MinTrianglePairsPerPacket + 1 );
}

HIPRT_INLINE HIPRT_HOST_DEVICE size_t
getMaxBoxNodeCount( const size_t count, const uint32_t branchingFactor, const uint32_t maxFatLeafSize )
{
	const size_t maxLeafNodes	  = DivideRoundUp( count, maxFatLeafSize + 1 );
	const size_t maxInternalNodes = 1 + DivideRoundUp( maxLeafNodes, branchingFactor - 1 );
	return maxLeafNodes + maxInternalNodes;
}

template <typename PrimitiveNode>
HIPRT_INLINE HIPRT_HOST_DEVICE size_t getMaxPrimNodeCount( const size_t count )
{
	size_t primNodeCount = count;
	if constexpr ( is_same<PrimitiveNode, TrianglePacketNode>::value ) primNodeCount = getMaxTrianglePacketNodeCount( count );
	return primNodeCount;
}

HIPRT_INLINE HIPRT_HOST_DEVICE size_t
getMaxPrimNodeCount( const hiprtGeometryBuildInput& buildInput, const uint32_t rtip, const size_t count )
{
	size_t primNodeCount = count;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh && rtip >= 31 )
		primNodeCount = getMaxTrianglePacketNodeCount( count );
	return primNodeCount;
}

template <typename BoxNode, typename PrimitiveNode>
HIPRT_INLINE HIPRT_HOST_DEVICE size_t getMaxBoxNodeCount( const size_t count )
{
	const uint32_t branchingFactor = BoxNode::BranchingFactor;
	if ( count <= branchingFactor ) return 1;

	uint32_t maxFatLeafSize = 1;
	if constexpr ( is_same<PrimitiveNode, TrianglePacketNode>::value ) maxFatLeafSize = MaxFatLeafSize;

	return getMaxBoxNodeCount( count, branchingFactor, maxFatLeafSize );
}

template <typename BuildInput>
HIPRT_INLINE HIPRT_HOST_DEVICE size_t
getMaxBoxNodeCount( const BuildInput& buildInput, const uint32_t rtip, const size_t count )
{
	const uint32_t branchingFactor = rtip >= 31 ? 8 : 4;
	if ( count <= branchingFactor ) return 1;

	uint32_t maxFatLeafSize = 1;
	if constexpr ( is_same<BuildInput, hiprtGeometryBuildInput>::value )
		if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh && rtip >= 31 ) maxFatLeafSize = MaxFatLeafSize;

	return getMaxBoxNodeCount( count, branchingFactor, maxFatLeafSize );
}

HIPRT_INLINE HIPRT_HOST_DEVICE size_t getPrimCount( const hiprtGeometryBuildInput& buildInput )
{
	size_t primCount{};
	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		primCount = buildInput.primitive.triangleMesh.triangleCount;
		if ( buildInput.primitive.triangleMesh.trianglePairCount > 0 )
			primCount = buildInput.primitive.triangleMesh.trianglePairCount;
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

HIPRT_INLINE HIPRT_HOST_DEVICE size_t
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

HIPRT_INLINE HIPRT_HOST_DEVICE size_t getGeometryStorageBufferSize(
	const size_t primNodeCount, const size_t boxNodeCount, const size_t primNodeSize, const size_t boxNodeSize )
{
	return RoundUp( sizeof( GeomHeader ), DefaultAlignment ) + RoundUp( primNodeCount * primNodeSize, DefaultAlignment ) +
		   RoundUp( boxNodeCount * boxNodeSize, DefaultAlignment );
}

HIPRT_INLINE HIPRT_HOST_DEVICE size_t getSceneStorageBufferSize(
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

HIPRT_INLINE HIPRT_HOST_DEVICE bool
batchBuild( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return getPrimCount( buildInput ) <= buildOptions.batchBuildMaxPrimCount &&
		   ( buildOptions.buildFlags & 7 ) != hiprtBuildFlagBitCustomBvhImport;
}

HIPRT_INLINE HIPRT_HOST_DEVICE bool batchBuild( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return buildInput.instanceCount <= buildOptions.batchBuildMaxPrimCount &&
		   ( buildOptions.buildFlags & 7 ) != hiprtBuildFlagBitCustomBvhImport;
}
} // namespace hiprt
