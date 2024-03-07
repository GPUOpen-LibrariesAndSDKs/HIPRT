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
#include <hiprt/hiprt_vec.h>
#include <hiprt/impl/Math.h>
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/AabbList.h>
#include <hiprt/impl/Triangle.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/BvhBuilderUtil.h>
#include <hiprt/impl/Geometry.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/MortonCode.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/TriangleMesh.h>
#include <hiprt/impl/BvhConfig.h>
using namespace hiprt;

template <typename PrimitiveContainer, typename PrimitiveNode>
__device__ void SetupClusters(
	PrimitiveContainer& primitives,
	PrimitiveNode*		primNodes,
	ReferenceNode*		references,
	const uint32_t*		sortedMortonCodeValues,
	uint32_t*			nodeIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if ( index >= primitives.getCount() ) return;

	uint32_t leafType;
	if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
		leafType = TriangleType;
	else if constexpr ( is_same<PrimitiveNode, CustomNode>::value )
		leafType = CustomType;
	else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
		leafType = InstanceType;

	uint32_t primIndex = sortedMortonCodeValues[index];
	references[index]  = ReferenceNode( primIndex, primitives.fetchAabb( primIndex ) );
	nodeIndices[index] = encodeNodeIndex( index, leafType );

	if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
		primNodes[primIndex] = primitives.fetchTriangleNode( primIndex );
}

__device__ __forceinline__ uint32_t encodeOffset( const int32_t threadIndex, const int32_t neighbourIndex )
{
	const int32_t  sOffset = neighbourIndex - threadIndex;
	const uint32_t uOffset = abs( sOffset ) - 1;
	return ( uOffset << 1 ) | ( sOffset < 0 ? ( threadIndex & 1 ) ^ 1 : threadIndex & 1 );
}

__device__ __forceinline__ int32_t decodeOffset( const int32_t threadIndex, const uint32_t offset )
{
	const uint32_t off = ( offset >> 1 ) + 1;
	return threadIndex + ( ( offset ^ threadIndex ) & 1 ? -static_cast<int32_t>( off ) : static_cast<int32_t>( off ) );
}

extern "C" __global__ void SetupClusters_TriangleMesh_TriangleNode(
	TriangleMesh	primitives,
	TriangleNode*	primNodes,
	ReferenceNode*	references,
	const uint32_t* sortedMortonCodeValues,
	uint32_t*		nodeIndices )
{
	SetupClusters<TriangleMesh, TriangleNode>( primitives, primNodes, references, sortedMortonCodeValues, nodeIndices );
}

extern "C" __global__ void SetupClusters_AabbList_CustomNode(
	AabbList		primitives,
	CustomNode*		primNodes,
	ReferenceNode*	references,
	const uint32_t* sortedMortonCodeValues,
	uint32_t*		nodeIndices )
{
	SetupClusters<AabbList, CustomNode>( primitives, primNodes, references, sortedMortonCodeValues, nodeIndices );
}

extern "C" __global__ void SetupClusters_InstanceList_SRTFrame_InstanceNode(
	InstanceList<SRTFrame> primitives,
	InstanceNode*		   primNodes,
	ReferenceNode*		   references,
	const uint32_t*		   sortedMortonCodeValues,
	uint32_t*			   nodeIndices )
{
	SetupClusters<InstanceList<SRTFrame>, InstanceNode>(
		primitives, primNodes, references, sortedMortonCodeValues, nodeIndices );
}

extern "C" __global__ void SetupClusters_InstanceList_MatrixFrame_InstanceNode(
	InstanceList<MatrixFrame> primitives,
	InstanceNode*			  primNodes,
	ReferenceNode*			  references,
	const uint32_t*			  sortedMortonCodeValues,
	uint32_t*				  nodeIndices )
{
	SetupClusters<InstanceList<MatrixFrame>, InstanceNode>(
		primitives, primNodes, references, sortedMortonCodeValues, nodeIndices );
}

extern "C" __global__ void
BlockPloc( uint32_t numberOfClusters, uint32_t* nodeIndices, ScratchNode* scratchNodes, ReferenceNode* references )
{
	const uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if ( blockIdx.x > 0 ) return;
	if ( index > PlocMainBlockSize ) return;

	alignas( alignof( Aabb ) ) __shared__ uint8_t boxesCache[sizeof( Aabb ) * PlocMainBlockSize];
	__shared__ uint32_t							  distanceOffsetsBlock[PlocMainBlockSize];
	__shared__ uint32_t							  nodeIndicesBlock[PlocMainBlockSize];
	__shared__ uint32_t							  numberOfClustersBlock;
	__shared__ uint32_t							  nodeCounter;

	Aabb* boxesBlock = reinterpret_cast<Aabb*>( boxesCache );

	if ( index >= 0 && index < numberOfClusters )
	{
		uint32_t nodeIndex		= nodeIndices[index];
		Aabb	 box			= isLeafNode( nodeIndex ) ? references[getNodeAddr( nodeIndex )].aabb()
														  : scratchNodes[getNodeAddr( nodeIndex )].aabb();
		boxesBlock[index]		= box;
		nodeIndicesBlock[index] = nodeIndex;
	}
	else
	{
		boxesBlock[index]		= Aabb( make_float3( -FltMax ), make_float3( FltMax ) );
		nodeIndicesBlock[index] = InvalidValue;
	}
	__syncthreads();

	constexpr uint32_t OffsetMask = ( ( 1u << ( Log2( PlocRadius ) + 1 ) ) - 1 );

	while ( numberOfClusters > 1 )
	{
		uint32_t nodeIndex = nodeIndicesBlock[index];

		distanceOffsetsBlock[index] = InvalidValue;
		__syncthreads();

		uint32_t minDistanceOffset = InvalidValue;
		Aabb	 box			   = boxesBlock[index];

		for ( int32_t neighbourIndex = index + 1; neighbourIndex < min( numberOfClusters, index + PlocRadius + 1 );
			  ++neighbourIndex )
		{
			Aabb neighbourBox = boxesBlock[neighbourIndex];
			neighbourBox.grow( box );
			uint32_t distance = ( ( __float_as_uint( neighbourBox.area() ) << 1 ) & ~OffsetMask );

			const uint32_t offset0		   = encodeOffset( index, neighbourIndex );
			const uint32_t distanceOffset0 = distance | offset0;
			minDistanceOffset			   = min( minDistanceOffset, distanceOffset0 );

			const uint32_t offset1		   = encodeOffset( neighbourIndex, index );
			const uint32_t distanceOffset1 = distance | offset1;
			atomicMin( &distanceOffsetsBlock[neighbourIndex], distanceOffset1 );
		}

		atomicMin( &distanceOffsetsBlock[index], minDistanceOffset );

		if ( index == 0 ) nodeCounter = 0;
		__syncthreads();

		if ( index < numberOfClusters )
		{
			int32_t neighbourIndex			= decodeOffset( index, distanceOffsetsBlock[index] & OffsetMask );
			int32_t neighbourNeighbourIndex = decodeOffset( neighbourIndex, distanceOffsetsBlock[neighbourIndex] & OffsetMask );

			uint32_t leftChildIndex	 = nodeIndicesBlock[index];
			uint32_t rightChildIndex = nodeIndicesBlock[neighbourIndex];

			bool merging = false;
			if ( index == neighbourNeighbourIndex )
			{
				if ( index < neighbourIndex )
				{
					merging = true;
				}
				else
				{
					box		  = Aabb( make_float3( -FltMax ), make_float3( FltMax ) );
					nodeIndex = InvalidValue;
				}
			}

			uint32_t nodeAddr = numberOfClusters - 2 - warpOffset( merging, &nodeCounter );
			if ( merging )
			{
				box.grow( boxesBlock[neighbourIndex] );
				nodeIndex = encodeNodeIndex( nodeAddr, BoxType );

				scratchNodes[nodeAddr].m_childIndex0 = leftChildIndex;
				scratchNodes[nodeAddr].m_childIndex1 = rightChildIndex;
				scratchNodes[nodeAddr].m_box		 = box;
			}
		}
		__syncthreads();

		uint32_t blockSum		= blockScan( nodeIndex != InvalidValue, distanceOffsetsBlock );
		boxesBlock[index]		= Aabb( make_float3( -FltMax ), make_float3( FltMax ) );
		nodeIndicesBlock[index] = InvalidValue;
		__syncthreads();

		if ( index == blockDim.x - 1 ) numberOfClustersBlock = blockSum;
		if ( index < numberOfClusters )
		{
			if ( nodeIndex != InvalidValue )
			{
				const uint32_t newClusterIndex	  = blockSum - 1;
				boxesBlock[newClusterIndex]		  = box;
				nodeIndicesBlock[newClusterIndex] = nodeIndex;
			}
		}
		__syncthreads();

		numberOfClusters = numberOfClustersBlock;
	}
}

extern "C" __global__ void DevicePloc(
	uint32_t	   numberOfClusters,
	uint32_t*	   nodeIndices0,
	uint32_t*	   nodeIndices1,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	uint32_t*	   taskCounter,
	uint32_t*	   blockCounter,
	uint32_t*	   newBlockOffsetSum )
{
	const uint32_t index	   = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t blockOffset = blockDim.x * blockIdx.x;

	alignas( alignof( Aabb ) ) __shared__ uint8_t boxesCache[sizeof( Aabb ) * ( PlocMainBlockSize + 4 * PlocRadius )];
	__shared__ uint32_t							  distanceOffsetsCache[PlocMainBlockSize + 4 * PlocRadius];
	__shared__ uint32_t							  nodeIndicesCache[PlocMainBlockSize + 4 * PlocRadius];
	__shared__ uint32_t							  newBlockOffset;

	uint32_t* distanceOffsetsBlock = distanceOffsetsCache + 2 * PlocRadius;
	uint32_t* nodeIndicesBlock	   = nodeIndicesCache + 2 * PlocRadius;
	Aabb*	  boxesBlock		   = reinterpret_cast<Aabb*>( boxesCache ) + 2 * PlocRadius;

	for ( int32_t neighbourIndex = static_cast<int32_t>( threadIdx.x - 2 * PlocRadius );
		  neighbourIndex < static_cast<int32_t>( blockDim.x + 2 * PlocRadius );
		  neighbourIndex += blockDim.x )
	{
		int32_t clusterIndex = neighbourIndex + blockOffset;
		if ( clusterIndex >= 0 && clusterIndex < numberOfClusters )
		{
			uint32_t nodeIndex				 = nodeIndices0[clusterIndex];
			Aabb	 box					 = isLeafNode( nodeIndex ) ? references[getNodeAddr( nodeIndex )].aabb()
																	   : scratchNodes[getNodeAddr( nodeIndex )].aabb();
			boxesBlock[neighbourIndex]		 = box;
			nodeIndicesBlock[neighbourIndex] = nodeIndex;
		}
		else
		{
			boxesBlock[neighbourIndex]		 = Aabb( make_float3( -FltMax ), make_float3( FltMax ) );
			nodeIndicesBlock[neighbourIndex] = InvalidValue;
		}
		distanceOffsetsBlock[neighbourIndex] = InvalidValue;
	}
	__syncthreads();

	constexpr uint32_t OffsetMask = ( ( 1u << ( Log2( PlocRadius ) + 1 ) ) - 1 );

	for ( int32_t threadIndex = static_cast<int32_t>( threadIdx.x - 2 * PlocRadius );
		  threadIndex < static_cast<int32_t>( blockDim.x + PlocRadius );
		  threadIndex += blockDim.x )
	{
		uint32_t minDistanceOffset = InvalidValue;
		Aabb	 box			   = boxesBlock[threadIndex];

		for ( int32_t neighbourIndex = threadIndex + 1; neighbourIndex < threadIndex + static_cast<int32_t>( PlocRadius ) + 1;
			  ++neighbourIndex )
		{
			Aabb neighbourBox = boxesBlock[neighbourIndex];
			neighbourBox.grow( box );
			uint32_t distance = ( ( __float_as_uint( neighbourBox.area() ) << 1 ) & ~OffsetMask );

			const uint32_t offset0		   = encodeOffset( threadIndex, neighbourIndex );
			const uint32_t distanceOffset0 = distance | offset0;
			minDistanceOffset			   = min( minDistanceOffset, distanceOffset0 );

			const uint32_t offset1		   = encodeOffset( neighbourIndex, threadIndex );
			const uint32_t distanceOffset1 = distance | offset1;
			atomicMin( &distanceOffsetsBlock[neighbourIndex], distanceOffset1 );
		}

		atomicMin( &distanceOffsetsBlock[threadIndex], minDistanceOffset );
	}
	__syncthreads();

	uint32_t nodeIndex = InvalidValue;
	if ( index < numberOfClusters )
	{
		int32_t neighbourIndex			= decodeOffset( threadIdx.x, distanceOffsetsBlock[threadIdx.x] & OffsetMask );
		int32_t neighbourNeighbourIndex = decodeOffset( neighbourIndex, distanceOffsetsBlock[neighbourIndex] & OffsetMask );

		uint32_t leftChildIndex	 = nodeIndicesBlock[threadIdx.x];
		uint32_t rightChildIndex = nodeIndicesBlock[neighbourIndex];

		bool merging = false;
		if ( static_cast<int32_t>( threadIdx.x ) == neighbourNeighbourIndex )
		{
			if ( static_cast<int32_t>( threadIdx.x ) < neighbourIndex ) merging = true;
		}
		else
		{
			nodeIndex = leftChildIndex;
		}

		uint32_t nodeAddr = numberOfClusters - 2 - warpOffset( merging, taskCounter );
		if ( merging )
		{
			Aabb box = boxesBlock[threadIdx.x];
			box.grow( boxesBlock[neighbourIndex] );

			scratchNodes[nodeAddr].m_childIndex0 = leftChildIndex;
			scratchNodes[nodeAddr].m_childIndex1 = rightChildIndex;
			scratchNodes[nodeAddr].m_box		 = box;

			nodeIndex = encodeNodeIndex( nodeAddr, BoxType );
		}
	}
	__syncthreads();

	uint32_t blockSum = blockScan( nodeIndex != InvalidValue, nodeIndicesCache ); // aliasing
	if ( threadIdx.x == blockDim.x - 1 )
	{
		while ( atomicAdd( blockCounter, 0 ) < blockIdx.x )
			;
		newBlockOffset = atomicAdd( newBlockOffsetSum, blockSum );
		atomicAdd( blockCounter, 1 );
	}
	__syncthreads();

	if ( index < numberOfClusters )
	{
		if ( nodeIndex != InvalidValue )
		{
			const uint32_t newClusterIndex = newBlockOffset + blockSum - 1;
			nodeIndices1[newClusterIndex]  = nodeIndex;
		}
	}
}
