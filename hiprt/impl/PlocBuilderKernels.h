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
#include <hiprt/hiprt_math.h>
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

HIPRT_DEVICE uint32_t findParent( uint32_t i, uint32_t j, uint32_t n, const uint32_t* sortedMortonCodeKeys )
{
	if ( i == 0 && j == n ) return InvalidValue;
	if ( i == 0 || ( j != n && findHighestDifferentBit( j - 1, j, n, sortedMortonCodeKeys ) <
								   findHighestDifferentBit( i - 1, i, n, sortedMortonCodeKeys ) ) )
		return j - 1;
	else
		return i - 1;
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

template <typename PrimitiveContainer>
__device__ void SetupClusters(
	PrimitiveContainer& primitives, ReferenceNode* references, const uint32_t* sortedMortonCodeValues, uint32_t* nodeIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if ( index >= primitives.getCount() ) return;

	uint32_t leafType;
	if constexpr ( is_same<PrimitiveContainer, TriangleMesh>::value )
		leafType = TriangleType;
	else if constexpr ( is_same<PrimitiveContainer, AabbList>::value )
		leafType = CustomType;
	else if constexpr (
		is_same<PrimitiveContainer, InstanceList<SRTFrame>>::value ||
		is_same<PrimitiveContainer, InstanceList<MatrixFrame>>::value )
		leafType = InstanceType;

	uint32_t primIndex = sortedMortonCodeValues[index];
	references[index]  = ReferenceNode( primIndex, primitives.fetchAabb( primIndex ) );
	nodeIndices[index] = encodeNodeIndex( index, leafType );
}

extern "C" __global__ void SetupClusters_TriangleMesh(
	TriangleMesh primitives, ReferenceNode* references, const uint32_t* sortedMortonCodeValues, uint32_t* nodeIndices )
{
	SetupClusters<TriangleMesh>( primitives, references, sortedMortonCodeValues, nodeIndices );
}

extern "C" __global__ void SetupClusters_AabbList(
	AabbList primitives, ReferenceNode* references, const uint32_t* sortedMortonCodeValues, uint32_t* nodeIndices )
{
	SetupClusters<AabbList>( primitives, references, sortedMortonCodeValues, nodeIndices );
}

extern "C" __global__ void SetupClusters_InstanceList_SRTFrame(
	InstanceList<SRTFrame> primitives,
	ReferenceNode*		   references,
	const uint32_t*		   sortedMortonCodeValues,
	uint32_t*			   nodeIndices )
{
	SetupClusters<InstanceList<SRTFrame>>( primitives, references, sortedMortonCodeValues, nodeIndices );
}

extern "C" __global__ void SetupClusters_InstanceList_MatrixFrame(
	InstanceList<MatrixFrame> primitives,
	ReferenceNode*			  references,
	const uint32_t*			  sortedMortonCodeValues,
	uint32_t*				  nodeIndices )
{
	SetupClusters<InstanceList<MatrixFrame>>( primitives, references, sortedMortonCodeValues, nodeIndices );
}

// H-PLOC: Hierarchical Parallel Locally-Ordered Clustering for Bounding Volume Hierarchy Construction
// https://gpuopen.com/download/publications/HPLOC.pdf
// Disclaimer: This implementation is different than the one used in the paper.
extern "C" __global__ void HPloc(
	uint32_t		primCount,
	const uint32_t* sortedMortonCodeKeys,
	uint32_t*		updateCounters,
	uint32_t*		nodeIndices,
	ScratchNode*	scratchNodes,
	ReferenceNode*	references,
	uint32_t*		nodeCounter )
{
	const uint32_t index	 = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t laneIndex = threadIdx.x & ( WarpSize - 1 );
	const uint32_t warpIndex = threadIdx.x >> Log2( WarpSize );

	alignas( alignof( Aabb ) ) __shared__ uint8_t boxesCache[sizeof( Aabb ) * PlocMainBlockSize];
	__shared__ uint32_t							  distanceOffsetsBlock[PlocMainBlockSize];
	__shared__ uint32_t							  nodeIndicesBlock[PlocMainBlockSize];

	Aabb*	  boxesBlock		  = reinterpret_cast<Aabb*>( boxesCache );
	Aabb*	  boxesWarp			  = &boxesBlock[warpIndex * WarpSize];
	uint32_t* distanceOffsetsWarp = &distanceOffsetsBlock[warpIndex * WarpSize];
	uint32_t* nodeIndicesWarp	  = &nodeIndicesBlock[warpIndex * WarpSize];

	uint32_t i = index;
	uint32_t j = i + 1;
	uint32_t k, s;

	bool active = index < primCount;

	while ( __ballot( active ) != 0 )
	{
		__threadfence();

		if ( active )
		{
			uint32_t parentAddr = findParent( i, j, primCount, sortedMortonCodeKeys );
			if ( parentAddr == i - 1 )
			{
				k = atomicExch( &updateCounters[parentAddr], j );
				s = i;
				i = k;
			}
			else
			{
				k = atomicExch( &updateCounters[parentAddr], i );
				s = j;
				j = k;
			}

			if ( k == InvalidValue ) active = false;
		}

		__threadfence();

		const uint32_t size	 = j - i;
		const bool	   last	 = active && size == primCount;
		uint64_t	   merge = __ballot( ( active && size > WarpSize / 2 ) || last );

		while ( merge )
		{
			const uint32_t currentLane = __ffsll( static_cast<unsigned long long>( merge ) ) - 1;
			merge &= merge - 1;

			const uint32_t current_i   = __shfl( i, currentLane );
			const uint32_t current_j   = __shfl( j, currentLane );
			const uint32_t current_s   = __shfl( s, currentLane );
			const bool	   currentLast = __shfl( last, currentLane );

			uint32_t numLeft  = min( current_s - current_i, WarpSize / 2 );
			uint32_t numRight = min( current_j - current_s, WarpSize / 2 );

			uint32_t leftIndex = InvalidValue;
			if ( laneIndex < numLeft ) leftIndex = nodeIndices[current_i + laneIndex];
			uint32_t numValidLeft = __popcll( __ballot( leftIndex != InvalidValue ) );
			numLeft				  = min( numLeft, numValidLeft );

			uint32_t rightIndex = InvalidValue;
			if ( laneIndex < numRight ) rightIndex = nodeIndices[current_s + laneIndex];
			uint32_t numValidRight = __popcll( __ballot( rightIndex != InvalidValue ) );
			numRight			   = min( numRight, numValidRight );

			if ( laneIndex < numLeft ) nodeIndicesWarp[laneIndex] = leftIndex;
			if ( laneIndex < numRight ) nodeIndicesWarp[laneIndex + numLeft] = rightIndex;

			__threadfence_block();
			SyncWarp();

			uint32_t	   numberOfClusters = numLeft + numRight;
			const uint32_t threshold		= currentLast ? 1 : WarpSize / 2;
			if ( numberOfClusters > threshold )
			{
				uint32_t nodeIndex	 = nodeIndicesWarp[min( laneIndex, numberOfClusters - 1 )];
				Aabb	 box		 = isLeafNode( nodeIndex ) ? references[getNodeAddr( nodeIndex )].aabb()
															   : scratchNodes[getNodeAddr( nodeIndex )].aabb();
				boxesWarp[laneIndex] = box;
			}

			constexpr uint32_t OffsetMask = ( ( 1u << ( Log2( PlocRadius ) + 1 ) ) - 1 );

			while ( numberOfClusters > threshold )
			{
				distanceOffsetsWarp[laneIndex] = InvalidValue;
				SyncWarp();

				uint32_t minDistanceOffset = InvalidValue;
				Aabb	 box			   = boxesWarp[laneIndex];

				for ( uint32_t neighbourIndex = laneIndex + 1;
					  neighbourIndex <= laneIndex + PlocRadius && neighbourIndex < numberOfClusters;
					  ++neighbourIndex )
				{
					Aabb neighbourBox = boxesWarp[neighbourIndex];
					neighbourBox.grow( box );
					uint32_t distance = ( ( __float_as_uint( neighbourBox.area() ) << 1 ) & ~OffsetMask );

					const uint32_t offset0		   = encodeOffset( laneIndex, neighbourIndex );
					const uint32_t distanceOffset0 = distance | offset0;
					minDistanceOffset			   = min( minDistanceOffset, distanceOffset0 );

					const uint32_t offset1		   = encodeOffset( neighbourIndex, laneIndex );
					const uint32_t distanceOffset1 = distance | offset1;
					atomicMin( &distanceOffsetsWarp[neighbourIndex], distanceOffset1 );
				}
				atomicMin( &distanceOffsetsWarp[laneIndex], minDistanceOffset );

				SyncWarp();

				uint32_t nodeIndex = InvalidValue;
				if ( laneIndex < numberOfClusters )
				{
					int32_t neighbourIndex = decodeOffset( laneIndex, distanceOffsetsWarp[laneIndex] & OffsetMask );
					int32_t neighbourNeighbourIndex =
						decodeOffset( neighbourIndex, distanceOffsetsWarp[neighbourIndex] & OffsetMask );

					uint32_t leftChildIndex	 = nodeIndicesWarp[laneIndex];
					uint32_t rightChildIndex = nodeIndicesWarp[neighbourIndex];

					bool merging = false;
					if ( static_cast<int32_t>( laneIndex ) == neighbourNeighbourIndex )
					{
						if ( static_cast<int32_t>( laneIndex ) < neighbourIndex ) merging = true;
					}
					else
					{
						nodeIndex = leftChildIndex;
					}

					uint32_t nodeAddr = primCount - 2 - warpOffset( merging, nodeCounter );
					if ( merging )
					{
						box.grow( boxesWarp[neighbourIndex] );

						scratchNodes[nodeAddr].m_childIndex0 = leftChildIndex;
						scratchNodes[nodeAddr].m_childIndex1 = rightChildIndex;
						scratchNodes[nodeAddr].m_box		 = box;

						nodeIndex = encodeNodeIndex( nodeAddr, BoxType );
					}
				}

				const uint64_t warpBallot = __ballot( nodeIndex != InvalidValue ); // warp sync'd here
				const uint32_t newIndex	  = __popcll( warpBallot & ( ( 1ull << laneIndex ) - 1ull ) );
				numberOfClusters		  = __popcll( warpBallot );

				if ( nodeIndex != InvalidValue )
				{
					boxesWarp[newIndex]		  = box;
					nodeIndicesWarp[newIndex] = nodeIndex;
				}

				__threadfence_block();
				SyncWarp();
			}

			if ( laneIndex < WarpSize / 2 )
				nodeIndices[current_i + laneIndex] =
					( laneIndex < numberOfClusters ) ? nodeIndicesWarp[laneIndex] : InvalidValue;

			__threadfence();
		}

		if ( last ) active = false;

		__threadfence();
	}
}
