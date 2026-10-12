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
#include <hiprt/hiprt_types.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/AabbList.h>
#include <hiprt/impl/BvhCommon.h>
#include <hiprt/impl/Triangle.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/MortonCode.h>
#include <hiprt/impl/TriangleMesh.h>
#include <hiprt/impl/BvhConfig.h>
#include <hiprt/impl/MemoryArena.h>

#include <hiprt/impl/BvhBuilderKernels.h>
#include <hiprt/impl/LbvhBuilderKernels.h>

using namespace hiprt;

static constexpr size_t BatchBuildCacheAlignment = alignof( ReferenceNode ) > alignof( ScratchNode ) ? alignof( ReferenceNode )
                                                                                                     : alignof( ScratchNode );
static constexpr size_t BatchBuildCacheSize =
	RoundUp( ( BatchBuilderMaxBlockSize - 1 ) * sizeof( ScratchNode ), BatchBuildCacheAlignment ) +
	RoundUp( ( BatchBuilderMaxBlockSize ) * sizeof( ReferenceNode ), BatchBuildCacheAlignment ) +
	RoundUp( BatchBuilderMaxBlockSize * sizeof( ConversionTask ), BatchBuildCacheAlignment ) +
	RoundUp( ( BatchBuilderMaxBlockSize + 1 ) * sizeof( uint32_t ), BatchBuildCacheAlignment );

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE size_t getStorageBufferSize( const hiprtGeometryBuildInput& buildInput )
{
	const size_t primCount     = getPrimCount( buildInput );
	const size_t primNodeCount = getMaxPrimNodeCount( buildInput, Rtip, primCount );
	const size_t primNodeSize  = getPrimNodeSize( buildInput, sizeof( TriangleNode ) );
	const size_t boxNodeCount  = getMaxBoxNodeCount( buildInput, Rtip, BoxNode::BranchingFactor, primCount );
	return getGeometryStorageBufferSize( primNodeCount, boxNodeCount, primNodeSize, sizeof( BoxNode ) );
}

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE size_t getStorageBufferSize( const hiprtSceneBuildInput& buildInput )
{
	const size_t frameCount   = buildInput.frameCount;
	const size_t primCount    = buildInput.instanceCount;
	const size_t boxNodeCount = getMaxBoxNodeCount( buildInput, Rtip, BoxNode::BranchingFactor, primCount );
	return getSceneStorageBufferSize(
		primCount, primCount, boxNodeCount, sizeof( BoxNode ), sizeof( InstanceNode ), frameCount );
}

template <typename PrimitiveNode, typename PrimitiveContainer>
HIPRT_DEVICE HIPRT_INLINE void build(
	PrimitiveContainer& primitives, const uint32_t geomTypeIn, MemoryArena& storageMemoryArena, MemoryArena& sharedMemoryArena )
{
	using Header = typename conditional<is_same_v<PrimitiveNode, InstanceNode>, SceneHeader, GeomHeader>::type;

	const uint32_t maxBoxNodeCount =
		static_cast<uint32_t>( getMaxBoxNodeCount<BoxNode, PrimitiveNode>( primitives.getCount() ) );
	const uint32_t maxPrimNodeCount =
		static_cast<uint32_t>( getMaxPrimNodeCount<PrimitiveNode>( Rtip, primitives.getCount() ) );

	Header*        header    = storageMemoryArena.allocate<Header>();
	BoxNode*       boxNodes  = storageMemoryArena.allocate<BoxNode>( maxBoxNodeCount );
	PrimitiveNode* primNodes = storageMemoryArena.allocate<PrimitiveNode>( maxPrimNodeCount );

	const uint32_t index     = threadIdx.x;
	const uint32_t primCount = primitives.getCount();

	// STEP 0: Init data
	if constexpr ( is_same_v<Header, SceneHeader> )
	{
		Frame*    frames    = storageMemoryArena.allocate<Frame>( primitives.getFrameCount() );
		Instance* instances = storageMemoryArena.allocate<Instance>( primitives.getCount() );

		primitives.setFrames( frames );
		InitSceneData<>(
			index, storageMemoryArena.getStorageSize(), primitives, boxNodes, primNodes, instances, frames, header );
	}
	else
	{
		constexpr bool triangleGeom = is_same_v<PrimitiveNode, TriangleNode>;
		const uint32_t geomType     = ( geomTypeIn << 1 ) | ( triangleGeom ? 1 : 0 );
		InitGeomDataImpl( index, primCount, geomType, storageMemoryArena.getStorageSize(), boxNodes, primNodes, header );
	}

	// A single primitive => special case
	if ( primCount == 1 )
	{
		SingletonConstruction( index, primitives, boxNodes, primNodes );
		return;
	}

	Aabb primBox;
	if ( index < primCount )
		primBox = primitives.fetchAabb( index );
	else
		primBox = primitives.fetchAabb( primCount - 1 );

	const uint32_t warpsPerBlock = DivideRoundUp( static_cast<uint32_t>( blockDim.x ), WarpSize );

	ScratchNode*    scratchNodes   = sharedMemoryArena.allocate<ScratchNode>( blockDim.x - 1 );
	ReferenceNode*  references     = sharedMemoryArena.allocate<ReferenceNode>( blockDim.x );
	uint32_t*       updateCounters = sharedMemoryArena.allocate<uint32_t>( blockDim.x + 1 );
	ConversionTask* taskQueue      = sharedMemoryArena.allocate<ConversionTask>( blockDim.x );

	uint32_t* mortonCodeKeys   = reinterpret_cast<uint32_t*>( taskQueue ) + 0 * blockDim.x;
	uint32_t* mortonCodeValues = reinterpret_cast<uint32_t*>( taskQueue ) + 1 * blockDim.x;

	// STEP 1: Calculate centroid bounding box by reduction
	updateCounters[index] = InvalidValue;
	Aabb* blockBoxes      = reinterpret_cast<Aabb*>( scratchNodes );
	Aabb  centroidBox     = blockUnion( primBox, blockBoxes );
	__syncthreads();

	// STEP 2: Calculate Morton codes
	if ( index < primCount )
	{
		const float3 boxExtent        = centroidBox.extent();
		const float3 center           = primitives.fetchCenter( index );
		const float3 normalizedCenter = ( center - centroidBox.m_min ) / boxExtent;
		mortonCodeKeys[index]         = computeExtendedMortonCode( normalizedCenter, boxExtent );
		mortonCodeValues[index]       = index;
	}
	else
	{
		mortonCodeKeys[index]   = InvalidValue;
		mortonCodeValues[index] = InvalidValue;
	}
	__syncthreads();

	// STEP 3: Sort Morton codes
	uint32_t* blockCache = reinterpret_cast<uint32_t*>( scratchNodes );
	for ( uint32_t i = 0; i < 32; ++i )
	{
		const uint32_t mortonCodeKey   = mortonCodeKeys[index];
		const uint32_t mortonCodeValue = mortonCodeValues[index];
		const uint32_t bit             = ( mortonCodeKey >> i ) & 1;
		const uint32_t blockSum        = blockScan( bit == 0, blockCache );
		const uint32_t newIndex        = bit == 0 ? blockSum - 1 : blockCache[warpsPerBlock - 1] + index - blockSum;
		__syncthreads();
		mortonCodeKeys[newIndex]   = mortonCodeKey;
		mortonCodeValues[newIndex] = mortonCodeValue;
		__syncthreads();
	}

	// STEP 4: Emit topology and refit nodes
	EmitTopologyAndFitBounds( index, primitives, mortonCodeKeys, mortonCodeValues, updateCounters, scratchNodes, references );
	__syncthreads();
	const uint32_t rootAddr = updateCounters[primCount - 1];
	__syncthreads();

	// STEP 5: Compute fat leaves
	if constexpr ( is_same_v<PrimitiveNode, rtip31::TrianglePacketNode> )
	{
		uint32_t* triangleCounts = reinterpret_cast<uint32_t*>( taskQueue );
		uint32_t* parentAddrs    = triangleCounts + primCount;
		updateCounters[index]    = 0;
		ComputeParentAddrs( index, primCount, rootAddr, scratchNodes, parentAddrs );
		__syncthreads();
		ComputeFatLeaves( index, primCount, parentAddrs, scratchNodes, triangleCounts, updateCounters );
		__syncthreads();
	}

	// STEP 6: Collapse
	uint32_t* referenceIndices = updateCounters;
	uint32_t* taskCounter      = &referenceIndices[primitives.getCount()];
	if ( index == 0 )
	{
		*taskCounter     = 0;
		taskQueue[index] = ConversionTask( encodeNodeIndex( rootAddr, BoxType ), 0, 0 );
	}
	else
	{
		taskQueue[index] = ConversionTask{};
	}
	__syncthreads();

	{
		const uint32_t maxThreads = RoundUp( BoxNode::BranchingFactor * maxBoxNodeCount, blockDim.x );
		for ( uint32_t i = index; i < maxThreads; i += blockDim.x )
		{
			Collapse<PrimitiveNode>(
				i, maxBoxNodeCount, primCount, scratchNodes, references, header, boxNodes, taskQueue, referenceIndices );
			__syncthreads();
		}
	}

	// STEP 7: Compact tasks
	CompactTasks( index, header->m_boxNodeCount, taskQueue, taskCounter );
	__syncthreads();

	// STEP 8: Pack leaves
	const uint32_t taskCount = *taskCounter;
	if constexpr ( is_same_v<PrimitiveNode, rtip31::TrianglePacketNode> )
	{
		if ( index < WarpSize )
		{
			const uint32_t maxThreads = RoundUp( taskCount * PrimitiveNode::LanesPerLeafPacketTask, WarpSize );
			for ( uint32_t i = index; i < maxThreads; i += WarpSize )
			{
				PackLeaves( i, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
			}
		}
		__syncthreads();
	}
	else
	{
		PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
	}
}

HIPRT_KERNEL void __launch_bounds__( BatchBuilderMaxBlockSize ) BatchBuild_hiprtGeometryBuildInput(
	const uint32_t count, const hiprtGeometryBuildInput* buildInputs, hiprtDevicePtr* buffers )
{
	const uint32_t index = blockIdx.x + gridDim.x * blockIdx.y;
	if ( index < count )
	{
		alignas( BatchBuildCacheAlignment ) __shared__ uint8_t cache[BatchBuildCacheSize];
		MemoryArena sharedMemoryArena( cache, BatchBuildCacheSize, BatchBuildCacheAlignment );

		hiprtGeometryBuildInput buildInput = buildInputs[index];
		MemoryArena             storageMemoryArena( buffers[index], getStorageBufferSize( buildInput ), DefaultAlignment );

		switch ( buildInput.type )
		{
		case hiprtPrimitiveTypeTriangleMesh: {
			TriangleMesh mesh( buildInput.primitive.triangleMesh );
			build<TriangleNode>( mesh, buildInput.geomType, storageMemoryArena, sharedMemoryArena );
			break;
		}
		case hiprtPrimitiveTypeAABBList: {
			AabbList list( buildInput.primitive.aabbList );
			build<CustomNode>( list, buildInput.geomType, storageMemoryArena, sharedMemoryArena );
			break;
		}
		}
	}
}

HIPRT_KERNEL void __launch_bounds__( BatchBuilderMaxBlockSize )
	BatchBuild_hiprtSceneBuildInput( const uint32_t count, const hiprtSceneBuildInput* buildInputs, hiprtDevicePtr* buffers )
{
	const uint32_t index = blockIdx.x + gridDim.x * blockIdx.y;
	if ( index < count )
	{
		alignas( BatchBuildCacheAlignment ) __shared__ uint8_t cache[BatchBuildCacheSize];
		MemoryArena sharedMemoryArena( cache, BatchBuildCacheSize, BatchBuildCacheAlignment );

		hiprtSceneBuildInput buildInput = buildInputs[index];
		MemoryArena          storageMemoryArena( buffers[index], getStorageBufferSize( buildInput ), DefaultAlignment );

		switch ( buildInput.frameType )
		{
		case hiprtFrameTypeSRT: {
			InstanceList<hiprtFrameSRT> list( buildInput );
			build<InstanceNode>( list, hiprtInvalidValue, storageMemoryArena, sharedMemoryArena );
			break;
		}
		case hiprtFrameTypeMatrix: {
			InstanceList<hiprtFrameMatrix> list( buildInput );
			build<InstanceNode>( list, hiprtInvalidValue, storageMemoryArena, sharedMemoryArena );
			break;
		}
		}
	}
}
