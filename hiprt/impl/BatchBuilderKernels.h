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
#include <hiprt/impl/Geometry.h>
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

static constexpr size_t CacheAlignment = alignof( ReferenceNode ) > alignof( ScratchNode ) ? alignof( ReferenceNode )
																						   : alignof( ScratchNode );

static constexpr size_t CacheSize = RoundUp( ( BatchBuilderMaxBlockSize - 1 ) * sizeof( ScratchNode ), CacheAlignment ) +
									RoundUp( ( BatchBuilderMaxBlockSize ) * sizeof( ReferenceNode ), CacheAlignment ) +
									2 * RoundUp( BatchBuilderMaxBlockSize * sizeof( uint32_t ), CacheAlignment ) +
									RoundUp( BatchBuilderMaxBlockSize * sizeof( uint32_t ), CacheAlignment ) +
									RoundUp( BatchBuilderMaxBlockSize * sizeof( uint3 ), CacheAlignment );

HIPRT_DEVICE size_t getStorageBufferSize( const hiprtGeometryBuildInput& buildInput )
{
	const size_t primCount	  = getPrimCount( buildInput );
	const size_t primNodeSize = getPrimNodeSize( buildInput );
	const size_t boxNodeCount = DivideRoundUp( 2 * primCount, 3 );
	return getGeometryStorageBufferSize( primCount, boxNodeCount, primNodeSize );
}

HIPRT_DEVICE size_t getStorageBufferSize( const hiprtSceneBuildInput& buildInput )
{
	const size_t frameCount	  = buildInput.frameCount;
	const size_t primCount	  = buildInput.instanceCount;
	const size_t boxNodeCount = DivideRoundUp( 2 * primCount, 3 );
	return getSceneStorageBufferSize( primCount, primCount, boxNodeCount, frameCount );
}

template <typename PrimitiveNode, typename PrimitiveContainer>
HIPRT_DEVICE void
build( PrimitiveContainer& primitives, uint32_t geomType, MemoryArena& storageMemoryArena, MemoryArena& sharedMemoryArena )
{
	typedef typename conditional<is_same<PrimitiveNode, InstanceNode>::value, SceneHeader, GeomHeader>::type Header;

	Header*		   header	 = storageMemoryArena.allocate<Header>();
	BoxNode*	   boxNodes	 = storageMemoryArena.allocate<BoxNode>( DivideRoundUp( 2 * primitives.getCount(), 3 ) );
	PrimitiveNode* primNodes = storageMemoryArena.allocate<PrimitiveNode>( primitives.getCount() );

	uint32_t index	   = threadIdx.x;
	uint32_t primCount = primitives.getCount();

	// STEP 0: Init data
	if constexpr ( is_same<Header, SceneHeader>::value )
	{
		Frame*	  frames	= storageMemoryArena.allocate<Frame>( primitives.getFrameCount() );
		Instance* instances = storageMemoryArena.allocate<Instance>( primitives.getCount() );

		primitives.setFrames( frames );
		InitSceneData<>(
			index, storageMemoryArena.getStorageSize(), primitives, boxNodes, primNodes, instances, frames, header );
	}
	else
	{
		geomType <<= 1;
		if constexpr ( is_same<PrimitiveNode, TriangleNode>::value ) geomType |= 1;
		InitGeomDataImpl( index, primCount, storageMemoryArena.getStorageSize(), boxNodes, primNodes, geomType, header );
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

	ReferenceNode* references		= sharedMemoryArena.allocate<ReferenceNode>( blockDim.x );
	ScratchNode*   scratchNodes		= sharedMemoryArena.allocate<ScratchNode>( blockDim.x - 1 );
	uint32_t*	   mortonCodeKeys	= sharedMemoryArena.allocate<uint32_t>( blockDim.x );
	uint32_t*	   mortonCodeValues = sharedMemoryArena.allocate<uint32_t>( blockDim.x );
	uint32_t*	   updateCounters	= sharedMemoryArena.allocate<uint32_t>( blockDim.x );
	uint3*		   taskQueue		= sharedMemoryArena.allocate<uint3>( blockDim.x );

	// STEP 1: Calculate centroid bounding box by reduction
	updateCounters[index] = InvalidValue;
	Aabb* blockBoxes	  = reinterpret_cast<Aabb*>( scratchNodes );
	Aabb  centroidBox	  = blockUnion( primBox, blockBoxes );
	__syncthreads();

	// STEP 2: Calculate Morton codes
	if ( index < primCount )
	{
		float3 boxExtent		= centroidBox.extent();
		float3 center			= primitives.fetchCenter( index );
		float3 normalizedCenter = ( center - centroidBox.m_min ) / boxExtent;
		mortonCodeKeys[index]	= computeExtendedMortonCode( normalizedCenter, boxExtent );
		mortonCodeValues[index] = index;
	}
	else
	{
		mortonCodeKeys[index]	= InvalidValue;
		mortonCodeValues[index] = InvalidValue;
	}
	__syncthreads();

	// STEP 3: Sort Morton codes
	uint32_t* blockCache = reinterpret_cast<uint32_t*>( scratchNodes );
	for ( uint32_t i = 0; i < 32; ++i )
	{
		uint32_t mortonCodeKey	 = mortonCodeKeys[index];
		uint32_t mortonCodeValue = mortonCodeValues[index];
		uint32_t bit			 = ( mortonCodeKey >> i ) & 1;
		uint32_t blockSum		 = blockScan( bit == 0, blockCache );
		uint32_t newIndex		 = bit == 0 ? blockSum - 1 : blockCache[warpsPerBlock - 1] + index - blockSum;
		__syncthreads();
		mortonCodeKeys[newIndex]   = mortonCodeKey;
		mortonCodeValues[newIndex] = mortonCodeValue;
		__syncthreads();
	}

	// STEP 4: Emit topology and refit nodes
	EmitTopologyAndFitBounds( index, mortonCodeKeys, mortonCodeValues, updateCounters, primitives, scratchNodes, references );
	__syncthreads();

	// STEP 5: Collapse
	uint32_t rootAddr = updateCounters[primCount - 1];
	if ( index == 0 )
		taskQueue[index] = make_uint3( encodeNodeIndex( rootAddr, BoxType ), 0, 0 );
	else
		taskQueue[index] = make_uint3( InvalidValue, InvalidValue, InvalidValue );
	__syncthreads();

	uint32_t* taskCounter = &updateCounters[0];
	*taskCounter		  = 1;
	__syncthreads();
	Collapse( index, primCount, header, scratchNodes, references, boxNodes, primNodes, primitives, taskCounter, taskQueue );
}

extern "C" __global__ void
BatchBuild_hiprtGeometryBuildInput( uint32_t count, const hiprtGeometryBuildInput* buildInputs, hiprtDevicePtr* buffers )
{
	const uint32_t index = blockIdx.x + gridDim.x * blockIdx.y;
	if ( index < count )
	{
		alignas( CacheAlignment ) __shared__ uint8_t cache[CacheSize];
		MemoryArena									 sharedMemoryArena( cache, CacheSize, CacheAlignment );

		hiprtGeometryBuildInput buildInput = buildInputs[index];
		MemoryArena				storageMemoryArena( buffers[index], getStorageBufferSize( buildInput ), DefaultAlignment );

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

extern "C" __global__ void
BatchBuild_hiprtSceneBuildInput( uint32_t count, const hiprtSceneBuildInput* buildInputs, hiprtDevicePtr* buffers )
{
	const uint32_t index = blockIdx.x + gridDim.x * blockIdx.y;
	if ( index < count )
	{
		alignas( CacheAlignment ) __shared__ uint8_t cache[CacheSize];
		MemoryArena									 sharedMemoryArena( cache, CacheSize, CacheAlignment );

		hiprtSceneBuildInput buildInput = buildInputs[index];
		MemoryArena			 storageMemoryArena( buffers[index], getStorageBufferSize( buildInput ), DefaultAlignment );

		switch ( buildInput.frameType )
		{
		case hiprtFrameTypeSRT: {
			InstanceList<SRTFrame> list( buildInput );
			build<InstanceNode>( list, hiprtInvalidValue, storageMemoryArena, sharedMemoryArena );
			break;
		}
		case hiprtFrameTypeMatrix: {
			InstanceList<MatrixFrame> list( buildInput );
			build<InstanceNode>( list, hiprtInvalidValue, storageMemoryArena, sharedMemoryArena );
			break;
		}
		}
	}
}
