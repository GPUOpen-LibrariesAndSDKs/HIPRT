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
#include <hiprt/impl/Instance.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/MortonCode.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/TriangleMesh.h>
#include <hiprt/impl/BvhConfig.h>
using namespace hiprt;

HIPRT_DEVICE HIPRT_INLINE uint32_t getScratchChildIndexAndBox(
	uint32_t i, const ScratchNode& scratchNode, const ScratchNode* scratchNodes, const ReferenceNode* references, Aabb& box )
{
	uint32_t childIndex = scratchNode.getChildIndex( i );
	uint32_t childType	= getNodeType( childIndex );
	uint32_t childAddr	= getNodeAddr( childIndex );
	if ( childType == BoxType )
	{
		box = scratchNodes[childAddr].aabb();
	}
	else
	{
		box		   = references[childAddr].aabb();
		childIndex = encodeNodeIndex( references[childAddr].m_primIndex, childType );
	}
	return childIndex;
}

__device__ void InitGeomDataImpl(
	uint32_t	index,
	uint32_t	primCount,
	size_t		size,
	BoxNode*	boxNodes,
	void*		primNodes,
	uint32_t	geomType,
	GeomHeader* geomHeader )
{
	if ( index == 0 )
	{
		geomHeader->m_size			= size;
		geomHeader->m_boxNodes		= boxNodes;
		geomHeader->m_primNodes		= primNodes;
		geomHeader->m_boxNodeCount	= 1u;
		geomHeader->m_primNodeCount = primCount;
		geomHeader->m_geomType		= geomType;
	}
}

extern "C" __global__ void
InitGeomData( size_t size, uint32_t primCount, BoxNode* boxNodes, void* primNodes, uint32_t geomType, GeomHeader* geomHeader )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitGeomDataImpl( index, primCount, size, boxNodes, primNodes, geomType, geomHeader );
}

template <typename InstanceList>
__device__ void InitSceneData(
	uint32_t			  index,
	size_t				  size,
	InstanceList&		  instanceList,
	BoxNode*			  boxNodes,
	InstanceNode*		  primNodes,
	Instance*			  instances,
	uint32_t*			  masks,
	hiprtTransformHeader* transforms,
	Frame*				  frames,
	SceneHeader*		  sceneHeader )
{
	if ( index < instanceList.getCount() )
	{
		instances[index]  = instanceList.fetchInstance( index );
		masks[index]	  = instanceList.fetchMask( index );
		transforms[index] = instanceList.fetchTransformHeader( index );
	}

	if ( index < instanceList.getFrameCount() ) instanceList.convertFrame( index );

	if ( index == 0 )
	{
		sceneHeader->m_size			 = size;
		sceneHeader->m_boxNodes		 = boxNodes;
		sceneHeader->m_primNodes	 = primNodes;
		sceneHeader->m_instances	 = instances;
		sceneHeader->m_masks		 = masks;
		sceneHeader->m_transforms	 = transforms;
		sceneHeader->m_frames		 = frames;
		sceneHeader->m_boxNodeCount	 = 1u;
		sceneHeader->m_primNodeCount = instanceList.getCount();
	}
}

extern "C" __global__ void InitSceneData_InstanceList_SRTFrame(
	size_t				   size,
	InstanceList<SRTFrame> instanceList,
	BoxNode*			   boxNodes,
	InstanceNode*		   primNodes,
	Instance*			   instances,
	uint32_t*			   masks,
	hiprtTransformHeader*  transforms,
	Frame*				   frames,
	SceneHeader*		   sceneHeader )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitSceneData<InstanceList<SRTFrame>>(
		index, size, instanceList, boxNodes, primNodes, instances, masks, transforms, frames, sceneHeader );
}

extern "C" __global__ void InitSceneData_InstanceList_MatrixFrame(
	size_t					  size,
	InstanceList<MatrixFrame> instanceList,
	BoxNode*				  boxNodes,
	InstanceNode*			  primNodes,
	Instance*				  instances,
	uint32_t*				  masks,
	hiprtTransformHeader*	  transforms,
	Frame*					  frames,
	SceneHeader*			  sceneHeader )
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitSceneData<InstanceList<MatrixFrame>>(
		index, size, instanceList, boxNodes, primNodes, instances, masks, transforms, frames, sceneHeader );
}

template <typename PrimitiveContainer, typename PrimitiveNode>
__device__ void
SingletonConstruction( uint32_t index, PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	if ( index > 0 ) return;

	uint32_t leafType;
	if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
	{
		primNodes[0] = primitives.fetchTriangleNode( 0 );
		leafType	 = TriangleType;
	}
	else if constexpr ( is_same<PrimitiveNode, CustomNode>::value )
	{
		leafType = CustomType;
	}
	else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
	{
		leafType = InstanceType;
	}

	primNodes[0].m_parentAddr = 0;

	BoxNode root;
	root.m_box0 = primitives.fetchAabb( 0 );
	root.m_box1.reset();
	root.m_box2.reset();
	root.m_box3.reset();
	root.encodeChildIndex( 0, 0, leafType );
	root.m_childIndex1 = InvalidValue;
	root.m_childIndex2 = InvalidValue;
	root.m_childIndex3 = InvalidValue;
	root.m_parentAddr  = InvalidValue;
	root.m_childCount  = 1;
	boxNodes[0]		   = root;
}

extern "C" __global__ void
SingletonConstruction_TriangleMesh_TriangleNode( TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<TriangleMesh, TriangleNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void
SingletonConstruction_AabbList_CustomNode( AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<AabbList, CustomNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void SingletonConstruction_InstanceList_SRTFrame_InstanceNode(
	InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<SRTFrame>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void SingletonConstruction_InstanceList_MatrixFrame_InstanceNode(
	InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<MatrixFrame>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void PairTriangles( TriangleMesh mesh, int2* pairIndices, GeomHeader* header )
{
	const uint32_t index	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t laneIndex = threadIdx.x & ( WarpSize - 1 );

	// TODO: validate triangles
	bool	 valid		 = index < mesh.getCount();
	uint32_t pairedIndex = InvalidValue;
	uint64_t activeMask	 = __ballot( valid );

	int3 triIndices;
	if ( valid ) triIndices = mesh.fetchTriangleIndices( index );

	while ( activeMask )
	{
		activeMask = __shfl( activeMask, 0 );

		const uint64_t broadcastLane = __ffsll( static_cast<unsigned long long>( activeMask ) ) - 1;
		if ( laneIndex == broadcastLane ) valid = false;

		activeMask &= activeMask - 1;

		const uint32_t broadcastIndex	   = __shfl( index, broadcastLane );
		const int3	   triIndicesBroadcast = make_int3(
			__shfl( triIndices.x, broadcastLane ),
			__shfl( triIndices.y, broadcastLane ),
			__shfl( triIndices.z, broadcastLane ) );

		bool pairable = false;
		if ( index != broadcastIndex && valid )
			pairable = tryPairTriangles( triIndicesBroadcast, triIndices ).x != InvalidValue;

		const uint32_t firstPairedLane = __ffsll( static_cast<unsigned long long>( __ballot( pairable ) ) ) - 1;
		if ( firstPairedLane < WarpSize )
		{
			activeMask &= ~( 1u << firstPairedLane );
			if ( laneIndex == firstPairedLane ) valid = false;

			const uint32_t secondIndex = __shfl( index, firstPairedLane );
			if ( laneIndex == broadcastLane ) pairedIndex = secondIndex;
		}
		else if ( laneIndex == broadcastLane )
		{
			pairedIndex = index;
		}
	}

	bool	 pairing   = index < mesh.getCount() && pairedIndex != InvalidValue;
	uint32_t pairIndex = warpOffset( pairing, &header->m_primNodeCount );
	if ( pairing ) pairIndices[pairIndex] = make_int2( index, pairedIndex );
}

template <typename PrimitiveContainer>
__device__ void ComputeCentroidBox( PrimitiveContainer& primitives, Aabb* centroidBox )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	Aabb primBox;
	if ( index < primitives.getCount() )
		primBox = primitives.fetchAabb( index );
	else
		primBox = primitives.fetchAabb( primitives.getCount() - 1 );

	constexpr uint32_t							  WarpsPerBlock = DivideRoundUp( BvhBuilderReductionBlockSize, WarpSize );
	alignas( alignof( Aabb ) ) __shared__ uint8_t cache[sizeof( Aabb ) * WarpsPerBlock];
	Aabb*										  blockBoxes = reinterpret_cast<Aabb*>( cache );

	Aabb blockBox = blockUnion( primBox, blockBoxes );
	if ( threadIdx.x == 0 ) centroidBox->atomicGrow( blockBox );
}

extern "C" __global__ void ComputeCentroidBox_TriangleMesh( TriangleMesh primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<TriangleMesh>( primitives, centroidBox );
}

extern "C" __global__ void ComputeCentroidBox_AabbList( AabbList primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<AabbList>( primitives, centroidBox );
}

extern "C" __global__ void ComputeCentroidBox_InstanceList_SRTFrame( InstanceList<SRTFrame> primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<InstanceList<SRTFrame>>( primitives, centroidBox );
}

extern "C" __global__ void
ComputeCentroidBox_InstanceList_MatrixFrame( InstanceList<MatrixFrame> primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<InstanceList<MatrixFrame>>( primitives, centroidBox );
}

template <typename PrimitiveContainer>
__device__ void ComputeBox( PrimitiveContainer& primitives, Aabb* box )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	Aabb primBox;
	if ( index < primitives.getCount() )
		primBox = primitives.fetchAabb( index );
	else
		primBox = primitives.fetchAabb( primitives.getCount() - 1 );

	constexpr uint32_t							  WarpsPerBlock = DivideRoundUp( BvhBuilderReductionBlockSize, WarpSize );
	alignas( alignof( Aabb ) ) __shared__ uint8_t cache[sizeof( Aabb ) * WarpsPerBlock];
	Aabb*										  blockBoxes = reinterpret_cast<Aabb*>( cache );

	Aabb blockBox = blockUnion( primBox, blockBoxes );
	if ( threadIdx.x == 0 ) box->atomicGrow( blockBox );
}

extern "C" __global__ void ComputeBox_TriangleMesh( TriangleMesh primitives, Aabb* box )
{
	ComputeBox<TriangleMesh>( primitives, box );
}

extern "C" __global__ void ComputeBox_AabbList( AabbList primitives, Aabb* box ) { ComputeBox<AabbList>( primitives, box ); }

extern "C" __global__ void ComputeBox_InstanceList_SRTFrame( InstanceList<SRTFrame> primitives, Aabb* box )
{
	ComputeBox<InstanceList<SRTFrame>>( primitives, box );
}

extern "C" __global__ void ComputeBox_InstanceList_MatrixFrame( InstanceList<MatrixFrame> primitives, Aabb* box )
{
	ComputeBox<InstanceList<MatrixFrame>>( primitives, box );
}

template <typename PrimitiveContainer>
__device__ void
ComputeMortonCodes( PrimitiveContainer& primitives, Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	Aabb box = *centroidBox;

	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if ( index < primitives.getCount() )
	{
		float3 boxExtent		= box.extent();
		float3 center			= primitives.fetchCenter( index );
		float3 normalizedCenter = ( center - box.m_min ) / boxExtent;
		mortonCodeKeys[index]	= computeExtendedMortonCode( normalizedCenter, boxExtent );
		mortonCodeValues[index] = index;
	}
}

extern "C" __global__ void ComputeMortonCodes_TriangleMesh(
	TriangleMesh primitives, Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	ComputeMortonCodes<TriangleMesh>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

extern "C" __global__ void
ComputeMortonCodes_AabbList( AabbList primitives, Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	ComputeMortonCodes<AabbList>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

extern "C" __global__ void ComputeMortonCodes_InstanceList_SRTFrame(
	InstanceList<SRTFrame> primitives, Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	ComputeMortonCodes<InstanceList<SRTFrame>>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

extern "C" __global__ void ComputeMortonCodes_InstanceList_MatrixFrame(
	InstanceList<MatrixFrame> primitives, Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	ComputeMortonCodes<InstanceList<MatrixFrame>>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

extern "C" __global__ void ResetCounters( uint32_t primCount, BoxNode* boxNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if ( index < primCount ) boxNodes[index].m_updateCounter = 0;
}

template <typename InstanceList>
__device__ void ResetCountersAndUpdateFrames( InstanceList& instanceList, BoxNode* boxNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if ( index < instanceList.getCount() ) boxNodes[index].m_updateCounter = 0;
	if ( index < instanceList.getFrameCount() ) instanceList.convertFrame( index );
}

extern "C" __global__ void
ResetCountersAndUpdateFrames_InstanceList_SRTFrame( InstanceList<SRTFrame> instanceList, BoxNode* boxNodes )
{
	ResetCountersAndUpdateFrames<InstanceList<SRTFrame>>( instanceList, boxNodes );
}

extern "C" __global__ void
ResetCountersAndUpdateFrames_InstanceList_MatrixFrame( InstanceList<MatrixFrame> instanceList, BoxNode* boxNodes )
{
	ResetCountersAndUpdateFrames<InstanceList<MatrixFrame>>( instanceList, boxNodes );
}

template <typename PrimitiveContainer, typename PrimitiveNode>
__device__ void FitBounds( PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if ( index >= primitives.getCount() ) return;

	uint32_t parentAddr = primNodes[index].m_parentAddr;
	if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
	{
		primNodes[index]			  = primitives.fetchTriangleNode( index );
		primNodes[index].m_parentAddr = parentAddr;
	}

	index = parentAddr;
	while ( index != InvalidValue && atomicAdd( &boxNodes[index].m_updateCounter, 1 ) >= boxNodes[index].m_childCount - 1 )
	{
		__threadfence();

		BoxNode& node = boxNodes[index];

		if ( node.m_childIndex0 != InvalidValue )
		{
			uint32_t childAddr = node.getChildAddr( 0 );
			if ( node.getChildType( 0 ) != BoxType )
				node.m_box0 = primitives.fetchAabb( childAddr );
			else
				node.m_box0 = boxNodes[childAddr].aabb();
		}

		if ( node.m_childIndex1 != InvalidValue )
		{
			uint32_t childAddr = node.getChildAddr( 1 );
			if ( node.getChildType( 1 ) != BoxType )
				node.m_box1 = primitives.fetchAabb( childAddr );
			else
				node.m_box1 = boxNodes[childAddr].aabb();
		}

		if ( node.m_childIndex2 != InvalidValue )
		{
			uint32_t childAddr = node.getChildAddr( 2 );
			if ( node.getChildType( 2 ) != BoxType )
				node.m_box2 = primitives.fetchAabb( childAddr );
			else
				node.m_box2 = boxNodes[childAddr].aabb();
		}

		if ( node.m_childIndex3 != InvalidValue )
		{
			uint32_t childAddr = node.getChildAddr( 3 );
			if ( node.getChildType( 3 ) != BoxType )
				node.m_box3 = primitives.fetchAabb( childAddr );
			else
				node.m_box3 = boxNodes[childAddr].aabb();
		}

		index = boxNodes[index].m_parentAddr;

		__threadfence();
	}
}

extern "C" __global__ void
FitBounds_TriangleMesh_TriangleNode( TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	FitBounds<TriangleMesh, TriangleNode>( primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_AabbList_CustomNode( AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	FitBounds<AabbList, CustomNode>( primitives, boxNodes, primNodes );
}

extern "C" __global__ void
FitBounds_InstanceList_SRTFrame_InstanceNode( InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds<InstanceList<SRTFrame>, InstanceNode>( primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_InstanceList_MatrixFrame_InstanceNode(
	InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds<InstanceList<MatrixFrame>, InstanceNode>( primitives, boxNodes, primNodes );
}

template <typename PrimitiveNode, typename Header>
__device__ void BlockCollapse(
	uint32_t*	   rootAddr,
	Header*		   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	PrimitiveNode* primNodes,
	uint32_t*	   taskCounter,
	int2*		   taskQueue )
{
	__shared__ int2		taskQueueBlock[CollapseBlockSize * BranchingFactor];
	__shared__ uint32_t taskOffset;
	__shared__ uint32_t taskCount;
	__shared__ uint32_t newTaskCount;

	if ( threadIdx.x == 0 )
	{
		taskQueueBlock[0] = make_int2( rootAddr != nullptr ? *rootAddr : 0u, InvalidValue );
		taskOffset		  = 0;
		taskCount		  = 1;
		newTaskCount	  = 0;
	}
	__syncthreads();

	while ( true )
	{
		const uint32_t taskEnd = roundUp( taskCount, blockDim.x );
		for ( uint32_t taskIndex = threadIdx.x; taskIndex < taskEnd; taskIndex += blockDim.x )
		{
			int2 task;
			if ( taskIndex < taskCount ) task = taskQueueBlock[taskIndex];
			__syncthreads();

			if ( taskIndex < taskCount )
			{
				uint32_t scratchAddr = task.x;
				uint32_t parentAddr	 = task.y;

				ScratchNode scratchNode = scratchNodes[scratchAddr];

				BoxNode boxNode;
				boxNode.m_parentAddr  = parentAddr;
				boxNode.m_childIndex0 = getScratchChildIndexAndBox( 0, scratchNode, scratchNodes, references, boxNode.m_box0 );
				boxNode.m_childIndex1 = getScratchChildIndexAndBox( 1, scratchNode, scratchNodes, references, boxNode.m_box1 );

				Aabb*	  childBoxes   = &boxNode.m_box0;
				uint32_t* childIndices = &boxNode.m_childIndex0;
				for ( uint32_t i = 0; i < BranchingFactor - 2; ++i )
				{
					float	 maxArea  = 0.0f;
					uint32_t maxIndex = InvalidValue;
					for ( uint32_t j = 0; j < boxNode.m_childCount; ++j )
					{
						if ( boxNode.getChildType( j ) == BoxType )
						{
							float area = childBoxes[j].area();
							if ( area > maxArea )
							{
								maxArea	 = area;
								maxIndex = j;
							}
						}
					}

					if ( maxIndex == InvalidValue ) break;

					ScratchNode scratchChild = scratchNodes[getNodeAddr( childIndices[maxIndex] )];
					childIndices[maxIndex] =
						getScratchChildIndexAndBox( 0, scratchChild, scratchNodes, references, childBoxes[maxIndex] );
					childIndices[boxNode.m_childCount] = getScratchChildIndexAndBox(
						1, scratchChild, scratchNodes, references, childBoxes[boxNode.m_childCount] );
					++boxNode.m_childCount;
				}

				uint32_t internalCount = 0;
				for ( uint32_t i = 0; i < boxNode.m_childCount; ++i )
				{
					if ( isInternalNode( childIndices[i] ) ) ++internalCount;
				}

				uint32_t childOffset = atomicAdd( &newTaskCount, internalCount );

				const uint32_t nodeAddr = taskOffset + taskIndex;
				for ( uint32_t i = 0; i < boxNode.m_childCount; ++i )
				{
					if ( isInternalNode( childIndices[i] ) )
					{
						uint32_t newTaskIndex		 = childOffset++;
						uint32_t childAddr			 = taskOffset + taskCount + newTaskIndex;
						taskQueueBlock[newTaskIndex] = make_int2( getNodeAddr( childIndices[i] ), nodeAddr );
						boxNode.encodeChildIndex( i, childAddr, BoxType );
					}
					else if ( childIndices[i] != InvalidValue )
					{
						uint32_t childAddr				  = getNodeAddr( childIndices[i] );
						primNodes[childAddr].m_parentAddr = nodeAddr;
					}
				}
				boxNodes[nodeAddr] = boxNode;
			}
		}
		__syncthreads();

		if ( threadIdx.x == 0 )
		{
			taskOffset += taskCount;
			taskCount	 = newTaskCount;
			newTaskCount = 0;
		}
		__syncthreads();

		if ( taskCount == 0 || taskCount >= CollapseBlockSize ) break;
	}

	for ( uint32_t taskIndex = threadIdx.x; taskIndex < taskCount; taskIndex += blockDim.x )
		taskQueue[taskOffset + taskIndex] = taskQueueBlock[taskIndex];

	if ( threadIdx.x == 0 )
	{
		header->m_boxNodeCount = taskOffset + taskCount;
		*taskCounter		   = taskCount;
	}
}

extern "C" __global__ void BlockCollapse_TriangleNode(
	uint32_t*	   rootAddr,
	GeomHeader*	   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	TriangleNode*  primNodes,
	uint32_t*	   taskCounter,
	int2*		   taskQueue )
{
	BlockCollapse<TriangleNode>( rootAddr, header, scratchNodes, references, boxNodes, primNodes, taskCounter, taskQueue );
}

extern "C" __global__ void BlockCollapse_CustomNode(
	uint32_t*	   rootAddr,
	GeomHeader*	   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	CustomNode*	   primNodes,
	uint32_t*	   taskCounter,
	int2*		   taskQueue )
{
	BlockCollapse<CustomNode>( rootAddr, header, scratchNodes, references, boxNodes, primNodes, taskCounter, taskQueue );
}

extern "C" __global__ void BlockCollapse_InstanceNode(
	uint32_t*	   rootAddr,
	SceneHeader*   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	InstanceNode*  primNodes,
	uint32_t*	   taskCounter,
	int2*		   taskQueue )
{
	BlockCollapse<InstanceNode>( rootAddr, header, scratchNodes, references, boxNodes, primNodes, taskCounter, taskQueue );
}

template <typename PrimitiveNode, typename Header>
__device__ void DeviceCollapse(
	uint32_t	   index,
	uint32_t	   taskCount,
	uint32_t	   taskOffset,
	Header*		   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	PrimitiveNode* primNodes,
	int2*		   taskQueue )
{
	BoxNode	  boxNode;
	Aabb*	  childBoxes	= &boxNode.m_box0;
	uint32_t* childIndices	= &boxNode.m_childIndex0;
	uint32_t  internalCount = 0;

	const uint32_t nodeAddr = taskOffset + index;

	if ( index < taskCount )
	{
		int2	 task		 = taskQueue[nodeAddr];
		uint32_t scratchAddr = task.x;
		uint32_t parentAddr	 = task.y;

		ScratchNode scratchNode = scratchNodes[scratchAddr];

		boxNode.m_parentAddr  = parentAddr;
		boxNode.m_childIndex0 = getScratchChildIndexAndBox( 0, scratchNode, scratchNodes, references, boxNode.m_box0 );
		boxNode.m_childIndex1 = getScratchChildIndexAndBox( 1, scratchNode, scratchNodes, references, boxNode.m_box1 );

		for ( uint32_t i = 0; i < BranchingFactor - 2; ++i )
		{
			float	 maxArea  = 0.0f;
			uint32_t maxIndex = InvalidValue;
			for ( uint32_t j = 0; j < boxNode.m_childCount; ++j )
			{
				if ( boxNode.getChildType( j ) == BoxType )
				{
					float area = childBoxes[j].area();
					if ( area > maxArea )
					{
						maxArea	 = area;
						maxIndex = j;
					}
				}
			}

			if ( maxIndex == InvalidValue ) break;

			ScratchNode scratchChild = scratchNodes[getNodeAddr( childIndices[maxIndex] )];
			childIndices[maxIndex] =
				getScratchChildIndexAndBox( 0, scratchChild, scratchNodes, references, childBoxes[maxIndex] );
			childIndices[boxNode.m_childCount] =
				getScratchChildIndexAndBox( 1, scratchChild, scratchNodes, references, childBoxes[boxNode.m_childCount] );
			++boxNode.m_childCount;
		}

		for ( uint32_t i = 0; i < boxNode.m_childCount; ++i )
		{
			if ( isInternalNode( childIndices[i] ) ) ++internalCount;
		}
	}

	uint32_t childOffset = warpOffset( internalCount, &header->m_boxNodeCount );

	if ( index < taskCount )
	{
		for ( uint32_t i = 0; i < boxNode.m_childCount; ++i )
		{
			if ( isInternalNode( childIndices[i] ) )
			{
				uint32_t childAddr	 = childOffset++;
				taskQueue[childAddr] = make_int2( getNodeAddr( childIndices[i] ), nodeAddr );
				boxNode.encodeChildIndex( i, childAddr, BoxType );
			}
			else if ( childIndices[i] != InvalidValue )
			{
				uint32_t childAddr				  = getNodeAddr( childIndices[i] );
				primNodes[childAddr].m_parentAddr = nodeAddr;
			}
		}
		boxNodes[nodeAddr] = boxNode;
	}
}

extern "C" __global__ void DeviceCollapse_TriangleNode(
	uint32_t	   taskCount,
	uint32_t	   taskOffset,
	GeomHeader*	   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	TriangleNode*  primNodes,
	int2*		   taskQueue )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	DeviceCollapse<TriangleNode>(
		index, taskCount, taskOffset, header, scratchNodes, references, boxNodes, primNodes, taskQueue );
}

extern "C" __global__ void DeviceCollapse_CustomNode(
	uint32_t	   taskCount,
	uint32_t	   taskOffset,
	GeomHeader*	   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	CustomNode*	   primNodes,
	int2*		   taskQueue )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	DeviceCollapse<CustomNode>(
		index, taskCount, taskOffset, header, scratchNodes, references, boxNodes, primNodes, taskQueue );
}

extern "C" __global__ void DeviceCollapse_InstanceNode(
	uint32_t	   taskCount,
	uint32_t	   taskOffset,
	SceneHeader*   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	InstanceNode*  primNodes,
	int2*		   taskQueue )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	DeviceCollapse<InstanceNode>(
		index, taskCount, taskOffset, header, scratchNodes, references, boxNodes, primNodes, taskQueue );
}

extern "C" __global__ void ComputeCost( uint32_t nodeCount, BoxNode* boxNodes, float* costCounter )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	float cost = 0.0f;
	if ( index < nodeCount )
	{
		float rootAreaInv = 1.0f / boxNodes[0].area();
		if ( boxNodes[index].m_childIndex0 != InvalidValue )
			cost += ( boxNodes[index].getChildType( 0 ) == BoxType ? Ct : Ci ) * boxNodes[index].m_box0.area() * rootAreaInv;
		if ( boxNodes[index].m_childIndex1 != InvalidValue )
			cost += ( boxNodes[index].getChildType( 1 ) == BoxType ? Ct : Ci ) * boxNodes[index].m_box1.area() * rootAreaInv;
		if ( boxNodes[index].m_childIndex2 != InvalidValue )
			cost += ( boxNodes[index].getChildType( 2 ) == BoxType ? Ct : Ci ) * boxNodes[index].m_box2.area() * rootAreaInv;
		if ( boxNodes[index].m_childIndex3 != InvalidValue )
			cost += ( boxNodes[index].getChildType( 3 ) == BoxType ? Ct : Ci ) * boxNodes[index].m_box3.area() * rootAreaInv;
		if ( index == 0 ) cost += Ct;
	}

	constexpr uint32_t WarpsPerBlock = DivideRoundUp( BvhBuilderReductionBlockSize, WarpSize );
	__shared__ float   costCache[WarpsPerBlock];

	float blockCost = blockSum( cost, costCache );
	if ( threadIdx.x == 0 ) atomicAdd( costCounter, blockCost );
}
