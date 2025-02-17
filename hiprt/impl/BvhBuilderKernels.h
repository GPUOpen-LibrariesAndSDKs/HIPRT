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

HIPRT_DEVICE HIPRT_INLINE Aabb
getNodeBox( uint32_t nodeIndex, const ScratchNode* scratchNodes, const ReferenceNode* references )
{
	uint32_t nodeType = getNodeType( nodeIndex );
	uint32_t nodeAddr = getNodeAddr( nodeIndex );
	if ( nodeType == BoxType )
		return scratchNodes[nodeAddr].aabb();
	else
		return references[nodeAddr].aabb();
}

template <typename PrimitiveContainer, typename PrimitiveNode>
HIPRT_DEVICE HIPRT_INLINE Aabb
getNodeBox( uint32_t nodeIndex, PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	uint32_t nodeAddr = getNodeAddr( nodeIndex );
	if ( getNodeType( nodeIndex ) != BoxType )
	{
		if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
			return primNodes[nodeAddr].aabb();
		else
			return primitives.fetchAabb( primNodes[nodeAddr].m_primIndex );
	}
	else
	{
		return boxNodes[nodeAddr].aabb();
	}
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
		geomHeader->m_boxNodeCount	= 1;
		geomHeader->m_primNodeCount = primCount == 1 ? 1 : 0;
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
	uint32_t	  index,
	size_t		  size,
	InstanceList& instanceList,
	BoxNode*	  boxNodes,
	InstanceNode* primNodes,
	Instance*	  instances,
	Frame*		  frames,
	SceneHeader*  sceneHeader )
{
	if ( index < instanceList.getCount() )
	{
		hiprtInstance		 i = instanceList.fetchInstance( index );
		hiprtTransformHeader t = instanceList.fetchTransformHeader( index );
		Instance			 instance;
		instance.m_type		  = i.type;
		instance.m_frameIndex = t.frameIndex;
		instance.m_frameCount = t.frameCount;
		if ( i.type == hiprtInstanceTypeGeometry )
			instance.m_geometry = reinterpret_cast<GeomHeader*>( i.geometry );
		else
			instance.m_scene = reinterpret_cast<SceneHeader*>( i.scene );
		instances[index] = instance;
	}

	if ( index < instanceList.getFrameCount() ) instanceList.convertFrame( index );

	if ( index == 0 )
	{
		sceneHeader->m_size			 = size;
		sceneHeader->m_boxNodes		 = boxNodes;
		sceneHeader->m_primNodes	 = primNodes;
		sceneHeader->m_instances	 = instances;
		sceneHeader->m_frames		 = frames;
		sceneHeader->m_primCount	 = instanceList.getCount();
		sceneHeader->m_primNodeCount = instanceList.getCount() == 1 ? 1 : 0;
		sceneHeader->m_boxNodeCount	 = 1;
		sceneHeader->m_frameCount	 = instanceList.getFrameCount();
	}
}

extern "C" __global__ void InitSceneData_InstanceList_SRTFrame(
	size_t				   size,
	InstanceList<SRTFrame> instanceList,
	BoxNode*			   boxNodes,
	InstanceNode*		   primNodes,
	Instance*			   instances,
	Frame*				   frames,
	SceneHeader*		   sceneHeader )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitSceneData<InstanceList<SRTFrame>>( index, size, instanceList, boxNodes, primNodes, instances, frames, sceneHeader );
}

extern "C" __global__ void InitSceneData_InstanceList_MatrixFrame(
	size_t					  size,
	InstanceList<MatrixFrame> instanceList,
	BoxNode*				  boxNodes,
	InstanceNode*			  primNodes,
	Instance*				  instances,
	Frame*					  frames,
	SceneHeader*			  sceneHeader )
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitSceneData<InstanceList<MatrixFrame>>( index, size, instanceList, boxNodes, primNodes, instances, frames, sceneHeader );
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
		primNodes[0].m_primIndex = 0;
		leafType				 = CustomType;
	}
	else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
	{
		hiprtInstance instance	 = primitives.fetchInstance( 0 );
		primNodes[0].m_primIndex = 0;
		primNodes[0].m_mask		 = primitives.fetchMask( 0 );
		primNodes[0].m_type		 = instance.type;
		primNodes[0].m_static	 = primitives.getCount() == primitives.getFrameCount() ? 1 : 0;

		if ( instance.type == hiprtInstanceTypeScene )
			primNodes[0].m_scene = reinterpret_cast<SceneHeader*>( instance.scene );
		else
			primNodes[0].m_geometry = reinterpret_cast<GeomHeader*>( instance.geometry );

		if ( primitives.getFrameCount() == 1 )
		{
			primNodes[0].m_identity = primitives.copyInvTransformMatrix( 0, primNodes[0].m_matrix ) ? 1 : 0;
		}
		else
		{
			primNodes[0].m_transform = primitives.fetchTransformHeader( 0 );
			primNodes[0].m_identity	 = 0;
		}

		leafType = InstanceType;
	}

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

extern "C" __global__ void PairTriangles( TriangleMesh mesh, uint2* pairIndices, uint32_t* pairCounter )
{
	const uint32_t index	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t laneIndex = threadIdx.x & ( WarpSize - 1 );

	bool	 valid		 = index < mesh.getCount();
	uint32_t pairedIndex = InvalidValue;
	uint64_t activeMask	 = __ballot( valid );

	uint3 triIndices;
	if ( valid ) triIndices = mesh.fetchTriangleIndices( index );

	while ( activeMask )
	{
		activeMask = __shfl( activeMask, 0 );

		const uint64_t broadcastLane = __ffsll( static_cast<unsigned long long>( activeMask ) ) - 1;
		if ( laneIndex == broadcastLane ) valid = false;

		activeMask &= activeMask - 1;

		const uint32_t broadcastIndex	   = __shfl( index, broadcastLane );
		const uint3	   triIndicesBroadcast = {
			   __shfl( triIndices.x, broadcastLane ),
			   __shfl( triIndices.y, broadcastLane ),
			   __shfl( triIndices.z, broadcastLane ) };

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
	uint32_t pairIndex = warpOffset( pairing, pairCounter );
	if ( pairing ) pairIndices[pairIndex] = make_uint2( index, pairedIndex );
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

template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
__device__ void ResetCountersAndUpdateLeaves(
	const Header* header, PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if ( index < header->m_boxNodeCount ) boxNodes[index].m_updateCounter = 0;

	if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
	{
		if ( index < header->m_primNodeCount )
		{
			primNodes[index] = primitives.fetchTriangleNode( { primNodes[index].m_primIndex0, primNodes[index].m_primIndex1 } );
		}
	}
	else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
	{
		if ( index < primitives.getFrameCount() ) primitives.convertFrame( index );

		if ( index < header->m_primNodeCount )
		{
			const uint32_t		 primIndex = primNodes[index].m_primIndex;
			hiprtTransformHeader transform = primitives.fetchTransformHeader( primIndex );
			primNodes[index].m_mask		   = primitives.fetchMask( primIndex );
			if ( transform.frameCount == 1 )
				primNodes[index].m_identity =
					primitives.copyInvTransformMatrix( transform.frameIndex, primNodes[index].m_matrix ) ? 1 : 0;
			else
				primNodes[index].m_identity = 0;
		}
	}
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_TriangleMesh_TriangleNode(
	const GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_AabbList_CustomNode(
	const GeomHeader* header, AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_InstanceList_MatrixFrame_InstanceNode(
	const SceneHeader* header, InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_InstanceList_SRTFrame_InstanceNode(
	const SceneHeader* header, InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
__device__ void FitBounds( Header* header, PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if ( index >= header->m_boxNodeCount ) return;

	BoxNode	 node		   = boxNodes[index];
	uint32_t internalCount = 0;
	for ( uint32_t i = 0; i < node.m_childCount; ++i )
	{
		if ( node.getChildType( i ) == BoxType ) internalCount++;
	}

	if ( internalCount > 0 ) return;

	while ( true )
	{
		__threadfence();

		BoxNode& node = boxNodes[index];

		if ( node.m_childIndex0 != InvalidValue )
			node.m_box0 = getNodeBox( node.m_childIndex0, primitives, boxNodes, primNodes );

		if ( node.m_childIndex1 != InvalidValue )
			node.m_box1 = getNodeBox( node.m_childIndex1, primitives, boxNodes, primNodes );

		if ( node.m_childIndex2 != InvalidValue )
			node.m_box2 = getNodeBox( node.m_childIndex2, primitives, boxNodes, primNodes );

		if ( node.m_childIndex3 != InvalidValue )
			node.m_box3 = getNodeBox( node.m_childIndex3, primitives, boxNodes, primNodes );

		internalCount = 0;
		for ( uint32_t i = 0; i < node.m_childCount; ++i )
		{
			if ( node.getChildType( i ) == BoxType ) internalCount++;
		}

		__threadfence();

		if ( atomicAdd( &node.m_updateCounter, 1 ) < internalCount - 1 ) break;
	}
}

extern "C" __global__ void
FitBounds_TriangleMesh_TriangleNode( GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	FitBounds<TriangleMesh, TriangleNode>( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void
FitBounds_AabbList_CustomNode( GeomHeader* header, AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	FitBounds<AabbList, CustomNode>( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_InstanceList_SRTFrame_InstanceNode(
	SceneHeader* header, InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds<InstanceList<SRTFrame>, InstanceNode>( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_InstanceList_MatrixFrame_InstanceNode(
	SceneHeader* header, InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds<InstanceList<MatrixFrame>, InstanceNode>( header, primitives, boxNodes, primNodes );
}

template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
__device__ void Collapse(
	uint32_t			index,
	uint32_t			leafCount,
	Header*				header,
	ScratchNode*		scratchNodes,
	ReferenceNode*		references,
	BoxNode*			boxNodes,
	PrimitiveNode*		primNodes,
	PrimitiveContainer& primitives,
	uint32_t*			taskCounter,
	uint3*				taskQueue )
{
	bool done = index >= leafCount;
	while ( __any( !done ) )
	{
		__threadfence();

		if ( done ) continue;

		uint3	 task		= taskQueue[index];
		uint32_t nodeIndex	= task.x;
		uint32_t nodeAddr	= task.y;
		uint32_t parentAddr = task.z;

		// we need to check all three values
		if ( nodeIndex != InvalidValue && nodeAddr != InvalidValue && parentAddr != InvalidValue )
		{
			if ( isInternalNode( nodeIndex ) )
			{
				if ( nodeAddr == 0 ) parentAddr = InvalidValue;

				BoxNode boxNode;
				boxNode.m_parentAddr = parentAddr;

				Aabb*	  childBoxes   = &boxNode.m_box0;
				uint32_t* childIndices = &boxNode.m_childIndex0;

				ScratchNode scratchNode = scratchNodes[getNodeAddr( nodeIndex )];
				childIndices[0]			= scratchNode.m_childIndex0;
				childIndices[1]			= scratchNode.m_childIndex1;
				childBoxes[0]			= getNodeBox( scratchNode.m_childIndex0, scratchNodes, references );
				childBoxes[1]			= getNodeBox( scratchNode.m_childIndex1, scratchNodes, references );

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

					ScratchNode scratchChild		   = scratchNodes[getNodeAddr( childIndices[maxIndex] )];
					childIndices[maxIndex]			   = scratchChild.m_childIndex0;
					childIndices[boxNode.m_childCount] = scratchChild.m_childIndex1;
					childBoxes[maxIndex]			   = getNodeBox( scratchChild.m_childIndex0, scratchNodes, references );
					childBoxes[boxNode.m_childCount]   = getNodeBox( scratchChild.m_childIndex1, scratchNodes, references );
					++boxNode.m_childCount;
				}

				uint32_t internalCount = 0;
				for ( uint32_t i = 0; i < boxNode.m_childCount; ++i )
				{
					if ( isInternalNode( childIndices[i] ) ) ++internalCount;
				}

				uint32_t taskOffset		= atomicAdd( taskCounter, boxNode.m_childCount - 1 );
				uint32_t internalOffset = atomicAdd( &header->m_boxNodeCount, internalCount );
				uint32_t leafOffset		= atomicAdd( &header->m_primNodeCount, boxNode.m_childCount - internalCount );
				for ( uint32_t i = 0; i < boxNode.m_childCount; ++i )
				{
					uint32_t childIndex = childIndices[i];
					uint32_t childAddr	= isInternalNode( childIndices[i] ) ? internalOffset++ : leafOffset++;
					childIndices[i]		= encodeNodeIndex( childAddr, getNodeType( childIndex ) );

					uint32_t taskAddr	= i == 0 ? index : taskOffset++;
					taskQueue[taskAddr] = make_uint3( childIndex, childAddr, nodeAddr );
					__threadfence();
				}

				boxNodes[nodeAddr] = boxNode;
			}
			else
			{
				const ReferenceNode reference = references[getNodeAddr( nodeIndex )];
				if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
				{
					primNodes[nodeAddr] = primitives.fetchTriangleNode( reference.m_primIndex );
				}
				else if constexpr ( is_same<PrimitiveNode, CustomNode>::value )
				{
					primNodes[nodeAddr].m_primIndex = reference.m_primIndex;
				}
				else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
				{
					hiprtInstance		 instance	= primitives.fetchInstance( reference.m_primIndex );
					hiprtTransformHeader transform	= primitives.fetchTransformHeader( reference.m_primIndex );
					primNodes[nodeAddr].m_primIndex = reference.m_primIndex;
					primNodes[nodeAddr].m_mask		= primitives.fetchMask( reference.m_primIndex );
					primNodes[nodeAddr].m_type		= instance.type;
					primNodes[nodeAddr].m_static	= transform.frameCount == 1 ? 1 : 0;

					if ( instance.type == hiprtInstanceTypeScene )
						primNodes[nodeAddr].m_scene = reinterpret_cast<SceneHeader*>( instance.scene );
					else
						primNodes[nodeAddr].m_geometry = reinterpret_cast<GeomHeader*>( instance.geometry );

					if ( transform.frameCount == 1 )
					{
						primNodes[nodeAddr].m_identity =
							primitives.copyInvTransformMatrix( transform.frameIndex, primNodes[nodeAddr].m_matrix ) ? 1 : 0;
					}
					else
					{
						primNodes[nodeAddr].m_transform = transform;
						primNodes[nodeAddr].m_identity	= 0;
					}
				}
				done = true;
			}
		}

		__threadfence();
	}
}

extern "C" __global__ void Collapse_TriangleMesh_TriangleNode(
	uint32_t	   leafCount,
	GeomHeader*	   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	TriangleNode*  primNodes,
	TriangleMesh   primitives,
	uint32_t*	   taskCounter,
	uint3*		   taskQueue )
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<TriangleMesh, TriangleNode>(
		index, leafCount, header, scratchNodes, references, boxNodes, primNodes, primitives, taskCounter, taskQueue );
}

extern "C" __global__ void Collapse_AabbList_CustomNode(
	uint32_t	   leafCount,
	GeomHeader*	   header,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	CustomNode*	   primNodes,
	AabbList	   primitives,
	uint32_t*	   taskCounter,
	uint3*		   taskQueue )
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<AabbList, CustomNode>(
		index, leafCount, header, scratchNodes, references, boxNodes, primNodes, primitives, taskCounter, taskQueue );
}

extern "C" __global__ void Collapse_InstanceList_SRTFrame_InstanceNode(
	uint32_t			   leafCount,
	SceneHeader*		   header,
	ScratchNode*		   scratchNodes,
	ReferenceNode*		   references,
	BoxNode*			   boxNodes,
	InstanceNode*		   primNodes,
	InstanceList<SRTFrame> primitives,
	uint32_t*			   taskCounter,
	uint3*				   taskQueue )
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<InstanceList<SRTFrame>, InstanceNode>(
		index, leafCount, header, scratchNodes, references, boxNodes, primNodes, primitives, taskCounter, taskQueue );
}

extern "C" __global__ void Collapse_InstanceList_MatrixFrame_InstanceNode(
	uint32_t				  leafCount,
	SceneHeader*			  header,
	ScratchNode*			  scratchNodes,
	ReferenceNode*			  references,
	BoxNode*				  boxNodes,
	InstanceNode*			  primNodes,
	InstanceList<MatrixFrame> primitives,
	uint32_t*				  taskCounter,
	uint3*					  taskQueue )
{
	uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<InstanceList<MatrixFrame>, InstanceNode>(
		index, leafCount, header, scratchNodes, references, boxNodes, primNodes, primitives, taskCounter, taskQueue );
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
