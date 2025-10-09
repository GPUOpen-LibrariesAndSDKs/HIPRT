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
#include <hiprt/impl/Obb.h>
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/AabbList.h>
#include <hiprt/impl/Triangle.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/BvhBuilderUtil.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/Instance.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/MortonCode.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/TriangleMesh.h>
#include <hiprt/impl/BvhConfig.h>
using namespace hiprt;

HIPRT_DEVICE HIPRT_INLINE uint3 atomic_load( const uint3* addr )
{
	uint3 value;
#if HIPRT_RTIP >= 31
	value.x = __hip_atomic_load( &addr->x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
	value.y = __hip_atomic_load( &addr->y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
	value.z = __hip_atomic_load( &addr->z, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
#else
	value = *addr;
#endif
	return value;
}

HIPRT_DEVICE HIPRT_INLINE void atomic_store( uint3* addr, const uint3& value )
{
#if HIPRT_RTIP >= 31
	__hip_atomic_store( &addr->x, value.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
	__hip_atomic_store( &addr->y, value.y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
	__hip_atomic_store( &addr->z, value.z, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT );
#else
	*addr = value;
#endif
}

template <typename BinaryNode>
HIPRT_DEVICE HIPRT_INLINE Aabb
getNodeBox( const uint32_t nodeIndex, const BinaryNode* binaryNodes, const ReferenceNode* references )
{
	const uint32_t nodeType = getNodeType( nodeIndex );
	const uint32_t nodeAddr = getNodeAddr( nodeIndex );
	if ( nodeType == BoxType )
		return binaryNodes[nodeAddr].aabb();
	else
		return references[nodeAddr].aabb();
}

template <typename PrimitiveContainer, typename PrimitiveNode>
HIPRT_DEVICE HIPRT_INLINE Aabb
getNodeBox( const uint32_t nodeIndex, PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	const uint32_t nodeAddr = getNodeAddr( nodeIndex );
	const uint32_t nodeType = getNodeType( nodeIndex );
	if ( nodeType != BoxType )
	{
		if constexpr ( is_same<PrimitiveNode, TrianglePairNode>::value )
			return primNodes[nodeAddr].aabb();
		else if constexpr ( is_same<PrimitiveNode, TrianglePacketNode>::value )
			return primNodes[nodeAddr].aabb( typeToTriPairIndex( nodeType ) );
		else
			return primitives.fetchAabb( primNodes[nodeAddr].m_primIndex );
	}
	else
	{
		return boxNodes[nodeAddr].aabb();
	}
}

template <typename PrimitiveContainer, typename PrimitiveNode>
HIPRT_DEVICE HIPRT_INLINE Obb getNodeObb(
	const uint32_t		matrixIndex,
	const uint32_t		nodeIndex,
	const Aabb&			nodeBox,
	PrimitiveContainer& primitives,
	PrimitiveNode*		primNodes,
	Kdop*				kdops )
{
	const uint32_t nodeAddr = getNodeAddr( nodeIndex );
	const uint32_t nodeType = getNodeType( nodeIndex );
	if ( nodeType != BoxType )
	{
		if constexpr ( is_same<PrimitiveNode, TrianglePacketNode>::value )
			return primNodes[nodeAddr].obb( typeToTriPairIndex( nodeType ), matrixIndex, nodeBox );
		else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
			return primitives.fetchObb( primNodes[nodeAddr].m_primIndex, matrixIndex, nodeBox );
		else
			return Obb( matrixIndex ).grow( nodeBox );
	}
	else
	{
		return kdops[nodeAddr].obb( matrixIndex );
	}
}

template <auto IsLeafNode, typename BinaryNode>
HIPRT_DEVICE HIPRT_INLINE void openNodes(
	const BinaryNode* binaryNodes, const ReferenceNode* references, uint32_t& childCount, uint32_t& childIndex, Aabb& childBox )
{
	const uint32_t laneIndex	= threadIdx.x % WarpSize;
	const uint32_t sublaneIndex = laneIndex % BranchingFactor;
	const uint32_t subwarpIndex = laneIndex / BranchingFactor;
	const uint64_t subwarpMask	= ( ( 1 << BranchingFactor ) - 1 )
								 << static_cast<uint64_t>( ( BranchingFactor * subwarpIndex ) );

	bool done = childCount == BranchingFactor;
	while ( hiprt::ballot( !done ) )
	{
		sync_warp();

		float area = -FltMax;
		if ( !done )
		{
			if ( sublaneIndex < childCount )
			{
				if ( !IsLeafNode( childIndex ) ) area = childBox.area();
			}
		}

		float maxArea = area;
#pragma unroll
		for ( uint32_t i = 1; i < BranchingFactor; i <<= 1 )
			maxArea = hiprt::max( maxArea, shfl_xor( maxArea, i ) );
		if ( maxArea < 0.0f ) done = true;

		const uint32_t maxLaneIndex =
			__ffsll( static_cast<unsigned long long>( hiprt::ballot( maxArea == area ) ) & subwarpMask ) - 1;
		const uint32_t maxIndex		 = maxLaneIndex % BranchingFactor;
		const uint32_t maxChildIndex = shfl( childIndex, maxLaneIndex );

		if ( !done )
		{
			BinaryNode binaryChild = binaryNodes[getNodeAddr( maxChildIndex )];

			if ( sublaneIndex == maxIndex )
			{
				childIndex = binaryChild[0];
				childBox   = getNodeBox( binaryChild[0], binaryNodes, references );
			}

			if ( sublaneIndex == childCount )
			{
				childIndex = binaryChild[1];
				childBox   = getNodeBox( binaryChild[1], binaryNodes, references );
			}

			childCount++;

			if ( childCount == BranchingFactor ) done = true;
		}
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
		geomHeader->m_size			 = size;
		geomHeader->m_boxNodes		 = boxNodes;
		geomHeader->m_primNodes		 = primNodes;
		geomHeader->m_referenceCount = primCount == 1u ? 1u : 0u;
		geomHeader->m_boxNodeCount	 = 1u;
		geomHeader->m_primNodeCount	 = primCount == 1u ? 1u : 0u;
		geomHeader->m_geomType		 = geomType;
		geomHeader->m_rtip			 = Rtip;
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
		sceneHeader->m_size			  = size;
		sceneHeader->m_boxNodes		  = boxNodes;
		sceneHeader->m_primNodes	  = primNodes;
		sceneHeader->m_instances	  = instances;
		sceneHeader->m_frames		  = frames;
		sceneHeader->m_referenceCount = instanceList.getCount() == 1u ? 1u : 0u;
		sceneHeader->m_primCount	  = instanceList.getCount();
		sceneHeader->m_primNodeCount  = instanceList.getCount() == 1u ? 1u : 0u;
		sceneHeader->m_boxNodeCount	  = 1u;
		sceneHeader->m_frameCount	  = instanceList.getFrameCount();
		sceneHeader->m_rtip			  = Rtip;
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
		leafType = TriangleType;
	else if constexpr ( is_same<PrimitiveNode, CustomNode>::value )
		leafType = CustomType;
	else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
		leafType = InstanceType;

	primNodes[0] = primitives.fetchPrimNode( 0 );

	Aabb	 childBoxes[BranchingFactor];
	uint32_t childIndices[BranchingFactor];

	childBoxes[0]	= primitives.fetchAabb( 0 );
	childIndices[0] = encodeNodeIndex( 0, leafType );
	for ( uint32_t i = 1; i < BranchingFactor; ++i )
		childIndices[i] = InvalidValue;

	boxNodes[0].init( InvalidValue, 1, 0, 0, childIndices, childBoxes );
}

extern "C" __global__ void
SingletonConstruction_TriangleMesh_TrianglePairNode( TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<TriangleMesh, TriangleNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void
SingletonConstruction_TriangleMesh_TrianglePacketNode( TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
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

extern "C" __global__ void SingletonConstruction_InstanceList_SRTFrame_UserInstanceNode(
	InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<SRTFrame>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void SingletonConstruction_InstanceList_SRTFrame_HwInstanceNode(
	InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<SRTFrame>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void SingletonConstruction_InstanceList_MatrixFrame_UserInstanceNode(
	InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<MatrixFrame>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

extern "C" __global__ void SingletonConstruction_InstanceList_MatrixFrame_HwInstanceNode(
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
	uint64_t activeMask	 = hiprt::ballot( valid );

	uint3 triIndices;
	if ( valid ) triIndices = mesh.fetchTriangleIndices( index );

	while ( activeMask )
	{
		activeMask = shfl( activeMask, 0 );

		const uint64_t broadcastLane = __ffsll( static_cast<unsigned long long>( activeMask ) ) - 1;
		if ( laneIndex == broadcastLane ) valid = false;

		activeMask &= activeMask - 1;

		const uint32_t broadcastIndex	   = shfl( index, broadcastLane );
		const uint3	   triIndicesBroadcast = {
			   shfl( triIndices.x, broadcastLane ), shfl( triIndices.y, broadcastLane ), shfl( triIndices.z, broadcastLane ) };

		bool pairable = false;
		if ( index != broadcastIndex && valid )
			pairable = tryPairTriangles( triIndicesBroadcast, triIndices ).x != InvalidValue;

		const uint32_t firstPairedLane = __ffsll( static_cast<unsigned long long>( hiprt::ballot( pairable ) ) ) - 1;
		if ( firstPairedLane < WarpSize )
		{
			activeMask &= ~( 1u << firstPairedLane );
			if ( laneIndex == firstPairedLane ) valid = false;

			const uint32_t secondIndex = shfl( index, firstPairedLane );
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

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCentroidBox_TriangleMesh( TriangleMesh primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<TriangleMesh>( primitives, centroidBox );
}

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCentroidBox_AabbList( AabbList primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<AabbList>( primitives, centroidBox );
}

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCentroidBox_InstanceList_SRTFrame( InstanceList<SRTFrame> primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<InstanceList<SRTFrame>>( primitives, centroidBox );
}

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
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

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeBox_TriangleMesh( TriangleMesh primitives, Aabb* box )
{
	ComputeBox<TriangleMesh>( primitives, box );
}

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeBox_AabbList( AabbList primitives, Aabb* box )
{
	ComputeBox<AabbList>( primitives, box );
}

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeBox_InstanceList_SRTFrame( InstanceList<SRTFrame> primitives, Aabb* box )
{
	ComputeBox<InstanceList<SRTFrame>>( primitives, box );
}

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeBox_InstanceList_MatrixFrame( InstanceList<MatrixFrame> primitives, Aabb* box )
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

	if constexpr ( is_same<PrimitiveNode, TrianglePairNode>::value )
	{
		if ( index < header->m_primNodeCount )
		{
			primNodes[index] =
				primitives.fetchPrimNode( { primNodes[index].getPrimIndex( 0 ), primNodes[index].getPrimIndex( 1 ) } );
		}
	}
	else if constexpr ( is_same<PrimitiveNode, TrianglePacketNode>::value )
	{
		if ( index < header->m_primNodeCount )
		{
			uint32_t vertIndexCache[MaxVerticesPerTrianglePacket];
			uint32_t vertexCount = 0;

			PrimitiveNode  triPacketNode = primNodes[index];
			const uint32_t triPairCount	 = triPacketNode.getTrianglePairCount();
			for ( uint32_t triPairIndex = 0; triPairIndex < triPairCount; ++triPairIndex )
			{
				const uint2 pairIndices{
					triPacketNode.getPrimIndex( triPairIndex, 0 ), triPacketNode.getPrimIndex( triPairIndex, 1 ) };
				const uint3 indices0 = primitives.fetchTriangleIndices( pairIndices.x );
				uint4		indices	 = make_uint4( indices0, indices0.z );

				uint3 vertexMapping{};
				if ( pairIndices.x != pairIndices.y )
				{
					uint3 indices1 = primitives.fetchTriangleIndices( pairIndices.y );
					vertexMapping  = tryPairTriangles( indices0, indices1 );

					uint32_t vertexIndex = 0;
					if ( vertexMapping.x == 3 ) vertexIndex = indices1.x;
					if ( vertexMapping.y == 3 ) vertexIndex = indices1.y;
					if ( vertexMapping.z == 3 ) vertexIndex = indices1.z;
					indices.w = vertexIndex;
				}

				uint32_t newVertMask = 0;
				for ( uint32_t j = 0; j < 4; ++j )
				{
					if ( j == 3 && pairIndices.x == pairIndices.y ) break;

					bool contains = false;
					for ( uint32_t k = 0; k < vertexCount; ++k )
					{
						if ( vertIndexCache[k] == ( &indices.x )[j] )
						{
							contains = true;
							break;
						}
					}

					if ( !contains )
					{
						newVertMask |= 1 << j;
					}
				}

				const uint32_t oldVertCount = vertexCount;
				const uint32_t newVertCount = __popc( newVertMask );
				for ( uint32_t j = 0; j < 4; ++j )
				{
					if ( j == 3 && pairIndices.x == pairIndices.y ) break;

					bool contains = !( newVertMask & ( 1 << j ) );
					if ( !contains )
					{
						uint32_t vertexMask			= ( 1 << j ) - 1;
						uint32_t vertexIndex		= oldVertCount + __popc( newVertMask & vertexMask );
						vertIndexCache[vertexIndex] = ( &indices.x )[j];
					}
				}
				vertexCount += newVertCount;
			}

			for ( uint32_t j = 0; j < vertexCount; ++j )
			{
				const float3 vertex = primitives.fetchVertex( vertIndexCache[j] );
				triPacketNode.template writeVertex<false, true>( j, vertex );
			}

			primNodes[index] = triPacketNode;
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

extern "C" __global__ void ResetCountersAndUpdateLeaves_TriangleMesh_TrianglePairNode(
	const GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_TriangleMesh_TrianglePacketNode(
	const GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_AabbList_CustomNode(
	const GeomHeader* header, AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_InstanceList_MatrixFrame_UserInstanceNode(
	const SceneHeader* header, InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, UserInstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_InstanceList_MatrixFrame_HwInstanceNode(
	const SceneHeader* header, InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, HwInstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_InstanceList_SRTFrame_UserInstanceNode(
	const SceneHeader* header, InstanceList<SRTFrame> primitives, BoxNode* boxNodes, UserInstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void ResetCountersAndUpdateLeaves_InstanceList_SRTFrame_HwInstanceNode(
	const SceneHeader* header, InstanceList<SRTFrame> primitives, BoxNode* boxNodes, HwInstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
__device__ void FitBounds( Header* header, PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	const uint32_t threadIndex	= threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t laneIndex	= threadIdx.x % WarpSize;
	const uint32_t sublaneIndex = laneIndex % BranchingFactor;
	const uint32_t subwarpIndex = laneIndex / BranchingFactor;
	const uint64_t subwarpMask	= ( ( 1 << BranchingFactor ) - 1 )
								 << static_cast<uint64_t>( ( BranchingFactor * subwarpIndex ) );

	uint32_t index = threadIndex / BranchingFactor;

	bool done = index >= header->m_boxNodeCount;

	BoxNode	 node;
	uint32_t childCount{};
	bool	 internal = false;
	if ( !done )
	{
		node	   = boxNodes[index];
		childCount = node.getChildCount();
		internal   = sublaneIndex < childCount && node.getChildType( sublaneIndex ) == BoxType;
	}

	uint32_t internalCount = __popcll( hiprt::ballot( internal ) & subwarpMask );
	if ( internalCount > 0 ) done = true;

	while ( hiprt::any( !done ) )
	{
		__threadfence();

		Aabb	 childBox;
		uint32_t childIndex = InvalidValue;
		uint32_t childRange = InvalidValue;
		if ( !done && sublaneIndex < childCount )
		{
			childIndex = node.getChildIndex( sublaneIndex );
			childRange = node.getChildRange( sublaneIndex );
			childBox   = getNodeBox( childIndex, primitives, boxNodes, primNodes );
		}

		Aabb nodeBox = childBox;
#pragma unroll
		for ( uint32_t i = 1; i < BranchingFactor; i <<= 1 )
			nodeBox.grow( shflAabb( nodeBox, laneIndex ^ i ) );

		if ( !done )
		{
			if ( sublaneIndex < childCount )
				boxNodes[index].initBox( sublaneIndex, childCount, childIndex, childBox, nodeBox, childRange );
			index = node.getParentAddr();
			if ( index == InvalidValue ) done = true;
		}

		internal = false;
		if ( !done )
		{
			node	   = boxNodes[index];
			childCount = node.getChildCount();
			internal   = sublaneIndex < childCount && node.getChildType( sublaneIndex ) == BoxType;
		}

		internalCount = __popcll( hiprt::ballot( internal ) & subwarpMask );

		__threadfence();

		if ( !done && sublaneIndex == 0 && atomicAdd( &boxNodes[index].m_updateCounter, 1 ) < internalCount - 1 ) done = true;

		done = shfl( done, subwarpIndex * BranchingFactor );
	}
}

extern "C" __global__ void FitBounds_TriangleMesh_TrianglePairNode(
	GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TrianglePairNode* primNodes )
{
	FitBounds( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_TriangleMesh_TrianglePacketNode(
	GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TrianglePacketNode* primNodes )
{
	FitBounds( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void
FitBounds_AabbList_CustomNode( GeomHeader* header, AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	FitBounds( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_InstanceList_SRTFrame_UserInstanceNode(
	SceneHeader* header, InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_InstanceList_SRTFrame_HwInstanceNode(
	SceneHeader* header, InstanceList<SRTFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_InstanceList_MatrixFrame_UserInstanceNode(
	SceneHeader* header, InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds( header, primitives, boxNodes, primNodes );
}

extern "C" __global__ void FitBounds_InstanceList_MatrixFrame_HwInstanceNode(
	SceneHeader* header, InstanceList<MatrixFrame> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	FitBounds( header, primitives, boxNodes, primNodes );
}

// assuming that OBBs are AMD specific, there are no warp syncs
template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
__device__ void FitOrientedBounds(
	Header*				header,
	PrimitiveContainer& primitives,
	Box8Node*			boxNodes,
	PrimitiveNode*		primNodes,
	Kdop*				kdops,
	uint32_t*			updateCounters )
{
	const uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t laneIndex   = threadIdx.x % WarpSize;

	uint32_t index = threadIndex / WarpSize;

	if ( index >= header->m_boxNodeCount ) return;

	Box8Node node		= boxNodes[index];
	uint32_t childCount = node.getChildCount();
	bool	 internal	= laneIndex < childCount && node.getChildType( laneIndex ) == BoxType;

	uint32_t internalCount = __popcll( hiprt::ballot( internal ) );

	bool done = internalCount > 0;

	while ( hiprt::any( !done ) )
	{
		__threadfence();

		if ( index > 0 )
		{
			uint32_t minIndexLane = InvalidValue;
			float	 minAreaLane  = FltMax;
			for ( uint32_t j = laneIndex; j <= RotationCount; j += WarpSize )
			{
				Aabb obb;
				for ( uint32_t i = 0; i < childCount; ++i )
				{
					const Aabb	   childBox	  = node.getChildBox( i );
					const uint32_t childIndex = node.getChildIndex( i );
					obb.grow( getNodeObb( j, childIndex, childBox, primitives, primNodes, kdops ).aabb() );
				}

				kdops[index].m_boxes[j] = obb;

				if ( minAreaLane > obb.area() )
				{
					minAreaLane	 = obb.area();
					minIndexLane = j;
				}
			}

			const float	   minArea	= warpMin( minAreaLane );
			const uint32_t minIndex = __ffsll( static_cast<unsigned long long>( hiprt::ballot( minAreaLane == minArea ) ) ) - 1;
			const uint32_t matrixIndex = shfl( minIndexLane, minIndex );

			Aabb	 childBox;
			uint32_t childIndex;
			uint32_t childRange;
			if ( laneIndex < childCount )
			{
				childIndex = node.getChildIndex( laneIndex );
				childRange = node.getChildRange( laneIndex );
				childBox =
					getNodeObb( matrixIndex, childIndex, node.getChildBox( laneIndex ), primitives, primNodes, kdops ).aabb();
			}

			const Aabb nodeBox = warpUnion( childBox );

			if ( laneIndex < childCount )
			{
				boxNodes[index].initBox(
					laneIndex, childCount, childIndex, childBox, nodeBox, childRange, MatrixIndexToId[matrixIndex] );
			}

			// revert aabb if obb is not better
			if ( laneIndex == 0 )
			{
				// reconstructed quantized boxes
				float aabbArea = 0.0f;
				float obbArea  = 0.0f;
				for ( uint32_t j = 0; j < node.getChildCount(); ++j )
				{
					aabbArea += node.getChildBox( j ).area();
					obbArea += boxNodes[index].getChildBox( j ).area();
				}

				// compare to aabb surface area
				if ( aabbArea < ObbSurfaceAreaAlpha * obbArea ) boxNodes[index] = node;
			}
		}

		index = node.getParentAddr();
		if ( index == InvalidValue ) break;

		node	   = boxNodes[index];
		childCount = node.getChildCount();
		internal   = laneIndex < childCount && node.getChildType( laneIndex ) == BoxType;

		internalCount = __popcll( hiprt::ballot( internal ) );

		__threadfence();

		if ( laneIndex == 0 && atomicAdd( &updateCounters[index], 1 ) < internalCount - 1 ) done = true;

		done = shfl( done, 0 );
	}
}

extern "C" __global__ void FitOrientedBounds_TriangleMesh_TrianglePacketNode(
	GeomHeader*			header,
	TriangleMesh		primitives,
	Box8Node*			boxNodes,
	TrianglePacketNode* primNodes,
	Kdop*				kdops,
	uint32_t*			updateCounters )
{
	FitOrientedBounds( header, primitives, boxNodes, primNodes, kdops, updateCounters );
}

extern "C" __global__ void FitOrientedBounds_AabbList_CustomNode(
	GeomHeader* header, AabbList primitives, Box8Node* boxNodes, CustomNode* primNodes, Kdop* kdops, uint32_t* updateCounters )
{
	FitOrientedBounds( header, primitives, boxNodes, primNodes, kdops, updateCounters );
}

extern "C" __global__ void FitOrientedBounds_InstanceList_SRTFrame_UserInstanceNode(
	SceneHeader*		   header,
	InstanceList<SRTFrame> primitives,
	Box8Node*			   boxNodes,
	InstanceNode*		   primNodes,
	Kdop*				   kdops,
	uint32_t*			   updateCounters )
{
	FitOrientedBounds( header, primitives, boxNodes, primNodes, kdops, updateCounters );
}

extern "C" __global__ void FitOrientedBounds_InstanceList_SRTFrame_HwInstanceNode(
	SceneHeader*		   header,
	InstanceList<SRTFrame> primitives,
	Box8Node*			   boxNodes,
	InstanceNode*		   primNodes,
	Kdop*				   kdops,
	uint32_t*			   updateCounters )
{
	FitOrientedBounds( header, primitives, boxNodes, primNodes, kdops, updateCounters );
}

extern "C" __global__ void FitOrientedBounds_InstanceList_MatrixFrame_UserInstanceNode(
	SceneHeader*			  header,
	InstanceList<MatrixFrame> primitives,
	Box8Node*				  boxNodes,
	InstanceNode*			  primNodes,
	Kdop*					  kdops,
	uint32_t*				  updateCounters )
{
	FitOrientedBounds( header, primitives, boxNodes, primNodes, kdops, updateCounters );
}

extern "C" __global__ void FitOrientedBounds_InstanceList_MatrixFrame_HwInstanceNode(
	SceneHeader*			  header,
	InstanceList<MatrixFrame> primitives,
	Box8Node*				  boxNodes,
	InstanceNode*			  primNodes,
	Kdop*					  kdops,
	uint32_t*				  updateCounters )
{
	FitOrientedBounds( header, primitives, boxNodes, primNodes, kdops, updateCounters );
}

template <typename BinaryNode>
__device__ void
ComputeParentAddrs( uint32_t index, uint32_t leafCount, uint32_t rootAddr, BinaryNode* binaryNodes, uint32_t* parentAddrs )
{
	if ( index < leafCount - 1 )
	{
		BinaryNode binaryNode = binaryNodes[index];
		for ( uint32_t i = 0; i < 2; ++i )
		{
			if ( binaryNode.getChildType( i ) == BoxType ) parentAddrs[binaryNode.getChildAddr( i )] = index;
		}
		if ( index == rootAddr ) parentAddrs[rootAddr] = InvalidValue;
	}
}

extern "C" __global__ void
ComputeParentAddrs_ScratchNode( uint32_t leafCount, uint32_t rootAddr, ScratchNode* binaryNodes, uint32_t* parentAddrs )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeParentAddrs( index, leafCount, rootAddr, binaryNodes, parentAddrs );
}

extern "C" __global__ void
ComputeParentAddrs_ApiNode( uint32_t leafCount, uint32_t rootAddr, ApiNode* binaryNodes, uint32_t* parentAddrs )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeParentAddrs( index, leafCount, rootAddr, binaryNodes, parentAddrs );
}

template <typename BinaryNode>
__device__ void ComputeFatLeaves(
	uint32_t	index,
	uint32_t	leafCount,
	BinaryNode* binaryNodes,
	uint32_t*	parentAddrs,
	uint32_t*	triangleCounts,
	uint32_t*	updateCounters )
{
	if ( index >= leafCount ) return;

	if ( index >= leafCount - 1 ) return;
	BinaryNode node			 = binaryNodes[index];
	uint32_t   internalCount = 0;
	for ( uint32_t i = 0; i < 2; ++i )
	{
		if ( node.getChildType( i ) == BoxType ) internalCount++;
	}

	if ( internalCount > 0 ) return;

	while ( true )
	{
		__threadfence();

		BinaryNode& binaryNode = binaryNodes[index];

		uint32_t triangleCount = 0;
		for ( uint32_t i = 0; i < 2; ++i )
		{
			uint32_t childTriCount = 0;
			if ( binaryNode.getChildType( i ) == TriangleType )
				childTriCount = 1;
			else
				childTriCount = triangleCounts[binaryNode.getChildAddr( i )];

			if ( childTriCount <= MaxFatLeafSize ) binaryNode.setChildFatLeafFlag( i );

			triangleCount += childTriCount;
		}

		triangleCounts[index] = triangleCount;

		index = parentAddrs[index];
		if ( index == InvalidValue ) break;
		node = binaryNodes[index];

		internalCount = 0;
		for ( uint32_t i = 0; i < 2; ++i )
		{
			if ( node.getChildType( i ) == BoxType ) internalCount++;
		}

		__threadfence();

		if ( atomicAdd( &updateCounters[index], 1 ) < internalCount - 1 ) break;
	}
}

extern "C" __global__ void ComputeFatLeaves_ScratchNode(
	uint32_t leafCount, ScratchNode* binaryNodes, uint32_t* parentAddrs, uint32_t* triangleCounts, uint32_t* updateCounters )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeFatLeaves( index, leafCount, binaryNodes, parentAddrs, triangleCounts, updateCounters );
}

extern "C" __global__ void ComputeFatLeaves_ApiNode(
	uint32_t leafCount, ApiNode* binaryNodes, uint32_t* parentAddrs, uint32_t* triangleCounts, uint32_t* updateCounters )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeFatLeaves( index, leafCount, binaryNodes, parentAddrs, triangleCounts, updateCounters );
}

template <typename PrimitiveNode, typename BinaryNode, typename Header>
__device__ void Collapse(
	uint32_t	   index,
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	Header*		   header,
	BinaryNode*	   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t laneIndex	= threadIdx.x % WarpSize;
	const uint32_t taskIndex	= index / BranchingFactor;
	const uint32_t sublaneIndex = laneIndex % BranchingFactor;
	const uint32_t subwarpIndex = laneIndex / BranchingFactor;
	const uint64_t subwarpMask	= ( ( 1 << BranchingFactor ) - 1 )
								 << static_cast<uint64_t>( ( BranchingFactor * subwarpIndex ) );

	bool done = taskIndex >= maxBoxNodeCount || taskIndex >= referenceCount;

	while ( hiprt::any( !done ) )
	{
		sync_warp();
		__threadfence();

		if ( atomicAdd( &header->m_referenceCount, 0 ) == referenceCount ) done = true;

		uint32_t nodeIndex	= InvalidValue;
		uint32_t nodeAddr	= InvalidValue;
		uint32_t parentAddr = InvalidValue;
		uint3	 task		= make_uint3( InvalidValue );
		if ( !done )
		{
			task	   = atomic_load( &taskQueue[taskIndex] );
			nodeIndex  = task.x;
			nodeAddr   = task.y;
			parentAddr = task.z;
		}

		// we need to check all three values
		const bool valid = nodeIndex != InvalidValue && nodeAddr != InvalidValue && parentAddr != InvalidValue;

		// skip inactive warps
		if ( hiprt::all( !valid ) ) continue;

		Aabb	 childBox;
		uint32_t childIndex = InvalidValue;
		uint32_t childCount = 2;

		if ( nodeAddr == 0 ) parentAddr = InvalidValue;

		// fill inactive lanes with first valid node index
		const uint32_t firstValidLane = __ffsll( static_cast<unsigned long long>( hiprt::ballot( valid ) ) ) - 1;
		nodeIndex					  = shfl( nodeIndex, valid ? laneIndex : firstValidLane );

		BinaryNode binaryNode = binaryNodes[getNodeAddr( nodeIndex )];
		if ( sublaneIndex < 2 )
		{
			childIndex = binaryNode[sublaneIndex];
			childBox   = getNodeBox( binaryNode[sublaneIndex], binaryNodes, references );
		}

		// open internal nodes first
		if constexpr ( is_same<PrimitiveNode, TrianglePacketNode>::value )
			openNodes<isFatLeafNode>( binaryNodes, references, childCount, childIndex, childBox );

		// open fat leaves for the remaining slots
		openNodes<isLeafNode>( binaryNodes, references, childCount, childIndex, childBox );

		const bool active = valid && sublaneIndex < childCount;

		const bool	   internal		= isInternalNode( childIndex ) && !isFatLeafNode( childIndex );
		const uint32_t childAddr	= warpOffset( active && internal, &header->m_boxNodeCount );
		const uint32_t internalBase = shfl( childAddr, subwarpIndex * BranchingFactor );
		if ( active && internal )
		{
			atomic_store( &taskQueue[childAddr], { childIndex, childAddr, nodeAddr } );
			childIndex = encodeNodeIndex( childAddr, getNodeType( childIndex ) );
			__threadfence();
		}

		if ( valid )
		{
			if constexpr ( !is_same<PrimitiveNode, TrianglePacketNode>::value )
			{
				boxNodes[nodeAddr].init(
					sublaneIndex, parentAddr, childCount, internalBase, 0, childIndex, childBox, binaryNode.m_box );
			}
			else
			{
				const bool fatLeaf = isFatLeafNode( childIndex ) && !isLeafNode( childIndex );
				if ( fatLeaf ) childIndex = encodeNodeIndex( getNodeAddr( childIndex ), TriangleType );
				boxNodes[nodeAddr].init(
					sublaneIndex, parentAddr, childCount, internalBase, 0, childIndex, childBox, binaryNode.m_box );
				if ( fatLeaf ) childIndex = encodeNodeIndex( getNodeAddr( childIndex ), BoxType ) | FatLeafBit;
			}
		}

		task = make_uint3( InvalidValue );

		if constexpr ( !is_same<PrimitiveNode, TrianglePacketNode>::value )
		{
			const bool	   leaf			 = isLeafNode( childIndex );
			const uint64_t activeSubmask = hiprt::ballot( active && leaf ) & subwarpMask;
			const uint32_t rangeSize	 = __popcll( activeSubmask );
			const uint32_t rangeAddr	 = warpOffset( active && leaf, &header->m_referenceCount );
			if ( active && leaf ) referenceIndices[rangeAddr] = childIndex;
			if ( valid && sublaneIndex == 0 && activeSubmask != 0 ) task = { rangeAddr, nodeAddr, rangeSize };
		}
		else
		{
			uint32_t leafRangeSize = 0;
			uint32_t subtreeRefIndices[MaxFatLeafSize];

			const bool fatLeaf = isFatLeafNode( childIndex );
			if ( active && fatLeaf )
			{
				uint32_t prevLeafRangeSize = 0;
				leafRangeSize			   = 1;
				subtreeRefIndices[0]	   = childIndex;
				while ( prevLeafRangeSize != leafRangeSize )
				{
					prevLeafRangeSize = leafRangeSize;
					for ( uint32_t j = 0; j < prevLeafRangeSize; ++j )
					{
						if ( !isLeafNode( subtreeRefIndices[j] ) )
						{
							const uint32_t referenceIndex	   = subtreeRefIndices[j];
							subtreeRefIndices[j]			   = binaryNodes[getNodeAddr( referenceIndex )][0];
							subtreeRefIndices[leafRangeSize++] = binaryNodes[getNodeAddr( referenceIndex )][1];
						}
					}
				}
			}

			const uint32_t rangeBase   = warpOffset( leafRangeSize, &header->m_referenceCount );
			uint32_t	   rangeOffset = rangeBase;

			if ( active && fatLeaf )
			{
				for ( uint32_t j = 0; j < leafRangeSize; ++j )
				{
					uint32_t referenceIndex = subtreeRefIndices[j];
					referenceIndex &= ~FatLeafBit;
					referenceIndex |= j == 0 ? RangeStartBit : 0;
					referenceIndex |= j == leafRangeSize - 1 ? RangeEndBit : 0;
					referenceIndices[rangeOffset++] = referenceIndex;
				}
			}

			const uint64_t activeSubmask   = hiprt::ballot( active && fatLeaf ) & subwarpMask;
			const uint32_t lastActiveLane  = activeSubmask == 0 ? 0 : ( WarpSize - 1 ) - __clzll( activeSubmask );
			const uint32_t lastRangeOffset = shfl( rangeOffset, lastActiveLane );
			if ( valid && sublaneIndex == 0 && activeSubmask != 0 )
			{
				const uint32_t rangeSize = lastRangeOffset - rangeBase;
				task					 = { rangeBase, nodeAddr, rangeSize };
			}
		}
		sync_warp();

		if ( valid )
		{
			if ( sublaneIndex == 0 ) atomic_store( &taskQueue[taskIndex], task );
			done = true;
		}

		__threadfence();
	}
}

extern "C" __global__ void Collapse_TrianglePairNode_ScratchNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	GeomHeader*	   header,
	ScratchNode*   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<TrianglePairNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_TrianglePacketNode_ScratchNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	GeomHeader*	   header,
	ScratchNode*   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<TrianglePacketNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_CustomNode_ScratchNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	GeomHeader*	   header,
	ScratchNode*   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<CustomNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_UserInstanceNode_ScratchNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	SceneHeader*   header,
	ScratchNode*   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<UserInstanceNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_HwInstanceNode_ScratchNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	SceneHeader*   header,
	ScratchNode*   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<HwInstanceNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_TrianglePairNode_ApiNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	GeomHeader*	   header,
	ApiNode*	   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<TrianglePairNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_TrianglePacketNode_ApiNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	GeomHeader*	   header,
	ApiNode*	   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<TrianglePacketNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_CustomNode_ApiNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	GeomHeader*	   header,
	ApiNode*	   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<CustomNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_UserInstanceNode_ApiNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	SceneHeader*   header,
	ApiNode*	   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<UserInstanceNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

extern "C" __global__ void Collapse_HwInstanceNode_ApiNode(
	uint32_t	   maxBoxNodeCount,
	uint32_t	   referenceCount,
	SceneHeader*   header,
	ApiNode*	   binaryNodes,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<HwInstanceNode>(
		index, maxBoxNodeCount, referenceCount, header, binaryNodes, references, boxNodes, taskQueue, referenceIndices );
}

__device__ void CompactTasks( uint32_t index, uint32_t taskCount, uint3* taskQueue, uint32_t* taskCounter )
{
	uint3 task	= make_uint3( InvalidValue );
	bool  valid = false;
	if ( index < taskCount )
	{
		task  = taskQueue[index];
		valid = task.z != InvalidValue && task.z > 0;
	}
	__syncthreads();

	const uint32_t newIndex = warpOffset( valid, taskCounter );

	if ( valid ) taskQueue[newIndex] = task;
}

extern "C" __global__ void CompactTasks( uint32_t taskCount, uint3* taskQueue, uint32_t* taskCounter )
{
	__shared__ uint32_t newTaskCount;
	if ( threadIdx.x == 0 ) newTaskCount = 0;
	__syncthreads();

	for ( uint32_t taskIndex = threadIdx.x; taskIndex < RoundUp( taskCount, blockDim.x ); taskIndex += blockDim.x )
	{
		CompactTasks( taskIndex, taskCount, taskQueue, &newTaskCount );
		__syncthreads();
	}

	if ( threadIdx.x == 0 && blockIdx.x == 0 ) *taskCounter = newTaskCount;
}

template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
__device__ void PackLeaves(
	uint32_t			index,
	uint32_t			taskCount,
	Header*				header,
	ReferenceNode*		references,
	BoxNode*			boxNodes,
	PrimitiveNode*		primNodes,
	PrimitiveContainer& primitives,
	uint3*				taskQueue,
	uint32_t*			referenceIndices )
{
	if ( index >= taskCount ) return;

	uint3	 task			 = taskQueue[index];
	uint32_t referenceOffset = task.x;
	uint32_t nodeAddr		 = task.y;
	uint32_t leafCount		 = task.z;
	if ( leafCount == InvalidValue || leafCount == 0 ) return;

	const uint32_t primNodeBase	  = atomicAdd( &header->m_primNodeCount, leafCount );
	uint32_t	   primNodeOffset = primNodeBase;

	BoxNode& node = boxNodes[nodeAddr];
	node.setPrimNodeBase( primNodeBase );
	for ( uint32_t i = 0; i < node.getChildCount(); ++i )
	{
		uint32_t childIndex = node.getChildIndex( i );
		if ( isLeafNode( childIndex ) )
		{
			const uint32_t		referenceIndex = referenceIndices[referenceOffset++];
			const ReferenceNode reference	   = references[getNodeAddr( referenceIndex )];
			primNodes[primNodeOffset]		   = primitives.fetchPrimNode( reference.m_primIndex );
			childIndex						   = encodeNodeIndex( primNodeOffset, getNodeType( childIndex ) );
			node.patchChild( i, childIndex, 1 );
			primNodeOffset++;
		}
	}
}

template <>
__device__ void PackLeaves<TriangleMesh, TrianglePacketNode, GeomHeader>(
	uint32_t			index,
	uint32_t			taskCount,
	GeomHeader*			header,
	ReferenceNode*		references,
	BoxNode*			boxNodes,
	TrianglePacketNode* primNodes,
	TriangleMesh&		primitives,
	uint3*				taskQueue,
	uint32_t*			referenceIndices )
{
	if ( index >= taskCount ) return;

	uint3	 task		 = taskQueue[index];
	uint32_t rangeOffset = task.x;
	uint32_t nodeAddr	 = task.y;
	uint32_t rangeSize	 = task.z;
	if ( rangeSize == InvalidValue || rangeSize == 0 ) return;

	TrianglePacketCache triPacketCache;
	TrianglePairOffsets triPairOffsetCache[BranchingFactor];

	uint32_t		   primNodeCount = 1;
	TrianglePacketData packet{};

	const uint32_t rangeBase = rangeOffset;
	while ( rangeOffset < rangeBase + rangeSize )
	{
		uint32_t referenceIndex = referenceIndices[rangeOffset];
		referenceIndex &= ~RangeStartBit;
		referenceIndex &= ~RangeEndBit;

		// form triangle pair
		const ReferenceNode reference	= references[getNodeAddr( referenceIndex )];
		const uint2			pairIndices = primitives.fetchTrianglePairIndices( reference.m_primIndex );
		uint3				indices0	= primitives.fetchTriangleIndices( pairIndices.x );
		uint4				indices		= make_uint4( indices0, indices0.z );

		if ( pairIndices.x != pairIndices.y )
		{
			uint3 indices1		= primitives.fetchTriangleIndices( pairIndices.y );
			uint3 vertexMapping = tryPairTriangles( indices0, indices1 );

			uint32_t vertexIndex = 0;
			if ( vertexMapping.x == 3 ) vertexIndex = indices1.x;
			if ( vertexMapping.y == 3 ) vertexIndex = indices1.y;
			if ( vertexMapping.z == 3 ) vertexIndex = indices1.z;
			indices.w = vertexIndex;
		}

		// find new vertices
		uint32_t newVertMask = 0;
		for ( uint32_t j = 0; j < 4; ++j )
		{
			if ( j == 3 && pairIndices.x == pairIndices.y ) break;

			bool contains = false;
			for ( uint32_t k = 0; k < packet.m_vertCount; ++k )
			{
				if ( triPacketCache.m_vertexIndices[k] == ( &indices.x )[j] )
				{
					contains = true;
					break;
				}
			}

			if ( !contains ) newVertMask |= 1 << j;
		}

		const uint32_t oldVertCount = packet.m_vertCount;
		const uint32_t newVertCount = __popc( newVertMask );

		// try to fit
		if ( !packet.tryAddTrianglePair( pairIndices.x, pairIndices.y, newVertCount ) )
		{
			primNodeCount++;
			packet = TrianglePacketData{};
			continue;
		}

		// store new vertices
		for ( uint32_t j = 0; j < 4; ++j )
		{
			if ( j == 3 && pairIndices.x == pairIndices.y ) break;

			bool contains = !( newVertMask & ( 1 << j ) );
			if ( !contains )
			{
				uint32_t vertexMask							= ( 1 << j ) - 1;
				uint32_t vertexIndex						= oldVertCount + __popc( newVertMask & vertexMask );
				triPacketCache.m_vertexIndices[vertexIndex] = ( &indices.x )[j];
			}
		}

		rangeOffset++;
	}

	const uint32_t primNodeBase	  = atomicAdd( &header->m_primNodeCount, primNodeCount );
	uint32_t	   primNodeOffset = primNodeBase;

	uint32_t triPairOffset	 = 0;
	uint32_t triPacketOffset = 0;
	uint32_t leafIndex		 = 0;

	packet		= TrianglePacketData{};
	rangeOffset = rangeBase;

	while ( rangeOffset < rangeBase + rangeSize )
	{
		uint32_t   referenceIndex = referenceIndices[rangeOffset];
		const bool rangeStart	  = referenceIndex & RangeStartBit;
		const bool rangeEnd		  = referenceIndex & RangeEndBit;
		referenceIndex &= ~RangeStartBit;
		referenceIndex &= ~RangeEndBit;

		// form triangle pair
		const ReferenceNode reference	= references[getNodeAddr( referenceIndex )];
		const uint2			pairIndices = primitives.fetchTrianglePairIndices( reference.m_primIndex );
		uint3				indices0	= primitives.fetchTriangleIndices( pairIndices.x );
		uint4				indices		= make_uint4( indices0, indices0.z );

		uint3 vertexMapping{};
		if ( pairIndices.x != pairIndices.y )
		{
			uint3 indices1 = primitives.fetchTriangleIndices( pairIndices.y );
			vertexMapping  = tryPairTriangles( indices0, indices1 );

			uint32_t vertexIndex = 0;
			if ( vertexMapping.x == 3 ) vertexIndex = indices1.x;
			if ( vertexMapping.y == 3 ) vertexIndex = indices1.y;
			if ( vertexMapping.z == 3 ) vertexIndex = indices1.z;
			indices.w = vertexIndex;
		}

		// find new vertices
		uint32_t newVertMask = 0;
		uint4	 vertexIndicesInPacket{};
		for ( uint32_t j = 0; j < 4; ++j )
		{
			if ( j == 3 && pairIndices.x == pairIndices.y ) break;

			bool contains = false;
			for ( uint32_t k = 0; k < packet.m_vertCount; ++k )
			{
				if ( triPacketCache.m_vertexIndices[k] == ( &indices.x )[j] )
				{
					( &vertexIndicesInPacket.x )[j] = k;
					contains						= true;
					break;
				}
			}

			if ( !contains )
			{
				( &vertexIndicesInPacket.x )[j] = packet.m_vertCount + __popc( newVertMask );
				newVertMask |= 1 << j;
			}
		}

		const uint32_t newVertCount = __popc( newVertMask );

		// try to fit
		if ( !packet.tryAddTrianglePair( pairIndices.x, pairIndices.y, newVertCount ) )
		{
			// build packet
			TrianglePacketNode triPacketNode{};

			// write header
			TrianglePacketHeader hdr = packet.buildHeader();
			triPacketNode.writeHeader( hdr );

			// write indices & descriptors
			for ( uint32_t j = 0; j < packet.m_triPairCount; ++j )
			{
				// write indices
				triPacketNode.writePrimIndex( j, 0, hdr, triPacketCache.m_triPairData[j].m_pairIndices.x );
				triPacketNode.writePrimIndex( j, 1, hdr, triPacketCache.m_triPairData[j].m_pairIndices.y );

				// write descriptor
				triPacketNode.writeDescriptor( j, triPacketCache.m_triPairData[j].m_descriptor );
			}

			// write vertices
			for ( uint32_t j = 0; j < packet.m_vertCount; ++j )
			{
				const float3 vertex = primitives.fetchVertex( triPacketCache.m_vertexIndices[j] );
				triPacketNode.writeVertex( j, vertex );
			}

			// write packet
			primNodes[primNodeOffset++] = triPacketNode;

			packet = TrianglePacketData{};
			continue;
		}

		if ( rangeStart )
		{
			triPairOffset	= packet.m_triPairCount - 1;
			triPacketOffset = primNodeOffset;
		}

		if ( rangeEnd )
		{
			triPairOffsetCache[leafIndex++] = TrianglePairOffsets( triPairOffset, triPacketOffset );
		}

		// store new vertices
		for ( uint32_t j = 0; j < 4; ++j )
		{
			if ( j == 3 && pairIndices.x == pairIndices.y ) break;

			bool contains = !( newVertMask & ( 1 << j ) );
			if ( !contains )
			{
				uint32_t vertexIndexInPacket						= ( &vertexIndicesInPacket.x )[j];
				triPacketCache.m_vertexIndices[vertexIndexInPacket] = ( &indices.x )[j];
			}
		}

		uint3 triIndices0 = make_uint3( vertexIndicesInPacket );
		uint3 triIndices1{};
		if ( pairIndices.x != pairIndices.y )
		{
			triIndices1.x = ( &vertexIndicesInPacket.x )[vertexMapping.x];
			triIndices1.y = ( &vertexIndicesInPacket.x )[vertexMapping.y];
			triIndices1.z = ( &vertexIndicesInPacket.x )[vertexMapping.z];
		}

		// store triangle pair
		triPacketCache.m_triPairData[packet.m_triPairCount - 1] =
			TrianglePairData( pairIndices, triIndices0, triIndices1, rangeEnd );

		rangeOffset++;
	}

	// build packet
	{
		TrianglePacketNode triPacketNode{};

		// write header
		TrianglePacketHeader hdr = packet.buildHeader();
		triPacketNode.writeHeader( hdr );

		// write indices & descriptors
		for ( uint32_t j = 0; j < packet.m_triPairCount; ++j )
		{
			// write indices
			triPacketNode.writePrimIndex( j, 0, hdr, triPacketCache.m_triPairData[j].m_pairIndices.x );
			triPacketNode.writePrimIndex( j, 1, hdr, triPacketCache.m_triPairData[j].m_pairIndices.y );

			// write descriptor
			triPacketNode.writeDescriptor( j, triPacketCache.m_triPairData[j].m_descriptor );
		}

		// write vertices
		for ( uint32_t j = 0; j < packet.m_vertCount; ++j )
		{
			const float3 vertex = primitives.fetchVertex( triPacketCache.m_vertexIndices[j] );
			triPacketNode.writeVertex( j, vertex );
		}

		// write packet
		primNodes[primNodeOffset++] = triPacketNode;
	}

	// patch children
	BoxNode& node = boxNodes[nodeAddr];
	node.setPrimNodeBase( primNodeBase );
	const uint32_t leafCount = leafIndex;
	leafIndex				 = 0;
	for ( uint32_t i = 0; i < node.getChildCount(); ++i )
	{
		uint32_t childIndex = node.getChildIndex( i );
		if ( isLeafNode( childIndex ) )
		{
			uint32_t childType	   = triPairIndexToType( triPairOffsetCache[leafIndex].m_pairOffset );
			uint32_t childOffsset0 = triPairOffsetCache[leafIndex].m_packetOffset;
			uint32_t childOffsset1 = triPairOffsetCache[min( leafIndex + 1, leafCount - 1 )].m_packetOffset;
			uint32_t childRange	   = childOffsset1 - childOffsset0;
			node.patchChild( i, childType, childRange );
			leafIndex++;
		}
	}
}

__device__ void PackLeavesWarp(
	uint32_t			index,
	uint32_t			taskCount,
	GeomHeader*			header,
	ReferenceNode*		references,
	BoxNode*			boxNodes,
	TrianglePacketNode* primNodes,
	TriangleMesh&		primitives,
	uint3*				taskQueue,
	uint32_t*			referenceIndices )
{
	if ( threadIdx.x >= WarpSize ) return;

	constexpr uint32_t			   PacketTasksPerWarp = WarpSize / LanesPerLeafPacketTask;
	__shared__ TrianglePacketCache triPacketCache[PacketTasksPerWarp];
	__shared__ TrianglePairOffsets triPairOffsetCache[PacketTasksPerWarp][BranchingFactor];

	const uint32_t laneIndex	= threadIdx.x % WarpSize;
	const uint32_t taskIndex	= index / LanesPerLeafPacketTask;
	const uint32_t subwarpIndex = laneIndex / LanesPerLeafPacketTask;
	const uint32_t sublaneIndex = laneIndex % LanesPerLeafPacketTask;

	uint3	 task		 = taskQueue[min( taskIndex, taskCount - 1 )];
	uint32_t rangeOffset = task.x;
	uint32_t nodeAddr	 = task.y;
	uint32_t rangeSize	 = task.z;

	uint32_t primNodeCount = 0;

	const uint32_t rangeBase = rangeOffset;
	while ( hiprt::ballot( taskIndex < taskCount && rangeOffset < rangeBase + rangeSize ) )
	{
		TrianglePacketData packet{};

		while ( rangeOffset < rangeBase + rangeSize && packet.m_triPairCount < MaxTrianglePairsPerTrianglePacket )
		{
			uint32_t referenceIndex = referenceIndices[rangeOffset];
			referenceIndex &= ~RangeStartBit;
			referenceIndex &= ~RangeEndBit;

			// form triangle pair
			const ReferenceNode reference	= references[getNodeAddr( referenceIndex )];
			const uint2			pairIndices = primitives.fetchTrianglePairIndices( reference.m_primIndex );
			uint3				indices0	= primitives.fetchTriangleIndices( pairIndices.x );
			uint4				indices		= make_uint4( indices0, indices0.z );

			if ( pairIndices.x != pairIndices.y )
			{
				uint3 indices1		= primitives.fetchTriangleIndices( pairIndices.y );
				uint3 vertexMapping = tryPairTriangles( indices0, indices1 );

				uint32_t vertexIndex = 0;
				if ( vertexMapping.x == 3 ) vertexIndex = indices1.x;
				if ( vertexMapping.y == 3 ) vertexIndex = indices1.y;
				if ( vertexMapping.z == 3 ) vertexIndex = indices1.z;
				indices.w = vertexIndex;
			}

			// find new vertices
			const uint32_t sublaneVertIndex = laneIndex % LanesPerLeafPacketTask;
			const uint32_t subwarpVertIndex = laneIndex / LanesPerLeafPacketTask;
			const bool	   valid			= sublaneVertIndex < 3 || pairIndices.x != pairIndices.y;

			bool	 contains = false;
			uint32_t vertexIndexInPacket{};
			for ( uint32_t k = 0; k < packet.m_vertCount; ++k )
			{
				if ( triPacketCache[subwarpIndex].m_vertexIndices[k] == ( &indices.x )[sublaneVertIndex] )
				{
					vertexIndexInPacket = k;
					contains			= true;
					break;
				}
			}

			const uint32_t newVertMask =
				( hiprt::ballot( !contains && valid ) >> ( LanesPerLeafPacketTask * subwarpVertIndex ) ) & 0xf;
			const uint32_t oldVertCount = packet.m_vertCount;
			const uint32_t newVertCount = __popc( newVertMask );

			// try to fit
			if ( !packet.tryAddTrianglePair( pairIndices.x, pairIndices.y, newVertCount ) ) break;

			// store new vertices
			if ( !contains )
			{
				const uint32_t vertexMask = ( 1 << sublaneVertIndex ) - 1;
				vertexIndexInPacket		  = oldVertCount + __popc( newVertMask & vertexMask );
				triPacketCache[subwarpIndex].m_vertexIndices[vertexIndexInPacket] = ( &indices.x )[sublaneVertIndex];
			}

			rangeOffset++;
		}
		sync_warp();
		__threadfence_block();

		// count packets
		if ( taskIndex < taskCount && packet.m_triPairCount > 0 ) primNodeCount++;
		sync_warp();
	}
	sync_warp();

	const uint32_t primNodeBase =
		warpOffset( sublaneIndex == LanesPerLeafPacketTask - 1 ? primNodeCount : 0u, &header->m_primNodeCount );
	uint32_t primNodeOffset = primNodeBase;

	uint32_t triPairOffset	 = 0;
	uint32_t triPacketOffset = 0;
	uint32_t leafIndex		 = 0;

	rangeOffset = rangeBase;
	sync_warp();

	while ( hiprt::ballot( taskIndex < taskCount && rangeOffset < rangeBase + rangeSize ) )
	{
		sync_warp();
		TrianglePacketData packet{};

		while ( rangeOffset < rangeBase + rangeSize && packet.m_triPairCount < MaxTrianglePairsPerTrianglePacket )
		{
			uint32_t   referenceIndex = referenceIndices[rangeOffset];
			const bool rangeStart	  = referenceIndex & RangeStartBit;
			const bool rangeEnd		  = referenceIndex & RangeEndBit;
			referenceIndex &= ~RangeStartBit;
			referenceIndex &= ~RangeEndBit;

			// form triangle pair
			const ReferenceNode reference	= references[getNodeAddr( referenceIndex )];
			const uint2			pairIndices = primitives.fetchTrianglePairIndices( reference.m_primIndex );
			uint3				indices0	= primitives.fetchTriangleIndices( pairIndices.x );
			uint4				indices		= make_uint4( indices0, indices0.z );

			uint3 vertexMapping{};
			if ( pairIndices.x != pairIndices.y )
			{
				uint3 indices1 = primitives.fetchTriangleIndices( pairIndices.y );
				vertexMapping  = tryPairTriangles( indices0, indices1 );

				uint32_t vertexIndex = 0;
				if ( vertexMapping.x == 3 ) vertexIndex = indices1.x;
				if ( vertexMapping.y == 3 ) vertexIndex = indices1.y;
				if ( vertexMapping.z == 3 ) vertexIndex = indices1.z;
				indices.w = vertexIndex;
			}

			// find new vertices
			const uint32_t sublaneVertIndex = laneIndex % LanesPerLeafPacketTask;
			const uint32_t subwarpVertIndex = laneIndex / LanesPerLeafPacketTask;
			const bool	   valid			= sublaneVertIndex < 3 || pairIndices.x != pairIndices.y;

			bool	 contains = false;
			uint32_t vertexIndexInPacket{};
			for ( uint32_t k = 0; k < packet.m_vertCount; ++k )
			{
				if ( triPacketCache[subwarpIndex].m_vertexIndices[k] == ( &indices.x )[sublaneVertIndex] )
				{
					vertexIndexInPacket = k;
					contains			= true;
					break;
				}
			}

			const uint32_t newVertMask	= ( hiprt::ballot( !contains && valid ) >> ( 4 * subwarpVertIndex ) ) & 0xf;
			const uint32_t oldVertCount = packet.m_vertCount;
			const uint32_t newVertCount = __popc( newVertMask );

			// try to fit
			if ( !packet.tryAddTrianglePair( pairIndices.x, pairIndices.y, newVertCount ) ) break;

			if ( rangeStart )
			{
				triPairOffset	= packet.m_triPairCount - 1;
				triPacketOffset = primNodeOffset;
			}

			if ( rangeEnd )
			{
				triPairOffsetCache[subwarpIndex][leafIndex++] = TrianglePairOffsets( triPairOffset, triPacketOffset );
			}

			// store new vertices
			if ( !contains )
			{
				const uint32_t vertexMask = ( 1 << sublaneVertIndex ) - 1;
				vertexIndexInPacket		  = oldVertCount + __popc( newVertMask & vertexMask );
				triPacketCache[subwarpIndex].m_vertexIndices[vertexIndexInPacket] = ( &indices.x )[sublaneVertIndex];
			}

			// shuffle vertex indices in packet
			uint4 vertexIndicesInPacket;
			vertexIndicesInPacket.x = shfl( vertexIndexInPacket, subwarpVertIndex * LanesPerLeafPacketTask + 0 );
			vertexIndicesInPacket.y = shfl( vertexIndexInPacket, subwarpVertIndex * LanesPerLeafPacketTask + 1 );
			vertexIndicesInPacket.z = shfl( vertexIndexInPacket, subwarpVertIndex * LanesPerLeafPacketTask + 2 );
			vertexIndicesInPacket.w = shfl( vertexIndexInPacket, subwarpVertIndex * LanesPerLeafPacketTask + 3 );

			uint3 triIndices0 = make_uint3( vertexIndicesInPacket );
			uint3 triIndices1{};
			if ( pairIndices.x != pairIndices.y )
			{
				triIndices1.x = ( &vertexIndicesInPacket.x )[vertexMapping.x];
				triIndices1.y = ( &vertexIndicesInPacket.x )[vertexMapping.y];
				triIndices1.z = ( &vertexIndicesInPacket.x )[vertexMapping.z];
			}

			// store triangle pair
			if ( sublaneIndex == 0 )
				triPacketCache[subwarpIndex].m_triPairData[packet.m_triPairCount - 1] =
					TrianglePairData( pairIndices, triIndices0, triIndices1, rangeEnd );

			rangeOffset++;
			// not sure why but this fixes the issue on linux
			// otherwise triIndices1 are not correcly written to the final packet
			__threadfence_block();
		}
		sync_warp();
		__threadfence_block();

		// build packets
		uint64_t packetMask = hiprt::ballot( taskIndex < taskCount && packet.m_triPairCount > 0 && sublaneIndex == 0 );
		while ( packetMask )
		{
			const uint32_t halfWarpIndex = laneIndex / 16;
			const uint32_t halfLaneIndex = laneIndex % 16;

			const uint32_t broadcastLane0 = __ffsll( static_cast<unsigned long long>( packetMask ) ) - 1;
			packetMask ^= 1 << broadcastLane0;

			const uint32_t broadcastLane1 = __ffsll( static_cast<unsigned long long>( packetMask ) ) - 1;
			const bool	   secondValid	  = packetMask != 0;
			if ( secondValid ) packetMask ^= 1 << broadcastLane1;

			const uint32_t			 broadcastLane			 = ( halfWarpIndex == 0 ) ? broadcastLane0 : broadcastLane1;
			const uint32_t			 broadcastSubwarpIndex	 = broadcastLane / LanesPerLeafPacketTask;
			const uint32_t			 broadcastPrimNodeOffset = shfl( primNodeOffset, broadcastLane );
			const TrianglePacketData broadcastPacket		 = packet.shuffle( broadcastLane );

			// store current packet data to registers
			TrianglePairData halfLaneTriPairData;
			uint32_t		 halfLaneVertexIndex;
			if ( halfWarpIndex == 0 || secondValid )
			{
				if ( halfLaneIndex < 2 * broadcastPacket.m_triPairCount )
					halfLaneTriPairData = triPacketCache[broadcastSubwarpIndex].m_triPairData[halfLaneIndex / 2];
				if ( halfLaneIndex < broadcastPacket.m_vertCount )
					halfLaneVertexIndex = triPacketCache[broadcastSubwarpIndex].m_vertexIndices[halfLaneIndex];
			}
			sync_warp();

			// reuse shared memory
			TrianglePacketNode& triPacketNode =
				*reinterpret_cast<TrianglePacketNode*>( &triPacketCache[broadcastSubwarpIndex] );
			if ( halfWarpIndex == 0 || secondValid )
			{
				triPacketNode.m_data[halfLaneIndex + 0 * 16] = 0;
				triPacketNode.m_data[halfLaneIndex + 1 * 16] = 0;
			}
			sync_warp();

			// build two packets at once
			if ( halfWarpIndex == 0 || secondValid )
			{
				// write header
				TrianglePacketHeader hdr = broadcastPacket.buildHeader();
				if ( halfLaneIndex == 0 ) triPacketNode.writeHeader<true>( hdr );

				// write indices & descriptors
				if ( halfLaneIndex < 2 * broadcastPacket.m_triPairCount )
				{
					// write indices
					triPacketNode.writePrimIndex<true>(
						halfLaneIndex / 2,
						halfLaneIndex % 2,
						hdr,
						( &halfLaneTriPairData.m_pairIndices.x )[halfLaneIndex % 2] );

					// write descriptors
					if ( halfLaneIndex % 2 == 0 )
						triPacketNode.writeDescriptor<true>( halfLaneIndex / 2, halfLaneTriPairData.m_descriptor );
				}

				// write vertices
				if ( halfLaneIndex < broadcastPacket.m_vertCount )
				{
					const float3 vertex = primitives.fetchVertex( halfLaneVertexIndex );
					triPacketNode.writeVertex<true>( halfLaneIndex, vertex );
				}
			}
			sync_warp();

			// write packet
			if ( ( halfWarpIndex == 0 || secondValid ) )
			{
				primNodes[broadcastPrimNodeOffset].m_data[halfLaneIndex + 0 * 16] =
					triPacketNode.m_data[halfLaneIndex + 0 * 16];
				primNodes[broadcastPrimNodeOffset].m_data[halfLaneIndex + 1 * 16] =
					triPacketNode.m_data[halfLaneIndex + 1 * 16];
			}
			sync_warp();
		}
		sync_warp();

		if ( taskIndex < taskCount && packet.m_triPairCount > 0 ) primNodeOffset++;
	}
	sync_warp();

	// patch children
	if ( taskIndex < taskCount )
	{
		BoxNode& node = boxNodes[nodeAddr];
		node.setPrimNodeBase( primNodeBase );
		const uint32_t leafCount = leafIndex;
		leafIndex				 = 0;
		for ( uint32_t i = 0; i < node.getChildCount(); ++i )
		{
			uint32_t childIndex = node.getChildIndex( i );
			if ( isLeafNode( childIndex ) )
			{
				uint32_t childType	   = triPairIndexToType( triPairOffsetCache[subwarpIndex][leafIndex].m_pairOffset );
				uint32_t childOffsset0 = triPairOffsetCache[subwarpIndex][leafIndex].m_packetOffset;
				uint32_t childOffsset1 = triPairOffsetCache[subwarpIndex][min( leafIndex + 1, leafCount - 1 )].m_packetOffset;
				uint32_t childRange	   = childOffsset1 - childOffsset0;
				node.patchChild( i, childType, childRange );
				leafIndex++;
			}
		}
	}
}

extern "C" __global__ void __launch_bounds__( WarpSize ) PackLeaves_TriangleMesh_TrianglePacketNode(
	uint32_t			taskCount,
	GeomHeader*			header,
	ReferenceNode*		references,
	BoxNode*			boxNodes,
	TrianglePacketNode* primNodes,
	TriangleMesh		primitives,
	uint3*				taskQueue,
	uint32_t*			referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeavesWarp( index, taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, referenceIndices );
}

extern "C" __global__ void PackLeaves_TriangleMesh_TrianglePairNode(
	uint32_t	   taskCount,
	GeomHeader*	   header,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	TriangleNode*  primNodes,
	TriangleMesh   primitives,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, referenceIndices );
}

extern "C" __global__ void PackLeaves_AabbList_CustomNode(
	uint32_t	   taskCount,
	GeomHeader*	   header,
	ReferenceNode* references,
	BoxNode*	   boxNodes,
	CustomNode*	   primNodes,
	AabbList	   primitives,
	uint3*		   taskQueue,
	uint32_t*	   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, referenceIndices );
}

extern "C" __global__ void PackLeaves_InstanceList_SRTFrame_UserInstanceNode(
	uint32_t			   taskCount,
	SceneHeader*		   header,
	ReferenceNode*		   references,
	BoxNode*			   boxNodes,
	InstanceNode*		   primNodes,
	InstanceList<SRTFrame> primitives,
	uint3*				   taskQueue,
	uint32_t*			   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, referenceIndices );
}

extern "C" __global__ void PackLeaves_InstanceList_SRTFrame_HwInstanceNode(
	uint32_t			   taskCount,
	SceneHeader*		   header,
	ReferenceNode*		   references,
	BoxNode*			   boxNodes,
	InstanceNode*		   primNodes,
	InstanceList<SRTFrame> primitives,
	uint3*				   taskQueue,
	uint32_t*			   referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, referenceIndices );
}

extern "C" __global__ void PackLeaves_InstanceList_MatrixFrame_UserInstanceNode(
	uint32_t				  taskCount,
	SceneHeader*			  header,
	ReferenceNode*			  references,
	BoxNode*				  boxNodes,
	InstanceNode*			  primNodes,
	InstanceList<MatrixFrame> primitives,
	uint3*					  taskQueue,
	uint32_t*				  referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, referenceIndices );
}

extern "C" __global__ void PackLeaves_InstanceList_MatrixFrame_HwInstanceNode(
	uint32_t				  taskCount,
	SceneHeader*			  header,
	ReferenceNode*			  references,
	BoxNode*				  boxNodes,
	InstanceNode*			  primNodes,
	InstanceList<MatrixFrame> primitives,
	uint3*					  taskQueue,
	uint32_t*				  referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, referenceIndices );
}

template <uint32_t LeafType>
__device__ void PatchApiNodes( uint32_t index, uint32_t nodeCount, ApiNode* apiNodes )
{
	if ( index < nodeCount )
	{
		ApiNode& node = apiNodes[index];
		for ( uint32_t j = 0; j < 2; ++j )
		{
			if ( node.m_childTypes[j] == hiprtBvhNodeTypeLeaf )
				node.m_childTypes[j] = LeafType;
			else
				node.m_childTypes[j] = BoxType;
		}
	}
}

extern "C" __global__ void PatchApiNodes_TriangleMesh( uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<TriangleType>( index, nodeCount, apiNodes );
}

extern "C" __global__ void PatchApiNodes_AabbList( uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<CustomType>( index, nodeCount, apiNodes );
}

extern "C" __global__ void PatchApiNodes_InstanceList_SRTFrame( uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<InstanceType>( index, nodeCount, apiNodes );
}

extern "C" __global__ void PatchApiNodes_InstanceList_MatrixFrame( uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<InstanceType>( index, nodeCount, apiNodes );
}

extern "C" __global__ void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCost( uint32_t nodeCount, BoxNode* boxNodes, float* costCounter )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	float cost = 0.0f;
	if ( index < nodeCount )
	{
		float rootAreaInv = 1.0f / boxNodes[0].area();

		for ( uint32_t i = 0; i < boxNodes[index].getChildCount(); ++i )
			cost += ( boxNodes[index].getChildType( i ) == BoxType ? Ct : Ci ) * boxNodes[index].getChildBox( i ).area() *
					rootAreaInv;

		if ( index == 0 ) cost += Ct;
	}

	constexpr uint32_t WarpsPerBlock = DivideRoundUp( BvhBuilderReductionBlockSize, WarpSize );
	__shared__ float   costCache[WarpsPerBlock];

	float blockCost = blockSum( cost, costCache );
	if ( threadIdx.x == 0 ) atomicAdd( costCounter, blockCost );
}
