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
#include <hiprt/impl/BvhCommon.h>
using namespace hiprt;

template <typename PrimitiveNode>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE uint32_t getLeafType()
{
	if constexpr ( is_same_v<PrimitiveNode, TriangleNode> )
		return TriangleType;
	else if constexpr ( is_same_v<PrimitiveNode, CustomNode> )
		return CustomType;
	else if constexpr ( is_same_v<PrimitiveNode, InstanceNode> )
		return InstanceType;
	return InvalidType;
};

template <typename BinaryNode>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE Aabb
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
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE Aabb
getNodeBox( const uint32_t nodeIndex, PrimitiveContainer& primitives, const BoxNode* boxNodes, const PrimitiveNode* primNodes )
{
	const uint32_t nodeAddr = getNodeAddr( nodeIndex );
	const uint32_t nodeType = getNodeType( nodeIndex );
	if ( nodeType != BoxType )
	{
		if constexpr ( isTrianglePairNode<PrimitiveNode>() )
			return primNodes[nodeAddr].aabb();
		else if constexpr ( isTrianglePacketNode<PrimitiveNode>() )
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
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE Obb getNodeObb(
	const uint32_t            matrixIndex,
	const uint32_t            nodeIndex,
	const Aabb&               nodeBox,
	const PrimitiveContainer& primitives,
	const PrimitiveNode*      primNodes,
	const Kdop*               kdops )
{
	const uint32_t nodeAddr = getNodeAddr( nodeIndex );
	const uint32_t nodeType = getNodeType( nodeIndex );
	if ( nodeType != BoxType )
	{
		if constexpr ( isTrianglePacketNode<PrimitiveNode>() )
		{
			Obb obb = primNodes[nodeAddr].obb( typeToTriPairIndex( nodeType ), matrixIndex, nodeBox );
			if ( !obb.aabb().valid() ) obb = Obb( matrixIndex ).grow( nodeBox );
			return obb;
		}
		else if constexpr ( is_same_v<PrimitiveNode, InstanceNode> )
		{
			return primitives.fetchObb( primNodes[nodeAddr].m_primIndex, matrixIndex, nodeBox );
		}
		else
		{
			return Obb( matrixIndex ).grow( nodeBox );
		}
	}
	else
	{
		return kdops[nodeAddr].obb( matrixIndex );
	}
}

template <auto IsLeafNode, typename BinaryNode>
HIPRT_DEVICE HIPRT_INLINE void openNodes(
	const BinaryNode*    binaryNodes,
	const ReferenceNode* references,
	const uint32_t       maxChildCount,
	uint32_t&            childCount,
	uint32_t&            childIndex,
	Aabb&                childBox )
{
	const uint32_t laneIndex    = threadIdx.x % WarpSize;
	const uint32_t sublaneIndex = laneIndex % BoxNode::BranchingFactor;
	const uint32_t subwarpIndex = laneIndex / BoxNode::BranchingFactor;
	const uint64_t subwarpMask  = ( ( 1ull << BoxNode::BranchingFactor ) - 1ull )
	                             << static_cast<uint64_t>( ( BoxNode::BranchingFactor * subwarpIndex ) );

	bool done = childCount == maxChildCount;
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
		HIPRT_UNROLL
		for ( uint32_t i = 1; i < BoxNode::BranchingFactor; i <<= 1 )
			maxArea = hiprt::max( maxArea, shfl_xor( maxArea, i ) );
		if ( maxArea < 0.0f ) done = true;

		const uint32_t maxLaneIndex  = hiprt::ffsll( hiprt::ballot( maxArea == area ) & subwarpMask ) - 1;
		const uint32_t maxIndex      = maxLaneIndex % BoxNode::BranchingFactor;
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

			if ( childCount == maxChildCount ) done = true;
		}
	}
}

HIPRT_DEVICE HIPRT_INLINE void InitGeomDataImpl(
	const uint32_t index,
	const uint32_t primCount,
	const uint32_t geomType,
	const size_t   size,
	BoxNode*       boxNodes,
	void*          primNodes,
	GeomHeader*    geomHeader )
{
	if ( index == 0 )
	{
		geomHeader->m_size           = size;
		geomHeader->m_boxNodes       = boxNodes;
		geomHeader->m_primNodes      = primNodes;
		geomHeader->m_referenceCount = primCount == 1u ? 1u : 0u;
		geomHeader->m_boxNodeCount   = 1u;
		geomHeader->m_primNodeCount  = primCount == 1u ? 1u : 0u;
		geomHeader->m_geomType       = geomType;
		geomHeader->m_rtip           = Rtip;
	}
}

HIPRT_KERNEL void InitGeomData(
	const uint32_t primCount,
	const uint32_t geomType,
	const size_t   size,
	BoxNode*       boxNodes,
	void*          primNodes,
	GeomHeader*    geomHeader )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitGeomDataImpl( index, primCount, geomType, size, boxNodes, primNodes, geomHeader );
}

template <typename InstanceList>
HIPRT_DEVICE HIPRT_INLINE void InitSceneData(
	const uint32_t index,
	const size_t   size,
	InstanceList&  instanceList,
	BoxNode*       boxNodes,
	InstanceNode*  primNodes,
	Instance*      instances,
	Frame*         frames,
	SceneHeader*   sceneHeader )
{
	if ( index < instanceList.getCount() )
	{
		const hiprtInstance        i = instanceList.fetchInstance( index );
		const hiprtTransformHeader t = instanceList.fetchTransformHeader( index );
		Instance                   instance;
		instance.m_type       = i.type;
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
		sceneHeader->m_size           = size;
		sceneHeader->m_boxNodes       = boxNodes;
		sceneHeader->m_primNodes      = primNodes;
		sceneHeader->m_instances      = instances;
		sceneHeader->m_frames         = frames;
		sceneHeader->m_referenceCount = instanceList.getCount() == 1u ? 1u : 0u;
		sceneHeader->m_primCount      = instanceList.getCount();
		sceneHeader->m_primNodeCount  = instanceList.getCount() == 1u ? 1u : 0u;
		sceneHeader->m_boxNodeCount   = 1u;
		sceneHeader->m_frameCount     = instanceList.getFrameCount();
		sceneHeader->m_rtip           = Rtip;
	}
}

HIPRT_KERNEL void InitSceneData_InstanceList_hiprtFrameSRT(
	const size_t                size,
	InstanceList<hiprtFrameSRT> instanceList,
	BoxNode*                    boxNodes,
	InstanceNode*               primNodes,
	Instance*                   instances,
	Frame*                      frames,
	SceneHeader*                sceneHeader )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitSceneData<InstanceList<hiprtFrameSRT>>(
		index, size, instanceList, boxNodes, primNodes, instances, frames, sceneHeader );
}

HIPRT_KERNEL void InitSceneData_InstanceList_hiprtFrameMatrix(
	const size_t                   size,
	InstanceList<hiprtFrameMatrix> instanceList,
	BoxNode*                       boxNodes,
	InstanceNode*                  primNodes,
	Instance*                      instances,
	Frame*                         frames,
	SceneHeader*                   sceneHeader )
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	InitSceneData<InstanceList<hiprtFrameMatrix>>(
		index, size, instanceList, boxNodes, primNodes, instances, frames, sceneHeader );
}

template <typename PrimitiveContainer, typename PrimitiveNode>
HIPRT_DEVICE HIPRT_INLINE void
SingletonConstruction( const uint32_t index, const PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	if ( index > 0 ) return;

	primNodes[0] = primitives.template fetchPrimNode<PrimitiveNode>( 0 );

	Aabb     childBoxes[BoxNode::BranchingFactor];
	uint32_t childIndices[BoxNode::BranchingFactor];

	childBoxes[0]   = primitives.fetchAabb( 0 );
	childIndices[0] = encodeNodeIndex( 0, getLeafType<PrimitiveNode>() );
	for ( uint32_t i = 1; i < BoxNode::BranchingFactor; ++i )
		childIndices[i] = InvalidValue;

	boxNodes[0].init( InvalidValue, 1, 0, 0, childIndices, childBoxes );
}

HIPRT_KERNEL void SingletonConstruction_TriangleMesh_rtip11_TrianglePairNode(
	const TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<TriangleMesh, TriangleNode>( index, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void SingletonConstruction_TriangleMesh_rtip31_TrianglePacketNode(
	const TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<TriangleMesh, TriangleNode>( index, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void
SingletonConstruction_AabbList_CustomNode( const AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<AabbList, CustomNode>( index, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void SingletonConstruction_InstanceList_hiprtFrameSRT_rtip11_InstanceNode(
	const InstanceList<hiprtFrameSRT> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<hiprtFrameSRT>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void SingletonConstruction_InstanceList_hiprtFrameSRT_rtip31_InstanceNode(
	const InstanceList<hiprtFrameSRT> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<hiprtFrameSRT>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void SingletonConstruction_InstanceList_hiprtFrameMatrix_rtip11_InstanceNode(
	const InstanceList<hiprtFrameMatrix> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<hiprtFrameMatrix>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void SingletonConstruction_InstanceList_hiprtFrameMatrix_rtip31_InstanceNode(
	const InstanceList<hiprtFrameMatrix> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	SingletonConstruction<InstanceList<hiprtFrameMatrix>, InstanceNode>( index, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void PairTriangles( const TriangleMesh mesh, uint2* pairIndices, uint32_t* pairCounter )
{
	const uint32_t index     = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t laneIndex = threadIdx.x & ( WarpSize - 1 );

	bool     valid       = index < mesh.getCount();
	uint32_t pairedIndex = InvalidValue;
	uint64_t activeMask  = hiprt::ballot( valid );

	uint3 triIndices;
	if ( valid ) triIndices = mesh.fetchTriangleIndices( index );

	while ( activeMask )
	{
		activeMask = shfl( activeMask, 0 );

		const uint64_t broadcastLane = hiprt::ffsll( activeMask ) - 1;
		if ( laneIndex == broadcastLane ) valid = false;

		activeMask &= activeMask - 1;

		const uint32_t broadcastIndex      = shfl( index, broadcastLane );
		const uint3    triIndicesBroadcast = {
            shfl( triIndices.x, broadcastLane ), shfl( triIndices.y, broadcastLane ), shfl( triIndices.z, broadcastLane ) };

		bool pairable = false;
		if ( index != broadcastIndex && valid )
			pairable = tryPairTriangles( triIndicesBroadcast, triIndices ).x != InvalidValue;

		const uint32_t firstPairedLane = hiprt::ffsll( hiprt::ballot( pairable ) ) - 1;
		if ( firstPairedLane < WarpSize )
		{
			activeMask &= ~( 1ull << firstPairedLane );
			if ( laneIndex == firstPairedLane ) valid = false;

			const uint32_t secondIndex = shfl( index, firstPairedLane );
			if ( laneIndex == broadcastLane ) pairedIndex = secondIndex;
		}
		else if ( laneIndex == broadcastLane )
		{
			pairedIndex = index;
		}
	}

	bool     pairing   = index < mesh.getCount() && pairedIndex != InvalidValue;
	uint32_t pairIndex = warpOffset( pairing, pairCounter );
	if ( pairing ) pairIndices[pairIndex] = make_uint2( index, pairedIndex );
}

template <typename PrimitiveContainer>
HIPRT_DEVICE HIPRT_INLINE void ComputeCentroidBox( const PrimitiveContainer& primitives, Aabb* centroidBox )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	Aabb primBox;
	if ( index < primitives.getCount() )
		primBox = primitives.fetchAabb( index );
	else
		primBox = primitives.fetchAabb( primitives.getCount() - 1 );

	constexpr uint32_t                            WarpsPerBlock = DivideRoundUp( BvhBuilderReductionBlockSize, WarpSize );
	alignas( alignof( Aabb ) ) __shared__ uint8_t cache[sizeof( Aabb ) * WarpsPerBlock];
	Aabb*                                         blockBoxes = reinterpret_cast<Aabb*>( cache );

	Aabb blockBox = blockUnion( primBox, blockBoxes );
	if ( threadIdx.x == 0 ) centroidBox->atomicGrow( blockBox );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCentroidBox_TriangleMesh( const TriangleMesh primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<TriangleMesh>( primitives, centroidBox );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCentroidBox_AabbList( const AabbList primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<AabbList>( primitives, centroidBox );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCentroidBox_InstanceList_hiprtFrameSRT( const InstanceList<hiprtFrameSRT> primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<InstanceList<hiprtFrameSRT>>( primitives, centroidBox );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCentroidBox_InstanceList_hiprtFrameMatrix( const InstanceList<hiprtFrameMatrix> primitives, Aabb* centroidBox )
{
	ComputeCentroidBox<InstanceList<hiprtFrameMatrix>>( primitives, centroidBox );
}

template <typename PrimitiveContainer>
HIPRT_DEVICE HIPRT_INLINE void ComputeBox( const PrimitiveContainer& primitives, Aabb* box )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	Aabb primBox;
	if ( index < primitives.getCount() )
		primBox = primitives.fetchAabb( index );
	else
		primBox = primitives.fetchAabb( primitives.getCount() - 1 );

	constexpr uint32_t                            WarpsPerBlock = DivideRoundUp( BvhBuilderReductionBlockSize, WarpSize );
	alignas( alignof( Aabb ) ) __shared__ uint8_t cache[sizeof( Aabb ) * WarpsPerBlock];
	Aabb*                                         blockBoxes = reinterpret_cast<Aabb*>( cache );

	Aabb blockBox = blockUnion( primBox, blockBoxes );
	if ( threadIdx.x == 0 ) box->atomicGrow( blockBox );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeBox_TriangleMesh( const TriangleMesh primitives, Aabb* box )
{
	ComputeBox<TriangleMesh>( primitives, box );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize ) ComputeBox_AabbList( const AabbList primitives, Aabb* box )
{
	ComputeBox<AabbList>( primitives, box );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeBox_InstanceList_hiprtFrameSRT( const InstanceList<hiprtFrameSRT> primitives, Aabb* box )
{
	ComputeBox<InstanceList<hiprtFrameSRT>>( primitives, box );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeBox_InstanceList_hiprtFrameMatrix( const InstanceList<hiprtFrameMatrix> primitives, Aabb* box )
{
	ComputeBox<InstanceList<hiprtFrameMatrix>>( primitives, box );
}

template <typename PrimitiveContainer>
HIPRT_DEVICE HIPRT_INLINE void ComputeMortonCodes(
	const PrimitiveContainer& primitives, const Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	const Aabb box = *centroidBox;

	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	if ( index < primitives.getCount() )
	{
		const float3 boxExtent        = box.extent();
		const float3 center           = primitives.fetchCenter( index );
		const float3 normalizedCenter = ( center - box.m_min ) / boxExtent;
		mortonCodeKeys[index]         = computeExtendedMortonCode( normalizedCenter, boxExtent );
		mortonCodeValues[index]       = index;
	}
}

HIPRT_KERNEL void ComputeMortonCodes_TriangleMesh(
	const TriangleMesh primitives, const Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	ComputeMortonCodes<TriangleMesh>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

HIPRT_KERNEL void ComputeMortonCodes_AabbList(
	const AabbList primitives, const Aabb* centroidBox, uint32_t* mortonCodeKeys, uint32_t* mortonCodeValues )
{
	ComputeMortonCodes<AabbList>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

HIPRT_KERNEL void ComputeMortonCodes_InstanceList_hiprtFrameSRT(
	const InstanceList<hiprtFrameSRT> primitives,
	const Aabb*                       centroidBox,
	uint32_t*                         mortonCodeKeys,
	uint32_t*                         mortonCodeValues )
{
	ComputeMortonCodes<InstanceList<hiprtFrameSRT>>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

HIPRT_KERNEL void ComputeMortonCodes_InstanceList_hiprtFrameMatrix(
	const InstanceList<hiprtFrameMatrix> primitives,
	const Aabb*                          centroidBox,
	uint32_t*                            mortonCodeKeys,
	uint32_t*                            mortonCodeValues )
{
	ComputeMortonCodes<InstanceList<hiprtFrameMatrix>>( primitives, centroidBox, mortonCodeKeys, mortonCodeValues );
}

template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
HIPRT_DEVICE HIPRT_INLINE void ResetCountersAndUpdateLeaves(
	const Header* header, PrimitiveContainer& primitives, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if ( index < header->m_boxNodeCount ) boxNodes[index].m_updateCounter = 0;

	if constexpr ( isTrianglePairNode<PrimitiveNode>() )
	{
		if ( index < header->m_primNodeCount )
		{
			primNodes[index] = primitives.template fetchPrimNode<PrimitiveNode>(
				{ primNodes[index].getPrimIndex( 0 ), primNodes[index].getPrimIndex( 1 ) } );
		}
	}
	else if constexpr ( is_same_v<PrimitiveNode, rtip31::TrianglePacketNode> )
	{
		if ( index < header->m_primNodeCount )
		{
			// Write each vertex into the slot the BUILD assigned it, taken from the
			// packet's own pair descriptors, instead of re-deriving a vertex ordering
			// here. Re-deriving can produce more unique vertices than the build
			// reserved room for -- the vertex section only extends to
			// m_indexSectionMidpoint -- and the surplus then overwrites the primitive
			// index anchor/payload and the pair descriptors. Reusing the descriptors
			// makes the write bounded by construction: every slot referenced by a
			// descriptor is a slot the build allocated.
			PrimitiveNode  triPacketNode = primNodes[index];
			const uint32_t triPairCount  = triPacketNode.getTrianglePairCount();
			for ( uint32_t triPairIndex = 0; triPairIndex < triPairCount; ++triPairIndex )
			{
				const uint2 pairIndices{
					triPacketNode.getPrimIndex( triPairIndex, 0 ), triPacketNode.getPrimIndex( triPairIndex, 1 ) };

				for ( uint32_t t = 0; t < 2; ++t )
				{
					// A pair holding a single triangle repeats its primitive index.
					if ( t == 1 && pairIndices.x == pairIndices.y ) break;

					const uint32_t primIndex = t == 0 ? pairIndices.x : pairIndices.y;
					const uint3    indices   = primitives.fetchTriangleIndices( primIndex );
					const uint4    slots     = triPacketNode.readDescriptor( triPairIndex, t );

					triPacketNode.template writeVertex<false, true>( slots.x, primitives.fetchVertex( indices.x ) );
					triPacketNode.template writeVertex<false, true>( slots.y, primitives.fetchVertex( indices.y ) );
					triPacketNode.template writeVertex<false, true>( slots.z, primitives.fetchVertex( indices.z ) );
				}
			}

			primNodes[index] = triPacketNode;
		}
	}
	else if constexpr ( is_same_v<PrimitiveNode, InstanceNode> )
	{
		if ( index < primitives.getFrameCount() ) primitives.convertFrame( index );

		if ( index < header->m_primNodeCount )
		{
			const uint32_t       primIndex = primNodes[index].m_primIndex;
			hiprtTransformHeader transform = primitives.fetchTransformHeader( primIndex );
			primNodes[index].m_mask        = primitives.fetchMask( primIndex );
			if ( transform.frameCount == 1 )
				primNodes[index].m_identity =
					primitives.computeInvTransformMatrix( transform.frameIndex, primNodes[index].m_matrix ) ? 1 : 0;
			else
				primNodes[index].m_identity = 0;
		}
	}
}

HIPRT_KERNEL void ResetCountersAndUpdateLeaves_TriangleMesh_rtip11_TrianglePairNode(
	const GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void ResetCountersAndUpdateLeaves_TriangleMesh_rtip31_TrianglePacketNode(
	const GeomHeader* header, TriangleMesh primitives, BoxNode* boxNodes, TriangleNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void ResetCountersAndUpdateLeaves_AabbList_CustomNode(
	const GeomHeader* header, AabbList primitives, BoxNode* boxNodes, CustomNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void ResetCountersAndUpdateLeaves_InstanceList_hiprtFrameMatrix_rtip11_InstanceNode(
	const SceneHeader* header, InstanceList<hiprtFrameMatrix> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void ResetCountersAndUpdateLeaves_InstanceList_hiprtFrameMatrix_rtip31_InstanceNode(
	const SceneHeader* header, InstanceList<hiprtFrameMatrix> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void ResetCountersAndUpdateLeaves_InstanceList_hiprtFrameSRT_rtip11_InstanceNode(
	const SceneHeader* header, InstanceList<hiprtFrameSRT> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

HIPRT_KERNEL void ResetCountersAndUpdateLeaves_InstanceList_hiprtFrameSRT_rtip31_InstanceNode(
	const SceneHeader* header, InstanceList<hiprtFrameSRT> primitives, BoxNode* boxNodes, InstanceNode* primNodes )
{
	ResetCountersAndUpdateLeaves( header, primitives, boxNodes, primNodes );
}

template <typename PrimitiveContainer, typename PrimitiveNode, typename Header>
HIPRT_DEVICE HIPRT_INLINE void
FitBounds( const Header* header, const PrimitiveContainer& primitives, const PrimitiveNode* primNodes, BoxNode* boxNodes )
{
	const uint32_t threadIndex  = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t laneIndex    = threadIdx.x % WarpSize;
	const uint32_t sublaneIndex = laneIndex % BoxNode::BranchingFactor;
	const uint32_t subwarpIndex = laneIndex / BoxNode::BranchingFactor;
	const uint64_t subwarpMask  = ( ( 1ull << BoxNode::BranchingFactor ) - 1ull )
	                             << static_cast<uint64_t>( ( BoxNode::BranchingFactor * subwarpIndex ) );

	uint32_t index = threadIndex / BoxNode::BranchingFactor;

	bool done = index >= header->m_boxNodeCount;

	BoxNode  node;
	uint32_t childCount{};
	bool     internal = false;
	if ( !done )
	{
		node       = boxNodes[index];
		childCount = node.getChildCount();
		internal   = sublaneIndex < childCount && node.getChildType( sublaneIndex ) == BoxType;
	}

	uint32_t internalCount = hiprt::popcll( hiprt::ballot( internal ) & subwarpMask );
	if ( internalCount > 0 ) done = true;

	while ( hiprt::any( !done ) )
	{
		__threadfence();

		Aabb     childBox;
		uint32_t childIndex = InvalidValue;
		uint32_t childRange = InvalidValue;
		if ( !done && sublaneIndex < childCount )
		{
			childIndex = node.getChildIndex( sublaneIndex );
			childRange = node.getChildRange( sublaneIndex );
			childBox   = getNodeBox( childIndex, primitives, boxNodes, primNodes );
		}

		Aabb nodeBox = childBox;
		HIPRT_UNROLL
		for ( uint32_t i = 1; i < BoxNode::BranchingFactor; i <<= 1 )
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
			node       = boxNodes[index];
			childCount = node.getChildCount();
			internal   = sublaneIndex < childCount && node.getChildType( sublaneIndex ) == BoxType;
		}

		internalCount = hiprt::popcll( hiprt::ballot( internal ) & subwarpMask );

		__threadfence();

		if ( !done && sublaneIndex == 0 && atomicAdd( &boxNodes[index].m_updateCounter, 1 ) < internalCount - 1 ) done = true;

		done = shfl( done, subwarpIndex * BoxNode::BranchingFactor );
	}
}

HIPRT_KERNEL void FitBounds_TriangleMesh_rtip11_TrianglePairNode(
	GeomHeader* header, TriangleMesh primitives, TriangleNode* primNodes, BoxNode* boxNodes )
{
	FitBounds( header, primitives, primNodes, boxNodes );
}

HIPRT_KERNEL void FitBounds_TriangleMesh_rtip31_TrianglePacketNode(
	GeomHeader* header, TriangleMesh primitives, TriangleNode* primNodes, BoxNode* boxNodes )
{
	FitBounds( header, primitives, primNodes, boxNodes );
}

HIPRT_KERNEL void
FitBounds_AabbList_CustomNode( GeomHeader* header, AabbList primitives, CustomNode* primNodes, BoxNode* boxNodes )
{
	FitBounds( header, primitives, primNodes, boxNodes );
}

HIPRT_KERNEL void FitBounds_InstanceList_hiprtFrameSRT_rtip11_InstanceNode(
	SceneHeader* header, InstanceList<hiprtFrameSRT> primitives, InstanceNode* primNodes, BoxNode* boxNodes )
{
	FitBounds( header, primitives, primNodes, boxNodes );
}

HIPRT_KERNEL void FitBounds_InstanceList_hiprtFrameSRT_rtip31_InstanceNode(
	SceneHeader* header, InstanceList<hiprtFrameSRT> primitives, InstanceNode* primNodes, BoxNode* boxNodes )
{
	FitBounds( header, primitives, primNodes, boxNodes );
}

HIPRT_KERNEL void FitBounds_InstanceList_hiprtFrameMatrix_rtip11_InstanceNode(
	SceneHeader* header, InstanceList<hiprtFrameMatrix> primitives, InstanceNode* primNodes, BoxNode* boxNodes )
{
	FitBounds( header, primitives, primNodes, boxNodes );
}

HIPRT_KERNEL void FitBounds_InstanceList_hiprtFrameMatrix_rtip31_InstanceNode(
	SceneHeader* header, InstanceList<hiprtFrameMatrix> primitives, InstanceNode* primNodes, BoxNode* boxNodes )
{
	FitBounds( header, primitives, primNodes, boxNodes );
}

// assuming that OBBs are AMD specific, there are no warp syncs
template <typename PrimitiveContainer, typename PrimitiveNode, typename OrientedBoxNode, typename Header>
HIPRT_DEVICE HIPRT_INLINE void FitOrientedBounds(
	const Header*             header,
	const PrimitiveContainer& primitives,
	const PrimitiveNode*      primNodes,
	OrientedBoxNode*          boxNodes,
	Kdop*                     kdops,
	uint32_t*                 updateCounters )
{
	const uint32_t threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t laneIndex   = threadIdx.x % WarpSize;

	uint32_t index = threadIndex / WarpSize;

	if ( index >= header->m_boxNodeCount ) return;

	OrientedBoxNode node       = boxNodes[index];
	uint32_t        childCount = node.getChildCount();
	bool            internal   = laneIndex < childCount && node.getChildType( laneIndex ) == BoxType;

	uint32_t internalCount = hiprt::popcll( hiprt::ballot( internal ) );

	bool done = internalCount > 0;

	while ( hiprt::any( !done ) )
	{
		__threadfence();

		if ( index > 0 )
		{
			uint32_t minIndexLane = InvalidValue;
			float    minAreaLane  = FltMax;
			for ( uint32_t j = laneIndex; j <= RotationCount; j += WarpSize )
			{
				Aabb obb;
				for ( uint32_t i = 0; i < childCount; ++i )
				{
					const Aabb     childBox   = node.getChildBox( i );
					const uint32_t childIndex = node.getChildIndex( i );
					obb.grow( getNodeObb( j, childIndex, childBox, primitives, primNodes, kdops ).aabb() );
				}

				kdops[index].m_boxes[j] = obb;

				if ( minAreaLane > obb.area() )
				{
					minAreaLane  = obb.area();
					minIndexLane = j;
				}
			}

			const float    minArea     = warpMin( minAreaLane );
			const uint32_t minIndex    = hiprt::ffsll( hiprt::ballot( minAreaLane == minArea ) ) - 1;
			const uint32_t matrixIndex = shfl( minIndexLane, minIndex );

			Aabb     childBox;
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

		node       = boxNodes[index];
		childCount = node.getChildCount();
		internal   = laneIndex < childCount && node.getChildType( laneIndex ) == BoxType;

		internalCount = hiprt::popcll( hiprt::ballot( internal ) );

		__threadfence();

		if ( laneIndex == 0 && atomicAdd( &updateCounters[index], 1 ) < internalCount - 1 ) done = true;

		done = shfl( done, 0 );
	}
}

HIPRT_KERNEL void FitOrientedBounds_TriangleMesh_rtip31_TrianglePacketNode_rtip31_BoxNode(
	const GeomHeader*                 header,
	const TriangleMesh                primitives,
	const rtip31::TrianglePacketNode* primNodes,
	rtip31::BoxNode*                  boxNodes,
	Kdop*                             kdops,
	uint32_t*                         updateCounters )
{
	FitOrientedBounds( header, primitives, primNodes, boxNodes, kdops, updateCounters );
}

HIPRT_KERNEL void FitOrientedBounds_AabbList_CustomNode_rtip31_BoxNode(
	const GeomHeader* header,
	const AabbList    primitives,
	const CustomNode* primNodes,
	rtip31::BoxNode*  boxNodes,
	Kdop*             kdops,
	uint32_t*         updateCounters )
{
	FitOrientedBounds( header, primitives, primNodes, boxNodes, kdops, updateCounters );
}

HIPRT_KERNEL void FitOrientedBounds_InstanceList_hiprtFrameSRT_rtip31_InstanceNode_rtip31_BoxNode(
	const SceneHeader*                header,
	const InstanceList<hiprtFrameSRT> primitives,
	const InstanceNode*               primNodes,
	rtip31::BoxNode*                  boxNodes,
	Kdop*                             kdops,
	uint32_t*                         updateCounters )
{
	FitOrientedBounds( header, primitives, primNodes, boxNodes, kdops, updateCounters );
}

HIPRT_KERNEL void FitOrientedBounds_InstanceList_hiprtFrameMatrix_rtip31_InstanceNode_rtip31_BoxNode(
	const SceneHeader*                   header,
	const InstanceList<hiprtFrameMatrix> primitives,
	const InstanceNode*                  primNodes,
	rtip31::BoxNode*                     boxNodes,
	Kdop*                                kdops,
	uint32_t*                            updateCounters )
{
	FitOrientedBounds( header, primitives, primNodes, boxNodes, kdops, updateCounters );
}

template <typename BinaryNode>
HIPRT_DEVICE HIPRT_INLINE void ComputeParentAddrs(
	const uint32_t    index,
	const uint32_t    leafCount,
	const uint32_t    rootAddr,
	const BinaryNode* binaryNodes,
	uint32_t*         parentAddrs )
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

HIPRT_KERNEL void ComputeParentAddrs_ScratchNode(
	const uint32_t leafCount, const uint32_t rootAddr, const ScratchNode* binaryNodes, uint32_t* parentAddrs )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeParentAddrs( index, leafCount, rootAddr, binaryNodes, parentAddrs );
}

HIPRT_KERNEL void ComputeParentAddrs_ApiNode(
	const uint32_t leafCount, const uint32_t rootAddr, const ApiNode* binaryNodes, uint32_t* parentAddrs )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeParentAddrs( index, leafCount, rootAddr, binaryNodes, parentAddrs );
}

template <typename BinaryNode>
HIPRT_DEVICE HIPRT_INLINE void ComputeFatLeaves(
	const uint32_t  index,
	const uint32_t  leafCount,
	const uint32_t* parentAddrs,
	BinaryNode*     binaryNodes,
	uint32_t*       triangleCounts,
	uint32_t*       updateCounters )
{
	uint32_t nodeIndex = index;

	if ( nodeIndex >= leafCount ) return;

	if ( nodeIndex >= leafCount - 1 ) return;
	BinaryNode node          = binaryNodes[nodeIndex];
	uint32_t   internalCount = 0;
	for ( uint32_t i = 0; i < 2; ++i )
	{
		if ( node.getChildType( i ) == BoxType ) internalCount++;
	}

	if ( internalCount > 0 ) return;

	while ( true )
	{
		__threadfence();

		BinaryNode& binaryNode = binaryNodes[nodeIndex];

		uint32_t triangleCount = 0;
		for ( uint32_t i = 0; i < 2; ++i )
		{
			uint32_t childTriCount = 0;
			if ( binaryNode.getChildType( i ) == TriangleType )
				childTriCount = 1;
			else
				childTriCount = triangleCounts[binaryNode.getChildAddr( i )];

			if ( childTriCount <= getMaxFatLeafSize( Rtip ) ) binaryNode.setChildFatLeafFlag( i );

			triangleCount += childTriCount;
		}

		triangleCounts[nodeIndex] = triangleCount;

		nodeIndex = parentAddrs[nodeIndex];
		if ( nodeIndex == InvalidValue ) break;
		node = binaryNodes[nodeIndex];

		internalCount = 0;
		for ( uint32_t i = 0; i < 2; ++i )
		{
			if ( node.getChildType( i ) == BoxType ) internalCount++;
		}

		__threadfence();

		if ( atomicAdd( &updateCounters[nodeIndex], 1 ) < internalCount - 1 ) break;
	}
}

HIPRT_KERNEL void ComputeFatLeaves_ScratchNode(
	const uint32_t  leafCount,
	const uint32_t* parentAddrs,
	ScratchNode*    binaryNodes,
	uint32_t*       triangleCounts,
	uint32_t*       updateCounters )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeFatLeaves( index, leafCount, parentAddrs, binaryNodes, triangleCounts, updateCounters );
}

HIPRT_KERNEL void ComputeFatLeaves_ApiNode(
	const uint32_t  leafCount,
	const uint32_t* parentAddrs,
	ApiNode*        binaryNodes,
	uint32_t*       triangleCounts,
	uint32_t*       updateCounters )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	ComputeFatLeaves( index, leafCount, parentAddrs, binaryNodes, triangleCounts, updateCounters );
}

template <typename PrimitiveNode, typename BinaryNode, typename Header>
HIPRT_DEVICE HIPRT_INLINE void Collapse(
	const uint32_t       index,
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const BinaryNode*    binaryNodes,
	const ReferenceNode* references,
	Header*              header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t laneIndex    = threadIdx.x % WarpSize;
	const uint32_t taskIndex    = index / BoxNode::BranchingFactor;
	const uint32_t sublaneIndex = laneIndex % BoxNode::BranchingFactor;
	const uint32_t subwarpIndex = laneIndex / BoxNode::BranchingFactor;
	const uint64_t subwarpShift = static_cast<uint64_t>( ( BoxNode::BranchingFactor * subwarpIndex ) );
	const uint64_t subwarpMask  = ( ( 1ull << BoxNode::BranchingFactor ) - 1ull ) << subwarpShift;

	bool done = taskIndex >= maxBoxNodeCount || taskIndex >= referenceCount;

	while ( hiprt::any( !done ) )
	{
		sync_warp();
		__threadfence();

		if ( atomicAdd( &header->m_referenceCount, 0 ) == referenceCount ) done = true;

		uint32_t       nodeIndex  = InvalidValue;
		uint32_t       nodeAddr   = InvalidValue;
		uint32_t       parentAddr = InvalidValue;
		ConversionTask task{};
		if ( !done )
		{
			task       = taskQueue[taskIndex].atomic_load();
			nodeIndex  = task.m_nodeIndex;
			nodeAddr   = task.m_nodeAddr;
			parentAddr = task.m_parentAddr;
		}

		// we need to check all three values
		const bool valid = task.valid();

		// skip inactive warps
		if ( hiprt::all( !valid ) ) continue;

		Aabb     childBox;
		uint32_t childIndex = InvalidValue;
		uint32_t childCount = 2;

		if ( nodeAddr == 0 ) parentAddr = InvalidValue;

		// fill inactive lanes with first valid node index
		const uint32_t firstValidLane = hiprt::ffsll( hiprt::ballot( valid ) ) - 1;
		nodeIndex                     = shfl( nodeIndex, valid ? laneIndex : firstValidLane );

		BinaryNode binaryNode = binaryNodes[getNodeAddr( nodeIndex )];
		if ( sublaneIndex < 2 )
		{
			childIndex = binaryNode[sublaneIndex];
			childBox   = getNodeBox( binaryNode[sublaneIndex], binaryNodes, references );
		}

		// open internal nodes first
		const uint32_t maxChildCount = nodeAddr == 0 ? InstanceNode::BranchingFactor : BoxNode::BranchingFactor;
		if constexpr ( isTrianglePacketNode<PrimitiveNode>() )
			openNodes<isFatLeafNode>( binaryNodes, references, maxChildCount, childCount, childIndex, childBox );

		// open fat leaves for the remaining slots
		openNodes<isLeafNode>( binaryNodes, references, maxChildCount, childCount, childIndex, childBox );

		const bool active = valid && sublaneIndex < childCount;

		const bool     internal     = isInternalNode( childIndex ) && !isFatLeafNode( childIndex );
		const uint32_t childAddr    = warpOffset( active && internal, &header->m_boxNodeCount );
		const uint32_t internalBase = shfl( childAddr, subwarpIndex * BoxNode::BranchingFactor );
		if ( active && internal )
		{
			taskQueue[childAddr].atomic_store( ConversionTask( childIndex, childAddr, nodeAddr ) );
			childIndex = encodeNodeIndex( childAddr, getNodeType( childIndex ) );
			__threadfence();
		}

		const bool fatLeaf  = isFatLeafNode( childIndex ) && !isLeafNode( childIndex );
		const bool leaf     = isLeafNode( childIndex );
		uint32_t   leafMask = InvalidValue;
		{
			const uint64_t activeSubmask = hiprt::ballot( active && ( leaf || fatLeaf ) ) & subwarpMask;
			leafMask                     = static_cast<uint32_t>( activeSubmask >> subwarpShift );
		}

		if ( valid )
		{
			const uint32_t leafType = getLeafType<PrimitiveNode>();
			if constexpr ( !isTrianglePacketNode<PrimitiveNode>() )
			{
				boxNodes[nodeAddr].init(
					sublaneIndex,
					parentAddr,
					childCount,
					leafType,
					leafMask,
					internalBase,
					0,
					childIndex,
					childBox,
					binaryNode.m_box );
			}
			else
			{
				if ( fatLeaf ) childIndex = encodeNodeIndex( getNodeAddr( childIndex ), TriangleType );
				boxNodes[nodeAddr].init(
					sublaneIndex,
					parentAddr,
					childCount,
					leafType,
					leafMask,
					internalBase,
					0,
					childIndex,
					childBox,
					binaryNode.m_box );
				if ( fatLeaf ) childIndex = encodeNodeIndex( getNodeAddr( childIndex ), BoxType ) | FatLeafBit;
			}
		}

		task = ConversionTask{};

		if constexpr ( !isTrianglePacketNode<PrimitiveNode>() )
		{
			const uint64_t activeSubmask = hiprt::ballot( active && leaf ) & subwarpMask;
			const uint32_t rangeSize     = hiprt::popcll( activeSubmask );
			const uint32_t rangeOffset   = warpOffset( active && leaf, &header->m_referenceCount );
			if ( active && leaf ) referenceIndices[rangeOffset] = childIndex;
			if ( valid && sublaneIndex == 0 && activeSubmask != 0 ) task = ConversionTask( rangeOffset, nodeAddr, rangeSize );
		}
		else
		{
			uint32_t leafRangeSize = 0;
			uint32_t subtreeRefIndices[PrimitiveNode::MaxFatLeafSize];

			const bool fatLeaf = isFatLeafNode( childIndex );
			if ( active && fatLeaf )
			{
				uint32_t prevLeafRangeSize = 0;
				leafRangeSize              = 1;
				subtreeRefIndices[0]       = childIndex;
				while ( prevLeafRangeSize != leafRangeSize )
				{
					prevLeafRangeSize = leafRangeSize;
					for ( uint32_t j = 0; j < prevLeafRangeSize; ++j )
					{
						if ( !isLeafNode( subtreeRefIndices[j] ) )
						{
							const uint32_t referenceIndex      = subtreeRefIndices[j];
							subtreeRefIndices[j]               = binaryNodes[getNodeAddr( referenceIndex )][0];
							subtreeRefIndices[leafRangeSize++] = binaryNodes[getNodeAddr( referenceIndex )][1];
						}
					}
				}
			}

			const uint32_t rangeBase   = warpOffset( leafRangeSize, &header->m_referenceCount );
			uint32_t       rangeOffset = rangeBase;

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
				task                     = ConversionTask( rangeBase, nodeAddr, rangeSize );
			}
		}
		sync_warp();

		if ( valid )
		{
			if ( sublaneIndex == 0 ) taskQueue[taskIndex].atomic_store( task );
			done = true;
		}

		__threadfence();
	}
}

HIPRT_KERNEL void Collapse_rtip11_TrianglePairNode_ScratchNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ScratchNode*   binaryNodes,
	const ReferenceNode* references,
	GeomHeader*          header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip11::TrianglePairNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_rtip31_TrianglePacketNode_ScratchNode(
	const uint32_t maxBoxNodeCount,
	const uint32_t referenceCount,

	const ScratchNode*   binaryNodes,
	const ReferenceNode* references,
	GeomHeader*          header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip31::TrianglePacketNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_CustomNode_ScratchNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ScratchNode*   binaryNodes,
	const ReferenceNode* references,
	GeomHeader*          header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<CustomNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_rtip11_InstanceNode_ScratchNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ScratchNode*   binaryNodes,
	const ReferenceNode* references,
	SceneHeader*         header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip11::InstanceNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_rtip31_InstanceNode_ScratchNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ScratchNode*   binaryNodes,
	const ReferenceNode* references,
	SceneHeader*         header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip31::InstanceNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_rtip11_TrianglePairNode_ApiNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ApiNode*       binaryNodes,
	const ReferenceNode* references,
	GeomHeader*          header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip11::TrianglePairNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_rtip31_TrianglePacketNode_ApiNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ApiNode*       binaryNodes,
	const ReferenceNode* references,
	GeomHeader*          header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip31::TrianglePacketNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_CustomNode_ApiNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ApiNode*       binaryNodes,
	const ReferenceNode* references,
	GeomHeader*          header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<CustomNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_rtip11_InstanceNode_ApiNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ApiNode*       binaryNodes,
	const ReferenceNode* references,
	SceneHeader*         header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip11::InstanceNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_KERNEL void Collapse_rtip31_InstanceNode_ApiNode(
	const uint32_t       maxBoxNodeCount,
	const uint32_t       referenceCount,
	const ApiNode*       binaryNodes,
	const ReferenceNode* references,
	SceneHeader*         header,
	BoxNode*             boxNodes,
	ConversionTask*      taskQueue,
	uint32_t*            referenceIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	Collapse<rtip31::InstanceNode>(
		index, maxBoxNodeCount, referenceCount, binaryNodes, references, header, boxNodes, taskQueue, referenceIndices );
}

HIPRT_DEVICE HIPRT_INLINE void
CompactTasks( const uint32_t index, const uint32_t taskCount, ConversionTask* taskQueue, uint32_t* taskCounter )
{
	ConversionTask task{};
	bool           valid = false;
	if ( index < taskCount )
	{
		task  = taskQueue[index];
		valid = task.m_rangeSize != InvalidValue && task.m_rangeSize > 0;
	}
	__syncthreads();

	const uint32_t newIndex = warpOffset( valid, taskCounter );

	if ( valid ) taskQueue[newIndex] = task;
}

HIPRT_KERNEL void CompactTasks( const uint32_t taskCount, ConversionTask* taskQueue, uint32_t* taskCounter )
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
HIPRT_DEVICE HIPRT_INLINE void PackLeaves(
	const uint32_t            index,
	const uint32_t            taskCount,
	const ConversionTask*     taskQueue,
	const uint32_t*           referenceIndices,
	const ReferenceNode*      references,
	const PrimitiveContainer& primitives,
	Header*                   header,
	BoxNode*                  boxNodes,
	PrimitiveNode*            primNodes )
{
	if ( index >= taskCount ) return;

	const ConversionTask task        = taskQueue[index];
	const uint32_t       nodeAddr    = task.m_nodeAddr;
	const uint32_t       rangeSize   = task.m_rangeSize;
	uint32_t             rangeOffset = task.m_rangeOffset;
	if ( rangeSize == InvalidValue || rangeSize == 0 ) return;

	const uint32_t primNodeBase   = atomicAdd( &header->m_primNodeCount, rangeSize );
	uint32_t       primNodeOffset = primNodeBase;

	BoxNode& node = boxNodes[nodeAddr];
	for ( uint32_t i = 0; i < node.getChildCount(); ++i )
	{
		const uint32_t childIndex = node.getChildIndex( i );
		if ( isLeafNode( childIndex ) )
		{
			const uint32_t      referenceIndex = referenceIndices[rangeOffset++];
			const ReferenceNode reference      = references[getNodeAddr( referenceIndex )];
			primNodes[primNodeOffset++]        = primitives.template fetchPrimNode<PrimitiveNode>( reference.m_primIndex );
		}
	}

	// patch parent
	node.patchParent( primNodeBase );
}

// assuming that triangle packets are AMD specific, there are no warp syncs
HIPRT_DEVICE HIPRT_INLINE void PackLeaves(
	const uint32_t              index,
	const uint32_t              taskCount,
	const ConversionTask*       taskQueue,
	const uint32_t*             referenceIndices,
	const ReferenceNode*        references,
	const TriangleMesh&         primitives,
	GeomHeader*                 header,
	rtip31::BoxNode*            boxNodes,
	rtip31::TrianglePacketNode* primNodes )
{
	if ( threadIdx.x >= WarpSize ) return;

	constexpr uint32_t PacketTasksPerWarp = WarpSize / rtip31::TrianglePacketNode::LanesPerLeafPacketTask;
	__shared__ rtip31::TrianglePacketCache triPacketCache[PacketTasksPerWarp];
	__shared__ rtip31::TrianglePairOffsets triPairOffsetCache[PacketTasksPerWarp][rtip31::BoxNode::BranchingFactor];

	const uint32_t laneIndex    = threadIdx.x % WarpSize;
	const uint32_t taskIndex    = index / rtip31::TrianglePacketNode::LanesPerLeafPacketTask;
	const uint32_t subwarpIndex = laneIndex / rtip31::TrianglePacketNode::LanesPerLeafPacketTask;
	const uint32_t sublaneIndex = laneIndex % rtip31::TrianglePacketNode::LanesPerLeafPacketTask;

	const ConversionTask task        = taskQueue[min( taskIndex, taskCount - 1 )];
	const uint32_t       nodeAddr    = task.m_nodeAddr;
	const uint32_t       rangeSize   = task.m_rangeSize;
	uint32_t             rangeOffset = task.m_rangeOffset;

	uint32_t primNodeCount = 0;

	const uint32_t rangeBase = rangeOffset;
	while ( hiprt::ballot( taskIndex < taskCount && rangeOffset < rangeBase + rangeSize ) )
	{
		rtip31::TrianglePacketData packet{};

		while ( rangeOffset < rangeBase + rangeSize &&
		        packet.getTrianglePairCount() < rtip31::TrianglePacketNode::MaxTrianglePairs )
		{
			uint32_t referenceIndex = referenceIndices[rangeOffset];
			referenceIndex &= ~RangeStartBit;
			referenceIndex &= ~RangeEndBit;

			// form triangle pair
			const ReferenceNode reference   = references[getNodeAddr( referenceIndex )];
			const uint2         pairIndices = primitives.fetchTrianglePairIndices( reference.m_primIndex );
			uint3               indices0    = primitives.fetchTriangleIndices( pairIndices.x );
			uint4               indices     = make_uint4( indices0, indices0.z );

			if ( pairIndices.x != pairIndices.y )
			{
				uint3 indices1      = primitives.fetchTriangleIndices( pairIndices.y );
				uint3 vertexMapping = tryPairTriangles( indices0, indices1 );

				uint32_t vertexIndex = 0;
				if ( vertexMapping.x == 3 ) vertexIndex = indices1.x;
				if ( vertexMapping.y == 3 ) vertexIndex = indices1.y;
				if ( vertexMapping.z == 3 ) vertexIndex = indices1.z;
				indices.w = vertexIndex;
			}

			// find new vertices
			const bool valid = sublaneIndex < 3 || pairIndices.x != pairIndices.y;

			bool     contains = false;
			uint32_t vertexIndexInPacket{};
			for ( uint32_t k = 0; k < packet.m_vertCount; ++k )
			{
				if ( triPacketCache[subwarpIndex].m_vertexIndices[k] == component( indices, sublaneIndex ) )
				{
					vertexIndexInPacket = k;
					contains            = true;
					break;
				}
			}

			const uint32_t newVertMask = ( hiprt::ballot( !contains && valid ) >>
			                               ( rtip31::TrianglePacketNode::LanesPerLeafPacketTask * subwarpIndex ) ) &
			                             0xf;
			const uint32_t oldVertCount = packet.m_vertCount;
			const uint32_t newVertCount = hiprt::popc( newVertMask );

			// try to fit
			if ( !packet.tryAddTrianglePair( pairIndices.x, pairIndices.y, newVertCount ) ) break;

			// store new vertices
			if ( !contains )
			{
				const uint32_t vertexMask = ( 1 << sublaneIndex ) - 1;
				vertexIndexInPacket       = oldVertCount + hiprt::popc( newVertMask & vertexMask );
				triPacketCache[subwarpIndex].m_vertexIndices[vertexIndexInPacket] = component( indices, sublaneIndex );
			}

			rangeOffset++;

			// not sure why but this fence is needed on linux
			__threadfence_block();
		}
		__threadfence_block();

		// count packets
		if ( taskIndex < taskCount && packet.m_triPairCount > 0 ) primNodeCount++;
	}

	const uint32_t primNodeBase = warpOffset(
		sublaneIndex == rtip31::TrianglePacketNode::LanesPerLeafPacketTask - 1 ? primNodeCount : 0u, &header->m_primNodeCount );
	uint32_t primNodeOffset = primNodeBase;

	uint32_t triPairOffset   = 0;
	uint32_t triPacketOffset = 0;
	uint32_t leafIndex       = 0;

	rangeOffset = rangeBase;

	while ( hiprt::ballot( taskIndex < taskCount && rangeOffset < rangeBase + rangeSize ) )
	{
		rtip31::TrianglePacketData packet{};

		while ( rangeOffset < rangeBase + rangeSize &&
		        packet.getTrianglePairCount() < rtip31::TrianglePacketNode::MaxTrianglePairs )
		{
			uint32_t   referenceIndex = referenceIndices[rangeOffset];
			const bool rangeStart     = referenceIndex & RangeStartBit;
			const bool rangeEnd       = referenceIndex & RangeEndBit;
			referenceIndex &= ~RangeStartBit;
			referenceIndex &= ~RangeEndBit;

			// form triangle pair
			const ReferenceNode reference   = references[getNodeAddr( referenceIndex )];
			const uint2         pairIndices = primitives.fetchTrianglePairIndices( reference.m_primIndex );
			uint3               indices0    = primitives.fetchTriangleIndices( pairIndices.x );
			uint4               indices     = make_uint4( indices0, indices0.z );

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
			const bool valid = sublaneIndex < 3 || pairIndices.x != pairIndices.y;

			bool     contains = false;
			uint32_t vertexIndexInPacket{};
			for ( uint32_t k = 0; k < packet.m_vertCount; ++k )
			{
				if ( triPacketCache[subwarpIndex].m_vertexIndices[k] == component( indices, sublaneIndex ) )
				{
					vertexIndexInPacket = k;
					contains            = true;
					break;
				}
			}

			const uint32_t newVertMask = ( hiprt::ballot( !contains && valid ) >>
			                               ( rtip31::TrianglePacketNode::LanesPerLeafPacketTask * subwarpIndex ) ) &
			                             0xf;
			const uint32_t oldVertCount = packet.m_vertCount;
			const uint32_t newVertCount = hiprt::popc( newVertMask );

			// try to fit
			if ( !packet.tryAddTrianglePair( pairIndices.x, pairIndices.y, newVertCount ) ) break;

			if ( rangeStart )
			{
				triPairOffset   = packet.m_triPairCount - 1;
				triPacketOffset = primNodeOffset - primNodeBase;
			}

			if ( rangeEnd )
			{
				triPairOffsetCache[subwarpIndex][leafIndex++] = rtip31::TrianglePairOffsets( triPairOffset, triPacketOffset );
			}

			// store new vertices
			if ( !contains )
			{
				const uint32_t vertexMask = ( 1 << sublaneIndex ) - 1;
				vertexIndexInPacket       = oldVertCount + hiprt::popc( newVertMask & vertexMask );
				triPacketCache[subwarpIndex].m_vertexIndices[vertexIndexInPacket] = component( indices, sublaneIndex );
			}

			// shuffle vertex indices in packet
			uint4 vertexIndicesInPacket;
			vertexIndicesInPacket.x =
				shfl( vertexIndexInPacket, subwarpIndex * rtip31::TrianglePacketNode::LanesPerLeafPacketTask + 0 );
			vertexIndicesInPacket.y =
				shfl( vertexIndexInPacket, subwarpIndex * rtip31::TrianglePacketNode::LanesPerLeafPacketTask + 1 );
			vertexIndicesInPacket.z =
				shfl( vertexIndexInPacket, subwarpIndex * rtip31::TrianglePacketNode::LanesPerLeafPacketTask + 2 );
			vertexIndicesInPacket.w =
				shfl( vertexIndexInPacket, subwarpIndex * rtip31::TrianglePacketNode::LanesPerLeafPacketTask + 3 );

			uint3 triIndices0 = make_uint3( vertexIndicesInPacket );
			uint3 triIndices1{};
			// tryPairTriangles reports InvalidValue when the two triangles cannot share vertex slots, which
			// indexes past the packet's four slots. Degenerate triangles make its verdict order-dependent,
			// so a caller-supplied pair can fail here even though it passed the caller's own pairing test.
			if ( pairIndices.x != pairIndices.y && vertexMapping.x != InvalidValue )
			{
				triIndices1.x = component( vertexIndicesInPacket, vertexMapping.x );
				triIndices1.y = component( vertexIndicesInPacket, vertexMapping.y );
				triIndices1.z = component( vertexIndicesInPacket, vertexMapping.z );
			}

			// store triangle pair
			if ( sublaneIndex == 0 )
				triPacketCache[subwarpIndex].m_triPairData[packet.m_triPairCount - 1] =
					rtip31::TrianglePairData( pairIndices, triIndices0, triIndices1, rangeEnd );

			rangeOffset++;
			// not sure why but this fixes the issue on linux
			// otherwise triIndices1 are not correcly written to the final packet
			__threadfence_block();
		}
		__threadfence_block();

		// build packets
		uint64_t packetMask = hiprt::ballot( taskIndex < taskCount && packet.m_triPairCount > 0 && sublaneIndex == 0 );
		while ( packetMask )
		{
			const uint32_t halfWarpIndex = laneIndex / 16;
			const uint32_t halfLaneIndex = laneIndex % 16;

			const uint32_t broadcastLane0 = hiprt::ffsll( packetMask ) - 1;
			packetMask ^= 1ull << broadcastLane0;

			const uint32_t broadcastLane1 = hiprt::ffsll( packetMask ) - 1;
			const bool     secondValid    = packetMask != 0;
			if ( secondValid ) packetMask ^= 1ull << broadcastLane1;

			const uint32_t broadcastLane           = ( halfWarpIndex == 0 ) ? broadcastLane0 : broadcastLane1;
			const uint32_t broadcastSubwarpIndex   = broadcastLane / rtip31::TrianglePacketNode::LanesPerLeafPacketTask;
			const uint32_t broadcastPrimNodeOffset = shfl( primNodeOffset, broadcastLane );
			const rtip31::TrianglePacketData broadcastPacket = packet.shuffle( broadcastLane );

			// store current packet data to registers
			rtip31::TrianglePairData halfLaneTriPairData;
			uint32_t                 halfLaneVertexIndex;
			if ( halfWarpIndex == 0 || secondValid )
			{
				if ( halfLaneIndex < 2 * broadcastPacket.m_triPairCount )
					halfLaneTriPairData = triPacketCache[broadcastSubwarpIndex].m_triPairData[halfLaneIndex / 2];
				if ( halfLaneIndex < broadcastPacket.m_vertCount )
					halfLaneVertexIndex = triPacketCache[broadcastSubwarpIndex].m_vertexIndices[halfLaneIndex];
			}

			// reuse shared memory
			rtip31::TrianglePacketNode& triPacketNode =
				*reinterpret_cast<rtip31::TrianglePacketNode*>( &triPacketCache[broadcastSubwarpIndex] );
			if ( halfWarpIndex == 0 || secondValid )
			{
				triPacketNode.m_data[halfLaneIndex + 0 * 16] = 0;
				triPacketNode.m_data[halfLaneIndex + 1 * 16] = 0;
			}

			// build two packets at once
			if ( halfWarpIndex == 0 || secondValid )
			{
				// write header
				rtip31::TrianglePacketHeader hdr = broadcastPacket.buildHeader();
				if ( halfLaneIndex == 0 ) triPacketNode.writeHeader<true>( hdr );

				// write indices & descriptors
				if ( halfLaneIndex < 2 * broadcastPacket.m_triPairCount )
				{
					// write indices
					triPacketNode.writePrimIndex<true>(
						halfLaneIndex / 2,
						halfLaneIndex % 2,
						hdr,
						component( halfLaneTriPairData.m_pairIndices, halfLaneIndex % 2 ) );

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

			// write packet
			if ( ( halfWarpIndex == 0 || secondValid ) )
			{
				primNodes[broadcastPrimNodeOffset].m_data[halfLaneIndex + 0 * 16] =
					triPacketNode.m_data[halfLaneIndex + 0 * 16];
				primNodes[broadcastPrimNodeOffset].m_data[halfLaneIndex + 1 * 16] =
					triPacketNode.m_data[halfLaneIndex + 1 * 16];
			}
		}

		if ( taskIndex < taskCount && packet.m_triPairCount > 0 ) primNodeOffset++;
	}

	// patch parent
	if ( taskIndex < taskCount && sublaneIndex == 0 )
		boxNodes[nodeAddr].patchParent( primNodeBase, leafIndex, triPairOffsetCache[subwarpIndex] );
}

HIPRT_KERNEL void __launch_bounds__( WarpSize ) PackLeaves_TriangleMesh_rtip31_TrianglePacketNode(
	const uint32_t              taskCount,
	const ConversionTask*       taskQueue,
	const uint32_t*             referenceIndices,
	const ReferenceNode*        references,
	const TriangleMesh          primitives,
	GeomHeader*                 header,
	rtip31::BoxNode*            boxNodes,
	rtip31::TrianglePacketNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
}

HIPRT_KERNEL void PackLeaves_TriangleMesh_rtip11_TrianglePairNode(
	const uint32_t        taskCount,
	const ConversionTask* taskQueue,
	const uint32_t*       referenceIndices,
	const ReferenceNode*  references,
	const TriangleMesh    primitives,
	GeomHeader*           header,
	BoxNode*              boxNodes,
	TriangleNode*         primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
}

HIPRT_KERNEL void PackLeaves_AabbList_CustomNode(
	const uint32_t        taskCount,
	const ConversionTask* taskQueue,
	const uint32_t*       referenceIndices,
	const ReferenceNode*  references,
	const AabbList        primitives,
	GeomHeader*           header,
	BoxNode*              boxNodes,
	CustomNode*           primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
}

HIPRT_KERNEL void PackLeaves_InstanceList_hiprtFrameSRT_rtip11_InstanceNode(
	const uint32_t                    taskCount,
	const ConversionTask*             taskQueue,
	const uint32_t*                   referenceIndices,
	const ReferenceNode*              references,
	const InstanceList<hiprtFrameSRT> primitives,
	SceneHeader*                      header,
	BoxNode*                          boxNodes,
	InstanceNode*                     primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
}

HIPRT_KERNEL void PackLeaves_InstanceList_hiprtFrameSRT_rtip31_InstanceNode(
	const uint32_t                    taskCount,
	const ConversionTask*             taskQueue,
	const uint32_t*                   referenceIndices,
	const ReferenceNode*              references,
	const InstanceList<hiprtFrameSRT> primitives,
	SceneHeader*                      header,
	BoxNode*                          boxNodes,
	InstanceNode*                     primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
}

HIPRT_KERNEL void PackLeaves_InstanceList_hiprtFrameMatrix_rtip11_InstanceNode(
	const uint32_t                       taskCount,
	const ConversionTask*                taskQueue,
	const uint32_t*                      referenceIndices,
	const ReferenceNode*                 references,
	const InstanceList<hiprtFrameMatrix> primitives,
	SceneHeader*                         header,
	BoxNode*                             boxNodes,
	InstanceNode*                        primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
}

HIPRT_KERNEL void PackLeaves_InstanceList_hiprtFrameMatrix_rtip31_InstanceNode(
	const uint32_t                       taskCount,
	const ConversionTask*                taskQueue,
	const uint32_t*                      referenceIndices,
	const ReferenceNode*                 references,
	const InstanceList<hiprtFrameMatrix> primitives,
	SceneHeader*                         header,
	BoxNode*                             boxNodes,
	InstanceNode*                        primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PackLeaves( index, taskCount, taskQueue, referenceIndices, references, primitives, header, boxNodes, primNodes );
}

template <uint32_t LeafType>
HIPRT_DEVICE HIPRT_INLINE void PatchApiNodes( const uint32_t index, const uint32_t nodeCount, ApiNode* apiNodes )
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

HIPRT_KERNEL void PatchApiNodes_TriangleMesh( const uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<TriangleType>( index, nodeCount, apiNodes );
}

HIPRT_KERNEL void PatchApiNodes_AabbList( const uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<CustomType>( index, nodeCount, apiNodes );
}

HIPRT_KERNEL void PatchApiNodes_InstanceList_hiprtFrameSRT( const uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<InstanceType>( index, nodeCount, apiNodes );
}

HIPRT_KERNEL void PatchApiNodes_InstanceList_hiprtFrameMatrix( const uint32_t nodeCount, ApiNode* apiNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	PatchApiNodes<InstanceType>( index, nodeCount, apiNodes );
}

HIPRT_KERNEL void __launch_bounds__( BvhBuilderReductionBlockSize )
	ComputeCost( const uint32_t nodeCount, const BoxNode* boxNodes, float* costCounter )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

	float cost = 0.0f;
	if ( index < nodeCount )
	{
		const float rootAreaInv = 1.0f / boxNodes[0].area();

		for ( uint32_t i = 0; i < boxNodes[index].getChildCount(); ++i )
			cost += ( boxNodes[index].getChildType( i ) == BoxType ? Ct : Ci ) * boxNodes[index].getChildBox( i ).area() *
			        rootAreaInv;

		if ( index == 0 ) cost += Ct;
	}

	constexpr uint32_t WarpsPerBlock = DivideRoundUp( BvhBuilderReductionBlockSize, WarpSize );
	__shared__ float   costCache[WarpsPerBlock];

	const float blockCost = blockSum( cost, costCache );
	if ( threadIdx.x == 0 ) atomicAdd( costCounter, blockCost );
}
