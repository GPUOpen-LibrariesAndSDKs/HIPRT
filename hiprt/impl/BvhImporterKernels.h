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
#include <hiprt/impl/Geometry.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/MortonCode.h>
#include <hiprt/impl/TriangleMesh.h>
#include <hiprt/impl/ApiNodeList.h>
using namespace hiprt;

template <typename PrimitiveContainer, typename PrimitiveNode>
__device__ void SetupLeaves( PrimitiveContainer& primitives, PrimitiveNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if ( index < primitives.getCount() )
	{
		if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
		{
			primNodes[index] = primitives.fetchTriangleNode( index );
		}
		else if constexpr ( is_same<PrimitiveNode, CustomNode>::value )
		{
			primNodes[index].m_primIndex = index;
		}
		else if constexpr ( is_same<PrimitiveNode, InstanceNode>::value )
		{
			hiprtInstance		 instance  = primitives.fetchInstance( index );
			hiprtTransformHeader transform = primitives.fetchTransformHeader( index );
			primNodes[index].m_primIndex   = index;
			primNodes[index].m_mask		   = primitives.fetchMask( index );
			primNodes[index].m_type		   = instance.type;
			primNodes[index].m_static	   = transform.frameCount == 1 ? 1 : 0;
			if ( instance.type == hiprtInstanceTypeScene )
				primNodes[index].m_scene = reinterpret_cast<SceneHeader*>( instance.scene );
			else
				primNodes[index].m_geometry = reinterpret_cast<GeomHeader*>( instance.geometry );
			if ( transform.frameCount == 1 )
				primNodes[index].m_identity =
					primitives.copyInvTransformMatrix( transform.frameIndex, primNodes[index].m_matrix ) ? 1 : 0;
			else
				primNodes[index].m_transform = transform;
		}
	}
}

extern "C" __global__ void SetupLeaves_TriangleMesh_TriangleNode( TriangleMesh primitives, TriangleNode* primNodes )
{
	SetupLeaves<TriangleMesh, TriangleNode>( primitives, primNodes );
}

extern "C" __global__ void SetupLeaves_AabbList_CustomNode( AabbList primitives, CustomNode* primNodes )
{
	SetupLeaves<AabbList, CustomNode>( primitives, primNodes );
}

extern "C" __global__ void
SetupLeaves_InstanceList_SRTFrame_InstanceNode( InstanceList<SRTFrame> primitives, InstanceNode* primNodes )
{
	SetupLeaves<InstanceList<SRTFrame>, InstanceNode>( primitives, primNodes );
}

extern "C" __global__ void
SetupLeaves_InstanceList_MatrixFrame_InstanceNode( InstanceList<MatrixFrame> primitives, InstanceNode* primNodes )
{
	SetupLeaves<InstanceList<MatrixFrame>, InstanceNode>( primitives, primNodes );
}

template <typename PrimitiveContainer, typename PrimitiveNode>
__device__ void Convert( PrimitiveContainer& primitives, ApiNodeList nodes, BoxNode* boxNodes, PrimitiveNode* primNodes )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	uint32_t leafType;
	if constexpr ( is_same<PrimitiveContainer, TriangleMesh>::value )
		leafType = TriangleType;
	else if constexpr ( is_same<PrimitiveContainer, AabbList>::value )
		leafType = CustomType;
	else
		leafType = InstanceType;

	if ( index < nodes.getCount() )
	{
		ApiNode node = nodes.fetchNode( index );

		Aabb*	  childBoxes   = &boxNodes[index].m_box0;
		uint32_t* childIndices = &boxNodes[index].m_childIndex0;
		uint32_t  childCount   = 0;
		for ( uint32_t i = 0; i < BranchingFactor; ++i )
		{
			uint32_t childIndex = node.m_childIndices[i];
			if ( childIndex != InvalidValue )
			{
				childIndices[i] = encodeNodeIndex( childIndex, node.m_leafFlags[i] ? leafType : BoxType );
				childBoxes[i]	= node.getChildBox( i );
				childCount++;
			}
			else
			{
				childIndices[i] = InvalidValue;
				childBoxes[i].reset();
			}
		}
		boxNodes[index].m_childCount = childCount;

		if ( index == 0 ) boxNodes[index].m_parentAddr = InvalidValue;
	}
}

extern "C" __global__ void
Convert_TriangleMesh_TriangleNode( TriangleMesh primitives, ApiNodeList nodes, BoxNode* boxNodes, TriangleNode* primNodes )
{
	Convert<TriangleMesh, TriangleNode>( primitives, nodes, boxNodes, primNodes );
}

extern "C" __global__ void
Convert_AabbList_CustomNode( AabbList primitives, ApiNodeList nodes, BoxNode* boxNodes, CustomNode* primNodes )
{
	Convert<AabbList, CustomNode>( primitives, nodes, boxNodes, primNodes );
}

extern "C" __global__ void Convert_InstanceList_SRTFrame_InstanceNode(
	InstanceList<SRTFrame> primtives, ApiNodeList nodes, BoxNode* boxNodes, InstanceNode* primNodes )
{
	Convert<InstanceList<SRTFrame>, InstanceNode>( primtives, nodes, boxNodes, primNodes );
}

extern "C" __global__ void Convert_InstanceList_MatrixFrame(
	InstanceList<MatrixFrame> primtives, ApiNodeList nodes, BoxNode* boxNodes, InstanceNode* primNodes )
{
	Convert<InstanceList<MatrixFrame>, InstanceNode>( primtives, nodes, boxNodes, primNodes );
}
