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
#include <hiprt/impl/BvhConfig.h>

using namespace hiprt;

HIPRT_DEVICE uint32_t findParent(
	uint32_t		nodeAddr,
	uint32_t		nodeType,
	uint32_t		i,
	uint32_t		j,
	uint32_t		n,
	ScratchNode*	scratchNodes,
	const uint32_t* sortedMortonCodeKeys )
{
	if ( i == 0 && j == n ) return InvalidValue;
	if ( i == 0 || ( j != n && findHighestDifferentBit( j - 1, j, n, sortedMortonCodeKeys ) <
								   findHighestDifferentBit( i - 1, i, n, sortedMortonCodeKeys ) ) )
	{
		scratchNodes[j - 1].encodeChildIndex( 0, nodeAddr, nodeType );
		return j - 1;
	}
	else
	{
		scratchNodes[i - 1].encodeChildIndex( 1, nodeAddr, nodeType );
		return i - 1;
	}
}

template <typename PrimitiveContainer>
__device__ void EmitTopologyAndFitBounds(
	uint32_t			index,
	const uint32_t*		sortedMortonCodeKeys,
	const uint32_t*		sortedMortonCodeValues,
	uint32_t*			updateCounters,
	PrimitiveContainer& primitives,
	ScratchNode*		scratchNodes,
	ReferenceNode*		references )
{
	uint32_t primCount = primitives.getCount();
	uint32_t i		   = index;
	uint32_t j		   = i + 1;
	uint32_t k;

	if ( index >= primCount ) return;

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

	uint32_t parentAddr = findParent( index, leafType, i, j, primCount, scratchNodes, sortedMortonCodeKeys );
	index				= parentAddr;
	while ( ( k = atomicExch( &updateCounters[index], parentAddr == i - 1 ? j : i ) ) != InvalidValue )
	{
		__threadfence();

		ScratchNode& node = scratchNodes[index];
		if ( parentAddr == i - 1 )
			i = k;
		else
			j = k;

		Aabb box;
		if ( node.getChildType( 0 ) != BoxType )
			box.grow( references[node.getChildAddr( 0 )].aabb() );
		else
			box.grow( scratchNodes[node.getChildAddr( 0 )].aabb() );

		if ( node.getChildType( 1 ) != BoxType )
			box.grow( references[node.getChildAddr( 1 )].aabb() );
		else
			box.grow( scratchNodes[node.getChildAddr( 1 )].aabb() );

		parentAddr = findParent( index, BoxType, i, j, primCount, scratchNodes, sortedMortonCodeKeys );
		node.m_box = box;

		if ( parentAddr == InvalidValue )
		{
			updateCounters[primCount - 1] = index; // save root index
			break;
		}

		index = parentAddr;

		__threadfence();
	}
}

extern "C" __global__ void EmitTopologyAndFitBounds_TriangleMesh(
	const uint32_t* sortedMortonCodeKeys,
	const uint32_t* sortedMortonCodeValues,
	uint32_t*		updateCounters,
	TriangleMesh	primitives,
	ScratchNode*	scratchNodes,
	ReferenceNode*	references )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	EmitTopologyAndFitBounds<TriangleMesh>(
		index, sortedMortonCodeKeys, sortedMortonCodeValues, updateCounters, primitives, scratchNodes, references );
}

extern "C" __global__ void EmitTopologyAndFitBounds_AabbList(
	const uint32_t* sortedMortonCodeKeys,
	const uint32_t* sortedMortonCodeValues,
	uint32_t*		updateCounters,
	AabbList		primitives,
	ScratchNode*	scratchNodes,
	ReferenceNode*	references )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	EmitTopologyAndFitBounds<AabbList>(
		index, sortedMortonCodeKeys, sortedMortonCodeValues, updateCounters, primitives, scratchNodes, references );
}

extern "C" __global__ void EmitTopologyAndFitBounds_InstanceList_SRTFrame(
	const uint32_t*		   sortedMortonCodeKeys,
	const uint32_t*		   sortedMortonCodeValues,
	uint32_t*			   updateCounters,
	InstanceList<SRTFrame> primitives,
	ScratchNode*		   scratchNodes,
	ReferenceNode*		   references )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	EmitTopologyAndFitBounds<InstanceList<SRTFrame>>(
		index, sortedMortonCodeKeys, sortedMortonCodeValues, updateCounters, primitives, scratchNodes, references );
}

extern "C" __global__ void EmitTopologyAndFitBounds_InstanceList_MatrixFrame(
	const uint32_t*			  sortedMortonCodeKeys,
	const uint32_t*			  sortedMortonCodeValues,
	uint32_t*				  updateCounters,
	InstanceList<MatrixFrame> primitives,
	ScratchNode*			  scratchNodes,
	ReferenceNode*			  references )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;
	EmitTopologyAndFitBounds<InstanceList<MatrixFrame>>(
		index, sortedMortonCodeKeys, sortedMortonCodeValues, updateCounters, primitives, scratchNodes, references );
}
