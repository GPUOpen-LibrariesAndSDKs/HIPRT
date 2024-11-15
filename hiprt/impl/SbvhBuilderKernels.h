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
#include <hiprt/impl/BvhBuilderUtil.h>
#include <hiprt/impl/Triangle.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Geometry.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/SbvhCommon.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/TriangleMesh.h>
#include <hiprt/impl/BvhConfig.h>
using namespace hiprt;

template <typename PrimitiveContainer>
__device__ void SetupLeavesAndReferences(
	PrimitiveContainer& primitives,
	ReferenceNode*		references,
	Task*				taskQueue,
	Aabb*				box,
	uint32_t*			referenceIndices,
	uint32_t*			taskIndices )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if ( index < primitives.getCount() )
	{
		references[index]		= ReferenceNode( index, primitives.fetchAabb( index ) );
		referenceIndices[index] = index;
		taskIndices[index]		= 0;
	}

	if ( index == 0 ) taskQueue[0] = Task( *box );
}

extern "C" __global__ void SetupLeavesAndReferences_TriangleMesh(
	TriangleMesh   primitives,
	ReferenceNode* references,
	Task*		   taskQueue,
	Aabb*		   box,
	uint32_t*	   referenceIndices,
	uint32_t*	   taskIndices )
{
	SetupLeavesAndReferences<TriangleMesh>( primitives, references, taskQueue, box, referenceIndices, taskIndices );
}

extern "C" __global__ void SetupLeavesAndReferences_AabbList(
	AabbList	   primitives,
	ReferenceNode* references,
	Task*		   taskQueue,
	Aabb*		   box,
	uint32_t*	   referenceIndices,
	uint32_t*	   taskIndices )
{
	SetupLeavesAndReferences<AabbList>( primitives, references, taskQueue, box, referenceIndices, taskIndices );
}

extern "C" __global__ void SetupLeavesAndReferences_InstanceList_SRTFrame(
	InstanceList<SRTFrame> primitives,
	ReferenceNode*		   references,
	Task*				   taskQueue,
	Aabb*				   box,
	uint32_t*			   referenceIndices,
	uint32_t*			   taskIndices )
{
	SetupLeavesAndReferences<InstanceList<SRTFrame>>( primitives, references, taskQueue, box, referenceIndices, taskIndices );
}

extern "C" __global__ void SetupLeavesAndReferences_InstanceList_MatrixFrame(
	InstanceList<MatrixFrame> primitives,
	ReferenceNode*			  references,
	Task*					  taskQueue,
	Aabb*					  box,
	uint32_t*				  referenceIndices,
	uint32_t*				  taskIndices )
{
	SetupLeavesAndReferences<InstanceList<MatrixFrame>>(
		primitives, references, taskQueue, box, referenceIndices, taskIndices );
}

template <bool spatialSplits>
__device__ void ResetBins( uint32_t taskCount, uint32_t binCount, Bin* objectBins, Bin* spatialBins )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	if ( index < 3 * binCount * taskCount )
	{
		objectBins[index].reset();
		if constexpr ( spatialSplits ) spatialBins[index].reset();
	}
}

extern "C" __global__ void ResetBins_true( uint32_t taskCount, uint32_t binCount, Bin* objectBins, Bin* spatialBins )
{
	ResetBins<true>( taskCount, binCount, objectBins, spatialBins );
}

extern "C" __global__ void ResetBins_false( uint32_t taskCount, uint32_t binCount, Bin* objectBins, Bin* spatialBins )
{
	ResetBins<false>( taskCount, binCount, objectBins, spatialBins );
}

extern "C" __global__ void BinReferencesObject(
	uint32_t	   activeRefCount,
	uint32_t	   binCount,
	uint32_t	   taskOffset,
	uint32_t	   taskCount,
	uint32_t*	   referenceIndices,
	uint32_t*	   taskIndices,
	Task*		   taskQueue,
	ReferenceNode* references,
	Bin*		   binsGlobal )
{
	const uint32_t indexStart = blockIdx.x * blockDim.x;
	const uint32_t index	  = indexStart + threadIdx.x;
	const uint32_t indexEnd	  = min( indexStart + blockDim.x, activeRefCount );

	const uint32_t firstReferenceIndex = referenceIndices[indexStart];
	const uint32_t lastReferenceIndex  = referenceIndices[indexEnd - 1];

	const uint32_t firstTaskIndex = taskIndices[firstReferenceIndex];
	const uint32_t lastTaskIndex  = taskIndices[lastReferenceIndex];

	alignas( alignof( Bin ) ) __shared__ uint8_t binBuffer[3 * SbvhMaxBinCount * sizeof( Bin )];
	Bin*										 binCache = reinterpret_cast<Bin*>( binBuffer );
	Bin*										 bins	  = binsGlobal;

	if ( firstTaskIndex == lastTaskIndex )
	{
		bins = binCache;
		for ( uint32_t binIndex = threadIdx.x; binIndex < 3 * binCount; binIndex += blockDim.x )
			binCache[binIndex].reset();
		__syncthreads();
	}

	if ( index < activeRefCount )
	{
		uint32_t referenceIndex = referenceIndices[index];
		uint32_t taskIndex		= taskIndices[referenceIndex];

		Task		  task = taskQueue[taskIndex + taskOffset];
		ReferenceNode ref  = references[referenceIndex];

		float3 k = ( 1.0f - SbvhEpsilon ) * ( static_cast<float>( binCount ) / ( task.m_box.m_max - task.m_box.m_min ) );
		uint3  binIndex =
			clamp( make_uint3( k * ( ref.m_box.center() - task.m_box.m_min ) ), make_uint3( 0 ), make_uint3( binCount - 1 ) );
		uint3 binAddr = binIndex + make_uint3( 0, 1, 2 ) * binCount;
		if ( firstTaskIndex != lastTaskIndex ) binAddr = taskIndex + taskCount * binAddr;

		bins[binAddr.x].m_box.atomicGrow( ref.m_box );
		bins[binAddr.y].m_box.atomicGrow( ref.m_box );
		bins[binAddr.z].m_box.atomicGrow( ref.m_box );

		atomicAdd( &bins[binAddr.x].m_counter, 1 );
		atomicAdd( &bins[binAddr.y].m_counter, 1 );
		atomicAdd( &bins[binAddr.z].m_counter, 1 );
	}

	if ( firstTaskIndex == lastTaskIndex )
	{
		__syncthreads();
		for ( uint32_t binIndex = threadIdx.x; binIndex < binCount; binIndex += blockDim.x )
		{
			uint3 binOffset = make_uint3( binIndex ) + make_uint3( 0, 1, 2 ) * binCount;
			uint3 binAddr	= firstTaskIndex + taskCount * binOffset;

			if ( binCache[binOffset.x].m_counter > 0 )
			{
				atomicAdd( &binsGlobal[binAddr.x].m_counter, binCache[binOffset.x].m_counter );
				binsGlobal[binAddr.x].m_box.atomicGrow( binCache[binOffset.x].m_box );
			}

			if ( binCache[binOffset.y].m_counter > 0 )
			{
				atomicAdd( &binsGlobal[binAddr.y].m_counter, binCache[binOffset.y].m_counter );
				binsGlobal[binAddr.y].m_box.atomicGrow( binCache[binOffset.y].m_box );
			}

			if ( binCache[binOffset.z].m_counter > 0 )
			{
				atomicAdd( &binsGlobal[binAddr.z].m_counter, binCache[binOffset.z].m_counter );
				binsGlobal[binAddr.z].m_box.atomicGrow( binCache[binOffset.z].m_box );
			}
		}
	}
}

template <typename PrimitiveContainer>
__device__ void BinReferencesSpatial(
	uint32_t			activeRefCount,
	uint32_t			binCount,
	uint32_t			taskOffset,
	uint32_t			taskCount,
	float				overlapThreshold,
	float				edgeThreshold,
	uint32_t*			referenceIndices,
	uint32_t*			taskIndices,
	Task*				taskQueue,
	PrimitiveContainer& primitives,
	ReferenceNode*		references,
	Bin*				binsGlobal )
{
	const uint32_t indexStart = blockIdx.x * blockDim.x;
	const uint32_t index	  = indexStart + threadIdx.x;
	const uint32_t indexEnd	  = min( indexStart + blockDim.x, activeRefCount );

	const uint32_t firstReferenceIndex = referenceIndices[indexStart];
	const uint32_t lastReferenceIndex  = referenceIndices[indexEnd - 1];

	const uint32_t firstTaskIndex = taskIndices[firstReferenceIndex];
	const uint32_t lastTaskIndex  = taskIndices[lastReferenceIndex];

	alignas( alignof( Bin ) ) __shared__ uint8_t binBuffer[3 * SbvhMaxBinCount * sizeof( Bin )];
	Bin*										 binCache = reinterpret_cast<Bin*>( binBuffer );
	Bin*										 bins	  = binsGlobal;

	if ( firstTaskIndex == lastTaskIndex )
	{
		bins = binCache;
		for ( uint32_t binIndex = threadIdx.x; binIndex < 3 * binCount; binIndex += blockDim.x )
			binCache[binIndex].reset();
		__syncthreads();
	}

	if ( index < activeRefCount )
	{
		uint32_t referenceIndex = referenceIndices[index];
		uint32_t taskIndex		= taskIndices[referenceIndex];

		Task task = taskQueue[taskIndex + taskOffset];

		Aabb overlap = task.m_box0;
		overlap.intersect( task.m_box1 );

		if ( overlap.area() >= overlapThreshold )
		{
			ReferenceNode ref = references[referenceIndex];

			for ( uint32_t axisIndex = 0; axisIndex < 3; ++axisIndex )
			{
				if ( ptr( task.m_box.m_max )[axisIndex] - ptr( task.m_box.m_min )[axisIndex] < edgeThreshold ) continue;

				uint32_t firstBin = binCount - 1;
				uint32_t lastBin  = firstBin;
				float	 binSize  = ( ptr( task.m_box.m_max )[axisIndex] - ptr( task.m_box.m_min )[axisIndex] ) /
								static_cast<float>( binCount );
				for ( uint32_t i = 0; i < binCount; ++i )
				{
					float position = ptr( task.m_box.m_min )[axisIndex] + binSize * static_cast<float>( i + 1 );
					if ( firstBin == binCount - 1 && ptr( ref.m_box.m_min )[axisIndex] < position ) firstBin = i;
					if ( lastBin == binCount - 1 && ptr( ref.m_box.m_max )[axisIndex] <= position ) lastBin = i;
				}
				if ( firstBin > lastBin ) firstBin = lastBin;

				uint32_t	  curBinAddr;
				ReferenceNode curRef = ref;
				ReferenceNode leftRef( ref.m_primIndex );
				ReferenceNode rightRef( ref.m_primIndex );

				for ( uint32_t i = firstBin; i < lastBin; i++ )
				{
					float position = ptr( task.m_box.m_min )[axisIndex] + binSize * static_cast<float>( i + 1 );
					if constexpr ( is_same<PrimitiveContainer, TriangleMesh>::value )
					{
						TrianglePair triPair = primitives.fetchTriangleNode( ref.m_primIndex ).m_triPair;
						triPair.split( axisIndex, position, curRef.m_box, leftRef.m_box, rightRef.m_box );
						if ( !leftRef.m_box.valid() || !rightRef.m_box.valid() )
						{
							leftRef.m_box						   = curRef.m_box;
							rightRef.m_box						   = curRef.m_box;
							ptr( leftRef.m_box.m_max )[axisIndex]  = position;
							ptr( rightRef.m_box.m_min )[axisIndex] = position;
						}
					}
					else
					{
						leftRef.m_box						   = curRef.m_box;
						rightRef.m_box						   = curRef.m_box;
						ptr( leftRef.m_box.m_max )[axisIndex]  = position;
						ptr( rightRef.m_box.m_min )[axisIndex] = position;
					}

					curBinAddr = i + axisIndex * binCount;
					if ( firstTaskIndex != lastTaskIndex ) curBinAddr = taskIndex + taskCount * curBinAddr;
					bins[curBinAddr].m_box.atomicGrow( leftRef.m_box );

					curRef = rightRef;
				}

				curBinAddr = firstBin + axisIndex * binCount;
				if ( firstTaskIndex != lastTaskIndex ) curBinAddr = taskIndex + taskCount * curBinAddr;
				atomicAdd( &bins[curBinAddr].m_enter, 1 );

				curBinAddr = lastBin + axisIndex * binCount;
				if ( firstTaskIndex != lastTaskIndex ) curBinAddr = taskIndex + taskCount * curBinAddr;
				atomicAdd( &bins[curBinAddr].m_exit, 1 );
				bins[curBinAddr].m_box.atomicGrow( curRef.m_box );
			}
		}
	}

	if ( firstTaskIndex == lastTaskIndex )
	{
		__syncthreads();
		for ( uint32_t binIndex = threadIdx.x; binIndex < binCount; binIndex += blockDim.x )
		{
			uint3 binOffset = make_uint3( binIndex ) + make_uint3( 0, 1, 2 ) * binCount;
			uint3 binAddr	= firstTaskIndex + taskCount * binOffset;

			if ( binCache[binOffset.x].m_box.valid() ) binsGlobal[binAddr.x].m_box.atomicGrow( binCache[binOffset.x].m_box );
			if ( binCache[binOffset.y].m_box.valid() ) binsGlobal[binAddr.y].m_box.atomicGrow( binCache[binOffset.y].m_box );
			if ( binCache[binOffset.z].m_box.valid() ) binsGlobal[binAddr.z].m_box.atomicGrow( binCache[binOffset.z].m_box );

			if ( binCache[binOffset.x].m_enter > 0 ) atomicAdd( &binsGlobal[binAddr.x].m_enter, binCache[binOffset.x].m_enter );
			if ( binCache[binOffset.y].m_enter > 0 ) atomicAdd( &binsGlobal[binAddr.y].m_enter, binCache[binOffset.y].m_enter );
			if ( binCache[binOffset.z].m_enter > 0 ) atomicAdd( &binsGlobal[binAddr.z].m_enter, binCache[binOffset.z].m_enter );

			if ( binCache[binOffset.x].m_exit > 0 ) atomicAdd( &binsGlobal[binAddr.x].m_exit, binCache[binOffset.x].m_exit );
			if ( binCache[binOffset.y].m_exit > 0 ) atomicAdd( &binsGlobal[binAddr.y].m_exit, binCache[binOffset.y].m_exit );
			if ( binCache[binOffset.z].m_exit > 0 ) atomicAdd( &binsGlobal[binAddr.z].m_exit, binCache[binOffset.z].m_exit );
		}
	}
}

extern "C" __global__ void BinReferencesSpatial_TriangleMesh(
	uint32_t	   activeRefCount,
	uint32_t	   binCount,
	uint32_t	   taskOffset,
	uint32_t	   taskCount,
	float		   overlapThreshold,
	float		   edgeThreshold,
	uint32_t*	   referenceIndices,
	uint32_t*	   taskIndices,
	Task*		   taskQueue,
	TriangleMesh   primitives,
	ReferenceNode* references,
	Bin*		   spatialBins )
{
	BinReferencesSpatial<TriangleMesh>(
		activeRefCount,
		binCount,
		taskOffset,
		taskCount,
		overlapThreshold,
		edgeThreshold,
		referenceIndices,
		taskIndices,
		taskQueue,
		primitives,
		references,
		spatialBins );
}

extern "C" __global__ void BinReferencesSpatial_AabbList(
	uint32_t	   activeRefCount,
	uint32_t	   binCount,
	uint32_t	   taskOffset,
	uint32_t	   taskCount,
	float		   overlapThreshold,
	float		   edgeThreshold,
	uint32_t*	   referenceIndices,
	uint32_t*	   taskIndices,
	Task*		   taskQueue,
	AabbList	   primitives,
	ReferenceNode* references,
	Bin*		   spatialBins )
{
	BinReferencesSpatial<AabbList>(
		activeRefCount,
		binCount,
		taskOffset,
		taskCount,
		overlapThreshold,
		edgeThreshold,
		referenceIndices,
		taskIndices,
		taskQueue,
		primitives,
		references,
		spatialBins );
}

extern "C" __global__ void BinReferencesSpatial_InstanceList_SRTFrame(
	uint32_t			   activeRefCount,
	uint32_t			   binCount,
	uint32_t			   taskOffset,
	uint32_t			   taskCount,
	float				   overlapThreshold,
	float				   edgeThreshold,
	uint32_t*			   referenceIndices,
	uint32_t*			   taskIndices,
	Task*				   taskQueue,
	InstanceList<SRTFrame> primitives,
	ReferenceNode*		   references,
	Bin*				   spatialBins )
{
	BinReferencesSpatial<InstanceList<SRTFrame>>(
		activeRefCount,
		binCount,
		taskOffset,
		taskCount,
		overlapThreshold,
		edgeThreshold,
		referenceIndices,
		taskIndices,
		taskQueue,
		primitives,
		references,
		spatialBins );
}

extern "C" __global__ void BinReferencesSpatial_InstanceList_MatrixFrame(
	uint32_t				  activeRefCount,
	uint32_t				  binCount,
	uint32_t				  taskOffset,
	uint32_t				  taskCount,
	float					  overlapThreshold,
	float					  edgeThreshold,
	uint32_t*				  referenceIndices,
	uint32_t*				  taskIndices,
	Task*					  taskQueue,
	InstanceList<MatrixFrame> primitives,
	ReferenceNode*			  references,
	Bin*					  spatialBins )
{
	BinReferencesSpatial<InstanceList<MatrixFrame>>(
		activeRefCount,
		binCount,
		taskOffset,
		taskCount,
		overlapThreshold,
		edgeThreshold,
		referenceIndices,
		taskIndices,
		taskQueue,
		primitives,
		references,
		spatialBins );
}

extern "C" __global__ void
FindObjectSplit( uint32_t taskCount, uint32_t binCount, uint32_t nodeCount, Bin* bins, Task* taskQueue )
{
	const uint32_t taskIndex = blockDim.x * blockIdx.x + threadIdx.x;

	if ( taskIndex < taskCount )
	{
		Bin leftBin, rightBin;
		Bin rightBins[SbvhMaxBinCount];

		float	 bestCost  = FltMax;
		uint32_t bestAxis  = InvalidValue;
		uint32_t bestIndex = InvalidValue;

		uint32_t nodeAddr = nodeCount - taskCount + taskIndex;
		Task	 task	  = taskQueue[nodeAddr];

		uint32_t nodeSize;
		for ( uint32_t axisIndex = 0; axisIndex < 3; ++axisIndex )
		{
			uint32_t binAddr		= taskIndex + taskCount * ( binCount - 1 + axisIndex * binCount );
			rightBins[binCount - 1] = bins[binAddr];
			for ( int32_t binIndex = binCount - 2; binIndex >= 0; --binIndex )
			{
				binAddr				= taskIndex + taskCount * ( binIndex + axisIndex * binCount );
				rightBins[binIndex] = rightBins[binIndex + 1];
				rightBins[binIndex].include( bins[binAddr] );
			}
			nodeSize = rightBins[0].m_counter;

			binAddr		   = taskIndex + taskCount * axisIndex * binCount;
			Bin curLeftBin = bins[binAddr];
			for ( uint32_t binIndex = 0; binIndex < binCount - 1; ++binIndex )
			{
				if ( curLeftBin.m_counter > 0 && rightBins[binIndex + 1].m_counter > 0 )
				{
					float cost = curLeftBin.cost() + rightBins[binIndex + 1].cost();
					if ( bestCost > cost )
					{
						bestCost  = cost;
						bestAxis  = axisIndex;
						bestIndex = binIndex;
						leftBin	  = curLeftBin;
						rightBin  = rightBins[binIndex + 1];
					}
				}
				binAddr = taskIndex + taskCount * ( binIndex + 1 + axisIndex * binCount );
				curLeftBin.include( bins[binAddr] );
			}
		}

		if ( bestIndex == InvalidValue )
		{
			bestCost		   = task.m_box.area() * nodeSize;
			bestAxis		   = 3;
			bestIndex		   = nodeSize >> 1;
			leftBin.m_counter  = bestIndex;
			rightBin.m_counter = nodeSize - bestIndex;
			leftBin.m_box	   = task.m_box;
			rightBin.m_box	   = task.m_box;
		}

		task.m_split.setSplitInfo( bestAxis, bestIndex, leftBin.m_counter == 1, rightBin.m_counter == 1, false );
		task.m_box0			= leftBin.m_box;
		task.m_counter0		= leftBin.m_counter;
		task.m_box1			= rightBin.m_box;
		task.m_counter1		= rightBin.m_counter;
		task.m_cost			= bestCost;
		taskQueue[nodeAddr] = task;
	}
}

template <bool spatialSplits>
__device__ void SplitReferences(
	uint32_t	 taskCount,
	uint32_t	 binCount,
	uint32_t	 nodeCount,
	uint32_t	 referenceCount,
	uint32_t	 maxReferenceCount,
	float		 overlapThreshold,
	float		 edgeThreshold,
	Bin*		 bins,
	ScratchNode* scratchNodes,
	Task*		 taskQueue,
	uint32_t*	 taskCounter,
	uint32_t*	 referenceCounter,
	uint32_t*	 refOffsetCounter )
{
	const uint32_t taskIndex = blockDim.x * blockIdx.x + threadIdx.x;
	const uint32_t taskEnd	 = RoundUp( taskCount, WarpSize );

	if ( taskIndex < taskEnd )
	{
		uint32_t nodeAddr;
		Task	 task;

		if ( taskIndex < taskCount )
		{
			nodeAddr = nodeCount - taskCount + taskIndex;
			task	 = taskQueue[nodeAddr];
		}

		if constexpr ( spatialSplits )
		{
			uint32_t duplicateCount = 0;
			Bin		 leftBin, rightBin;
			Bin		 rightBins[SbvhMaxBinCount];

			float	 bestCost  = FltMax;
			uint32_t bestAxis  = InvalidValue;
			uint32_t bestIndex = InvalidValue;

			if ( taskIndex < taskCount )
			{
				bestCost	 = task.m_cost;
				Aabb overlap = task.m_box0;
				overlap.intersect( task.m_box1 );

				if ( overlap.area() >= overlapThreshold )
				{
					for ( uint32_t axisIndex = 0; axisIndex < 3; ++axisIndex )
					{
						if ( ptr( task.m_box.m_max )[axisIndex] - ptr( task.m_box.m_min )[axisIndex] < edgeThreshold ) continue;

						uint32_t binAddr		= taskIndex + taskCount * ( binCount - 1 + axisIndex * binCount );
						rightBins[binCount - 1] = bins[binAddr];

						for ( int32_t binIndex = binCount - 2; binIndex >= 0; --binIndex )
						{
							binAddr				= taskIndex + taskCount * ( binIndex + axisIndex * binCount );
							rightBins[binIndex] = rightBins[binIndex + 1];
							rightBins[binIndex].include( bins[binAddr] );
						}

						binAddr		   = taskIndex + taskCount * axisIndex * binCount;
						Bin curLeftBin = bins[binAddr];
						for ( uint32_t binIndex = 0; binIndex < binCount - 1; ++binIndex )
						{
							if ( curLeftBin.m_enter > 0 && rightBins[binIndex + 1].m_exit > 0 )
							{
								float cost = curLeftBin.leftCost() + rightBins[binIndex + 1].rightCost();
								if ( bestCost > cost )
								{
									bestCost		  = cost;
									bestAxis		  = axisIndex;
									bestIndex		  = binIndex;
									leftBin			  = curLeftBin;
									leftBin.m_counter = leftBin.m_enter;
									rightBin		  = rightBins[binIndex + 1];
								}
							}
							binAddr = taskIndex + taskCount * ( binIndex + 1 + axisIndex * binCount );
							curLeftBin.include( bins[binAddr] );
						}
					}

					if ( bestIndex != InvalidValue )
						duplicateCount = leftBin.m_enter + rightBin.m_exit - task.m_counter0 - task.m_counter1;
				}
			}

			uint32_t referenceOffset = warpOffset( duplicateCount, referenceCounter );

			if ( taskIndex < taskCount )
			{
				if ( bestIndex != InvalidValue )
				{
					if ( referenceCount + referenceOffset + duplicateCount <= maxReferenceCount )
					{
						task.m_split.setSplitInfo( bestAxis, bestIndex, leftBin.m_enter == 1, rightBin.m_exit == 1, true );
						task.m_box0		= leftBin.m_box;
						task.m_counter0 = leftBin.m_enter;
						task.m_box1		= rightBin.m_box;
						task.m_counter1 = rightBin.m_exit;
					}
				}
			}
		}

		uint32_t outputTaskCount = ( task.m_counter0 > 1 ) + ( task.m_counter1 > 1 );
		uint32_t taskOffset		 = warpOffset( outputTaskCount, taskCounter );

		uint32_t refCount = 0;
		if ( task.m_counter0 > 1 ) refCount += task.m_counter0;
		if ( task.m_counter1 > 1 ) refCount += task.m_counter1;

		uint32_t leftRefOffset	= warpOffset( refCount, refOffsetCounter );
		uint32_t rightRefOffset = leftRefOffset;
		if ( task.m_counter0 > 1 ) rightRefOffset += task.m_counter0;

		if ( taskIndex < taskCount )
		{
			ScratchNode node;
			node.m_box = task.m_box;

			uint32_t nodeOffset = taskOffset;
			if ( task.m_counter0 > 1 )
			{
				uint32_t leftNodeAddr	= nodeCount + ( nodeOffset++ );
				node.m_childIndex0		= encodeNodeIndex( leftNodeAddr, BoxType );
				taskQueue[leftNodeAddr] = Task( task.m_box0, leftRefOffset );
			}

			if ( task.m_counter1 > 1 )
			{
				uint32_t rightNodeAddr	 = nodeCount + nodeOffset;
				node.m_childIndex1		 = encodeNodeIndex( rightNodeAddr, BoxType );
				taskQueue[rightNodeAddr] = Task( task.m_box1, rightRefOffset );
			}
			scratchNodes[nodeAddr] = node;

			task.m_refOffset	= 0;
			task.m_taskOffset	= taskOffset;
			taskQueue[nodeAddr] = task;
		}
	}
}

extern "C" __global__ void SplitReferences_true(
	uint32_t	 taskCount,
	uint32_t	 binCount,
	uint32_t	 nodeCount,
	uint32_t	 referenceCount,
	uint32_t	 maxReferenceCount,
	float		 overlapThreshold,
	float		 edgeThreshold,
	Bin*		 bins,
	ScratchNode* scratchNodes,
	Task*		 taskQueue,
	uint32_t*	 taskCounter,
	uint32_t*	 referenceCounter,
	uint32_t*	 refOffsetCounter )
{
	SplitReferences<true>(
		taskCount,
		binCount,
		nodeCount,
		referenceCount,
		maxReferenceCount,
		overlapThreshold,
		edgeThreshold,
		bins,
		scratchNodes,
		taskQueue,
		taskCounter,
		referenceCounter,
		refOffsetCounter );
}

extern "C" __global__ void SplitReferences_false(
	uint32_t	 taskCount,
	uint32_t	 binCount,
	uint32_t	 nodeCount,
	uint32_t	 referenceCount,
	uint32_t	 maxReferenceCount,
	float		 overlapThreshold,
	float		 edgeThreshold,
	Bin*		 bins,
	ScratchNode* scratchNodes,
	Task*		 taskQueue,
	uint32_t*	 taskCounter,
	uint32_t*	 referenceCounter,
	uint32_t*	 refOffsetCounter )
{
	SplitReferences<false>(
		taskCount,
		binCount,
		nodeCount,
		referenceCount,
		maxReferenceCount,
		overlapThreshold,
		edgeThreshold,
		bins,
		scratchNodes,
		taskQueue,
		taskCounter,
		referenceCounter,
		refOffsetCounter );
}

template <typename PrimitiveContainer>
__device__ void DistributeReferences(
	uint32_t			activeRefCount,
	uint32_t			referenceCount,
	uint32_t			binCount,
	uint32_t			nodeCount,
	uint32_t			taskCount,
	uint32_t			taskOffset,
	uint32_t*			referenceIndices0,
	uint32_t*			referenceIndices1,
	uint32_t*			taskIndices,
	Task*				taskQueue,
	PrimitiveContainer& primitives,
	ScratchNode*		scratchNodes,
	ReferenceNode*		references,
	uint32_t*			referenceCounter )
{
	const uint32_t index = threadIdx.x + blockIdx.x * blockDim.x;

	uint32_t leafType;
	if constexpr ( is_same<PrimitiveContainer, TriangleMesh>::value )
		leafType = TriangleType;
	else if constexpr ( is_same<PrimitiveContainer, AabbList>::value )
		leafType = CustomType;
	else if constexpr (
		is_same<PrimitiveContainer, InstanceList<SRTFrame>>::value ||
		is_same<PrimitiveContainer, InstanceList<MatrixFrame>>::value )
		leafType = InstanceType;

	if ( index < activeRefCount )
	{
		uint32_t referenceIndex = referenceIndices0[index];
		uint32_t taskIndex		= taskIndices[referenceIndex];
		uint32_t nodeAddr		= taskIndex + taskOffset;

		Task		  task = taskQueue[nodeAddr];
		ReferenceNode ref  = references[referenceIndex];

		uint32_t splitAxis	  = task.m_split.m_splitAxis;
		uint32_t splitIndex	  = task.m_split.m_splitIndex;
		bool	 leftLeaf	  = task.m_split.m_leftLeaf;
		bool	 rightLeaf	  = task.m_split.m_rightLeaf;
		bool	 spatialSplit = task.m_split.m_spatialSplit;

		if ( !spatialSplit )
		{
			float3 k = ( 1.0f - SbvhEpsilon ) * ( static_cast<float>( binCount ) / ( task.m_box.m_max - task.m_box.m_min ) );
			uint3  binIndex = clamp(
				 make_uint3( k * ( ref.m_box.center() - task.m_box.m_min ) ), make_uint3( 0 ), make_uint3( binCount - 1 ) );

			bool onLeft;
			if ( splitAxis < 3 )
				onLeft = ptr( binIndex )[splitAxis] <= splitIndex;
			else
				onLeft = atomicAdd( &taskQueue[nodeAddr].m_refOffset, 1 ) < splitIndex;

			uint32_t newTaskIndex = task.m_taskOffset;
			if ( !onLeft && !leftLeaf ) ++newTaskIndex;

			if ( ( !onLeft || leftLeaf ) && ( onLeft || rightLeaf ) )
			{
				if ( onLeft )
					scratchNodes[nodeAddr].m_childIndex0 = encodeNodeIndex( referenceIndex, leafType );
				else
					scratchNodes[nodeAddr].m_childIndex1 = encodeNodeIndex( referenceIndex, leafType );
				ScratchNode node = scratchNodes[nodeAddr];
			}
			else
			{
				uint32_t newIndex			= atomicAdd( &taskQueue[newTaskIndex + nodeCount].m_refOffset, 1 );
				taskIndices[referenceIndex] = newTaskIndex;
				referenceIndices1[newIndex] = referenceIndex;
			}
		}

		else
		{
			float	 binSize  = ( ptr( task.m_box.m_max )[splitAxis] - ptr( task.m_box.m_min )[splitAxis] ) / float( binCount );
			uint32_t firstBin = binCount - 1;
			uint32_t lastBin  = firstBin;
			for ( uint32_t i = 0; i < binCount; ++i )
			{
				float position = ptr( task.m_box.m_min )[splitAxis] + binSize * static_cast<float>( i + 1 );
				if ( firstBin == binCount - 1 && ptr( ref.m_box.m_min )[splitAxis] < position ) firstBin = i;
				if ( lastBin == binCount - 1 && ptr( ref.m_box.m_max )[splitAxis] <= position ) lastBin = i;
			}
			if ( firstBin > lastBin ) firstBin = lastBin;

			float position	 = ptr( task.m_box.m_min )[splitAxis] + binSize * static_cast<float>( splitIndex + 1 );
			bool  duplicated = firstBin <= splitIndex && lastBin > splitIndex;
			if ( duplicated )
			{
				ReferenceNode leftRef( ref.m_primIndex );
				ReferenceNode rightRef( ref.m_primIndex );
				if constexpr ( is_same<PrimitiveContainer, TriangleMesh>::value )
				{
					TrianglePair triPair = primitives.fetchTriangleNode( ref.m_primIndex ).m_triPair;
					triPair.split( splitAxis, position, ref.m_box, leftRef.m_box, rightRef.m_box );
					if ( !leftRef.m_box.valid() || !rightRef.m_box.valid() )
					{
						leftRef.m_box						   = ref.m_box;
						rightRef.m_box						   = ref.m_box;
						ptr( leftRef.m_box.m_max )[splitAxis]  = position;
						ptr( rightRef.m_box.m_min )[splitAxis] = position;
					}
				}
				else
				{
					leftRef.m_box						   = ref.m_box;
					rightRef.m_box						   = ref.m_box;
					ptr( leftRef.m_box.m_max )[splitAxis]  = position;
					ptr( rightRef.m_box.m_min )[splitAxis] = position;
				}

				uint32_t referenceOffset	  = atomicAdd( referenceCounter, 1 );
				uint32_t newReferenceIndex	  = referenceCount + referenceOffset;
				references[referenceIndex]	  = leftRef;
				references[newReferenceIndex] = rightRef;

				uint32_t newTaskIndex = task.m_taskOffset;
				if ( leftLeaf )
				{
					scratchNodes[nodeAddr].m_childIndex0 = encodeNodeIndex( referenceIndex, leafType );
					ScratchNode node					 = scratchNodes[nodeAddr];
				}
				else
				{
					uint32_t newIndex			= atomicAdd( &taskQueue[newTaskIndex + nodeCount].m_refOffset, 1 );
					taskIndices[referenceIndex] = newTaskIndex;
					referenceIndices1[newIndex] = referenceIndex;
				}

				newTaskIndex = task.m_taskOffset;
				if ( !leftLeaf ) ++newTaskIndex;
				if ( rightLeaf )
				{
					scratchNodes[nodeAddr].m_childIndex1 = encodeNodeIndex( newReferenceIndex, leafType );
					ScratchNode node					 = scratchNodes[nodeAddr];
				}
				else
				{
					uint32_t newIndex			   = atomicAdd( &taskQueue[newTaskIndex + nodeCount].m_refOffset, 1 );
					taskIndices[newReferenceIndex] = newTaskIndex;
					referenceIndices1[newIndex]	   = newReferenceIndex;
				}
			}
			else
			{
				uint32_t newTaskIndex = task.m_taskOffset;
				bool	 onLeft		  = ptr( ref.m_box.m_max )[splitAxis] <= position;
				if ( !onLeft && !leftLeaf ) ++newTaskIndex;

				if ( ( !onLeft || leftLeaf ) && ( onLeft || rightLeaf ) )
				{
					if ( onLeft )
						scratchNodes[nodeAddr].m_childIndex0 = encodeNodeIndex( referenceIndex, leafType );
					else
						scratchNodes[nodeAddr].m_childIndex1 = encodeNodeIndex( referenceIndex, leafType );
					ScratchNode node = scratchNodes[nodeAddr];
				}
				else
				{
					uint32_t newIndex			= atomicAdd( &taskQueue[newTaskIndex + nodeCount].m_refOffset, 1 );
					taskIndices[referenceIndex] = newTaskIndex;
					referenceIndices1[newIndex] = referenceIndex;
				}
			}
		}
	}
}

extern "C" __global__ void DistributeReferences_TriangleMesh(
	uint32_t	   activeRefCount,
	uint32_t	   referenceCount,
	uint32_t	   binCount,
	uint32_t	   nodeCount,
	uint32_t	   taskCount,
	uint32_t	   taskOffset,
	uint32_t*	   referenceIndices0,
	uint32_t*	   referenceIndices1,
	uint32_t*	   taskIndices,
	Task*		   taskQueue,
	TriangleMesh   primitives,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	uint32_t*	   referenceCounter )
{
	DistributeReferences<TriangleMesh>(
		activeRefCount,
		referenceCount,
		binCount,
		nodeCount,
		taskCount,
		taskOffset,
		referenceIndices0,
		referenceIndices1,
		taskIndices,
		taskQueue,
		primitives,
		scratchNodes,
		references,
		referenceCounter );
}

extern "C" __global__ void DistributeReferences_AabbList(
	uint32_t	   activeRefCount,
	uint32_t	   referenceCount,
	uint32_t	   binCount,
	uint32_t	   nodeCount,
	uint32_t	   taskCount,
	uint32_t	   taskOffset,
	uint32_t*	   referenceIndices0,
	uint32_t*	   referenceIndices1,
	uint32_t*	   taskIndices,
	Task*		   taskQueue,
	AabbList	   primitives,
	ScratchNode*   scratchNodes,
	ReferenceNode* references,
	uint32_t*	   referenceCounter )
{
	DistributeReferences<AabbList>(
		activeRefCount,
		referenceCount,
		binCount,
		nodeCount,
		taskCount,
		taskOffset,
		referenceIndices0,
		referenceIndices1,
		taskIndices,
		taskQueue,
		primitives,
		scratchNodes,
		references,
		referenceCounter );
}

extern "C" __global__ void DistributeReferences_InstanceList_SRTFrame(
	uint32_t			   activeRefCount,
	uint32_t			   referenceCount,
	uint32_t			   binCount,
	uint32_t			   nodeCount,
	uint32_t			   taskCount,
	uint32_t			   taskOffset,
	uint32_t*			   referenceIndices0,
	uint32_t*			   referenceIndices1,
	uint32_t*			   taskIndices,
	Task*				   taskQueue,
	InstanceList<SRTFrame> primitives,
	ScratchNode*		   scratchNodes,
	ReferenceNode*		   references,
	uint32_t*			   referenceCounter )
{
	DistributeReferences<InstanceList<SRTFrame>>(
		activeRefCount,
		referenceCount,
		binCount,
		nodeCount,
		taskCount,
		taskOffset,
		referenceIndices0,
		referenceIndices1,
		taskIndices,
		taskQueue,
		primitives,
		scratchNodes,
		references,
		referenceCounter );
}

extern "C" __global__ void DistributeReferences_InstanceList_MatrixFrame(
	uint32_t				  activeRefCount,
	uint32_t				  referenceCount,
	uint32_t				  binCount,
	uint32_t				  nodeCount,
	uint32_t				  taskCount,
	uint32_t				  taskOffset,
	uint32_t*				  referenceIndices0,
	uint32_t*				  referenceIndices1,
	uint32_t*				  taskIndices,
	Task*					  taskQueue,
	InstanceList<MatrixFrame> primitives,
	ScratchNode*			  scratchNodes,
	ReferenceNode*			  references,
	uint32_t*				  referenceCounter )
{
	DistributeReferences<InstanceList<MatrixFrame>>(
		activeRefCount,
		referenceCount,
		binCount,
		nodeCount,
		taskCount,
		taskOffset,
		referenceIndices0,
		referenceIndices1,
		taskIndices,
		taskQueue,
		primitives,
		scratchNodes,
		references,
		referenceCounter );
}
