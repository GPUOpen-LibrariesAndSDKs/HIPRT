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
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/Obb.h>
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/Kernel.h>
#include <hiprt/impl/MemoryArena.h>
#include <hiprt/impl/SbvhCommon.h>
#include <hiprt/impl/Timer.h>
#include <hiprt/impl/Utility.h>
#include <hiprt/impl/BvhConfig.h>

#if defined( HIPRT_LOAD_FROM_STRING )
#include <hiprt/cache/Kernels.h>
#include <hiprt/cache/KernelArgs.h>
#endif

namespace hiprt
{
class SbvhBuilder
{
  public:
	static constexpr uint32_t ReductionBlockSize = BvhBuilderReductionBlockSize;
	static constexpr uint32_t MinBinCount		 = SbvhMinBinCount;
	static constexpr uint32_t MaxBinCount		 = SbvhMaxBinCount;
	static constexpr float	  Alpha				 = SbvhAlpha;
	static constexpr float	  Beta				 = SbvhBeta;
	static constexpr float	  Gamma				 = SbvhGamma;
	static constexpr float	  Epsilon			 = SbvhEpsilon;

	enum Times
	{
		PairTrianglesTime,
		ComputeBoxTime,
		SetupReferencesTime,
		ResetBinsTime,
		BinReferencesObjectTime,
		FindObjectSplitTime,
		BinReferencesSpatialTime,
		SplitTime,
		DistributeReferencesTime,
		ComputeParentAddrsTime,
		ComputeFatLeavesTime,
		CollapseTime,
		CompactTasksTime,
		PackLeavesTime,
		FitOrientedBoundsTime
	};

	SbvhBuilder()								 = delete;
	SbvhBuilder& operator=( const SbvhBuilder& ) = delete;

	template <typename BuildInput>
	static size_t
	getTemporaryBufferSize( Context& context, const BuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static size_t
	getTemporaryBufferSize( Context& context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static size_t
	getTemporaryBufferSize( Context& context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static size_t
	getStorageBufferSize( Context& context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static size_t
	getStorageBufferSize( Context& context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static void build(
		Context&					   context,
		const hiprtGeometryBuildInput& buildInput,
		const hiprtBuildOptions		   buildOptions,
		hiprtDevicePtr				   temporaryBuffer,
		oroStream					   stream,
		hiprtDevicePtr				   buffer );

	static void build(
		Context&					context,
		const hiprtSceneBuildInput& buildInput,
		const hiprtBuildOptions		buildOptions,
		hiprtDevicePtr				temporaryBuffer,
		oroStream					stream,
		hiprtDevicePtr				buffer );

	template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
	static void build(
		Context&				context,
		PrimitiveContainer&		primitives,
		const hiprtBuildOptions buildOptions,
		uint32_t				geomType,
		MemoryArena&			temporaryMemoryArena,
		oroStream				stream,
		MemoryArena&			storageMemoryArena );

	static void update(
		Context&					   context,
		const hiprtGeometryBuildInput& buildInput,
		const hiprtBuildOptions		   buildOptions,
		hiprtDevicePtr				   temporaryBuffer,
		oroStream					   stream,
		hiprtDevicePtr				   buffer );

	static void update(
		Context&					context,
		const hiprtSceneBuildInput& buildInput,
		const hiprtBuildOptions		buildOptions,
		hiprtDevicePtr				temporaryBuffer,
		oroStream					stream,
		hiprtDevicePtr				buffer );

	template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
	static void update(
		Context&				context,
		PrimitiveContainer&		primitives,
		const hiprtBuildOptions buildOptions,
		oroStream				stream,
		MemoryArena&			storageMemoryArena );
};

template <typename BuildInput>
size_t
SbvhBuilder::getTemporaryBufferSize( Context& context, const BuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	size_t count{};
	if constexpr ( std::is_same<BuildInput, hiprtGeometryBuildInput>::value )
		count = getPrimCount( buildInput );
	else
		count = buildInput.instanceCount;
	const bool	spatialSplits = !( buildOptions.buildFlags & hiprtBuildFlagBitDisableSpatialSplits );
	const bool	kdops = context.getRtip() >= 31 && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableOrientedBoundingBoxes );
	const float alpha = !spatialSplits ? 1.0f : Alpha;
	const size_t maxReferenceCount = alpha * count;
	const size_t boxNodeCount	   = getMaxBoxNodeCount( buildInput, context.getRtip(), maxReferenceCount );
	const size_t size =
		RoundUp( sizeof( Aabb ), DefaultAlignment ) + RoundUp( maxReferenceCount * sizeof( Task ), DefaultAlignment ) +
		RoundUp( maxReferenceCount * sizeof( ScratchNode ), DefaultAlignment ) +
		RoundUp( maxReferenceCount * sizeof( ReferenceNode ), DefaultAlignment ) +
		RoundUp( maxReferenceCount * sizeof( uint32_t ), DefaultAlignment ) +
		3 * RoundUp( sizeof( uint32_t ), DefaultAlignment ) +
		( !spatialSplits ? 1 : 2 ) * RoundUp( ( maxReferenceCount / 2 ) * sizeof( Bin ) * 3 * MinBinCount, DefaultAlignment );
	const size_t obbSize = kdops ? RoundUp( boxNodeCount * sizeof( Kdop ), DefaultAlignment ) +
									   RoundUp( boxNodeCount * sizeof( uint32_t ), DefaultAlignment )
								 : 0;
	return std::max( size, obbSize );
}

template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
void SbvhBuilder::build(
	Context&				context,
	PrimitiveContainer&		primitives,
	const hiprtBuildOptions buildOptions,
	uint32_t				geomType,
	MemoryArena&			temporaryMemoryArena,
	oroStream				stream,
	MemoryArena&			storageMemoryArena )
{
	using Header = typename std::conditional<
		std::is_same<PrimitiveNode, UserInstanceNode>::value || std::is_same<PrimitiveNode, HwInstanceNode>::value,
		SceneHeader,
		GeomHeader>::type;

	bool		   spatialSplits	 = !( buildOptions.buildFlags & hiprtBuildFlagBitDisableSpatialSplits );
	const float	   alpha			 = spatialSplits ? Alpha : 1.0f;
	const uint32_t maxReferenceCount = alpha * primitives.getCount();
	const uint32_t maxBoxNodeCount	 = static_cast<uint32_t>( getMaxBoxNodeCount<BoxNode, PrimitiveNode>( maxReferenceCount ) );
	const uint32_t maxPrimNodeCount	 = static_cast<uint32_t>( getMaxPrimNodeCount<PrimitiveNode>( maxReferenceCount ) );

	Header*		   header	 = storageMemoryArena.allocate<Header>();
	BoxNode*	   boxNodes	 = storageMemoryArena.allocate<BoxNode>( maxBoxNodeCount );
	PrimitiveNode* primNodes = storageMemoryArena.allocate<PrimitiveNode>( maxPrimNodeCount );

	Aabb*		   box			= temporaryMemoryArena.allocate<Aabb>();
	Task*		   taskQueue	= temporaryMemoryArena.allocate<Task>( maxReferenceCount );
	uint32_t*	   taskIndices	= temporaryMemoryArena.allocate<uint32_t>( maxReferenceCount );
	ScratchNode*   scratchNodes = temporaryMemoryArena.allocate<ScratchNode>( maxReferenceCount );
	ReferenceNode* references	= temporaryMemoryArena.allocate<ReferenceNode>( maxReferenceCount );
	Bin*		   objectBins	= temporaryMemoryArena.allocate<Bin>( 3 * MinBinCount * ( maxReferenceCount / 2 ) );
	Bin*		   spatialBins =
		  spatialSplits ? temporaryMemoryArena.allocate<Bin>( 3 * MinBinCount * ( maxReferenceCount / 2 ) ) : nullptr;
	uint32_t* taskCounter	   = temporaryMemoryArena.allocate<uint32_t>();
	uint32_t* referenceCounter = temporaryMemoryArena.allocate<uint32_t>();
	uint32_t* refOffsetCounter = temporaryMemoryArena.allocate<uint32_t>();

	uint32_t* referenceIndices[2];
	referenceIndices[0] = reinterpret_cast<uint32_t*>( boxNodes ) + 0 * maxReferenceCount;
	referenceIndices[1] = reinterpret_cast<uint32_t*>( boxNodes ) + 1 * maxReferenceCount;

	Timer					 timer;
	Compiler&				 compiler = context.getCompiler();
	std::vector<const char*> opts;
	// opts.push_back( "-G" );

	const std::string headerParam			 = Compiler::kernelNameSufix( Traits<Header>::TYPE_NAME );
	const std::string containerParam		 = Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );
	const std::string primNodeParam			 = Compiler::kernelNameSufix( Traits<PrimitiveNode>::TYPE_NAME );
	const std::string binNodeParam			 = Compiler::kernelNameSufix( Traits<ScratchNode>::TYPE_NAME );
	const std::string containerPrimNodeParam = containerParam + "_" + primNodeParam;
	const std::string primNodeBinNodeParam	 = primNodeParam + "_" + binNodeParam;

	bool pairTriangles = false;
	if constexpr (
		std::is_same<PrimitiveNode, TrianglePairNode>::value || std::is_same<PrimitiveNode, TrianglePacketNode>::value )
		pairTriangles = primitives.pairable() && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableTrianglePairing );

	// STEP 0: Init data
	if constexpr ( std::is_same<Header, SceneHeader>::value )
	{
		Instance* instances = storageMemoryArena.allocate<Instance>( primitives.getCount() );
		Frame*	  frames	= storageMemoryArena.allocate<Frame>( primitives.getFrameCount() );

		primitives.setFrames( frames );
		Kernel initDataKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"InitSceneData_" + containerParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		initDataKernel.setArgs(
			{ storageMemoryArena.getStorageSize(), primitives, boxNodes, primNodes, instances, frames, header } );
		initDataKernel.launch( std::max( primitives.getFrameCount(), primitives.getCount() ), stream );
	}
	else
	{
		geomType <<= 1;
		if constexpr (
			std::is_same<PrimitiveNode, TrianglePairNode>::value || std::is_same<PrimitiveNode, TrianglePacketNode>::value )
			geomType |= 1;
		const uint32_t primCount	  = pairTriangles ? 0u : primitives.getCount();
		Kernel		   initDataKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"InitGeomData",
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		initDataKernel.setArgs( { storageMemoryArena.getStorageSize(), primCount, boxNodes, primNodes, geomType, header } );
		initDataKernel.launch( 1, stream );
	}

	// A single primitive => special case
	if ( primitives.getCount() == 1 )
	{
		Kernel singletonConstructionKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"SingletonConstruction_" + containerPrimNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		singletonConstructionKernel.setArgs( { primitives, boxNodes, primNodes } );
		singletonConstructionKernel.launch( 1, stream );
		return;
	}

	// STEP 1: Pair triangles
	if constexpr (
		std::is_same<PrimitiveNode, TrianglePairNode>::value || std::is_same<PrimitiveNode, TrianglePacketNode>::value )
	{
		if ( pairTriangles )
		{
			uint2* pairIndices = temporaryMemoryArena.allocate<uint2>( primitives.getCount() );
			checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( taskCounter ), 0, sizeof( uint32_t ), stream ) );
			Kernel pairTrianglesKernel = compiler.getKernel(
				context,
				Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
				"PairTriangles",
				opts,
				GET_ARG_LIST( BvhBuilderKernels ) );
			pairTrianglesKernel.setArgs( { primitives, pairIndices, taskCounter } );
			timer.measure( PairTrianglesTime, [&]() { pairTrianglesKernel.launch( primitives.getCount(), stream ); } );

			uint32_t pairCount = 0;
			checkOro(
				oroMemcpyDtoHAsync( &pairCount, reinterpret_cast<oroDeviceptr>( taskCounter ), sizeof( uint32_t ), stream ) );
			checkOro( oroStreamSynchronize( stream ) );
			primitives.setPairs( pairCount, pairIndices );
		}
	}

	// STEP 2: Calculate bounding box by reduction
	Aabb emptyBox;
	checkOro( oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( box ), &emptyBox, sizeof( Aabb ), stream ) );

	Kernel computeBoxKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"ComputeBox_" + containerParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	computeBoxKernel.setArgs( { primitives, box } );
	timer.measure( ComputeBoxTime, [&]() { computeBoxKernel.launch( primitives.getCount(), ReductionBlockSize, stream ); } );

	Aabb sceneBox;
	checkOro( oroMemcpyDtoHAsync( &sceneBox, reinterpret_cast<oroDeviceptr>( box ), sizeof( Aabb ), stream ) );
	checkOro( oroStreamSynchronize( stream ) );

	// STEP 3: Setup references
	Kernel setupReferencesKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/SbvhBuilderKernels.h",
		"SetupLeavesAndReferences_" + containerParam,
		opts,
		GET_ARG_LIST( SbvhBuilderKernels ) );
	setupReferencesKernel.setArgs( { primitives, references, taskQueue, box, referenceIndices[0], taskIndices } );
	timer.measure( SetupReferencesTime, [&]() { setupReferencesKernel.launch( primitives.getCount(), stream ); } );

	// STEP 4: Construction
	uint32_t referenceCount	  = primitives.getCount();
	uint32_t activeRefCount	  = referenceCount;
	uint32_t nodeCount		  = 1;
	uint32_t taskCount		  = 1;
	float	 overlapThreshold = sceneBox.area() * Beta;
	float3	 sceneExtent	  = sceneBox.extent();
	float	 maxExtent		  = std::max( std::max( sceneExtent.x, sceneExtent.y ), sceneExtent.z );
	float	 edgeThreshold	  = maxExtent * Gamma;
	bool	 swapBuffers	  = false;

	while ( taskCount > 0 )
	{
		uint32_t taskOffset = nodeCount - taskCount;
		uint32_t binCount	= std::min( MinBinCount * ( maxReferenceCount / 2 ) / taskCount, MaxBinCount );

		std::string spatialSplitsString = spatialSplits ? "true" : "false";
		std::string spatialSplitsContainerParam =
			spatialSplitsString + "_" + Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );

		// Reset bins
		Kernel resetBinsKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/SbvhBuilderKernels.h",
			"ResetBins_" + spatialSplitsString,
			opts,
			GET_ARG_LIST( SbvhBuilderKernels ) );
		resetBinsKernel.setArgs( { taskCount, binCount, objectBins, spatialBins } );
		timer.measure( ResetBinsTime, [&]() { resetBinsKernel.launch( 3 * binCount * taskCount, stream ); } );

		// Object bin references
		Kernel binReferencesObjectKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/SbvhBuilderKernels.h",
			"BinReferencesObject",
			opts,
			GET_ARG_LIST( SbvhBuilderKernels ) );
		binReferencesObjectKernel.setArgs(
			{ activeRefCount,
			  binCount,
			  taskOffset,
			  taskCount,
			  referenceIndices[swapBuffers],
			  taskIndices,
			  taskQueue,
			  references,
			  objectBins } );
		timer.measure( BinReferencesObjectTime, [&]() { binReferencesObjectKernel.launch( activeRefCount, stream ); } );

		// Find object split
		Kernel findObjectSplitKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/SbvhBuilderKernels.h",
			"FindObjectSplit",
			opts,
			GET_ARG_LIST( SbvhBuilderKernels ) );
		findObjectSplitKernel.setArgs( { taskCount, binCount, nodeCount, objectBins, taskQueue } );
		timer.measure( FindObjectSplitTime, [&]() { findObjectSplitKernel.launch( taskCount, stream ); } );

		// Spatial bin references
		if ( spatialSplits )
		{
			Kernel binReferencesSpatialKernel = compiler.getKernel(
				context,
				Utility::getRootDir() / "hiprt/impl/SbvhBuilderKernels.h",
				"BinReferencesSpatial_" + containerParam,
				opts,
				GET_ARG_LIST( SbvhBuilderKernels ) );
			binReferencesSpatialKernel.setArgs(
				{ activeRefCount,
				  binCount,
				  taskOffset,
				  taskCount,
				  overlapThreshold,
				  edgeThreshold,
				  referenceIndices[swapBuffers],
				  taskIndices,
				  taskQueue,
				  primitives,
				  references,
				  spatialBins } );
			timer.measure( BinReferencesSpatialTime, [&]() { binReferencesSpatialKernel.launch( activeRefCount, stream ); } );
		}

		// Split
		checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( taskCounter ), 0, sizeof( uint32_t ), stream ) );
		checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( referenceCounter ), 0, sizeof( uint32_t ), stream ) );
		checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( refOffsetCounter ), 0, sizeof( uint32_t ), stream ) );
		Kernel splitKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/SbvhBuilderKernels.h",
			"SplitReferences_" + spatialSplitsString,
			opts,
			GET_ARG_LIST( SbvhBuilderKernels ) );
		splitKernel.setArgs(
			{ taskCount,
			  binCount,
			  nodeCount,
			  referenceCount,
			  maxReferenceCount,
			  overlapThreshold,
			  edgeThreshold,
			  spatialBins,
			  scratchNodes,
			  taskQueue,
			  taskCounter,
			  referenceCounter,
			  refOffsetCounter } );
		timer.measure( SplitTime, [&]() { splitKernel.launch( taskCount, stream ); } );

		uint32_t newReferenceCountEstimate;
		checkOro( oroMemcpyDtoHAsync(
			&newReferenceCountEstimate, reinterpret_cast<oroDeviceptr>( referenceCounter ), sizeof( uint32_t ), stream ) );
		checkOro( oroStreamSynchronize( stream ) );

		// Distribute references
		checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( referenceCounter ), 0, sizeof( uint32_t ), stream ) );
		Kernel distributeReferencesKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/SbvhBuilderKernels.h",
			"DistributeReferences_" + containerParam,
			opts,
			GET_ARG_LIST( SbvhBuilderKernels ) );
		distributeReferencesKernel.setArgs(
			{ activeRefCount,
			  referenceCount,
			  binCount,
			  nodeCount,
			  taskCount,
			  taskOffset,
			  referenceIndices[swapBuffers],
			  referenceIndices[!swapBuffers],
			  taskIndices,
			  taskQueue,
			  primitives,
			  scratchNodes,
			  references,
			  referenceCounter } );
		timer.measure( DistributeReferencesTime, [&]() { distributeReferencesKernel.launch( referenceCount, stream ); } );

		uint32_t newTaskCount;
		uint32_t newReferenceCount;
		checkOro(
			oroMemcpyDtoHAsync( &newTaskCount, reinterpret_cast<oroDeviceptr>( taskCounter ), sizeof( uint32_t ), stream ) );
		checkOro( oroMemcpyDtoHAsync(
			&newReferenceCount, reinterpret_cast<oroDeviceptr>( referenceCounter ), sizeof( uint32_t ), stream ) );
		checkOro( oroMemcpyDtoHAsync(
			&activeRefCount, reinterpret_cast<oroDeviceptr>( refOffsetCounter ), sizeof( uint32_t ), stream ) );
		checkOro( oroStreamSynchronize( stream ) );

		if ( referenceCount + newReferenceCountEstimate >= maxReferenceCount ) spatialSplits = false;

		nodeCount += newTaskCount;
		taskCount = newTaskCount;
		referenceCount += newReferenceCount;
		swapBuffers = !swapBuffers;
	}

	// STEP 5: Compute fat leaves
	if constexpr ( std::is_same<PrimitiveNode, TrianglePacketNode>::value )
	{
		uint32_t rootAddr = 0;

		uint32_t* updateCounters = reinterpret_cast<uint32_t*>( taskQueue );
		uint32_t* parentAddrs	 = updateCounters + referenceCount;
		uint32_t* triangleCounts = parentAddrs + referenceCount;
		checkOro( oroMemsetD8Async(
			reinterpret_cast<oroDeviceptr>( updateCounters ), 0, sizeof( uint32_t ) * referenceCount, stream ) );

		Kernel computeParentAddrsKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"ComputeParentAddrs_" + binNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		computeParentAddrsKernel.setArgs( { referenceCount, rootAddr, scratchNodes, parentAddrs } );
		timer.measure( ComputeParentAddrsTime, [&]() { computeParentAddrsKernel.launch( referenceCount, stream ); } );

		Kernel computeFatLeaves = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"ComputeFatLeaves_" + binNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		computeFatLeaves.setArgs( { referenceCount, scratchNodes, parentAddrs, triangleCounts, updateCounters } );
		timer.measure( ComputeFatLeavesTime, [&]() { computeFatLeaves.launch( referenceCount, stream ); } );
	}

	// STEP 6: Collapse
	uint3 rootCollapseTask = { RootIndex, 0, 0 };
	checkOro( oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( taskQueue ), &rootCollapseTask, sizeof( uint3 ), stream ) );
	checkOro( oroMemsetD8Async(
		reinterpret_cast<oroDeviceptr>( reinterpret_cast<uint3*>( taskQueue ) + 1 ),
		0xFF,
		sizeof( uint3 ) * ( referenceCount - 1 ),
		stream ) );

	Kernel collapseKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"Collapse_" + primNodeBinNodeParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	collapseKernel.setArgs(
		{ maxBoxNodeCount, referenceCount, header, scratchNodes, references, boxNodes, taskQueue, taskIndices } );
	timer.measure( CollapseTime, [&]() { collapseKernel.launch( context.getBranchingFactor() * maxBoxNodeCount, stream ); } );

	uint32_t boxNodeCount{};
	checkOro( oroMemcpyDtoHAsync(
		&boxNodeCount, reinterpret_cast<oroDeviceptr>( &header->m_boxNodeCount ), sizeof( uint32_t ), stream ) );
	checkOro( oroStreamSynchronize( stream ) );

	Kernel compactTasksKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"CompactTasks",
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	compactTasksKernel.setArgs( { boxNodeCount, taskQueue, taskCounter } );
	timer.measure( CompactTasksTime, [&]() {
		compactTasksKernel.launch( BvhBuilderCompactionBlockSize, BvhBuilderCompactionBlockSize, stream );
	} );

	checkOro( oroMemcpyDtoHAsync( &taskCount, reinterpret_cast<oroDeviceptr>( taskCounter ), sizeof( uint32_t ), stream ) );
	checkOro( oroStreamSynchronize( stream ) );

	Kernel packLeavesKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"PackLeaves_" + containerPrimNodeParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	packLeavesKernel.setArgs( { taskCount, header, references, boxNodes, primNodes, primitives, taskQueue, taskIndices } );
	timer.measure( PackLeavesTime, [&]() {
		if constexpr ( std::is_same<PrimitiveNode, TrianglePacketNode>::value )
			packLeavesKernel.launch( taskCount * LanesPerLeafPacketTask, context.getWarpSize(), stream );
		else
			packLeavesKernel.launch( taskCount, stream );
	} );

	// STEP 7: Fit oriented bounding boxes
	if ( context.getRtip() >= 31 && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableOrientedBoundingBoxes ) )
	{
		uint32_t* updateCounters = reinterpret_cast<uint32_t*>( box );
		Kdop*	  kdops			 = reinterpret_cast<Kdop*>( updateCounters + boxNodeCount );
		checkOro( oroMemsetD8Async(
			reinterpret_cast<oroDeviceptr>( updateCounters ), 0, sizeof( uint32_t ) * boxNodeCount, stream ) );
		Kernel fitOrientedBoundsKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"FitOrientedBounds_" + containerPrimNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		fitOrientedBoundsKernel.setArgs( { header, primitives, boxNodes, primNodes, kdops, updateCounters } );
		timer.measure(
			FitOrientedBoundsTime, [&]() { fitOrientedBoundsKernel.launch( context.getWarpSize() * boxNodeCount, stream ); } );
	}

	// STEP 8: BVH cost
	if constexpr ( LogBvhCost )
	{
		uint32_t boxNodeCount;
		checkOro( oroMemcpyDtoHAsync(
			&boxNodeCount, reinterpret_cast<oroDeviceptr>( &header->m_boxNodeCount ), sizeof( uint32_t ), stream ) );
		checkOro( oroStreamSynchronize( stream ) );

		checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( taskCounter ), 0, sizeof( float ), stream ) );
		Kernel computeCostKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"ComputeCost",
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		computeCostKernel.setArgs( { boxNodeCount, boxNodes, taskCounter } );
		computeCostKernel.launch( boxNodeCount, ReductionBlockSize, stream );

		float cost;
		checkOro( oroMemcpyDtoHAsync( &cost, reinterpret_cast<oroDeviceptr>( taskCounter ), sizeof( float ), stream ) );
		checkOro( oroStreamSynchronize( stream ) );
		std::cout << "Bvh cost: " << cost << std::endl;
	}

	if constexpr ( Timer::EnableTimer )
	{
		const float time = timer.getTimeRecord( PairTrianglesTime ) + timer.getTimeRecord( ComputeBoxTime ) +
						   timer.getTimeRecord( SetupReferencesTime ) + timer.getTimeRecord( ResetBinsTime ) +
						   timer.getTimeRecord( BinReferencesObjectTime ) + timer.getTimeRecord( BinReferencesSpatialTime ) +
						   timer.getTimeRecord( FindObjectSplitTime ) + timer.getTimeRecord( SplitTime ) +
						   timer.getTimeRecord( DistributeReferencesTime ) + timer.getTimeRecord( ComputeParentAddrsTime ) +
						   timer.getTimeRecord( ComputeFatLeavesTime ) + timer.getTimeRecord( CollapseTime ) +
						   timer.getTimeRecord( CompactTasksTime ) + timer.getTimeRecord( PackLeavesTime ) +
						   timer.getTimeRecord( FitOrientedBoundsTime );
		std::cout << "Sbvh total construction time: " << time << " ms" << std::endl;
		std::cout << "\tpair triangles time: " << timer.getTimeRecord( PairTrianglesTime ) << " ms" << std::endl;
		std::cout << "\tcompute box time: " << timer.getTimeRecord( ComputeBoxTime ) << " ms" << std::endl;
		std::cout << "\tsetup references time: " << timer.getTimeRecord( SetupReferencesTime ) << " ms" << std::endl;
		std::cout << "\treset bins time: " << timer.getTimeRecord( ResetBinsTime ) << " ms" << std::endl;
		std::cout << "\tbin references object time: " << timer.getTimeRecord( BinReferencesObjectTime ) << " ms" << std::endl;
		std::cout << "\tbin references spatial time: " << timer.getTimeRecord( BinReferencesSpatialTime ) << " ms" << std::endl;
		std::cout << "\tfind object split time: " << timer.getTimeRecord( FindObjectSplitTime ) << " ms" << std::endl;
		std::cout << "\tsplit time: " << timer.getTimeRecord( SplitTime ) << " ms" << std::endl;
		std::cout << "\tdistribute references time: " << timer.getTimeRecord( DistributeReferencesTime ) << " ms" << std::endl;
		std::cout << "\tcompute parent addrs time: " << timer.getTimeRecord( ComputeParentAddrsTime ) << " ms" << std::endl;
		std::cout << "\tcompute fat leaves time: " << timer.getTimeRecord( ComputeFatLeavesTime ) << " ms" << std::endl;
		std::cout << "\tcollapse time: " << timer.getTimeRecord( CollapseTime ) << " ms" << std::endl;
		std::cout << "\tcompact tasks time: " << timer.getTimeRecord( CompactTasksTime ) << " ms" << std::endl;
		std::cout << "\tpack leaves time: " << timer.getTimeRecord( PackLeavesTime ) << " ms" << std::endl;
		std::cout << "\tcompute oriented bounding boxes time: " << timer.getTimeRecord( FitOrientedBoundsTime ) << " ms"
				  << std::endl;
	}
}

template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
void SbvhBuilder::update(
	Context&				context,
	PrimitiveContainer&		primitives,
	const hiprtBuildOptions buildOptions,
	oroStream				stream,
	MemoryArena&			storageMemoryArena )
{
	using Header = typename std::conditional<
		std::is_same<PrimitiveNode, UserInstanceNode>::value || std::is_same<PrimitiveNode, HwInstanceNode>::value,
		SceneHeader,
		GeomHeader>::type;

	Header* header = storageMemoryArena.allocate<Header>();

	Header h;
	checkOro( oroMemcpyDtoHAsync( &h, reinterpret_cast<oroDeviceptr>( header ), sizeof( Header ), stream ) );
	checkOro( oroStreamSynchronize( stream ) );

	if ( !( buildOptions.buildFlags & hiprtBuildFlagBitDisableSpatialSplits ) || primitives.getCount() != h.m_referenceCount )
		throw std::runtime_error( "Update is not supported for high-quality build using spatial splits. You can disable "
								  "spatial split by the 'hiprtBuildFlagBitDisableSpatialSplits' build flag." );

	if ( context.getRtip() >= 31 && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableOrientedBoundingBoxes ) )
		throw std::runtime_error(
			"Update is not supported for high-quality build using oriented bounding boxes. You can disable "
			"oriented bounding boxes by the 'hiprtBuildFlagBitDisableOrientedBoundingBoxes' build flag." );

	BoxNode*	   boxNodes	 = reinterpret_cast<BoxNode*>( h.m_boxNodes );
	PrimitiveNode* primNodes = reinterpret_cast<PrimitiveNode*>( h.m_primNodes );

	std::string containerParam		   = Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );
	std::string containerPrimNodeParam = containerParam + "_" + Compiler::kernelNameSufix( Traits<PrimitiveNode>::TYPE_NAME );

	Compiler&				 compiler = context.getCompiler();
	std::vector<const char*> opts;

	uint32_t resetThreadCount = primitives.getCount();
	if constexpr ( std::is_same<Header, SceneHeader>::value )
	{
		Instance* instances = storageMemoryArena.allocate<Instance>( primitives.getCount() );
		Frame*	  frames	= storageMemoryArena.allocate<Frame>( primitives.getFrameCount() );
		primitives.setFrames( h.m_frames );
		resetThreadCount = std::max( primitives.getCount(), primitives.getFrameCount() );
	}

	Kernel resetCountersAndUpdateLeavesKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"ResetCountersAndUpdateLeaves_" + containerPrimNodeParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	resetCountersAndUpdateLeavesKernel.setArgs( { header, primitives, boxNodes, primNodes } );
	resetCountersAndUpdateLeavesKernel.launch( resetThreadCount, stream );

	Kernel fitBoundsKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"FitBounds_" + containerPrimNodeParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	fitBoundsKernel.setArgs( { header, primitives, boxNodes, primNodes } );
	fitBoundsKernel.launch( context.getBranchingFactor() * h.m_boxNodeCount, stream );
}
} // namespace hiprt
