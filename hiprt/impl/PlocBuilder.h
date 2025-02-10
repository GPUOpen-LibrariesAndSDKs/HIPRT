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
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/Geometry.h>
#include <hiprt/impl/Kernel.h>
#include <hiprt/impl/MemoryArena.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/Timer.h>
#include <hiprt/impl/RadixSort.h>
#include <hiprt/impl/Utility.h>
#include <hiprt/impl/BvhConfig.h>

#if defined( HIPRT_LOAD_FROM_STRING )
#include <hiprt/cache/Kernels.h>
#include <hiprt/cache/KernelArgs.h>
#endif

namespace hiprt
{
class PlocBuilder
{
  public:
	static constexpr uint32_t ReductionBlockSize = BvhBuilderReductionBlockSize;
	static constexpr uint32_t MainBlockSize		 = PlocMainBlockSize;
	static constexpr uint32_t Radius			 = PlocRadius;

	enum Times
	{
		PairTrianglesTime,
		ComputeCentroidBoxTime,
		ComputeMortonCodesTime,
		SortTime,
		SetupClustersTime,
		PlocTime,
		CollapseTime
	};

	PlocBuilder()								 = delete;
	PlocBuilder& operator=( const PlocBuilder& ) = delete;

	static size_t getTemporaryBufferSize( const size_t count );

	static size_t getTemporaryBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static size_t getTemporaryBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static size_t getStorageBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static size_t getStorageBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions );

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

	template <typename PrimitiveNode, typename PrimitiveContainer>
	static void build(
		Context&				context,
		PrimitiveContainer&		primitives,
		const hiprtBuildOptions buildOptions,
		const uint32_t			geomType,
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

	template <typename PrimitiveNode, typename PrimitiveContainer>
	static void update(
		Context&				context,
		PrimitiveContainer&		primitives,
		const hiprtBuildOptions buildOptions,
		oroStream				stream,
		MemoryArena&			storageMemoryArena );
};

template <typename PrimitiveNode, typename PrimitiveContainer>
void PlocBuilder::build(
	Context&				context,
	PrimitiveContainer&		primitives,
	const hiprtBuildOptions buildOptions,
	uint32_t				geomType,
	MemoryArena&			temporaryMemoryArena,
	oroStream				stream,
	MemoryArena&			storageMemoryArena )
{
	typedef typename std::conditional<std::is_same<PrimitiveNode, InstanceNode>::value, SceneHeader, GeomHeader>::type Header;

	Header*		   header	 = storageMemoryArena.allocate<Header>();
	BoxNode*	   boxNodes	 = storageMemoryArena.allocate<BoxNode>( DivideRoundUp( 2 * primitives.getCount(), 3 ) );
	PrimitiveNode* primNodes = storageMemoryArena.allocate<PrimitiveNode>( primitives.getCount() );

	Aabb* centroidBox = temporaryMemoryArena.allocate<Aabb>();

	ScratchNode*   scratchNodes = temporaryMemoryArena.allocate<ScratchNode>( primitives.getCount() );
	ReferenceNode* references	= temporaryMemoryArena.allocate<ReferenceNode>( primitives.getCount() );
	uint32_t*	   taskCounter	= temporaryMemoryArena.allocate<uint32_t>();
	uint3*		   taskQueue	= temporaryMemoryArena.allocate<uint3>( primitives.getCount() );

	uint32_t* nodeIndices	 = reinterpret_cast<uint32_t*>( taskQueue ) + 0 * primitives.getCount();
	uint32_t* updateCounters = reinterpret_cast<uint32_t*>( taskQueue ) + 1 * primitives.getCount();

	uint32_t* mortonCodeKeys[2];
	mortonCodeKeys[0] = reinterpret_cast<uint32_t*>( boxNodes ) + 0 * primitives.getCount();
	mortonCodeKeys[1] = reinterpret_cast<uint32_t*>( boxNodes ) + 1 * primitives.getCount();

	uint32_t* mortonCodeValues[2];
	mortonCodeValues[0] = reinterpret_cast<uint32_t*>( boxNodes ) + 2 * primitives.getCount();
	mortonCodeValues[1] = reinterpret_cast<uint32_t*>( boxNodes ) + 3 * primitives.getCount();

	RadixSort sort( context.getDevice(), stream, context.getOrochiUtils() );
	Timer	  timer;

	Compiler&				 compiler = context.getCompiler();
	std::vector<const char*> opts;
	// opts.push_back( "-G" );

	std::string containerParam	   = Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );
	std::string nodeParam		   = Compiler::kernelNameSufix( Traits<PrimitiveNode>::TYPE_NAME );
	std::string containerNodeParam = containerParam + "_" + nodeParam;

	bool pairTriangles = false;
	if constexpr ( std::is_same<PrimitiveNode, TriangleNode>::value )
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
		if constexpr ( std::is_same<PrimitiveNode, TriangleNode>::value ) geomType |= 1;
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
			"SingletonConstruction_" + containerNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		singletonConstructionKernel.setArgs( { primitives, boxNodes, primNodes } );
		singletonConstructionKernel.launch( 1, stream );
		return;
	}

	// STEP 1: Pair triangles
	if constexpr ( std::is_same<PrimitiveNode, TriangleNode>::value )
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

	// STEP 2: Calculate centroid bounding box by reduction
	Aabb emptyBox;
	checkOro( oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( centroidBox ), &emptyBox, sizeof( Aabb ), stream ) );

	Kernel computeCentroidBoxKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"ComputeCentroidBox_" + containerParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	computeCentroidBoxKernel.setArgs( { primitives, centroidBox } );
	timer.measure( ComputeCentroidBoxTime, [&]() {
		computeCentroidBoxKernel.launch( primitives.getCount(), ReductionBlockSize, stream );
	} );

	// STEP 3: Calculate Morton codes
	Kernel computeMortonCodesKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"ComputeMortonCodes_" + containerParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	computeMortonCodesKernel.setArgs( { primitives, centroidBox, mortonCodeKeys[0], mortonCodeValues[0] } );
	timer.measure( ComputeMortonCodesTime, [&]() { computeMortonCodesKernel.launch( primitives.getCount(), stream ); } );

	// STEP 4: Sort Morton codes
	timer.measure( SortTime, [&]() {
		sort.sort(
			mortonCodeKeys[0], mortonCodeValues[0], mortonCodeKeys[1], mortonCodeValues[1], primitives.getCount(), stream );
	} );

	// STEP 5: Setup initial clusters from leaves
	Kernel setupClustersKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/PlocBuilderKernels.h",
		"SetupClusters_" + containerParam,
		opts,
		GET_ARG_LIST( PlocBuilderKernels ) );
	setupClustersKernel.setArgs( { primitives, references, mortonCodeValues[1], nodeIndices } );
	timer.measure( SetupClustersTime, [&]() { setupClustersKernel.launch( primitives.getCount(), stream ); } );

	// STEP 6: Clustering
	checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( taskCounter ), 0, sizeof( uint32_t ), stream ) );
	checkOro( oroMemsetD8Async(
		reinterpret_cast<oroDeviceptr>( updateCounters ), 0xFF, sizeof( uint32_t ) * primitives.getCount(), stream ) );

	Kernel hplocKernel = compiler.getKernel(
		context, Utility::getRootDir() / "hiprt/impl/PlocBuilderKernels.h", "HPloc", opts, GET_ARG_LIST( PlocBuilderKernels ) );
	hplocKernel.setArgs(
		{ primitives.getCount(), mortonCodeKeys[1], updateCounters, nodeIndices, scratchNodes, references, taskCounter } );
	timer.measure( PlocTime, [&]() { hplocKernel.launch( primitives.getCount(), MainBlockSize, stream ); } );

	// STEP 7: Collapse
	uint32_t one			  = 1;
	uint3	 rootCollapseTask = { encodeNodeIndex( 0, BoxType ), 0, 0 };
	checkOro( oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( taskQueue ), &rootCollapseTask, sizeof( uint3 ), stream ) );
	checkOro( oroMemsetD8Async(
		reinterpret_cast<oroDeviceptr>( taskQueue + 1 ), 0xFF, sizeof( uint3 ) * ( primitives.getCount() - 1 ), stream ) );
	checkOro( oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( taskCounter ), &one, sizeof( uint32_t ), stream ) );

	Kernel collapseKernel = compiler.getKernel(
		context,
		"../hiprt/impl/BvhBuilderKernels.h",
		"Collapse_" + containerNodeParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	collapseKernel.setArgs(
		{ primitives.getCount(), header, scratchNodes, references, boxNodes, primNodes, primitives, taskCounter, taskQueue } );
	timer.measure( CollapseTime, [&]() { collapseKernel.launch( primitives.getCount(), stream ); } );

	// STEP 8: BVH cost
	if constexpr ( LogBvhCost )
	{
		uint32_t nodeCount;
		checkOro( oroMemcpyDtoHAsync(
			&nodeCount, reinterpret_cast<oroDeviceptr>( &header->m_boxNodeCount ), sizeof( uint32_t ), stream ) );
		checkOro( oroStreamSynchronize( stream ) );
		checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( taskCounter ), 0, sizeof( float ), stream ) );
		Kernel computeCostKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"ComputeCost",
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		computeCostKernel.setArgs( { nodeCount, boxNodes, taskCounter } );
		computeCostKernel.launch( nodeCount, ReductionBlockSize, stream );

		float cost;
		checkOro( oroMemcpyDtoHAsync( &cost, reinterpret_cast<oroDeviceptr>( taskCounter ), sizeof( float ), stream ) );
		checkOro( oroStreamSynchronize( stream ) );
		std::cout << "Bvh cost: " << cost << std::endl;
	}

	if constexpr ( Timer::EnableTimer )
	{
		float time = timer.getTimeRecord( PairTrianglesTime ) + timer.getTimeRecord( ComputeCentroidBoxTime ) +
					 timer.getTimeRecord( ComputeMortonCodesTime ) + timer.getTimeRecord( SortTime ) +
					 timer.getTimeRecord( SetupClustersTime ) + timer.getTimeRecord( PlocTime ) +
					 timer.getTimeRecord( CollapseTime );
		std::cout << "Ploc total construction time: " << time << " ms" << std::endl;
		std::cout << "\tpair triangles time: " << timer.getTimeRecord( PairTrianglesTime ) << " ms" << std::endl;
		std::cout << "\tcompute centroid box time: " << timer.getTimeRecord( ComputeCentroidBoxTime ) << " ms" << std::endl;
		std::cout << "\tcompute morton codes time: " << timer.getTimeRecord( ComputeMortonCodesTime ) << " ms" << std::endl;
		std::cout << "\tsort time: " << timer.getTimeRecord( SortTime ) << " ms" << std::endl;
		std::cout << "\tsetup clusters time: " << timer.getTimeRecord( SetupClustersTime ) << " ms" << std::endl;
		std::cout << "\tploc time time: " << timer.getTimeRecord( PlocTime ) << " ms" << std::endl;
		std::cout << "\tcollapse time: " << timer.getTimeRecord( CollapseTime ) << " ms" << std::endl;
	}
}

template <typename PrimitiveNode, typename PrimitiveContainer>
void PlocBuilder::update(
	Context&				context,
	PrimitiveContainer&		primitives,
	const hiprtBuildOptions buildOptions,
	oroStream				stream,
	MemoryArena&			storageMemoryArena )
{
	typedef typename std::conditional<std::is_same<PrimitiveNode, InstanceNode>::value, SceneHeader, GeomHeader>::type Header;

	Header* header = storageMemoryArena.allocate<Header>();

	Header h;
	checkOro( oroMemcpyDtoHAsync( &h, reinterpret_cast<oroDeviceptr>( header ), sizeof( Header ), stream ) );
	checkOro( oroStreamSynchronize( stream ) );

	BoxNode*	   boxNodes	 = reinterpret_cast<BoxNode*>( h.m_boxNodes );
	PrimitiveNode* primNodes = reinterpret_cast<PrimitiveNode*>( h.m_primNodes );

	std::string containerParam		   = Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );
	std::string containerPrimNodeParam = containerParam + "_" + Compiler::kernelNameSufix( Traits<PrimitiveNode>::TYPE_NAME );

	Compiler&				 compiler = context.getCompiler();
	std::vector<const char*> opts;

	uint32_t resetThreadCount = primitives.getCount();
	if constexpr ( std::is_same<Header, SceneHeader>::value )
	{
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
	fitBoundsKernel.launch( h.m_boxNodeCount, stream );
}
} // namespace hiprt
