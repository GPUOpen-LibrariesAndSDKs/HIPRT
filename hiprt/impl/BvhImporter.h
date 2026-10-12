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
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/Kernel.h>
#include <hiprt/impl/MemoryArena.h>
#include <hiprt/impl/NodeList.h>
#include <hiprt/impl/Utility.h>
#include <hiprt/impl/BvhConfig.h>
#if defined( HIPRT_LOAD_FROM_STRING )
#include <hiprt/cache/Kernels.h>
#include <hiprt/cache/KernelArgs.h>
#endif

namespace hiprt
{
class BvhImporter
{
  public:
	static constexpr uint32_t ReductionBlockSize = BvhBuilderReductionBlockSize;

	BvhImporter( void )                          = delete;
	BvhImporter& operator=( const BvhImporter& ) = delete;

	template <typename BuildInput>
	[[nodiscard]] static size_t
	getTemporaryBufferSize( const Context& context, const BuildInput& buildInput, const hiprtBuildOptions buildOptions );

	[[nodiscard]] static size_t getTemporaryBufferSize(
		const Context& context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	[[nodiscard]] static size_t getTemporaryBufferSize(
		const Context& context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	[[nodiscard]] static size_t getStorageBufferSize(
		const Context& context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	[[nodiscard]] static size_t getStorageBufferSize(
		const Context& context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions );

	static void build(
		Context&                       context,
		const hiprtGeometryBuildInput& buildInput,
		const hiprtBuildOptions        buildOptions,
		hiprtDevicePtr                 temporaryBuffer,
		oroStream                      stream,
		hiprtDevicePtr                 buffer );

	static void build(
		Context&                    context,
		const hiprtSceneBuildInput& buildInput,
		const hiprtBuildOptions     buildOptions,
		hiprtDevicePtr              temporaryBuffer,
		const oroStream             stream,
		hiprtDevicePtr              buffer );

	template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
	static void build(
		Context&                context,
		PrimitiveContainer&     primitives,
		NodeList&               nodes,
		const hiprtBuildOptions buildOptions,
		const uint32_t          geomType,
		MemoryArena&            temporaryMemoryArena,
		const oroStream         stream,
		MemoryArena&            storageMemoryArena );

	static void update(
		Context&                       context,
		const hiprtGeometryBuildInput& buildInput,
		const hiprtBuildOptions        buildOptions,
		hiprtDevicePtr                 temporaryBuffer,
		const oroStream                stream,
		hiprtDevicePtr                 buffer );

	static void update(
		Context&                    context,
		const hiprtSceneBuildInput& buildInput,
		const hiprtBuildOptions     buildOptions,
		hiprtDevicePtr              temporaryBuffer,
		const oroStream             stream,
		hiprtDevicePtr              buffer );

	template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
	static void update(
		Context&                context,
		PrimitiveContainer&     primitives,
		const NodeList&         nodes,
		const hiprtBuildOptions buildOptions,
		const oroStream         stream,
		MemoryArena&            storageMemoryArena );
};

template <typename BuildInput>
[[nodiscard]] size_t BvhImporter::getTemporaryBufferSize(
	const Context& context, const BuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const bool kdops = context.getRtip() >= 31 && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableOrientedBoundingBoxes );
	const size_t count        = buildInput.nodeList.nodeCount;
	const size_t boxNodeCount = getMaxBoxNodeCount( buildInput, context.getRtip(), context.getBranchingFactor(), count );
	const size_t size         = RoundUp( sizeof( uint32_t ) * count, DefaultAlignment ) +
	                    RoundUp( sizeof( uint32_t ), DefaultAlignment ) +
	                    RoundUp( sizeof( ConversionTask ) * count, DefaultAlignment );
	const size_t obbSize = kdops ? RoundUp( boxNodeCount * sizeof( Kdop ), DefaultAlignment ) +
	                                   RoundUp( boxNodeCount * sizeof( uint32_t ), DefaultAlignment )
	                             : 0;
	return std::max( size, obbSize );
}

template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
void BvhImporter::build(
	Context&                context,
	PrimitiveContainer&     primitives,
	NodeList&               nodes,
	const hiprtBuildOptions buildOptions,
	const uint32_t          geomTypeIn,
	MemoryArena&            temporaryMemoryArena,
	const oroStream         stream,
	MemoryArena&            storageMemoryArena )
{
	using Header = typename std::conditional<isInstanceNode<PrimitiveNode>(), SceneHeader, GeomHeader>::type;

	const uint32_t maxBoxNodeCount =
		static_cast<uint32_t>( getMaxBoxNodeCount<BoxNode, PrimitiveNode>( nodes.getReferenceCount() ) );
	const uint32_t maxPrimNodeCount =
		static_cast<uint32_t>( getMaxPrimNodeCount<PrimitiveNode>( context.getRtip(), nodes.getReferenceCount() ) );

	Header*        header    = storageMemoryArena.allocate<Header>();
	BoxNode*       boxNodes  = storageMemoryArena.allocate<BoxNode>( maxBoxNodeCount );
	PrimitiveNode* primNodes = storageMemoryArena.allocate<PrimitiveNode>( maxPrimNodeCount );
	if ( reinterpret_cast<uintptr_t>( boxNodes ) - reinterpret_cast<uintptr_t>( header ) != sizeof( Header ) )
		throw std::runtime_error( "Box nodes must be appended just after the header." );

	uint32_t*       taskCounter      = temporaryMemoryArena.allocate<uint32_t>();
	uint32_t*       referenceIndices = temporaryMemoryArena.allocate<uint32_t>( nodes.getReferenceCount() );
	ConversionTask* taskQueue        = temporaryMemoryArena.allocate<ConversionTask>( nodes.getReferenceCount() );

	Compiler&                compiler = context.getCompiler();
	std::vector<const char*> opts;
	// opts.push_back("-G");

	const std::string containerParam                = Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );
	const std::string primNodeParam                 = Compiler::kernelNameSufix( Traits<PrimitiveNode>::TYPE_NAME );
	const std::string boxNodeParam                  = Compiler::kernelNameSufix( Traits<BoxNode>::TYPE_NAME );
	const std::string binNodeParam                  = Compiler::kernelNameSufix( Traits<ApiNode>::TYPE_NAME );
	const std::string containerPrimNodeBoxNodeParam = containerParam + "_" + primNodeParam + "_" + boxNodeParam;
	const std::string containerPrimNodeParam        = containerParam + "_" + primNodeParam;
	const std::string primNodeBinNodeParam          = primNodeParam + "_" + binNodeParam;

	// STEP 0: Init data
	if constexpr ( std::is_same_v<Header, SceneHeader> )
	{
		Instance* instances = storageMemoryArena.allocate<Instance>( primitives.getCount() );
		Frame*    frames    = storageMemoryArena.allocate<Frame>( primitives.getFrameCount() );

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
		constexpr bool triangleGeom   = isTriangleNode<PrimitiveNode>();
		const uint32_t geomType       = ( geomTypeIn << 1 ) | ( triangleGeom ? 1 : 0 );
		Kernel         initDataKernel = compiler.getKernel(
            context,
            Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
            "InitGeomData",
            opts,
            GET_ARG_LIST( BvhBuilderKernels ) );
		initDataKernel.setArgs(
			{ primitives.getCount(), geomType, storageMemoryArena.getStorageSize(), boxNodes, primNodes, header } );
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

	// STEP 1: Patch API nodes
	Kernel patchApiNodesKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"PatchApiNodes_" + containerParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	patchApiNodesKernel.setArgs( { nodes.getReferenceCount() - 1, nodes.getApiNodes() } );
	patchApiNodesKernel.launch( nodes.getReferenceCount() - 1, stream );

	// STEP 2: Compute fat leaves
	if constexpr ( isTrianglePacketNode<PrimitiveNode>() )
	{
		uint32_t rootAddr = 0;

		uint32_t* updateCounters = reinterpret_cast<uint32_t*>( taskQueue );
		uint32_t* parentAddrs    = updateCounters + nodes.getReferenceCount();
		uint32_t* triangleCounts = parentAddrs + 2 * nodes.getReferenceCount();
		checkOro( oroMemsetD8Async(
			reinterpret_cast<oroDeviceptr>( updateCounters ), 0, sizeof( uint32_t ) * nodes.getReferenceCount(), stream ) );

		Kernel computeParentAddrsKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"ComputeParentAddrs_" + binNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		computeParentAddrsKernel.setArgs( { nodes.getReferenceCount(), rootAddr, nodes.getApiNodes(), parentAddrs } );
		computeParentAddrsKernel.launch( nodes.getReferenceCount(), stream );

		Kernel computeFatLeaves = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"ComputeFatLeaves_" + binNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		computeFatLeaves.setArgs(
			{ nodes.getReferenceCount(), parentAddrs, nodes.getApiNodes(), triangleCounts, updateCounters } );
		computeFatLeaves.launch( nodes.getReferenceCount(), stream );
	}

	// STEP 3: Collapse
	ConversionTask rootCollapseTask( RootIndex, 0, 0 );
	checkOro( oroMemcpyHtoDAsync(
		reinterpret_cast<oroDeviceptr>( taskQueue ), &rootCollapseTask, sizeof( ConversionTask ), stream ) );
	checkOro( oroMemsetD8Async(
		reinterpret_cast<oroDeviceptr>( reinterpret_cast<ConversionTask*>( taskQueue ) + 1 ),
		0xFF,
		sizeof( ConversionTask ) * ( nodes.getReferenceCount() - 1 ),
		stream ) );

	Kernel collapseKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"Collapse_" + primNodeBinNodeParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	collapseKernel.setArgs(
		{ maxBoxNodeCount,
	      nodes.getReferenceCount(),
	      nodes.getApiNodes(),
	      nodes.getReferenceNodes(),
	      header,
	      boxNodes,
	      taskQueue,
	      referenceIndices } );
	collapseKernel.launch( context.getBranchingFactor() * maxBoxNodeCount, stream );

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
	compactTasksKernel.launch( BvhBuilderCompactionBlockSize, BvhBuilderCompactionBlockSize, stream );

	uint32_t taskCount{};
	checkOro( oroMemcpyDtoHAsync( &taskCount, reinterpret_cast<oroDeviceptr>( taskCounter ), sizeof( uint32_t ), stream ) );
	checkOro( oroStreamSynchronize( stream ) );

	Kernel packLeavesKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
		"PackLeaves_" + containerPrimNodeParam,
		opts,
		GET_ARG_LIST( BvhBuilderKernels ) );
	packLeavesKernel.setArgs(
		{ taskCount, taskQueue, referenceIndices, nodes.getReferenceNodes(), primitives, header, boxNodes, primNodes } );
	if constexpr ( isTrianglePacketNode<PrimitiveNode>() )
		packLeavesKernel.launch( taskCount * context.getLanesPerLeafPacketTask(), context.getWarpSize(), stream );
	else
		packLeavesKernel.launch( taskCount, stream );

	// STEP 4: Fit oriented bounding boxes
	if ( context.getRtip() >= 31 && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableOrientedBoundingBoxes ) )
	{
		MemoryArena obbMemoryArena(
			temporaryMemoryArena.getData(), temporaryMemoryArena.getStorageSize(), temporaryMemoryArena.getAlignment() );
		uint32_t* updateCounters = obbMemoryArena.allocate<uint32_t>( boxNodeCount );
		Kdop*     kdops          = obbMemoryArena.allocate<Kdop>( boxNodeCount );
		checkOro( oroMemsetD8Async(
			reinterpret_cast<oroDeviceptr>( updateCounters ), 0, sizeof( uint32_t ) * boxNodeCount, stream ) );
		Kernel fitOrientedBoundsKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"FitOrientedBounds_" + containerPrimNodeBoxNodeParam,
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		fitOrientedBoundsKernel.setArgs( { header, primitives, primNodes, boxNodes, kdops, updateCounters } );
		fitOrientedBoundsKernel.launch( context.getWarpSize() * boxNodeCount, stream );
	}

	// STEP 5: BVH cost
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
}

template <typename BoxNode, typename PrimitiveNode, typename PrimitiveContainer>
void BvhImporter::update(
	Context&                context,
	PrimitiveContainer&     primitives,
	const NodeList&         nodes,
	const hiprtBuildOptions buildOptions,
	const oroStream         stream,
	MemoryArena&            storageMemoryArena )
{
	using Header = typename std::conditional<isInstanceNode<PrimitiveNode>(), SceneHeader, GeomHeader>::type;

	Header* header = storageMemoryArena.allocate<Header>();

	Header headerHost{};
	checkOro( oroMemcpyDtoHAsync( &headerHost, reinterpret_cast<oroDeviceptr>( header ), sizeof( Header ), stream ) );
	checkOro( oroStreamSynchronize( stream ) );

	if ( !( buildOptions.buildFlags & hiprtBuildFlagBitDisableSpatialSplits ) ||
	     primitives.getCount() != headerHost.m_referenceCount )
		throw std::runtime_error( "Update is not supported for an imported BVH with spatial splits." );

	if ( context.getRtip() >= 31 && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableOrientedBoundingBoxes ) )
		throw std::runtime_error(
			"Update is not supported for an imported BVH with oriented bounding boxes fit. You can disable "
			"oriented bounding boxes by the 'hiprtBuildFlagBitDisableOrientedBoundingBoxes' build flag." );

	BoxNode*       boxNodes  = reinterpret_cast<BoxNode*>( headerHost.m_boxNodes );
	PrimitiveNode* primNodes = reinterpret_cast<PrimitiveNode*>( headerHost.m_primNodes );

	const std::string containerParam = Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );
	const std::string containerPrimNodeParam =
		containerParam + "_" + Compiler::kernelNameSufix( Traits<PrimitiveNode>::TYPE_NAME );

	Compiler&                compiler = context.getCompiler();
	std::vector<const char*> opts;

	uint32_t resetThreadCount = primitives.getCount();
	if constexpr ( std::is_same_v<Header, SceneHeader> )
	{
		primitives.setFrames( headerHost.m_frames );
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
	fitBoundsKernel.setArgs( { header, primitives, primNodes, boxNodes } );
	fitBoundsKernel.launch( context.getBranchingFactor() * headerHost.m_boxNodeCount, stream );
}
} // namespace hiprt
