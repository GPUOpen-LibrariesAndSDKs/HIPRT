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
#include <hiprt/impl/Geometry.h>
#include <hiprt/impl/Kernel.h>
#include <hiprt/impl/MemoryArena.h>
#include <hiprt/impl/ApiNodeList.h>
#include <hiprt/impl/Scene.h>
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

	BvhImporter( void )							 = delete;
	BvhImporter& operator=( const BvhImporter& ) = delete;

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
		const ApiNodeList&		nodes,
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

	template <typename PrimitiveNode, typename PrimitiveContainer>
	static void update(
		Context&				context,
		PrimitiveContainer&		primitives,
		const ApiNodeList&		nodes,
		const hiprtBuildOptions buildOptions,
		oroStream				stream,
		MemoryArena&			storageMemoryArena );
};

template <typename PrimitiveNode, typename PrimitiveContainer>
void BvhImporter::build(
	Context&				context,
	PrimitiveContainer&		primitives,
	const ApiNodeList&		nodes,
	const hiprtBuildOptions buildOptions,
	uint32_t				geomType,
	MemoryArena&			temporaryMemoryArena,
	oroStream				stream,
	MemoryArena&			storageMemoryArena )
{
	typedef typename std::conditional<std::is_same<PrimitiveNode, InstanceNode>::value, SceneHeader, GeomHeader>::type Header;

	Header*		   header	 = storageMemoryArena.allocate<Header>();
	BoxNode*	   boxNodes	 = storageMemoryArena.allocate<BoxNode>( nodes.getCount() );
	PrimitiveNode* primNodes = storageMemoryArena.allocate<PrimitiveNode>( primitives.getCount() );

	Compiler&				 compiler = context.getCompiler();
	std::vector<const char*> opts;
	// opts.push_back("-G");

	std::string containerParam	   = Compiler::kernelNameSufix( Traits<PrimitiveContainer>::TYPE_NAME );
	std::string nodeParam		   = Compiler::kernelNameSufix( Traits<PrimitiveNode>::TYPE_NAME );
	std::string containerNodeParam = containerParam + "_" + nodeParam;

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
		Kernel initDataKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"InitGeomData",
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		initDataKernel.setArgs(
			{ storageMemoryArena.getStorageSize(), primitives.getCount(), boxNodes, primNodes, geomType, header } );
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

	// STEP 1: Setup leaves
	Kernel setupLeavesKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhImporterKernels.h",
		"SetupLeaves_" + containerNodeParam,
		opts,
		GET_ARG_LIST( BvhImporterKernels ) );
	setupLeavesKernel.setArgs( { primitives, primNodes } );
	setupLeavesKernel.launch( primitives.getCount(), stream );

	// STEP 2: Convert to internal format
	Kernel convertKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BvhImporterKernels.h",
		"Convert_" + containerNodeParam,
		opts,
		GET_ARG_LIST( BvhImporterKernels ) );
	convertKernel.setArgs( { primitives, nodes, boxNodes, primNodes } );
	convertKernel.launch( nodes.getCount(), stream );

	if constexpr ( LogBvhCost )
	{
		uint32_t nodeCount	 = nodes.getCount();
		float*	 costCounter = nullptr;
		checkOro( oroMalloc( reinterpret_cast<oroDeviceptr*>( &costCounter ), sizeof( float ) ) );
		checkOro( oroMemsetD8Async( reinterpret_cast<oroDeviceptr>( costCounter ), 0, sizeof( float ), stream ) );
		Kernel computeCostKernel = compiler.getKernel(
			context,
			Utility::getRootDir() / "hiprt/impl/BvhBuilderKernels.h",
			"ComputeCost",
			opts,
			GET_ARG_LIST( BvhBuilderKernels ) );
		computeCostKernel.setArgs( { nodeCount, boxNodes, costCounter } );
		computeCostKernel.launch( nodeCount, ReductionBlockSize, stream );

		float cost;
		checkOro( oroMemcpyDtoHAsync( &cost, reinterpret_cast<oroDeviceptr>( costCounter ), sizeof( float ), stream ) );
		checkOro( oroStreamSynchronize( stream ) );
		checkOro( oroFree( reinterpret_cast<oroDeviceptr>( costCounter ) ) );

		std::cout << "Bvh cost: " << cost << std::endl;
	}
}

template <typename PrimitiveNode, typename PrimitiveContainer>
void BvhImporter::update(
	Context&				context,
	PrimitiveContainer&		primitives,
	const ApiNodeList&		nodes,
	const hiprtBuildOptions buildOptions,
	oroStream				stream,
	MemoryArena&			storageMemoryArena )
{
	throw std::runtime_error( "Not supported. Use a different BVH builder." );
}
} // namespace hiprt
