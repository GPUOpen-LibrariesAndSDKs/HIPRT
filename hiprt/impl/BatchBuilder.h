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
#include <hiprt/impl/Utility.h>
#include <hiprt/impl/BvhConfig.h>

#if defined( HIPRT_LOAD_FROM_STRING )
#include <hiprt/cache/Kernels.h>
#include <hiprt/cache/KernelArgs.h>
#endif

namespace hiprt
{
class BatchBuilder
{
  public:
	static constexpr uint32_t MaxBlockSize = BatchBuilderMaxBlockSize;

	enum Times
	{
		BatchBuildTime
	};

	BatchBuilder()								   = delete;
	BatchBuilder& operator=( const BatchBuilder& ) = delete;

	template <typename BuildInput>
	static size_t getTemporaryBufferSize( const std::vector<BuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
	{
		return RoundUp( sizeof( BuildInput ) * buildInputs.size(), DefaultAlignment ) +
			   RoundUp( sizeof( hiprtDevicePtr ) * buildInputs.size(), DefaultAlignment );
	}

	static size_t getStorageBufferSize( const hiprtGeometryBuildInput& buildInputs, const hiprtBuildOptions buildOptions );

	static size_t getStorageBufferSize( const hiprtSceneBuildInput& buildInputs, const hiprtBuildOptions buildOptions );

	template <typename BuildInput>
	static void build(
		Context&					   context,
		const std::vector<BuildInput>& buildInputs,
		const hiprtBuildOptions		   buildOptions,
		hiprtDevicePtr				   temporaryBuffer,
		oroStream					   stream,
		std::vector<hiprtDevicePtr>&   buffers );
};

template <typename BuildInput>
void BatchBuilder::build(
	Context&					   context,
	const std::vector<BuildInput>& buildInputs,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	oroStream					   stream,
	std::vector<hiprtDevicePtr>&   buffers )
{
	const auto	tempSize = getTemporaryBufferSize( buildInputs, buildOptions );
	MemoryArena temporaryMemoryArena( temporaryBuffer, tempSize, DefaultAlignment );

	BuildInput*		buildInputsDev = temporaryMemoryArena.allocate<BuildInput>( buildInputs.size() );
	hiprtDevicePtr* buffersDev	   = temporaryMemoryArena.allocate<hiprtDevicePtr>( buildInputs.size() );

	checkOro( oroMemcpyHtoDAsync(
		(oroDeviceptr)buildInputsDev,
		const_cast<BuildInput*>( buildInputs.data() ),
		buildInputs.size() * sizeof( BuildInput ),
		stream ) );

	checkOro(
		oroMemcpyHtoDAsync( (oroDeviceptr)buffersDev, buffers.data(), buffers.size() * sizeof( hiprtDevicePtr ), stream ) );

	Timer timer;

	Compiler&				 compiler = context.getCompiler();
	std::vector<const char*> opts;
	// opts.push_back("-G");

	std::string buildInputParam = Compiler::kernelNameSufix( Traits<BuildInput>::TYPE_NAME );

	uint32_t gridSize  = context.getMaxGridSize();
	uint32_t gridSizeY = std::max( 1u, DivideRoundUp( static_cast<uint32_t>( buildInputs.size() ), gridSize ) );
	uint32_t gridSizeX = DivideRoundUp( static_cast<uint32_t>( buildInputs.size() ), gridSizeY );

	uint32_t blockSize = 64;
	while ( blockSize < buildOptions.batchBuildMaxPrimCount )
		blockSize *= 2;

	Kernel batchBuildKernel = compiler.getKernel(
		context,
		Utility::getRootDir() / "hiprt/impl/BatchBuilderKernels.h",
		"BatchBuild_" + buildInputParam,
		opts,
		GET_ARG_LIST( BatchBuilderKernels ) );
	batchBuildKernel.setArgs( { buildInputs.size(), buildInputsDev, buffersDev } );
	timer.measure( BatchBuildTime, [&]() { batchBuildKernel.launch( gridSizeX, gridSizeY, 1, blockSize, 1, 1, 0, stream ); } );

	if constexpr ( Timer::EnableTimer )
		std::cout << "Batch construction time: " << timer.getTimeRecord( BatchBuildTime ) << " ms" << std::endl;
}
} // namespace hiprt
