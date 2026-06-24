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

#include <hiprt/hiprt_hybrid.h>
#include <hiprt/hiprt_cpu.h>
#include <hiprt/impl/HybridContext.h>

#include <Orochi/Orochi.h>

#include <exception>
#include <iostream>

using namespace hiprt;

namespace
{
ContextBase* asContext( hiprtContext context ) noexcept
{
	return reinterpret_cast<ContextBase*>( context );
}

uint32_t computeGpuRayCount( uint32_t rayCount, const hiprtHybridTraceConfig& config ) noexcept
{
	if ( rayCount == 0 )
		return 0;

	uint32_t gpuCount = static_cast<uint32_t>( static_cast<float>( rayCount ) * config.gpuFraction );
	if ( gpuCount > rayCount )
		gpuCount = rayCount;

	const uint32_t cpuCount = rayCount - gpuCount;
	if ( cpuCount > 0 && cpuCount < config.minCpuBatch && rayCount > config.minCpuBatch )
	{
		gpuCount = rayCount - config.minCpuBatch;
		if ( gpuCount > rayCount )
			gpuCount = rayCount;
	}

	return gpuCount;
}

// Batch kernel: (handle, numRays, rays, hits).
hiprtError launchBatchKernel(
	oroFunction		func,
	void*			handle,
	uint32_t		numRays,
	hiprtRay*		rays,
	hiprtHit*		hits,
	hiprtApiStream	stream )
{
	if ( func == nullptr )
		return hiprtErrorInvalidParameter;
	if ( numRays == 0 )
		return hiprtSuccess;

	constexpr uint32_t blockSize = 256u;
	const uint32_t	   gridSize  = ( numRays + blockSize - 1 ) / blockSize;

	void* args[] = { &handle, &numRays, &rays, &hits };
	const oroError err = oroModuleLaunchKernel(
		func, gridSize, 1, 1, blockSize, 1, 1, 0, reinterpret_cast<oroStream>( stream ), args, 0 );
	return ( err == oroSuccess ) ? hiprtSuccess : hiprtErrorInternal;
}

hiprtError syncStream( hiprtApiStream stream )
{
	const oroStream gpuStream = reinterpret_cast<oroStream>( stream );
	const oroError	err		  = ( gpuStream != nullptr ) ? oroStreamSynchronize( gpuStream ) : oroDeviceSynchronize();
	return ( err == oroSuccess ) ? hiprtSuccess : hiprtErrorInternal;
}

// Caller kernel, or compile/cache internal kernel for kernelKind.
oroFunction resolveBatchKernel(
	HybridContext*					   hybrid,
	const hiprtHybridTraceGpuInput*	   gpuInput,
	HybridContext::HybridBatchKernel   kernelKind )
{
	if ( gpuInput != nullptr && gpuInput->traceKernel != nullptr )
	{
		hiprtApiFunction funcApi = gpuInput->traceKernel;
		return *reinterpret_cast<oroFunction*>( &funcApi );
	}

	try
	{
		return hybrid->getBatchKernel( kernelKind );
	}
	catch ( const std::exception& e )
	{
		std::cerr << e.what() << std::endl;
		return nullptr;
	}
}

// GPU batch first (sync), then CPU batch on the remainder.
template <typename HandleT, typename CpuFn>
hiprtError dispatchHybrid(
	hiprtContext					 context,
	HandleT							 handle,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput*	 gpuInput,
	HybridContext::HybridBatchKernel kernelKind,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream,
	CpuFn							 cpuFn )
{
	if ( context == nullptr || handle == nullptr || rays == nullptr || hits == nullptr )
		return hiprtErrorInvalidParameter;

	HybridContext* hybrid = dynamic_cast<HybridContext*>( asContext( context ) );
	if ( hybrid == nullptr )
		return hiprtErrorInvalidParameter;

	const uint32_t gpuCount = computeGpuRayCount( rayCount, config );
	const uint32_t cpuCount = rayCount - gpuCount;

	if ( gpuCount > 0 )
	{
		oroFunction func = resolveBatchKernel( hybrid, gpuInput, kernelKind );
		if ( func == nullptr )
			return hiprtErrorInternal;

		const hiprtError e = launchBatchKernel( func, handle, gpuCount, rays, hits, stream );
		if ( e != hiprtSuccess )
			return e;
		const hiprtError s = syncStream( stream );
		if ( s != hiprtSuccess )
			return s;
	}

	if ( cpuCount > 0 )
		cpuFn( rays + gpuCount, hits + gpuCount, cpuCount );

	return hiprtSuccess;
}
} // namespace

static hiprtError traceHybridClosestScene(
	hiprtContext					context,
	hiprtScene						scene,
	const hiprtHybridTraceConfig&	config,
	const hiprtHybridTraceGpuInput*	gpuInput,
	hiprtRay*						rays,
	hiprtHit*						hits,
	uint32_t						rayCount,
	hiprtApiStream					stream )
{
	if ( scene != nullptr && hiprtGetInternalCpuDataFromScene( scene ) == nullptr )
		return hiprtErrorInvalidParameter;

	return dispatchHybrid(
		context, scene, config, gpuInput, HybridContext::HybridBatchKernel::SceneClosest, rays, hits, rayCount, stream,
		[context, scene]( hiprtRay* r, hiprtHit* h, uint32_t c ) {
			hiprtGeomTraversalClosestCPU::traceBatch( context, scene, r, h, c );
		} );
}

hiprtError hiprtTraceHybridClosest(
	hiprtContext					 context,
	hiprtScene						 scene,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream )
{
	return traceHybridClosestScene( context, scene, config, &gpuInput, rays, hits, rayCount, stream );
}

hiprtError hiprtTraceHybridClosest(
	hiprtContext				  context,
	hiprtScene					  scene,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream )
{
	return traceHybridClosestScene( context, scene, config, nullptr, rays, hits, rayCount, stream );
}

static hiprtError traceHybridClosestGeom(
	hiprtContext					context,
	hiprtGeometry					geometry,
	const hiprtHybridTraceConfig&	config,
	const hiprtHybridTraceGpuInput*	gpuInput,
	hiprtRay*						rays,
	hiprtHit*						hits,
	uint32_t						rayCount,
	hiprtApiStream					stream )
{
	if ( geometry != nullptr && hiprtGetInternalCpuDataFromGeometry( geometry ) == nullptr )
		return hiprtErrorInvalidParameter;

	return dispatchHybrid(
		context, geometry, config, gpuInput, HybridContext::HybridBatchKernel::GeomClosest, rays, hits, rayCount, stream,
		[context, geometry]( hiprtRay* r, hiprtHit* h, uint32_t c ) {
			hiprtGeomTraversalClosestCPU::traceBatch( context, geometry, r, h, c );
		} );
}

hiprtError hiprtTraceHybridClosest(
	hiprtContext					 context,
	hiprtGeometry					 geometry,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream )
{
	return traceHybridClosestGeom( context, geometry, config, &gpuInput, rays, hits, rayCount, stream );
}

hiprtError hiprtTraceHybridClosest(
	hiprtContext				  context,
	hiprtGeometry				  geometry,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream )
{
	return traceHybridClosestGeom( context, geometry, config, nullptr, rays, hits, rayCount, stream );
}

static hiprtError traceHybridAnyHitScene(
	hiprtContext					context,
	hiprtScene						scene,
	const hiprtHybridTraceConfig&	config,
	const hiprtHybridTraceGpuInput*	gpuInput,
	hiprtRay*						rays,
	hiprtHit*						hits,
	uint32_t						rayCount,
	hiprtApiStream					stream )
{
	if ( scene != nullptr && hiprtGetInternalCpuDataFromScene( scene ) == nullptr )
		return hiprtErrorInvalidParameter;

	return dispatchHybrid(
		context, scene, config, gpuInput, HybridContext::HybridBatchKernel::SceneAnyHit, rays, hits, rayCount, stream,
		[context, scene]( hiprtRay* r, hiprtHit* h, uint32_t c ) {
			hiprtGeomTraversalAnyHitCPU::traceBatch( context, scene, r, h, c );
		} );
}

hiprtError hiprtTraceHybridAnyHit(
	hiprtContext					 context,
	hiprtScene						 scene,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream )
{
	return traceHybridAnyHitScene( context, scene, config, &gpuInput, rays, hits, rayCount, stream );
}

hiprtError hiprtTraceHybridAnyHit(
	hiprtContext				  context,
	hiprtScene					  scene,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream )
{
	return traceHybridAnyHitScene( context, scene, config, nullptr, rays, hits, rayCount, stream );
}

static hiprtError traceHybridAnyHitGeom(
	hiprtContext					context,
	hiprtGeometry					geometry,
	const hiprtHybridTraceConfig&	config,
	const hiprtHybridTraceGpuInput*	gpuInput,
	hiprtRay*						rays,
	hiprtHit*						hits,
	uint32_t						rayCount,
	hiprtApiStream					stream )
{
	if ( geometry != nullptr && hiprtGetInternalCpuDataFromGeometry( geometry ) == nullptr )
		return hiprtErrorInvalidParameter;

	return dispatchHybrid(
		context, geometry, config, gpuInput, HybridContext::HybridBatchKernel::GeomAnyHit, rays, hits, rayCount, stream,
		[context, geometry]( hiprtRay* r, hiprtHit* h, uint32_t c ) {
			hiprtGeomTraversalAnyHitCPU::traceBatch( context, geometry, r, h, c );
		} );
}

hiprtError hiprtTraceHybridAnyHit(
	hiprtContext					 context,
	hiprtGeometry					 geometry,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream )
{
	return traceHybridAnyHitGeom( context, geometry, config, &gpuInput, rays, hits, rayCount, stream );
}

hiprtError hiprtTraceHybridAnyHit(
	hiprtContext				  context,
	hiprtGeometry				  geometry,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream )
{
	return traceHybridAnyHitGeom( context, geometry, config, nullptr, rays, hits, rayCount, stream );
}
