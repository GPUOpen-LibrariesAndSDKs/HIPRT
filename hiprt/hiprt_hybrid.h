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

#include <hiprt/hiprt.h>

struct hiprtHybridTraceConfig
{
	float	 gpuFraction = 0.8f;
	uint32_t minCpuBatch = 1024u;
};

// Optional: pass your own GPU kernel. Default overloads compile one internally.
struct hiprtHybridTraceGpuInput
{
	hiprtApiFunction	   traceKernel	 = nullptr;
	hiprtGlobalStackBuffer globalStack = {};
};

// Hybrid closest-hit (scene). Requires hybrid context; rays/hits must be host-visible.
HIPRT_API hiprtError hiprtTraceHybridClosest(
	hiprtContext					 context,
	hiprtScene						 scene,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream );

HIPRT_API hiprtError hiprtTraceHybridClosest(
	hiprtContext					 context,
	hiprtGeometry					 geometry,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream );

// Hybrid any-hit (scene/geometry). Only hit.hasHit() is meaningful on CPU path.
HIPRT_API hiprtError hiprtTraceHybridAnyHit(
	hiprtContext					 context,
	hiprtScene						 scene,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream );

HIPRT_API hiprtError hiprtTraceHybridAnyHit(
	hiprtContext					 context,
	hiprtGeometry					 geometry,
	const hiprtHybridTraceConfig&	 config,
	const hiprtHybridTraceGpuInput&	 gpuInput,
	hiprtRay*						 rays,
	hiprtHit*						 hits,
	uint32_t						 rayCount,
	hiprtApiStream					 stream );

// Simple API: kernel compiled and cached inside the hybrid context.
HIPRT_API hiprtError hiprtTraceHybridClosest(
	hiprtContext				  context,
	hiprtScene					  scene,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream );

HIPRT_API hiprtError hiprtTraceHybridClosest(
	hiprtContext				  context,
	hiprtGeometry				  geometry,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream );

HIPRT_API hiprtError hiprtTraceHybridAnyHit(
	hiprtContext				  context,
	hiprtScene					  scene,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream );

HIPRT_API hiprtError hiprtTraceHybridAnyHit(
	hiprtContext				  context,
	hiprtGeometry				  geometry,
	const hiprtHybridTraceConfig& config,
	hiprtRay*					  rays,
	hiprtHit*					  hits,
	uint32_t					  rayCount,
	hiprtApiStream				  stream );
