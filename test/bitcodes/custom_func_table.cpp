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

#define HIPRT_BITCODE_LINKING
#define HIPRT_EXPORTS

#if defined( __CUDACC__ )
#include <cuda_runtime.h>
#include <cmath>
#else
#include <hip/hip_runtime.h>
#endif

#include <hiprt/hiprt_device.h>

__device__ bool duplicityFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit );
__device__ bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit );
__device__ bool intersectSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit );
__device__ bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit );

HIPRT_DEVICE bool intersectFunc(
	uint32_t					geomType,
	uint32_t					rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&				ray,
	void*						payload,
	hiprtHit&					hit )
{
	const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
	const void*	   data	 = tableHeader.funcDataSets[index].intersectFuncData;
	switch ( index )
	{

	case 0: {
		return intersectCircle( ray, data, payload, hit );
	}
	case 1: {
		return intersectSphere( ray, data, payload, hit );
	}

	default: {
		return false;
	}
	}
}

HIPRT_DEVICE bool filterFunc(
	uint32_t					geomType,
	uint32_t					rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&				ray,
	void*						payload,
	const hiprtHit&				hit )
{
	const uint32_t index = tableHeader.numGeomTypes * rayType + geomType;
	const void*	   data	 = tableHeader.funcDataSets[index].filterFuncData;
	switch ( index )
	{

	case 2: {
		return duplicityFilter( ray, data, payload, hit );
	}
	case 3: {
		return cutoutFilter( ray, data, payload, hit );
	}

	default: {
		return false;
	}
	}
}
