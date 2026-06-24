#pragma once

#include <test/shared.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 1
#endif

extern "C" __global__ void HybridRowsKernel(
	hiprtGeometry geom,
	uint8_t*	  image,
	uint32_t	  width,
	uint32_t	  height,
	uint32_t	  yBegin,
	uint32_t	  yEnd,
	uint8_t		  hitValue )
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= width || y >= height || y < yBegin || y >= yEnd )
		return;

	const uint32_t index = x + y * width;

	hiprtRay ray{};
	ray.origin	  = { x / static_cast<float>( width ), y / static_cast<float>( height ), -1.0f };
	ray.direction = { 0.0f, 0.0f, 1.0f };
	ray.minT	  = 0.0f;
	ray.maxT	  = 1e30f;

	hiprtGeomTraversalClosest tr( geom, ray );
	const hiprtHit hit = tr.getNextHit();

	const uint8_t shade		 = hit.hasHit() ? hitValue : 0;
	image[index * 4 + 0]	 = shade;
	image[index * 4 + 1]	 = shade;
	image[index * 4 + 2]	 = shade;
	image[index * 4 + 3]	 = 255;
}
