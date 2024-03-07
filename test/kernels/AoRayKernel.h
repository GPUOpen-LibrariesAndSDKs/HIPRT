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

#include <test/shared.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif

#ifndef SHARED_STACK_SIZE
#define SHARED_STACK_SIZE 1
#endif

__device__ bool filter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	if ( ( hit.primID + hit.instanceID ) & 1 ) return true;
	return false;
}

extern "C" __global__ void __launch_bounds__( 64 ) AoRayKernel(
	hiprtScene			   scene,
	uint8_t*			   image,
	int2				   resolution,
	hiprtGlobalStackBuffer globalStackBuffer,
	Camera				   camera,
	uint32_t*			   matIndices,
	Material*			   materials,
	uint32_t*			   matOffsetPerInstance,
	uint32_t*			   indices,
	uint32_t*			   indxOffsets,
	float3*				   normals,
	uint32_t*			   normOffset,
	uint32_t*			   numOfLights,
	Light*				   lights,
	float				   aoRadius,
	hiprtFuncTable		   table )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	constexpr uint32_t Spp		 = 512;
	constexpr uint32_t AoSamples = 32;

	int3   color{};
	float3 diffuseColor = make_float3( 1.0f );
	float  ao			= 0.0f;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	for ( uint32_t p = 0; p < Spp; p++ )
	{
		uint32_t seed = tea<16>( x + y * resolution.x, p ).x;

		hiprtRay													ray = generateRay( x, y, resolution, camera, seed, true );
		hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, ray, stack, instanceStack );
		{
			hiprtHit hit = tr.getNextHit();

			if ( hit.hasHit() )
			{
				const float3 surfacePt = ray.origin + hit.t * ( 1.0f - 1.0e-2f ) * ray.direction;

				const uint32_t matOffset = matOffsetPerInstance[hit.instanceID] + hit.primID;
				const uint32_t matIndex	 = matIndices[matOffset];

				float3 Ng = hiprtVectorObjectToWorld( hit.normal, scene, hit.instanceID );
				if ( hiprt::dot( ray.direction, Ng ) > 0.0f ) Ng = -Ng;
				Ng = hiprt::normalize( Ng );

				if ( matIndex == hiprtInvalidValue || !materials[matIndex].light() )
				{
					hiprtRay aoRay;
					aoRay.origin = surfacePt;
					aoRay.maxT	 = aoRadius;
					hiprtHit aoHit;

					for ( uint32_t i = 0; i < AoSamples; i++ )
					{
						aoRay.direction = sampleHemisphereCosine( Ng, seed );
						hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> tr(
							scene, aoRay, stack, instanceStack, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
						aoHit = tr.getNextHit();
						ao += !aoHit.hasHit() ? 1.0f : 0.0f;
					}
				}
			}
		}
	}

	ao = ao / ( Spp * AoSamples );

	color.x = ( ao * diffuseColor.x ) * 255;
	color.y = ( ao * diffuseColor.y ) * 255;
	color.z = ( ao * diffuseColor.z ) * 255;

	image[index * 4 + 0] = color.x;
	image[index * 4 + 1] = color.y;
	image[index * 4 + 2] = color.z;
	image[index * 4 + 3] = 255;
}
