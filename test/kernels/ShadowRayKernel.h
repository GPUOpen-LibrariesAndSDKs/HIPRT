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

__device__ float3 sampleLightVertex( const Light& light, float3 x, float3& lVtxOut, float3& lNormalOut, float& pdf, float2 xi )
{

	float3 le{};
	lNormalOut	   = hiprt::cross( light.m_lv1 - light.m_lv0, light.m_lv2 - light.m_lv0 );
	float area	   = sqrtf( hiprt::dot( lNormalOut, lNormalOut ) ) / 2.0f;
	lNormalOut	   = hiprt::normalize( lNormalOut );
	const float2 w = make_float2( 1.f - sqrtf( xi.x ), xi.y * sqrtf( xi.x ) );
	lVtxOut		   = light.m_lv0 + w.x * ( light.m_lv1 - light.m_lv0 ) + w.y * ( light.m_lv2 - light.m_lv0 );

	// evaluate light surface integral part here for convenience
	float3 r = lVtxOut - x;
	// light surface cos term
	float cos = fabs( hiprt::dot( lNormalOut, -hiprt::normalize( r ) ) );
	if ( sqrtf( hiprt::dot( r, r ) ) < 0.0001f || cos < 0.0001f )
	{
		pdf = 0.0f;
		return le;
	}

	if ( hiprt::dot( r, lNormalOut ) > 0.0f )
	{
		pdf = 0.0f;
		return le;
	}

	// inverse of cos(theta) / r ^2 / p_i as we will be dividing by pdf
	pdf = hiprt::dot( r, r ) * ( 1.0f / area ) / cos;
	return light.m_le;
}

extern "C" __global__ void __launch_bounds__( 64 ) ShadowRayKernel(
	hiprtScene			   scene,
	uint8_t*			   image,
	uint2				   resolution,
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
	float				   aoRadius )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	uint32_t seed = tea<16>( x, y ).x;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	hiprtRay													ray = generateRay( x, y, resolution, camera, seed, false );
	hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, ray, stack, instanceStack );
	hiprtHit													hit = tr.getNextHit();

	uint3 color{};
	if ( hit.hasHit() )
	{
		const uint32_t idxOffset = indxOffsets[hit.instanceID];
		const uint32_t idx0		 = indices[idxOffset + ( ( hit.primID * 3 ) + 0 )];
		const uint32_t idx1		 = indices[idxOffset + ( ( hit.primID * 3 ) + 1 )];
		const uint32_t idx2		 = indices[idxOffset + ( ( hit.primID * 3 ) + 2 )];

		const uint32_t nOffset = normOffset[hit.instanceID];
		const float3   n0	   = normals[nOffset + idx0];
		const float3   n1	   = normals[nOffset + idx1];
		const float3   n2	   = normals[nOffset + idx2];

		float3 Ns = ( 1.0f - hit.uv.x - hit.uv.y ) * n0 + hit.uv.x * n1 + hit.uv.y * n2;
		float3 Ng = hiprtVectorObjectToWorld( hit.normal, scene, hit.instanceID );
		if ( hiprt::dot( ray.direction, Ng ) > 0.f ) Ng = -Ng;
		Ng = hiprt::normalize( Ng );

		if ( hiprt::dot( Ng, Ns ) < 0.0f ) Ns = Ns - 2.0f * hiprt::dot( Ng, Ns ) * Ng;
		Ns = hiprt::normalize( Ns );

		const uint32_t matOffset = matOffsetPerInstance[hit.instanceID] + hit.primID;
		const uint32_t matIndex	 = matIndices[matOffset];

		const float3 diffuseColor  = materials[matIndex].m_diffuse;
		const float3 emissiveColor = materials[matIndex].m_emission;
		float3		 finalColor	   = emissiveColor;

		const float3 surfacePt = ray.origin + hit.t * ray.direction;
		if ( matIndex == hiprtInvalidValue || !materials[matIndex].light() )
		{
			constexpr uint32_t Spp = 32;
			float3			   est{};
			for ( uint32_t l = 0; l < numOfLights[0]; l++ )
			{
				for ( uint32_t p = 0; p < Spp; p++ )
				{

					float3 lightVtx;
					float3 lNormal;
					Light  light = lights[l];
					float  pdf	 = 0.0f;

					float3 le = sampleLightVertex(
						light, surfacePt, lightVtx, lNormal, pdf, make_float2( randf( seed ), randf( seed ) ) );

					float3 lightDir = lightVtx - surfacePt;
					float3 lightPt	= lightVtx + 1.0e-3f * lNormal;
					float3 lightVec = lightPt - surfacePt;

					hiprtRay shadowRay;
					shadowRay.origin	= surfacePt + 1.0e-3f * Ng;
					shadowRay.direction = hiprt::normalize( lightDir );
					shadowRay.maxT		= 0.99f * sqrtf( hiprt::dot( lightVec, lightVec ) );

					hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> tr( scene, shadowRay, stack, instanceStack );
					hiprtHit												   hitShadow	   = tr.getNextHit();
					int														   lightVisibility = hitShadow.hasHit() ? 0 : 1;

					if ( pdf != 0.0f )
						est += lightVisibility * le * max( 0.0f, hiprt::dot( Ng, hiprt::normalize( lightDir ) ) ) / pdf;
				}
			}

			finalColor = 1.0f / Spp * est * diffuseColor / hiprt::Pi;
		}

		color.x = finalColor.x * 255;
		color.y = finalColor.y * 255;
		color.z = finalColor.z * 255;
	}

	image[index * 4 + 0] = color.x;
	image[index * 4 + 1] = color.y;
	image[index * 4 + 2] = color.z;
	image[index * 4 + 3] = 255;
}
