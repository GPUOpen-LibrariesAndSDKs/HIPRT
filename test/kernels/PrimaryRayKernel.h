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

template <uint32_t Option>
__device__ uint3
getColor( hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices, Material* materials, uint32_t* matOffsetPerInstance )
{
	return uint3{};
}

template <>
__device__ uint3 getColor<VisualizeColor>(
	hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices, Material* materials, uint32_t* matOffsetPerInstance )
{
	const uint32_t matOffset	= matOffsetPerInstance[hit.instanceID] + hit.primID;
	const uint32_t matIndex		= matIndices[matOffset];
	float3		   diffuseColor = materials[matIndex].m_diffuse;
	uint3		   color;
	color.x = diffuseColor.x * 255;
	color.y = diffuseColor.y * 255;
	color.z = diffuseColor.z * 255;
	return color;
}

template <>
__device__ uint3 getColor<VisualizeUv>(
	hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices, Material* materials, uint32_t* matOffsetPerInstance )
{
	uint3 color;
	color.x = hiprt::clamp( static_cast<uint32_t>( hit.uv.x * 255 ), 0, 255 );
	color.y = hiprt::clamp( static_cast<uint32_t>( hit.uv.y * 255 ), 0, 255 );
	color.z = 0;
	return color;
}

template <>
__device__ uint3 getColor<VisualizeId>(
	hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices, Material* materials, uint32_t* matOffsetPerInstance )
{
	uint3 color;
	color.x = tea<16>( hit.primID, 0 ).x % 255;
	color.y = tea<16>( hit.instanceID, 0 ).x % 255;
	color.z = tea<16>( hit.instanceID, hit.primID ).x % 255;
	return color;
}

template <>
__device__ uint3 getColor<VisualizeHitDist>(
	hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices, Material* materials, uint32_t* matOffsetPerInstance )
{
	float t = hit.t / 50.0f;
	uint3 color;
	color.x = hiprt::clamp( static_cast<uint32_t>( t * 255 ), 0, 255 );
	color.y = hiprt::clamp( static_cast<uint32_t>( t * 255 ), 0, 255 );
	color.z = hiprt::clamp( static_cast<uint32_t>( t * 255 ), 0, 255 );
	return color;
}

template <>
__device__ uint3 getColor<VisualizeNormal>(
	hiprtScene scene, const hiprtHit& hit, uint32_t* matIndices, Material* materials, uint32_t* matOffsetPerInstance )
{
	float3 n = hiprt::normalize( hiprtVectorObjectToWorld( hit.normal, scene, hit.instanceID ) );
	uint3  color;
	color.x = ( ( n.x + 1.0f ) * 0.5f ) * 255;
	color.y = ( ( n.y + 1.0f ) * 0.5f ) * 255;
	color.z = ( ( n.z + 1.0f ) * 0.5f ) * 255;
	return color;
}

template <uint32_t Option>
__device__ void PrimaryRayKernel(
	hiprtScene			   scene,
	uint8_t*			   image,
	uint2				   resolution,
	hiprtGlobalStackBuffer globalStackBuffer,
	const Camera&		   camera,
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

	uint32_t seed = tea<16>( x + y * resolution.x, 0 ).x;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	hiprtRay													ray = generateRay( x, y, resolution, camera, seed, false );
	hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, ray, stack, instanceStack );
	{
		hiprtHit hit = tr.getNextHit();
		uint3	 color{};
		if ( hit.hasHit() ) color = getColor<Option>( scene, hit, matIndices, materials, matOffsetPerInstance );

		image[index * 4 + 0] = color.x;
		image[index * 4 + 1] = color.y;
		image[index * 4 + 2] = color.z;
		image[index * 4 + 3] = 255;
	}
}

extern "C" __global__ void PrimaryRayKernel_0(
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
	PrimaryRayKernel<VisualizeColor>(
		scene,
		image,
		resolution,
		globalStackBuffer,
		camera,
		matIndices,
		materials,
		matOffsetPerInstance,
		indices,
		indxOffsets,
		normals,
		normOffset,
		numOfLights,
		lights,
		aoRadius );
}

extern "C" __global__ void PrimaryRayKernel_1(
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
	PrimaryRayKernel<VisualizeUv>(
		scene,
		image,
		resolution,
		globalStackBuffer,
		camera,
		matIndices,
		materials,
		matOffsetPerInstance,
		indices,
		indxOffsets,
		normals,
		normOffset,
		numOfLights,
		lights,
		aoRadius );
}

extern "C" __global__ void PrimaryRayKernel_2(
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
	PrimaryRayKernel<VisualizeId>(
		scene,
		image,
		resolution,
		globalStackBuffer,
		camera,
		matIndices,
		materials,
		matOffsetPerInstance,
		indices,
		indxOffsets,
		normals,
		normOffset,
		numOfLights,
		lights,
		aoRadius );
}

extern "C" __global__ void PrimaryRayKernel_3(
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
	PrimaryRayKernel<VisualizeHitDist>(
		scene,
		image,
		resolution,
		globalStackBuffer,
		camera,
		matIndices,
		materials,
		matOffsetPerInstance,
		indices,
		indxOffsets,
		normals,
		normOffset,
		numOfLights,
		lights,
		aoRadius );
}

extern "C" __global__ void PrimaryRayKernel_4(
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
	PrimaryRayKernel<VisualizeNormal>(
		scene,
		image,
		resolution,
		globalStackBuffer,
		camera,
		matIndices,
		materials,
		matOffsetPerInstance,
		indices,
		indxOffsets,
		normals,
		normOffset,
		numOfLights,
		lights,
		aoRadius );
}

extern "C" __global__ void PrimaryRayKernel_5(
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
	PrimaryRayKernel<VisualizeAo>(
		scene,
		image,
		resolution,
		globalStackBuffer,
		camera,
		matIndices,
		materials,
		matOffsetPerInstance,
		indices,
		indxOffsets,
		normals,
		normOffset,
		numOfLights,
		lights,
		aoRadius );
}