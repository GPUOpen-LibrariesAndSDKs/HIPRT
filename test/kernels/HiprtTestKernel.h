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

HIPRT_DEVICE bool intersectSphere( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float3 orig = ray.origin;
	const float3 dir  = ray.direction;

	const float4 sphere = reinterpret_cast<const float4*>( data )[hit.primID];
	const float3 center = make_float3( sphere );
	const float	 radius = sphere.w;

	const float3 O = orig - center;
	const float3 D = hiprt::normalize( dir );

	const float b	 = hiprt::dot( O, D );
	const float c	 = hiprt::dot( O, O ) - radius * radius;
	const float disc = b * b - c;
	if ( disc > 0.0f )
	{
		const float sdisc = sqrtf( disc );
		const float root  = ( -b - sdisc );
		hit.t			  = root;
		hit.normal		  = ( O + (root)*D ) / radius;
		return true;
	}

	return false;
}

HIPRT_DEVICE bool intersectCircle( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit )
{
	const float*	o = reinterpret_cast<const float*>( data );
	constexpr float R = 0.1f;

	const float2 c = { o[hit.primID] - ray.origin.x, 0.5f - ray.origin.y };
	const float	 d = sqrtf( c.x * c.x + c.y * c.y );

	int2 colors[] = { { 255, 0 }, { 0, 255 }, { 255, 255 } };
	if ( payload )
	{
		int2* color = reinterpret_cast<int2*>( payload );
		*color		= colors[hit.primID];
	}

	bool hasHit = d < R;
	return hasHit;
}

HIPRT_DEVICE bool duplicityFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	uint32_t* processed = reinterpret_cast<uint32_t*>( payload );
	if ( processed[hit.primID] ) return true;
	processed[hit.primID] = 1u;
	return false;
}

HIPRT_DEVICE bool cutoutFilter( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit )
{
	const float	  SCALE = 16.0f;
	const float2& uv	= hit.uv;
	float2		  texCoord[2];
	texCoord[0] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 0.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 1.0f );
	texCoord[1] = ( 1.0f - uv.x - uv.y ) * make_float2( 0.0f, 0.0f ) + uv.x * make_float2( 1.0f, 1.0f ) +
				  uv.y * make_float2( 1.0f, 0.0f );
	if ( ( static_cast<uint32_t>( SCALE * texCoord[hit.primID].x ) + static_cast<uint32_t>( SCALE * texCoord[hit.primID].y ) ) &
		 1 )
		return true;
	return false;
}

extern "C" __global__ void
TraceKernel( hiprtScene scene, uint32_t numOfRays, hiprtGlobalStackBuffer globalStackBuffer, hiprtRay* rays, hiprtHit* hits )
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index >= numOfRays ) return;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, rays[index], stack, instanceStack );
	hits[index] = tr.getNextHit();
}

extern "C" __global__ void CornellBoxKernel(
	hiprtGeometry geom, uint8_t* image, hiprtFuncTable table, int2 resolution, uint32_t* matIndices, float3* diffusColors )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { 278.0f, 273.0f, -900.0f };
	const float2 d = {
		2.0f * x / static_cast<float>( resolution.x ) - 1.0f, 2.0f * ( 1.0f - y / static_cast<float>( resolution.y ) ) - 1.0f };
	const float3 uvw = { -387.817566f, 387.817566f, 1230.0f };

	ray.origin	  = o;
	ray.direction = { uvw.x * d.x, uvw.y * d.y, uvw.z };
	ray.direction /=
		sqrtf( ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y + ray.direction.z * ray.direction.z );

	const uint32_t MaxPrims = 32u;
	uint32_t	   processed[MaxPrims];
	for ( uint32_t i = 0; i < MaxPrims; ++i )
		processed[i] = 0u;

	hiprtGeomTraversalAnyHit tr( geom, ray, hiprtTraversalHintDefault, processed, table );
	while ( true )
	{
		hiprtHit hit = tr.getNextHit();

		int3 color{};
		if ( hit.hasHit() )
		{
			const uint32_t matIndex		= matIndices[hit.primID];
			const float	   alpha		= 1.0f / 3.0f;
			const float3   diffuseColor = alpha * diffusColors[matIndex];
			color.x						= diffuseColor.x * 255;
			color.y						= diffuseColor.y * 255;
			color.z						= diffuseColor.z * 255;
		}

		image[index * 4 + 0] = min( 255, static_cast<uint32_t>( image[index * 4 + 0] ) + color.x );
		image[index * 4 + 1] = min( 255, static_cast<uint32_t>( image[index * 4 + 1] ) + color.y );
		image[index * 4 + 2] = min( 255, static_cast<uint32_t>( image[index * 4 + 2] ) + color.z );
		image[index * 4 + 3] = 255;

		if ( tr.getCurrentState() == hiprtTraversalStateFinished ) break;
	}
}

extern "C" __global__ void MeshIntersectionKernel( hiprtGeometry geom, uint8_t* image, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ), y / static_cast<float>( resolution.y ), -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;

	hiprtGeomTraversalClosest tr( geom, ray );
	hiprtHit				  hit = tr.getNextHit();

	image[index * 4 + 0] = hit.hasHit() ? ( static_cast<float>( x ) / resolution.x ) * 255 : 0;
	image[index * 4 + 1] = hit.hasHit() ? ( static_cast<float>( y ) / resolution.y ) * 255 : 0;
	image[index * 4 + 2] = 0;
	image[index * 4 + 3] = 255;
}

extern "C" __global__ void PairTrianglesKernel( hiprtScene scene, uint8_t* image, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ), y / static_cast<float>( resolution.y ), -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;

	const int3 colors[] = { { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 } };
	int3	   color	= { 0, 0, 0 };

	hiprtSceneTraversalAnyHit tr( scene, ray );
	while ( true )
	{
		hiprtHit hit = tr.getNextHit();
		if ( hit.hasHit() ) color += colors[hit.primID];
		if ( tr.getCurrentState() == hiprtTraversalStateFinished ) break;
	}

	image[index * 4 + 0] = color.x;
	image[index * 4 + 1] = color.y;
	image[index * 4 + 2] = color.z;
	image[index * 4 + 3] = 255;
}

extern "C" __global__ void CutoutKernel( hiprtGeometry geom, uint8_t* image, hiprtFuncTable table, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ), y / static_cast<float>( resolution.y ), -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;

	hiprtGeomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, nullptr, table );
	hiprtHit				  hit = tr.getNextHit();

	image[index * 4 + 0] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 1] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 2] = hit.hasHit() ? 255 : 0;
	image[index * 4 + 3] = 255;
}

extern "C" __global__ void CustomIntersectionKernel( hiprtGeometry geom, uint8_t* image, hiprtFuncTable table, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ), y / static_cast<float>( resolution.y ), -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;
	ray.maxT	   = 1000.0f;

	int2	 color;
	uint32_t hitIdx = hiprtInvalidValue;

	hiprtGeomCustomTraversalClosest tr( geom, ray, hiprtTraversalHintDefault, &color, table );
	hiprtHit						hit = tr.getNextHit();
	if ( hit.hasHit() ) hitIdx = hit.primID;

	image[index * 4 + 0] = hitIdx != hiprtInvalidValue ? color.x : 0;
	image[index * 4 + 1] = hitIdx != hiprtInvalidValue ? color.y : 0;
	image[index * 4 + 2] = 0;
	image[index * 4 + 3] = 255;
}

extern "C" __global__ void SceneIntersectionSingleton( hiprtScene scene, uint8_t* image, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ) - 0.5f, y / static_cast<float>( resolution.y ) - 0.5f, -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;
	ray.maxT	   = 1000.0f;

	hiprtSceneTraversalClosest tr( scene, ray );
	hiprtHit				   hit = tr.getNextHit();

	image[index * 4 + 0] = hit.hasHit() ? ( static_cast<float>( x ) / resolution.x ) * 255 : 0;
	image[index * 4 + 1] = hit.hasHit() ? ( static_cast<float>( y ) / resolution.y ) * 255 : 0;
	image[index * 4 + 2] = 0;
	image[index * 4 + 3] = 255;
}

extern "C" __global__ void SceneIntersectionKernel( hiprtScene scene, uint8_t* image, hiprtFuncTable table, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ), y / static_cast<float>( resolution.y ), -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;
	ray.maxT	   = 1000.0f;

	const float3 colors[2][3] = {
		{ { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f } },
		{ { 0.0f, 1.0f, 1.0f }, { 1.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 0.0f } },
	};

	hiprtSceneTraversalAnyHit tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
	while ( true )
	{
		hiprtHit hit = tr.getNextHit();

		int3 color = { 0, 0, 0 };
		if ( hit.hasHit() )
		{
			const uint32_t instanceID	= hit.instanceIDs[1] != hiprtInvalidValue ? hit.instanceIDs[1] : hit.instanceIDs[0];
			const float3   diffuseColor = colors[instanceID][hit.primID];
			color.x						= diffuseColor.x * 255;
			color.y						= diffuseColor.y * 255;
			color.z						= diffuseColor.z * 255;
		}

		image[index * 4 + 0] += color.x;
		image[index * 4 + 1] += color.y;
		image[index * 4 + 2] += color.z;
		image[index * 4 + 3] = 255;

		if ( tr.getCurrentState() == hiprtTraversalStateFinished ) break;
	}
}

extern "C" __global__ void MotionBlurKernel( hiprtScene scene, uint8_t* image, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	constexpr uint32_t Samples = 32u;

	hiprtRay	 ray;
	const float3 o = { x / static_cast<float>( resolution.x ) - 0.5f, y / static_cast<float>( resolution.y ) - 0.5f, -1.0f };
	const float3 d = { 0.0f, 0.0f, 1.0f };
	ray.origin	   = o;
	ray.direction  = d;

	const float3 colors[2] = { { 1.0f, 0.0f, 0.5f }, { 0.0f, 0.5f, 1.0f } };

	float3 color{};
	for ( uint32_t i = 0; i < Samples; ++i )
	{
		const float				   time = i / static_cast<float>( Samples );
		hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, time );
		hiprtHit				   hit = tr.getNextHit();
		if ( hit.hasHit() ) color += colors[hit.instanceID];
	}

	color = gammaCorrect( color / Samples );

	image[index * 4 + 0] = color.x * 255;
	image[index * 4 + 1] = color.y * 255;
	image[index * 4 + 2] = color.z * 255;
	image[index * 4 + 3] = 255;
}

extern "C" __global__ void MotionBlurSlerpKernel( hiprtScene scene, uint8_t* image, hiprtFuncTable table, int2 resolution )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	constexpr uint32_t Samples = 512u;

	const float3 U = { 1.576494f, 0.0f, 0.0f };
	const float3 V = { 0.0f, 1.576494f, 0.0f };
	const float3 W = { 0.0f, 0.0f, -5.0f };

	uint32_t seed = tea<16>( x, y ).x;

	hiprtRay	 ray;
	const float3 o = { 0.0f, 0.0f, 5.0f };
	const float2 d = 2.0f * make_float2(
								( static_cast<float>( x ) + randf( seed ) ) / static_cast<float>( resolution.x ),
								( static_cast<float>( y ) + randf( seed ) ) / static_cast<float>( resolution.y ) ) -
					 1.0f;
	ray.origin	  = o;
	ray.direction = hiprt::normalize( d.x * U + d.y * V + W );

	int2   payload;
	float3 color{};

	for ( uint32_t i = 0; i < Samples; ++i )
	{
		float					   time = randf( seed );
		hiprtSceneTraversalClosest tr( scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, &payload, table, 0, time );
		hiprtHit				   hit = tr.getNextHit();
		if ( hit.hasHit() )
			color += { 0.9f, 0.1f, 0.1f };
		else
			color += { 0.1f, 0.1f, 0.1f };
	}

	color = gammaCorrect( color / Samples );

	image[index * 4 + 0] = color.x * 255;
	image[index * 4 + 1] = color.y * 255;
	image[index * 4 + 2] = color.z * 255;
	image[index * 4 + 3] = 255;
}
