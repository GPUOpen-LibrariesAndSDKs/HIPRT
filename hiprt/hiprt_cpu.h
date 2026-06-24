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
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/CpuTypes.h>
#include <embree4/rtcore.h>
#include <embree4/rtcore_ray.h>

// Internal helpers
namespace hiprt_cpu_detail
{

inline RTCRayHit makeRTCRayHit( const hiprtRay& ray ) noexcept
{
	RTCRayHit rh{};
	rh.ray.org_x	 = ray.origin.x;
	rh.ray.org_y	 = ray.origin.y;
	rh.ray.org_z	 = ray.origin.z;
	rh.ray.dir_x	 = ray.direction.x;
	rh.ray.dir_y	 = ray.direction.y;
	rh.ray.dir_z	 = ray.direction.z;
	rh.ray.tnear	 = ray.minT;
	rh.ray.tfar		 = ray.maxT;
	rh.ray.mask		 = static_cast<unsigned>( -1 );
	rh.ray.flags	 = 0;
	rh.hit.geomID	 = RTC_INVALID_GEOMETRY_ID;
	rh.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
	return rh;
}

inline hiprtHit makeHiprtHit( const RTCRayHit& rh ) noexcept
{
	hiprtHit hit{};
	if ( rh.hit.geomID == RTC_INVALID_GEOMETRY_ID )
		return hit;

	hit.primID		= rh.hit.primID;
	hit.instanceID	= ( rh.hit.instID[0] != RTC_INVALID_GEOMETRY_ID )
						  ? rh.hit.instID[0]
						  : hiprtInvalidValue;
	hit.uv.x		= rh.hit.u;
	hit.uv.y		= rh.hit.v;
	hit.normal.x	= rh.hit.Ng_x;
	hit.normal.y	= rh.hit.Ng_y;
	hit.normal.z	= rh.hit.Ng_z;
	hit.t			= rh.ray.tfar;
	return hit;
}

inline void traceGeometryBatch(
	RTCScene scene, const hiprtRay* rays, hiprtHit* hits, uint32_t count ) noexcept
{
	if ( scene == nullptr )
	{
		for ( uint32_t i = 0; i < count; ++i )
			hits[i] = hiprtHit{};
		return;
	}

	for ( uint32_t i = 0; i < count; ++i )
	{
		RTCRayHit rh = makeRTCRayHit( rays[i] );
		rtcIntersect1( scene, &rh );
		hits[i] = makeHiprtHit( rh );
	}
}

inline RTCRay makeRTCRay( const hiprtRay& ray ) noexcept
{
	RTCRay r{};
	r.org_x = ray.origin.x;
	r.org_y = ray.origin.y;
	r.org_z = ray.origin.z;
	r.dir_x = ray.direction.x;
	r.dir_y = ray.direction.y;
	r.dir_z = ray.direction.z;
	r.tnear = ray.minT;
	r.tfar	= ray.maxT;
	r.mask	= static_cast<unsigned>( -1 );
	r.flags = 0;
	return r;
}

// Any-hit: rtcOccluded1 result encoded in hit.hasHit().
inline void traceGeometryAnyHitBatch(
	RTCScene scene, const hiprtRay* rays, hiprtHit* hits, uint32_t count ) noexcept
{
	if ( scene == nullptr )
	{
		for ( uint32_t i = 0; i < count; ++i )
			hits[i] = hiprtHit{};
		return;
	}

	for ( uint32_t i = 0; i < count; ++i )
	{
		RTCRay r = makeRTCRay( rays[i] );
		rtcOccluded1( scene, &r );

		hiprtHit hit{};
		if ( r.tfar < 0.0f ) // occluded
		{
			hit.primID	   = 0;
			hit.instanceID = hiprtInvalidValue;
		}
		hits[i] = hit;
	}
}

} // namespace hiprt_cpu_detail

// Host-side CPU batch traversal (Embree).
class hiprtGeomTraversalClosestCPU
{
  public:
	static void traceBatch(
		hiprtContext	 ctx,
		hiprtGeometry	 geom,
		const hiprtRay*	 rays,
		hiprtHit*		 hits,
		uint32_t		 count ) noexcept
	{
		(void)ctx;
		auto* data = reinterpret_cast<hiprt::CpuGeometryData*>( hiprtGetInternalCpuDataFromGeometry( geom ) );
		hiprt_cpu_detail::traceGeometryBatch(
			( data != nullptr ) ? data->rtcScene : nullptr, rays, hits, count );
	}

	static void traceBatch(
		hiprtContext	 ctx,
		hiprtScene		 scene,
		const hiprtRay*	 rays,
		hiprtHit*		 hits,
		uint32_t		 count ) noexcept
	{
		(void)ctx;
		auto* data = reinterpret_cast<hiprt::CpuSceneData*>( hiprtGetInternalCpuDataFromScene( scene ) );
		hiprt_cpu_detail::traceGeometryBatch(
			( data != nullptr ) ? data->rtcScene : nullptr, rays, hits, count );
	}
};

// Any-hit batch traversal (shadow/occlusion rays).
class hiprtGeomTraversalAnyHitCPU
{
  public:
	static void traceBatch(
		hiprtContext	 ctx,
		hiprtGeometry	 geom,
		const hiprtRay*	 rays,
		hiprtHit*		 hits,
		uint32_t		 count ) noexcept
	{
		(void)ctx;
		auto* data = reinterpret_cast<hiprt::CpuGeometryData*>( hiprtGetInternalCpuDataFromGeometry( geom ) );
		hiprt_cpu_detail::traceGeometryAnyHitBatch(
			( data != nullptr ) ? data->rtcScene : nullptr, rays, hits, count );
	}

	static void traceBatch(
		hiprtContext	 ctx,
		hiprtScene		 scene,
		const hiprtRay*	 rays,
		hiprtHit*		 hits,
		uint32_t		 count ) noexcept
	{
		(void)ctx;
		auto* data = reinterpret_cast<hiprt::CpuSceneData*>( hiprtGetInternalCpuDataFromScene( scene ) );
		hiprt_cpu_detail::traceGeometryAnyHitBatch(
			( data != nullptr ) ? data->rtcScene : nullptr, rays, hits, count );
	}
};
