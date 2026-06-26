#pragma once

#include <hiprt/hiprt_types.h>
#include <hiprt/impl/ContextBase.h>
#include <hiprt/impl/CpuContext.h>
#include <embree4/rtcore.h>

namespace hiprt
{

class hiprtGeomTraversalClosestCPU
{
  public:
    static void traceBatch(
        hiprtContext     ctx,
        hiprtGeometry    geom,
        const hiprtRay*  rays,
        hiprtHit*        hits,
        uint32_t         count )
    {
        auto* contextBase = reinterpret_cast<ContextBase*>(ctx);
        CpuGeometryData* cpuGeom = contextBase->getCpuGeom(geom);
        if (!cpuGeom || !cpuGeom->rtcScene) return;

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);

        for (uint32_t i = 0; i < count; ++i)
        {
            RTCRayHit rayhit;
            rayhit.ray.org_x = rays[i].origin.x;
            rayhit.ray.org_y = rays[i].origin.y;
            rayhit.ray.org_z = rays[i].origin.z;
            rayhit.ray.dir_x = rays[i].direction.x;
            rayhit.ray.dir_y = rays[i].direction.y;
            rayhit.ray.dir_z = rays[i].direction.z;
            rayhit.ray.tnear = rays[i].minT;
            rayhit.ray.tfar  = rays[i].maxT;
            rayhit.ray.mask  = -1;
            rayhit.ray.id    = i;
            rayhit.ray.flags = 0;

            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

            rtcIntersect1(cpuGeom->rtcScene, &rayhit, &args);

            if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
            {
                hits[i].primID = rayhit.hit.primID;
                hits[i].instanceID = hiprtInvalidValue;
                hits[i].uv.x = rayhit.hit.u;
                hits[i].uv.y = rayhit.hit.v;
                hits[i].normal.x = rayhit.hit.Ng_x;
                hits[i].normal.y = rayhit.hit.Ng_y;
                hits[i].normal.z = rayhit.hit.Ng_z;
                hits[i].t = rayhit.ray.tfar;
            }
            else
            {
                hits[i].primID = hiprtInvalidValue;
            }
        }
    }

    static void traceBatchAVX(
        hiprtContext     ctx,
        hiprtGeometry    geom,
        const hiprtRay*  rays,
        hiprtHit*        hits,
        uint32_t         count )
    {
        auto* contextBase = reinterpret_cast<ContextBase*>(ctx);
        CpuGeometryData* cpuGeom = contextBase->getCpuGeom(geom);
        if (!cpuGeom || !cpuGeom->rtcScene) return;

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);

        for (uint32_t i = 0; i < count; i += 8)
        {
            RTCRayHit8 rayhit8;
            int valid[8] = {0};

            uint32_t batchSize = std::min(count - i, 8u);

            for (uint32_t j = 0; j < batchSize; ++j)
            {
                valid[j] = -1;
                rayhit8.ray.org_x[j] = rays[i + j].origin.x;
                rayhit8.ray.org_y[j] = rays[i + j].origin.y;
                rayhit8.ray.org_z[j] = rays[i + j].origin.z;
                rayhit8.ray.dir_x[j] = rays[i + j].direction.x;
                rayhit8.ray.dir_y[j] = rays[i + j].direction.y;
                rayhit8.ray.dir_z[j] = rays[i + j].direction.z;
                rayhit8.ray.tnear[j] = rays[i + j].minT;
                rayhit8.ray.tfar[j]  = rays[i + j].maxT;
                rayhit8.ray.mask[j]  = -1;
                rayhit8.ray.id[j]    = i + j;
                rayhit8.ray.flags[j] = 0;

                rayhit8.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
                rayhit8.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
            }

            for (uint32_t j = batchSize; j < 8; ++j) {
                valid[j] = 0;
            }

            rtcIntersect8(valid, cpuGeom->rtcScene, &rayhit8, &args);

            for (uint32_t j = 0; j < batchSize; ++j)
            {
                if (rayhit8.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID)
                {
                    hits[i + j].primID = rayhit8.hit.primID[j];
                    hits[i + j].instanceID = hiprtInvalidValue;
                    hits[i + j].uv.x = rayhit8.hit.u[j];
                    hits[i + j].uv.y = rayhit8.hit.v[j];
                    hits[i + j].normal.x = rayhit8.hit.Ng_x[j];
                    hits[i + j].normal.y = rayhit8.hit.Ng_y[j];
                    hits[i + j].normal.z = rayhit8.hit.Ng_z[j];
                    hits[i + j].t = rayhit8.ray.tfar[j];
                }
                else
                {
                    hits[i + j].primID = hiprtInvalidValue;
                }
            }
        }
    }

    static void traceBatchAVX(
        hiprtContext     ctx,
        hiprtScene       scene,
        const hiprtRay*  rays,
        hiprtHit*        hits,
        uint32_t         count )
    {
        auto* contextBase = reinterpret_cast<ContextBase*>(ctx);
        CpuSceneData* cpuScene = contextBase->getCpuScene(scene);
        if (!cpuScene || !cpuScene->rtcScene) return;

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);

        for (uint32_t i = 0; i < count; i += 8)
        {
            RTCRayHit8 rayhit8;
            int valid[8] = {0};

            uint32_t batchSize = std::min(count - i, 8u);

            for (uint32_t j = 0; j < batchSize; ++j)
            {
                valid[j] = -1;
                rayhit8.ray.org_x[j] = rays[i + j].origin.x;
                rayhit8.ray.org_y[j] = rays[i + j].origin.y;
                rayhit8.ray.org_z[j] = rays[i + j].origin.z;
                rayhit8.ray.dir_x[j] = rays[i + j].direction.x;
                rayhit8.ray.dir_y[j] = rays[i + j].direction.y;
                rayhit8.ray.dir_z[j] = rays[i + j].direction.z;
                rayhit8.ray.tnear[j] = rays[i + j].minT;
                rayhit8.ray.tfar[j]  = rays[i + j].maxT;
                rayhit8.ray.mask[j]  = -1;
                rayhit8.ray.id[j]    = i + j;
                rayhit8.ray.flags[j] = 0;

                rayhit8.hit.geomID[j] = RTC_INVALID_GEOMETRY_ID;
                rayhit8.hit.instID[0][j] = RTC_INVALID_GEOMETRY_ID;
            }

            for (uint32_t j = batchSize; j < 8; ++j) {
                valid[j] = 0;
            }

            rtcIntersect8(valid, cpuScene->rtcScene, &rayhit8, &args);

            for (uint32_t j = 0; j < batchSize; ++j)
            {
                if (rayhit8.hit.geomID[j] != RTC_INVALID_GEOMETRY_ID)
                {
                    hits[i + j].primID = rayhit8.hit.primID[j];
                    hits[i + j].instanceID = rayhit8.hit.instID[0][j];
                    hits[i + j].uv.x = rayhit8.hit.u[j];
                    hits[i + j].uv.y = rayhit8.hit.v[j];
                    hits[i + j].normal.x = rayhit8.hit.Ng_x[j];
                    hits[i + j].normal.y = rayhit8.hit.Ng_y[j];
                    hits[i + j].normal.z = rayhit8.hit.Ng_z[j];
                    hits[i + j].t = rayhit8.ray.tfar[j];
                }
                else
                {
                    hits[i + j].primID = hiprtInvalidValue;
                }
            }
        }
    }

    static void traceBatch(
        hiprtContext     ctx,
        hiprtScene       scene,
        const hiprtRay*  rays,
        hiprtHit*        hits,
        uint32_t         count )
    {
        auto* contextBase = reinterpret_cast<ContextBase*>(ctx);
        CpuSceneData* cpuScene = contextBase->getCpuScene(scene);
        if (!cpuScene || !cpuScene->rtcScene) return;

        RTCIntersectArguments args;
        rtcInitIntersectArguments(&args);

        for (uint32_t i = 0; i < count; ++i)
        {
            RTCRayHit rayhit;
            rayhit.ray.org_x = rays[i].origin.x;
            rayhit.ray.org_y = rays[i].origin.y;
            rayhit.ray.org_z = rays[i].origin.z;
            rayhit.ray.dir_x = rays[i].direction.x;
            rayhit.ray.dir_y = rays[i].direction.y;
            rayhit.ray.dir_z = rays[i].direction.z;
            rayhit.ray.tnear = rays[i].minT;
            rayhit.ray.tfar  = rays[i].maxT;
            rayhit.ray.mask  = -1;
            rayhit.ray.id    = i;
            rayhit.ray.flags = 0;

            rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
            rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

            rtcIntersect1(cpuScene->rtcScene, &rayhit, &args);

            if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID)
            {
                hits[i].primID = rayhit.hit.primID;
                hits[i].instanceID = rayhit.hit.instID[0];
                hits[i].uv.x = rayhit.hit.u;
                hits[i].uv.y = rayhit.hit.v;
                hits[i].normal.x = rayhit.hit.Ng_x;
                hits[i].normal.y = rayhit.hit.Ng_y;
                hits[i].normal.z = rayhit.hit.Ng_z;
                hits[i].t = rayhit.ray.tfar;
            }
            else
            {
                hits[i].primID = hiprtInvalidValue;
            }
        }
    }
};

class hiprtGeomTraversalAnyHitCPU
{
  public:
    static void traceBatch(
        hiprtContext     ctx,
        hiprtGeometry    geom,
        const hiprtRay*  rays,
        hiprtHit*        hits,
        uint32_t         count )
    {
        auto* contextBase = reinterpret_cast<ContextBase*>(ctx);
        CpuGeometryData* cpuGeom = contextBase->getCpuGeom(geom);
        if (!cpuGeom || !cpuGeom->rtcScene) return;

        RTCOccludedArguments args;
        rtcInitOccludedArguments(&args);

        for (uint32_t i = 0; i < count; ++i)
        {
            RTCRay ray;
            ray.org_x = rays[i].origin.x;
            ray.org_y = rays[i].origin.y;
            ray.org_z = rays[i].origin.z;
            ray.dir_x = rays[i].direction.x;
            ray.dir_y = rays[i].direction.y;
            ray.dir_z = rays[i].direction.z;
            ray.tnear = rays[i].minT;
            ray.tfar  = rays[i].maxT;
            ray.mask  = -1;
            ray.id    = i;
            ray.flags = 0;

            rtcOccluded1(cpuGeom->rtcScene, &ray, &args);

            if (ray.tfar < 0.0f) // hit
            {
                hits[i].primID = 0; // Just some valid value to indicate hit
                hits[i].t = 0.0f;
            }
            else
            {
                hits[i].primID = hiprtInvalidValue;
            }
        }
    }

    static void traceBatch(
        hiprtContext     ctx,
        hiprtScene       scene,
        const hiprtRay*  rays,
        hiprtHit*        hits,
        uint32_t         count )
    {
        auto* contextBase = reinterpret_cast<ContextBase*>(ctx);
        CpuSceneData* cpuScene = contextBase->getCpuScene(scene);
        if (!cpuScene || !cpuScene->rtcScene) return;

        RTCOccludedArguments args;
        rtcInitOccludedArguments(&args);

        for (uint32_t i = 0; i < count; ++i)
        {
            RTCRay ray;
            ray.org_x = rays[i].origin.x;
            ray.org_y = rays[i].origin.y;
            ray.org_z = rays[i].origin.z;
            ray.dir_x = rays[i].direction.x;
            ray.dir_y = rays[i].direction.y;
            ray.dir_z = rays[i].direction.z;
            ray.tnear = rays[i].minT;
            ray.tfar  = rays[i].maxT;
            ray.mask  = -1;
            ray.id    = i;
            ray.flags = 0;

            rtcOccluded1(cpuScene->rtcScene, &ray, &args);

            if (ray.tfar < 0.0f) // hit
            {
                hits[i].primID = 0; // Just some valid value to indicate hit
                hits[i].t = 0.0f;
            }
            else
            {
                hits[i].primID = hiprtInvalidValue;
            }
        }
    }
};

} // namespace hiprt
