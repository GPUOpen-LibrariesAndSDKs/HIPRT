#pragma once

#include <hiprt/impl/ContextBase.h>
#include <stdexcept>
#include <string>
#include <embree4/rtcore.h>
#include <algorithm>

namespace hiprt
{

inline void aabbBoundsFunc(const struct RTCBoundsFunctionArguments* args) {
    const hiprtAABBListPrimitive* list = (const hiprtAABBListPrimitive*)args->geometryUserPtr;
    const char* ptr = (const char*)list->aabbs + args->primID * list->aabbStride;
    const float* minP = (const float*)ptr;
    const float* maxP = (const float*)(ptr + (list->aabbStride / 2));
    args->bounds_o->lower_x = minP[0];
    args->bounds_o->lower_y = minP[1];
    args->bounds_o->lower_z = minP[2];
    args->bounds_o->upper_x = maxP[0];
    args->bounds_o->upper_y = maxP[1];
    args->bounds_o->upper_z = maxP[2];
}

inline void aabbIntersectFunc(const struct RTCIntersectFunctionNArguments* args) {
    int* valid = args->valid;
    if (!valid[0]) return;
    
    const hiprtAABBListPrimitive* list = (const hiprtAABBListPrimitive*)args->geometryUserPtr;
    const char* ptr = (const char*)list->aabbs + args->primID * list->aabbStride;
    const float* minP = (const float*)ptr;
    const float* maxP = (const float*)(ptr + (list->aabbStride / 2));
    
    RTCRayHit* rayhit = (RTCRayHit*)args->rayhit;
    RTCRay* ray = &rayhit->ray;
    
    float tmin = (minP[0] - ray->org_x) / ray->dir_x;
    float tmax = (maxP[0] - ray->org_x) / ray->dir_x;
    if (tmin > tmax) std::swap(tmin, tmax);
    
    float tymin = (minP[1] - ray->org_y) / ray->dir_y;
    float tymax = (maxP[1] - ray->org_y) / ray->dir_y;
    if (tymin > tymax) std::swap(tymin, tymax);
    
    if ((tmin > tymax) || (tymin > tmax)) return;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;
    
    float tzmin = (minP[2] - ray->org_z) / ray->dir_z;
    float tzmax = (maxP[2] - ray->org_z) / ray->dir_z;
    if (tzmin > tzmax) std::swap(tzmin, tzmax);
    
    if ((tmin > tzmax) || (tzmin > tmax)) return;
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;
    
    if (tmin >= ray->tnear && tmin <= ray->tfar) {
        ray->tfar = tmin;
        rayhit->hit.geomID = args->geomID;
        rayhit->hit.primID = args->primID;
        rayhit->hit.Ng_x = 1.0f; rayhit->hit.Ng_y = 0.0f; rayhit->hit.Ng_z = 0.0f;
    }
}

struct CpuGeometryData
{
    RTCDevice rtcDevice;
    RTCScene  rtcScene;
    hiprtAABBListPrimitive* aabbListCopy = nullptr;
};

struct CpuSceneData
{
    RTCDevice rtcDevice;
    RTCScene  rtcScene;
};

class CpuContext : public ContextBase
{
  private:
    RTCDevice m_rtcDevice;

  public:
    CpuContext( const hiprtContextCreationInput& input )
    {
        std::string config = "";
        if (input.numCpuThreads > 0) {
            config += "threads=" + std::to_string(input.numCpuThreads);
        }
        m_rtcDevice = rtcNewDevice(config.c_str());
        if (!m_rtcDevice)
        {
            throw std::runtime_error("Failed to initialize Embree device");
        }
    }

    virtual ~CpuContext()
    {
        if (m_rtcDevice) {
            rtcReleaseDevice(m_rtcDevice);
        }
    }

    virtual std::vector<hiprtGeometry> createGeometries( const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        std::vector<hiprtGeometry> out;
        out.reserve( buildInputs.size() );

        for ( size_t i = 0; i < buildInputs.size(); ++i )
        {
            auto* data = new CpuGeometryData();
            data->rtcDevice = m_rtcDevice;
            data->rtcScene  = rtcNewScene( m_rtcDevice );
            out.push_back( reinterpret_cast<hiprtGeometry>( data ) );
        }

        return out;
    }

    virtual void destroyGeometries( const std::vector<hiprtGeometry>& geometries ) override
    {
        for ( auto geom : geometries )
        {
            auto* data = reinterpret_cast<CpuGeometryData*>( geom );
            if (data) {
                if (data->rtcScene) rtcReleaseScene(data->rtcScene);
                if (data->aabbListCopy) delete data->aabbListCopy;
                delete data;
            }
        }
    }

    virtual void buildGeometries(
        const std::vector<hiprtGeometryBuildInput>& buildInputs,
        const hiprtBuildOptions                     buildOptions,
        hiprtDevicePtr                              temporaryBuffer,
        oroStream                                   stream,
        std::vector<hiprtDevicePtr>&                buffers ) override
    {
        for ( size_t i = 0; i < buildInputs.size(); ++i )
        {
            auto* data = reinterpret_cast<CpuGeometryData*>( buffers[i] );
            const auto& input = buildInputs[i];

            if ( input.type == hiprtPrimitiveTypeTriangleMesh )
            {
                const auto& mesh = input.primitive.triangleMesh;

                RTCGeometry geom = rtcNewGeometry( m_rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE );
                rtcSetSharedGeometryBuffer(
                    geom,
                    RTC_BUFFER_TYPE_VERTEX,
                    0,
                    RTC_FORMAT_FLOAT3,
                    mesh.vertices,
                    0,
                    mesh.vertexStride,
                    mesh.vertexCount );
                
                if (mesh.triangleIndices) {
                    rtcSetSharedGeometryBuffer(
                        geom,
                        RTC_BUFFER_TYPE_INDEX,
                        0,
                        RTC_FORMAT_UINT3,
                        mesh.triangleIndices,
                        0,
                        mesh.triangleStride,
                        mesh.triangleCount );
                }

                rtcCommitGeometry( geom );
                rtcAttachGeometry( data->rtcScene, geom );
                rtcReleaseGeometry( geom );
            }
            else if ( input.type == hiprtPrimitiveTypeAABBList )
            {
                const auto& aabbList = input.primitive.aabbList;
                data->aabbListCopy = new hiprtAABBListPrimitive(aabbList);

                RTCGeometry geom = rtcNewGeometry( m_rtcDevice, RTC_GEOMETRY_TYPE_USER );
                rtcSetGeometryUserPrimitiveCount(geom, aabbList.aabbCount);
                rtcSetGeometryUserData(geom, data->aabbListCopy);
                rtcSetGeometryBoundsFunction(geom, aabbBoundsFunc, nullptr);
                rtcSetGeometryIntersectFunction(geom, aabbIntersectFunc);

                rtcCommitGeometry(geom);
                rtcAttachGeometry(data->rtcScene, geom);
                rtcReleaseGeometry(geom);
            }

            rtcCommitScene( data->rtcScene );
        }
    }

    virtual void updateGeometries(
        const std::vector<hiprtGeometryBuildInput>& buildInputs,
        const hiprtBuildOptions                     buildOptions,
        hiprtDevicePtr                              temporaryBuffer,
        oroStream                                   stream,
        std::vector<hiprtDevicePtr>&                buffers ) override
    {
        for ( size_t i = 0; i < buildInputs.size(); ++i )
        {
            auto* data = reinterpret_cast<CpuGeometryData*>( buffers[i] );
            if (data) {
                if (data->rtcScene) {
                    rtcReleaseScene(data->rtcScene);
                    data->rtcScene = rtcNewScene(m_rtcDevice);
                }
                if (data->aabbListCopy) {
                    delete data->aabbListCopy;
                    data->aabbListCopy = nullptr;
                }
            }
        }
        buildGeometries(buildInputs, buildOptions, temporaryBuffer, stream, buffers);
    }

    virtual size_t getGeometriesBuildTempBufferSize(
        const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        return 0; 
    }

    virtual std::vector<hiprtGeometry> compactGeometries( const std::vector<hiprtGeometry>& geometries, oroStream stream ) override
    {
        return geometries;
    }

    virtual std::vector<hiprtScene> createScenes( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        std::vector<hiprtScene> out;
        out.reserve( buildInputs.size() );

        for ( size_t i = 0; i < buildInputs.size(); ++i )
        {
            auto* data = new CpuSceneData();
            data->rtcDevice = m_rtcDevice;
            data->rtcScene  = rtcNewScene( m_rtcDevice );
            out.push_back( reinterpret_cast<hiprtScene>( data ) );
        }

        return out;
    }

    virtual void destroyScenes( const std::vector<hiprtScene>& scenes ) override
    {
        for ( auto scene : scenes )
        {
            auto* data = reinterpret_cast<CpuSceneData*>( scene );
            if (data) {
                if (data->rtcScene) rtcReleaseScene(data->rtcScene);
                delete data;
            }
        }
    }

    virtual void buildScenes(
        const std::vector<hiprtSceneBuildInput>& buildInputs,
        const hiprtBuildOptions                  buildOptions,
        hiprtDevicePtr                           temporaryBuffer,
        oroStream                                stream,
        std::vector<hiprtDevicePtr>&             buffers ) override
    {
        for ( size_t i = 0; i < buildInputs.size(); ++i )
        {
            auto* data = reinterpret_cast<CpuSceneData*>( buffers[i] );
            const auto& input = buildInputs[i];

            if (input.instances != nullptr) {
                hiprtInstance* instances = reinterpret_cast<hiprtInstance*>(input.instances);
                for (uint32_t j = 0; j < input.instanceCount; ++j) {
                    if (instances[j].type == hiprtInstanceTypeGeometry) {
                        CpuGeometryData* geomData = reinterpret_cast<CpuGeometryData*>(instances[j].geometry);
                        if (geomData) {
                            RTCGeometry instance = rtcNewGeometry(m_rtcDevice, RTC_GEOMETRY_TYPE_INSTANCE);
                            rtcSetGeometryInstancedScene(instance, geomData->rtcScene);
                            rtcCommitGeometry(instance);
                            rtcAttachGeometry(data->rtcScene, instance);
                            rtcReleaseGeometry(instance);
                        }
                    } else if (instances[j].type == hiprtInstanceTypeScene) {
                        CpuSceneData* sceneData = reinterpret_cast<CpuSceneData*>(instances[j].scene);
                        if (sceneData) {
                            RTCGeometry instance = rtcNewGeometry(m_rtcDevice, RTC_GEOMETRY_TYPE_INSTANCE);
                            rtcSetGeometryInstancedScene(instance, sceneData->rtcScene);
                            rtcCommitGeometry(instance);
                            rtcAttachGeometry(data->rtcScene, instance);
                            rtcReleaseGeometry(instance);
                        }
                    }
                }
            }

            rtcCommitScene( data->rtcScene );
        }
    }

    virtual void updateScenes(
        const std::vector<hiprtSceneBuildInput>& buildInputs,
        const hiprtBuildOptions                  buildOptions,
        hiprtDevicePtr                           temporaryBuffer,
        oroStream                                stream,
        std::vector<hiprtDevicePtr>&             buffers ) override
    {
        for ( size_t i = 0; i < buildInputs.size(); ++i )
        {
            auto* data = reinterpret_cast<CpuSceneData*>( buffers[i] );
            if (data) {
                if (data->rtcScene) {
                    rtcReleaseScene(data->rtcScene);
                    data->rtcScene = rtcNewScene(m_rtcDevice);
                }
            }
        }
        buildScenes(buildInputs, buildOptions, temporaryBuffer, stream, buffers);
    }

    virtual size_t getScenesBuildTempBufferSize( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        return 0;
    }

    virtual std::vector<hiprtScene> compactScenes( const std::vector<hiprtScene>& scenes, oroStream stream ) override
    {
        return scenes;
    }

    virtual hiprtFuncTable createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes ) override { return nullptr; }
    virtual void setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set ) override {}
    virtual void destroyFuncTable( hiprtFuncTable funcTable ) override {}
    virtual void createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut ) override {}
    virtual void destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer ) override {}
    virtual void saveGeometry( hiprtGeometry inGeometry, const std::string& filename ) override {
        throw std::runtime_error("CpuContext::saveGeometry not implemented");
    }
    virtual hiprtGeometry loadGeometry( const std::string& filename ) override {
        throw std::runtime_error("CpuContext::loadGeometry not implemented");
        return nullptr;
    }
    virtual void saveScene( hiprtScene inScene, const std::string& filename ) override {
        throw std::runtime_error("CpuContext::saveScene not implemented");
    }
    virtual hiprtScene loadScene( const std::string& filename ) override {
        throw std::runtime_error("CpuContext::loadScene not implemented");
        return nullptr;
    }
    virtual void exportGeometryAabb( hiprtGeometry inGeometry, hiprtFloat3& outAabbMin, hiprtFloat3& outAabbMax ) override
    {
        CpuGeometryData* data = reinterpret_cast<CpuGeometryData*>(inGeometry);
        if (data && data->rtcScene) {
            RTCBounds bounds;
            rtcGetSceneBounds(data->rtcScene, &bounds);
            outAabbMin = {bounds.lower_x, bounds.lower_y, bounds.lower_z};
            outAabbMax = {bounds.upper_x, bounds.upper_y, bounds.upper_z};
        } else {
            throw std::runtime_error("Invalid CPU geometry for AABB export");
        }
    }

    virtual void exportSceneAabb( hiprtScene inScene, hiprtFloat3& outAabbMin, hiprtFloat3& outAabbMax ) override
    {
        CpuSceneData* data = reinterpret_cast<CpuSceneData*>(inScene);
        if (data && data->rtcScene) {
            RTCBounds bounds;
            rtcGetSceneBounds(data->rtcScene, &bounds);
            outAabbMin = {bounds.lower_x, bounds.lower_y, bounds.lower_z};
            outAabbMax = {bounds.upper_x, bounds.upper_y, bounds.upper_z};
        } else {
            throw std::runtime_error("Invalid CPU scene for AABB export");
        }
    }
    
    virtual void buildKernels(
        const std::vector<const char*>&      funcNames,
        const std::string&                   src,
        const std::filesystem::path&         moduleName,
        std::vector<const char*>&            headers,
        std::vector<const char*>&            includeNames,
        std::vector<const char*>&            options,
        uint32_t                             numGeomTypes,
        uint32_t                             numRayTypes,
        const std::vector<hiprtFuncNameSet>& funcNameSets,
        std::vector<oroFunction>&            functions,
        oroModule&                           module,
        bool                                 cache ) override {}

    virtual void buildKernelsFromBitcode(
        const std::vector<const char*>&      funcNames,
        const std::filesystem::path&         moduleName,
        const std::string_view               bitcodeBinary,
        uint32_t                             numGeomTypes,
        uint32_t                             numRayTypes,
        const std::vector<hiprtFuncNameSet>& funcNameSets,
        std::vector<oroFunction>&            functions,
        bool                                 cache ) override {}

    virtual void setCacheDir( const std::filesystem::path& path ) override {}
    virtual void setLogLevel( hiprtLogLevel level ) override {}

    virtual CpuGeometryData* getCpuGeom( hiprtGeometry geom ) override
    {
        return reinterpret_cast<CpuGeometryData*>(geom);
    }

    virtual CpuSceneData* getCpuScene( hiprtScene scene ) override
    {
        return reinterpret_cast<CpuSceneData*>(scene);
    }
};

} // namespace hiprt
