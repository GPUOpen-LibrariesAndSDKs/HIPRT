#pragma once

#include <hiprt/impl/ContextBase.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/CpuContext.h>
#include <stdexcept>
#include <unordered_map>
#include <future>

namespace hiprt
{

class HybridContext : public ContextBase
{
  public:
    Context* m_gpu = nullptr;
    CpuContext* m_cpu = nullptr;

    std::unordered_map<uintptr_t, CpuGeometryData*> m_cpuGeomMap;
    std::unordered_map<uintptr_t, CpuSceneData*>    m_cpuSceneMap;

  public:
    HybridContext( const hiprtContextCreationInput& input )
    {
        hiprtContextCreationInput gpuInput = input;
        if (input.deviceType & hiprtDeviceAMD) gpuInput.deviceType = hiprtDeviceAMD;
        else if (input.deviceType & hiprtDeviceNVIDIA) gpuInput.deviceType = hiprtDeviceNVIDIA;

        m_gpu = new Context(gpuInput);
        try {
            m_cpu = new CpuContext(input);
        } catch (...) {
            delete m_gpu;
            throw;
        }
    }

    virtual ~HybridContext()
    {
        delete m_gpu;
        delete m_cpu;
    }

    virtual std::vector<hiprtGeometry> createGeometries( const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        auto gpuGeoms = m_gpu->createGeometries( buildInputs, buildOptions );
        auto cpuGeoms = m_cpu->createGeometries( buildInputs, buildOptions );

        for ( size_t i = 0; i < gpuGeoms.size(); ++i )
        {
            m_cpuGeomMap[reinterpret_cast<uintptr_t>( gpuGeoms[i] )] = reinterpret_cast<CpuGeometryData*>( cpuGeoms[i] );
        }

        return gpuGeoms;
    }

    virtual void destroyGeometries( const std::vector<hiprtGeometry>& geometries ) override
    {
        std::vector<hiprtGeometry> cpuGeometries;
        for (auto g : geometries) {
            cpuGeometries.push_back(reinterpret_cast<hiprtGeometry>(m_cpuGeomMap[reinterpret_cast<uintptr_t>(g)]));
            m_cpuGeomMap.erase(reinterpret_cast<uintptr_t>(g));
        }

        m_gpu->destroyGeometries( geometries );
        m_cpu->destroyGeometries( cpuGeometries );
    }

    virtual void buildGeometries(
        const std::vector<hiprtGeometryBuildInput>& buildInputs,
        const hiprtBuildOptions                     buildOptions,
        hiprtDevicePtr                              temporaryBuffer,
        oroStream                                   stream,
        std::vector<hiprtDevicePtr>&                buffers ) override
    {
        std::vector<hiprtDevicePtr> cpuBuffers;
        for (auto b : buffers) {
            cpuBuffers.push_back(reinterpret_cast<hiprtDevicePtr>(m_cpuGeomMap[reinterpret_cast<uintptr_t>(b)]));
        }
        
        auto cpuFuture = std::async(std::launch::async, [this, &buildInputs, buildOptions, &cpuBuffers]() {
            m_cpu->buildGeometries( buildInputs, buildOptions, nullptr, nullptr, cpuBuffers );
        });

        m_gpu->buildGeometries( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
        cpuFuture.wait();
    }

    virtual void updateGeometries(
        const std::vector<hiprtGeometryBuildInput>& buildInputs,
        const hiprtBuildOptions                     buildOptions,
        hiprtDevicePtr                              temporaryBuffer,
        oroStream                                   stream,
        std::vector<hiprtDevicePtr>&                buffers ) override
    {
        std::vector<hiprtDevicePtr> cpuBuffers;
        for (auto b : buffers) {
            cpuBuffers.push_back(reinterpret_cast<hiprtDevicePtr>(m_cpuGeomMap[reinterpret_cast<uintptr_t>(b)]));
        }

        auto cpuFuture = std::async(std::launch::async, [this, &buildInputs, buildOptions, &cpuBuffers]() {
            m_cpu->updateGeometries( buildInputs, buildOptions, nullptr, nullptr, cpuBuffers );
        });

        m_gpu->updateGeometries( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
        cpuFuture.wait();
    }

    virtual size_t getGeometriesBuildTempBufferSize(
        const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        return m_gpu->getGeometriesBuildTempBufferSize(buildInputs, buildOptions);
    }

    virtual std::vector<hiprtGeometry> compactGeometries( const std::vector<hiprtGeometry>& geometries, oroStream stream ) override
    {
        throw std::runtime_error("HybridContext::compactGeometries not implemented");
    }

    virtual std::vector<hiprtScene> createScenes( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        auto gpuScenes = m_gpu->createScenes( buildInputs, buildOptions );
        auto cpuScenes = m_cpu->createScenes( buildInputs, buildOptions );

        for ( size_t i = 0; i < gpuScenes.size(); ++i )
        {
            m_cpuSceneMap[reinterpret_cast<uintptr_t>( gpuScenes[i] )] = reinterpret_cast<CpuSceneData*>( cpuScenes[i] );
        }

        return gpuScenes;
    }

    virtual void destroyScenes( const std::vector<hiprtScene>& scenes ) override
    {
        std::vector<hiprtScene> cpuScenes;
        for (auto s : scenes) {
            cpuScenes.push_back(reinterpret_cast<hiprtScene>(m_cpuSceneMap[reinterpret_cast<uintptr_t>(s)]));
            m_cpuSceneMap.erase(reinterpret_cast<uintptr_t>(s));
        }

        m_gpu->destroyScenes( scenes );
        m_cpu->destroyScenes( cpuScenes );
    }

    virtual void buildScenes(
        const std::vector<hiprtSceneBuildInput>& buildInputs,
        const hiprtBuildOptions                  buildOptions,
        hiprtDevicePtr                           temporaryBuffer,
        oroStream                                stream,
        std::vector<hiprtDevicePtr>&             buffers ) override
    {
        std::vector<hiprtDevicePtr> cpuBuffers;
        for (auto b : buffers) {
            cpuBuffers.push_back(reinterpret_cast<hiprtDevicePtr>(m_cpuSceneMap[reinterpret_cast<uintptr_t>(b)]));
        }

        std::vector<hiprtSceneBuildInput> cpuBuildInputs = buildInputs;
        std::vector<std::vector<hiprtInstance>> cpuInstancesArray(buildInputs.size());

        for (size_t i = 0; i < buildInputs.size(); ++i) {
            if (buildInputs[i].instances != nullptr) {
                cpuInstancesArray[i].resize(buildInputs[i].instanceCount);
                oroMemcpyDtoH(cpuInstancesArray[i].data(), reinterpret_cast<oroDeviceptr>(buildInputs[i].instances), buildInputs[i].instanceCount * sizeof(hiprtInstance));
                for (uint32_t j = 0; j < buildInputs[i].instanceCount; ++j) {
                    if (cpuInstancesArray[i][j].type == hiprtInstanceTypeGeometry) {
                        cpuInstancesArray[i][j].geometry = reinterpret_cast<hiprtGeometry>(m_cpuGeomMap[reinterpret_cast<uintptr_t>(cpuInstancesArray[i][j].geometry)]);
                    } else if (cpuInstancesArray[i][j].type == hiprtInstanceTypeScene) {
                        cpuInstancesArray[i][j].scene = reinterpret_cast<hiprtScene>(m_cpuSceneMap[reinterpret_cast<uintptr_t>(cpuInstancesArray[i][j].scene)]);
                    }
                }
                cpuBuildInputs[i].instances = reinterpret_cast<hiprtDevicePtr>(cpuInstancesArray[i].data());
            }
        }

        auto cpuFuture = std::async( std::launch::async, [&]() {
            m_cpu->buildScenes( cpuBuildInputs, buildOptions, nullptr, nullptr, cpuBuffers );
        });

        m_gpu->buildScenes( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
        cpuFuture.wait();
    }

    virtual void updateScenes(
        const std::vector<hiprtSceneBuildInput>& buildInputs,
        const hiprtBuildOptions                  buildOptions,
        hiprtDevicePtr                           temporaryBuffer,
        oroStream                                stream,
        std::vector<hiprtDevicePtr>&             buffers ) override
    {
        std::vector<hiprtDevicePtr> cpuBuffers;
        for (auto b : buffers) {
            cpuBuffers.push_back(reinterpret_cast<hiprtDevicePtr>(m_cpuSceneMap[reinterpret_cast<uintptr_t>(b)]));
        }

        std::vector<hiprtSceneBuildInput> cpuBuildInputs = buildInputs;
        std::vector<std::vector<hiprtInstance>> cpuInstancesArray(buildInputs.size());

        for (size_t i = 0; i < buildInputs.size(); ++i) {
            if (buildInputs[i].instances != nullptr) {
                cpuInstancesArray[i].resize(buildInputs[i].instanceCount);
                oroMemcpyDtoH(cpuInstancesArray[i].data(), reinterpret_cast<oroDeviceptr>(buildInputs[i].instances), buildInputs[i].instanceCount * sizeof(hiprtInstance));
                for (uint32_t j = 0; j < buildInputs[i].instanceCount; ++j) {
                    if (cpuInstancesArray[i][j].type == hiprtInstanceTypeGeometry) {
                        cpuInstancesArray[i][j].geometry = reinterpret_cast<hiprtGeometry>(m_cpuGeomMap[reinterpret_cast<uintptr_t>(cpuInstancesArray[i][j].geometry)]);
                    } else if (cpuInstancesArray[i][j].type == hiprtInstanceTypeScene) {
                        cpuInstancesArray[i][j].scene = reinterpret_cast<hiprtScene>(m_cpuSceneMap[reinterpret_cast<uintptr_t>(cpuInstancesArray[i][j].scene)]);
                    }
                }
                cpuBuildInputs[i].instances = reinterpret_cast<hiprtDevicePtr>(cpuInstancesArray[i].data());
            }
        }

        auto cpuFuture = std::async(std::launch::async, [&]() {
            m_cpu->updateScenes( cpuBuildInputs, buildOptions, nullptr, nullptr, cpuBuffers );
        });

        m_gpu->updateScenes( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
        cpuFuture.wait();
    }

    virtual size_t getScenesBuildTempBufferSize( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override
    {
        return m_gpu->getScenesBuildTempBufferSize(buildInputs, buildOptions);
    }

    virtual std::vector<hiprtScene> compactScenes( const std::vector<hiprtScene>& scenes, oroStream stream ) override
    {
        throw std::runtime_error("HybridContext::compactScenes not implemented");
    }

    virtual hiprtFuncTable createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes ) override { return m_gpu->createFuncTable(numGeomTypes, numRayTypes); }
    virtual void setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set ) override { m_gpu->setFuncTable(funcTable, geomType, rayType, set); }
    virtual void destroyFuncTable( hiprtFuncTable funcTable ) override { m_gpu->destroyFuncTable(funcTable); }
    virtual void createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut ) override { m_gpu->createGlobalStackBuffer(input, stackBufferOut); }
    virtual void destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer ) override { m_gpu->destroyGlobalStackBuffer(stackBuffer); }
    
    virtual void saveGeometry( hiprtGeometry inGeometry, const std::string& filename ) override { throw std::runtime_error("HybridContext::saveGeometry not implemented"); }
    virtual hiprtGeometry loadGeometry( const std::string& filename ) override { throw std::runtime_error("HybridContext::loadGeometry not implemented"); return nullptr; }
    virtual void saveScene( hiprtScene inScene, const std::string& filename ) override { throw std::runtime_error("HybridContext::saveScene not implemented"); }
    virtual hiprtScene loadScene( const std::string& filename ) override { throw std::runtime_error("HybridContext::loadScene not implemented"); return nullptr; }
    
    virtual void exportGeometryAabb( hiprtGeometry inGeometry, hiprtFloat3& outAabbMin, hiprtFloat3& outAabbMax ) override { m_gpu->exportGeometryAabb(inGeometry, outAabbMin, outAabbMax); }
    virtual void exportSceneAabb( hiprtScene inScene, hiprtFloat3& outAabbMin, hiprtFloat3& outAabbMax ) override { m_gpu->exportSceneAabb(inScene, outAabbMin, outAabbMax); }
    
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
        bool                                 cache ) override
    {
        m_gpu->buildKernels(funcNames, src, moduleName, headers, includeNames, options, numGeomTypes, numRayTypes, funcNameSets, functions, module, cache);
    }

    virtual void buildKernelsFromBitcode(
        const std::vector<const char*>&      funcNames,
        const std::filesystem::path&         moduleName,
        const std::string_view               bitcodeBinary,
        uint32_t                             numGeomTypes,
        uint32_t                             numRayTypes,
        const std::vector<hiprtFuncNameSet>& funcNameSets,
        std::vector<oroFunction>&            functions,
        bool                                 cache ) override
    {
        m_gpu->buildKernelsFromBitcode(funcNames, moduleName, bitcodeBinary, numGeomTypes, numRayTypes, funcNameSets, functions, cache);
    }

    virtual void setCacheDir( const std::filesystem::path& path ) override { m_gpu->setCacheDir(path); }
    virtual void setLogLevel( hiprtLogLevel level ) override { m_gpu->setLogLevel(level); }

    virtual CpuGeometryData* getCpuGeom( hiprtGeometry geom ) override
    {
        auto it = m_cpuGeomMap.find(reinterpret_cast<uintptr_t>(geom));
        if (it != m_cpuGeomMap.end()) return it->second;
        return nullptr;
    }

    virtual CpuSceneData* getCpuScene( hiprtScene scene ) override
    {
        auto it = m_cpuSceneMap.find(reinterpret_cast<uintptr_t>(scene));
        if (it != m_cpuSceneMap.end()) return it->second;
        return nullptr;
    }
};

} // namespace hiprt
