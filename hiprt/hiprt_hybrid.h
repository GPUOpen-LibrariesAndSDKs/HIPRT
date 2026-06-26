#pragma once

#include <hiprt/hiprt_types.h>
#include <hiprt/hiprt_cpu.h>
#include <functional>
#include <future>
#include <vector>
#include <algorithm>
#include <Orochi/Orochi.h>

namespace hiprt
{

struct hiprtHybridTraceConfig
{
    float    gpuFraction = 0.8f;
    uint32_t minCpuBatch = 1024;
};

inline void hiprtTraceHybridClosest(
    hiprtContext                                                  context,
    hiprtScene                                                    scene,
    const hiprtHybridTraceConfig&                                 config,
    hiprtRay*                                                     raysGpuVisible,
    hiprtHit*                                                     hitsGpuVisible,
    uint32_t                                                      rayCount,
    hiprtApiStream                                                stream,
    std::function<void(uint32_t offset, uint32_t count)>          gpuLaunchCallback )
{
    const uint32_t gpuCount = static_cast<uint32_t>( rayCount * config.gpuFraction );
    const uint32_t cpuStart = gpuCount;

    // 1. Launch GPU kernel for rays [0 .. gpuCount)
    if (gpuCount > 0 && gpuLaunchCallback)
    {
        gpuLaunchCallback(0, gpuCount);
    }

    // 2. Launch CPU traceBatch for rays [gpuCount .. rayCount)
    if (rayCount > cpuStart)
    {
        uint32_t cpuCount = rayCount - cpuStart;
        const uint32_t chunkSize = 16384; 
        uint32_t numChunks = (cpuCount + chunkSize - 1) / chunkSize;
        std::vector<std::future<void>> futures;

        for (uint32_t i = 0; i < numChunks; ++i) {
            uint32_t offset = i * chunkSize;
            uint32_t size = std::min(chunkSize, cpuCount - offset);
            futures.push_back(std::async(std::launch::async, [=]() {
                hiprtGeomTraversalClosestCPU::traceBatch(
                    context,
                    scene,
                    raysGpuVisible + cpuStart + offset,
                    hitsGpuVisible + cpuStart + offset,
                    size );
            }));
        }

        for (auto& f : futures) {
            f.wait();
        }
    }

    // 3. Synchronize stream (the caller can do this, but if we have Orochi, we can't easily include it here without Orochi headers)
    if (gpuCount > 0 && stream)
    {
        oroStreamSynchronize(reinterpret_cast<oroStream>(stream));
    }
}

} // namespace hiprt
