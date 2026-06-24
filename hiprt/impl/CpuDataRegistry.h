#pragma once

#include <hiprt/hiprt_types.h>

namespace hiprt
{
struct CpuGeometryData;
struct CpuSceneData;

// Maps GPU and CPU handles for hybrid traversal. Thread-safe.

void registerCpuGeom( hiprtGeometry handle, CpuGeometryData* data );
void unregisterCpuGeom( hiprtGeometry handle );
CpuGeometryData* lookupCpuGeom( hiprtGeometry handle );

void registerCpuScene( hiprtScene handle, CpuSceneData* data );
void unregisterCpuScene( hiprtScene handle );
CpuSceneData* lookupCpuScene( hiprtScene handle );

} // namespace hiprt
