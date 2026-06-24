#pragma once

#include <embree4/rtcore.h>

namespace hiprt
{

struct CpuGeometryData
{
	RTCDevice rtcDevice = nullptr;
	RTCScene  rtcScene  = nullptr;
};

struct CpuSceneData
{
	RTCDevice rtcDevice = nullptr;
	RTCScene  rtcScene  = nullptr;
};

} // namespace hiprt
