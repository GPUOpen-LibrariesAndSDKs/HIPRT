
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <hiprt/hiprt_types.h>
#include <Orochi/Orochi.h>

namespace hiprt
{


class ContextBase
{
  public:
	virtual ~ContextBase() = default;

	virtual std::vector<hiprtGeometry>
	createGeometries( const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) = 0;
	virtual void buildGeometries(
		const std::vector<hiprtGeometryBuildInput>& buildInputs,
		const hiprtBuildOptions						buildOptions,
		hiprtDevicePtr								temporaryBuffer,
		oroStream									stream,
		std::vector<hiprtDevicePtr>&				buffers ) = 0;
	virtual void updateGeometries(
		const std::vector<hiprtGeometryBuildInput>& buildInputs,
		const hiprtBuildOptions						buildOptions,
		hiprtDevicePtr								temporaryBuffer,
		oroStream									stream,
		std::vector<hiprtDevicePtr>&				buffers )										 = 0;
	virtual void   destroyGeometries( const std::vector<hiprtGeometry>& geometries ) = 0;
	virtual size_t getGeometriesBuildTempBufferSize(
		const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )					   = 0;
	virtual std::vector<hiprtGeometry> compactGeometries( const std::vector<hiprtGeometry>& geometries, oroStream stream ) = 0;

	virtual std::vector<hiprtScene>
				 createScenes( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) = 0;
	virtual void buildScenes(
		const std::vector<hiprtSceneBuildInput>& buildInputs,
		const hiprtBuildOptions					 buildOptions,
		hiprtDevicePtr							 temporaryBuffer,
		oroStream								 stream,
		std::vector<hiprtDevicePtr>&			 buffers ) = 0;
	virtual void updateScenes(
		const std::vector<hiprtSceneBuildInput>& buildInputs,
		const hiprtBuildOptions					 buildOptions,
		hiprtDevicePtr							 temporaryBuffer,
		oroStream								 stream,
		std::vector<hiprtDevicePtr>&			 buffers )											   = 0;
	virtual void					destroyScenes( const std::vector<hiprtScene>& scenes ) = 0;
	virtual size_t					getScenesBuildTempBufferSize( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )				   = 0;
	virtual std::vector<hiprtScene> compactScenes( const std::vector<hiprtScene>& scenes, oroStream stream )								   = 0;

	virtual void		  saveGeometry( hiprtGeometry inGeometry, const std::string& filename )					 = 0;
	virtual hiprtGeometry loadGeometry( const std::string& filename )											 = 0;
	virtual void		  saveScene( hiprtScene inScene, const std::string& filename )							 = 0;
	virtual hiprtScene	  loadScene( const std::string& filename )												 = 0;
	virtual void		  exportGeometryAabb( hiprtGeometry inGeometry, float3& outAabbMin, float3& outAabbMax ) = 0;
	virtual void		  exportSceneAabb( hiprtScene inScene, float3& outAabbMin, float3& outAabbMax )			 = 0;

	virtual void buildKernels(
		const std::vector<const char*>&		 funcNames,
		const std::string&					 src,
		const std::filesystem::path&		 moduleName,
		std::vector<const char*>&			 headers,
		std::vector<const char*>&			 includeNames,
		std::vector<const char*>&			 options,
		uint32_t							 numGeomTypes,
		uint32_t							 numRayTypes,
		const std::vector<hiprtFuncNameSet>& funcNameSets,
		std::vector<oroFunction>&			 functions,
		oroModule&							 module,
		bool								 cache ) = 0;
	virtual void buildKernelsFromBitcode(
		const std::vector<const char*>&		 funcNames,
		const std::filesystem::path&		 moduleName,
		const std::string_view				 bitcodeBinary,
		uint32_t							 numGeomTypes,
		uint32_t							 numRayTypes,
		const std::vector<hiprtFuncNameSet>& funcNameSets,
		std::vector<oroFunction>&			 functions,
		bool								 cache ) = 0;
	virtual void createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut ) = 0;
	virtual void destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer )									= 0;

	virtual hiprtFuncTable createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes )										   = 0;
	virtual void		   setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set ) = 0;
	virtual void		   destroyFuncTable( hiprtFuncTable funcTable )														   = 0;

	virtual void setCacheDir( const std::filesystem::path& path ) = 0;

	virtual void setLogLevel( hiprtLogLevel level ) = 0;

	virtual struct CpuGeometryData* getCpuGeom( hiprtGeometry ) = 0;
	virtual struct CpuSceneData*	getCpuScene( hiprtScene )	= 0;
};
} // namespace hiprt