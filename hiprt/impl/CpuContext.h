#pragma once
#include "ContextBase.h"
#include "CpuTypes.h"

#include <Orochi/Orochi.h>
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/Compiler.h>
#include <hiprt/impl/Error.h>
#include <hiprt/impl/Logger.h>
#include <ParallelPrimitives/RadixSort.h>

#include <embree4/rtcore.h>


namespace hiprt{
class CpuContext : public ContextBase
{
  public:
	CpuContext( const hiprtContextCreationInput& input );
	~CpuContext() override;

	std::vector<hiprtGeometry>
	createGeometries( const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override;

	void destroyGeometries( const std::vector<hiprtGeometry>& geometries ) override;

	void buildGeometries(
		const std::vector<hiprtGeometryBuildInput>& buildInputs,
		const hiprtBuildOptions						buildOptions,
		hiprtDevicePtr								temporaryBuffer,
		oroStream									stream,
		std::vector<hiprtDevicePtr>&				buffers ) override;

	void updateGeometries(
		const std::vector<hiprtGeometryBuildInput>& buildInputs,
		const hiprtBuildOptions						buildOptions,
		hiprtDevicePtr								temporaryBuffer,
		oroStream									stream,
		std::vector<hiprtDevicePtr>&				buffers ) override;

	size_t getGeometriesBuildTempBufferSize(
		const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override;

	std::vector<hiprtGeometry> compactGeometries( const std::vector<hiprtGeometry>& geometries, oroStream stream ) override;

	std::vector<hiprtScene>
	createScenes( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override;

	void destroyScenes( const std::vector<hiprtScene>& scenes ) override;

	void buildScenes(
		const std::vector<hiprtSceneBuildInput>& buildInputs,
		const hiprtBuildOptions					 buildOptions,
		hiprtDevicePtr							 temporaryBuffer,
		oroStream								 stream,
		std::vector<hiprtDevicePtr>&			 buffers ) override;

	void updateScenes(
		const std::vector<hiprtSceneBuildInput>& buildInputs,
		const hiprtBuildOptions					 buildOptions,
		hiprtDevicePtr							 temporaryBuffer,
		oroStream								 stream,
		std::vector<hiprtDevicePtr>&			 buffers ) override;

	size_t getScenesBuildTempBufferSize(
		const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions ) override;

	std::vector<hiprtScene> compactScenes( const std::vector<hiprtScene>& scenes, oroStream stream ) override;

	hiprtFuncTable createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes ) override;
	void		   setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set ) override;
	void		   destroyFuncTable( hiprtFuncTable funcTable ) override;

	void setCacheDir( const std::filesystem::path& path ) override { (void)path; }

	void setLogLevel( hiprtLogLevel level ) override { (void)level; }

	void createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut ) override;
	void destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer ) override;

	void		  saveGeometry( hiprtGeometry inGeometry, const std::string& filename ) override;
	hiprtGeometry loadGeometry( const std::string& filename ) override;

	void	   saveScene( hiprtScene inScene, const std::string& filename ) override;
	hiprtScene loadScene( const std::string& filename ) override;

	void exportGeometryAabb( hiprtGeometry inGeometry, float3& outAabbMin, float3& outAabbMax ) override;
	void exportSceneAabb( hiprtScene inScene, float3& outAabbMin, float3& outAabbMax ) override;

	void buildKernels(
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
		bool								 cache ) override;

	void buildKernelsFromBitcode(
		const std::vector<const char*>&		 funcNames,
		const std::filesystem::path&		 moduleName,
		const std::string_view				 bitcodeBinary,
		uint32_t							 numGeomTypes,
		uint32_t							 numRayTypes,
		const std::vector<hiprtFuncNameSet>& funcNameSets,
		std::vector<oroFunction>&			 functions,
		bool								 cache ) override;

	struct CpuGeometryData* getCpuGeom( hiprtGeometry geometry ) override
	{
		return reinterpret_cast<CpuGeometryData*>( geometry );
	};
	struct CpuSceneData*	getCpuScene( hiprtScene scene )	override
	{
		return reinterpret_cast<CpuSceneData*>( scene );
	};

  private:
	void buildGeometryEntry( CpuGeometryData* data, const hiprtGeometryBuildInput& input, size_t index );
	void buildSceneEntry( CpuSceneData* data, const hiprtSceneBuildInput& input, size_t index );

	RTCDevice m_rtcDevice = nullptr;
};
}//namespace hiprt