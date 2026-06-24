#pragma once

#include <hiprt/impl/ContextBase.h>
#include <hiprt/impl/CpuContext.h>
#include <hiprt/impl/CpuTypes.h>
#include <hiprt/impl/GpuContext.h>

#include <mutex>
#include <unordered_map>

namespace hiprt
{

// GPU + CPU (Embree) context. Build ops update both sides; callers keep GPU handles.
class HybridContext : public ContextBase
{
  public:
	enum class HybridBatchKernel : int
	{
		SceneClosest = 0,
		GeomClosest,
		SceneAnyHit,
		GeomAnyHit,
		Count
	};

	explicit HybridContext( const hiprtContextCreationInput& input );
	~HybridContext() override;

	oroFunction getBatchKernel( HybridBatchKernel kind );

	// Geometry
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

	// Scene
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

	// GPU ops (delegate to GpuContext)
	hiprtFuncTable createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes ) override;
	void		   setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set ) override;
	void		   destroyFuncTable( hiprtFuncTable funcTable ) override;

	void createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut ) override;
	void destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer ) override;

	void		  saveGeometry( hiprtGeometry inGeometry, const std::string& filename ) override;
	hiprtGeometry loadGeometry( const std::string& filename ) override;
	void		  saveScene( hiprtScene inScene, const std::string& filename ) override;
	hiprtScene	  loadScene( const std::string& filename ) override;

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

	void setCacheDir( const std::filesystem::path& path ) override;
	void setLogLevel( hiprtLogLevel level ) override;

	// CPU data accessors
	CpuGeometryData* getCpuGeom( hiprtGeometry geometry ) override;
	CpuSceneData*	 getCpuScene( hiprtScene scene ) override;

  private:
	void dispatchCpuGeomBuild(
		const std::vector<hiprtGeometryBuildInput>& buildInputs,
		const std::vector<hiprtDevicePtr>&			gpuBuffers,
		bool										isUpdate );

	void dispatchCpuSceneBuild(
		const std::vector<hiprtSceneBuildInput>& buildInputs,
		const std::vector<hiprtDevicePtr>&		 gpuBuffers,
		bool									 isUpdate );

	GpuContext m_gpu;
	CpuContext m_cpu;

	std::unordered_map<hiprtGeometry, hiprtGeometry> m_gpuToCpuGeom;
	std::unordered_map<hiprtScene,    hiprtScene>    m_gpuToCpuScene;

	std::mutex	m_batchKernelMutex;
	oroFunction m_batchKernels[static_cast<int>( HybridBatchKernel::Count )] = {};
};

} // namespace hiprt
