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
#include <Orochi/Orochi.h>
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/Compiler.h>
#include <hiprt/impl/Error.h>
#include <ParallelPrimitives/RadixSort.h>

namespace hiprt
{
class Context
{
  public:
	Context( const hiprtContextCreationInput& input );
	~Context();

	std::vector<hiprtGeometry>
	createGeometries( const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions );

	void destroyGeometries( const std::vector<hiprtGeometry> geometries );

	void buildGeometries(
		const std::vector<hiprtGeometryBuildInput>& buildInputs,
		const hiprtBuildOptions						buildOptions,
		hiprtDevicePtr								temporaryBuffer,
		oroStream									stream,
		std::vector<hiprtDevicePtr>&				buffers );

	void updateGeometries(
		const std::vector<hiprtGeometryBuildInput>& buildInputs,
		const hiprtBuildOptions						buildOptions,
		hiprtDevicePtr								temporaryBuffer,
		oroStream									stream,
		std::vector<hiprtDevicePtr>&				buffers );

	size_t getGeometriesBuildTempBufferSize(
		const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions );

	std::vector<hiprtGeometry> compactGeometries( const std::vector<hiprtGeometry>& geometries, oroStream stream );

	std::vector<hiprtScene>
	createScenes( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions );

	void destroyScenes( const std::vector<hiprtScene> scenes );

	void buildScenes(
		const std::vector<hiprtSceneBuildInput>& buildInputs,
		const hiprtBuildOptions					 buildOptions,
		hiprtDevicePtr							 temporaryBuffer,
		oroStream								 stream,
		std::vector<hiprtDevicePtr>&			 buffers );

	void updateScenes(
		const std::vector<hiprtSceneBuildInput>& buildInputs,
		const hiprtBuildOptions					 buildOptions,
		hiprtDevicePtr							 temporaryBuffer,
		oroStream								 stream,
		std::vector<hiprtDevicePtr>&			 buffers );

	size_t
	getScenesBuildTempBufferSize( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions );

	std::vector<hiprtScene> compactScenes( const std::vector<hiprtScene>& scenes, oroStream stream );

	hiprtFuncTable createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes );
	void		   setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set );
	void		   destroyFuncTable( hiprtFuncTable funcTable );

	void createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut );
	void destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer );

	void		  saveGeometry( hiprtGeometry inGeometry, const std::string& filename );
	hiprtGeometry loadGeometry( const std::string& filename );

	void	   saveScene( hiprtScene inScene, const std::string& filename );
	hiprtScene loadScene( const std::string& filename );

	void exportGeometryAabb( hiprtGeometry inGeometry, float3& outAabbMin, float3& outAabbMax );
	void exportSceneAabb( hiprtScene inScene, float3& outAabbMin, float3& outAabbMax );

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
		bool								 cache );

	void buildKernelsFromBitcode(
		const std::vector<const char*>&		 funcNames,
		const std::filesystem::path&		 moduleName,
		const std::string_view				 bitcodeBinary,
		uint32_t							 numGeomTypes,
		uint32_t							 numRayTypes,
		const std::vector<hiprtFuncNameSet>& funcNameSets,
		std::vector<oroFunction>&			 functions,
		bool								 cache );

	void setCacheDir( const std::filesystem::path& path );

	uint32_t	getSMCount() const;
	uint32_t	getMaxBlockSize() const;
	uint32_t	getMaxGridSize() const;
	std::string getDeviceName() const;
	std::string getGcnArchName() const;
	std::string getDriverVersion() const;

	oroDevice	 getDevice() const noexcept;
	OrochiUtils& getOrochiUtils() { return m_oroutils; }
	Compiler&	 getCompiler() { return m_compiler; }

	bool enableHwi() const;

  private:
	oroDevice	m_device;
	oroCtx		m_ctxt;
	OrochiUtils m_oroutils;
	Compiler	m_compiler;

	std::mutex											m_poolMutex;
	std::map<std::pair<oroDeviceptr, size_t>, uint32_t> m_poolHeads;
};
} // namespace hiprt
