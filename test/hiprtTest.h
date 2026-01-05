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
#define NOMINMAX
#include <hiprt/hiprt.h>
#include <Orochi/Orochi.h>
#include <embree4/rtcore.h>
#include <optional>
#include <filesystem>
#include <contrib/cpp20/source_location.h>
#include <gtest/gtest.h>

#if defined( _MSC_VER )
#define ASSERT( cond )  \
	if ( !( cond ) )    \
	{                   \
		__debugbreak(); \
	}
#elif defined( __GNUC__ )
#include <signal.h>
#define ASSERT( cond )    \
	if ( !( cond ) )      \
	{                     \
		raise( SIGTRAP ); \
	}
#else
#define ASSERT( cond )
#endif

#include <test/shared.h>

void checkOro( oroError res, const source_location& location = source_location::current() );
void checkOrortc( orortcResult res, const source_location& location = source_location::current() );
void checkHiprt( hiprtError res, const source_location& location = source_location::current() );

std::string			  getEnvVariable( const std::string& key );
std::filesystem::path getRootDir();

namespace
{
#if defined( HIPRT_BITCODE_LINKING )
constexpr bool UseBitcode = true;
#else
constexpr bool UseBitcode = false;
#endif
} // namespace

struct CmdArguments
{
	uint32_t			  m_ww					   = 512u;
	uint32_t			  m_wh					   = 512u;
	std::filesystem::path m_referencePath		   = getRootDir() / "test/references/";
	uint32_t			  m_deviceIdx			   = 0u;
	bool				  m_usePrecompiledBitcodes = false;
};

extern CmdArguments g_parsedArgs;

struct Aabb;

class InitCommandlineArgs : public testing::Environment
{
  public:
	explicit InitCommandlineArgs( const CmdArguments command_line_arg ) { g_parsedArgs = command_line_arg; }
};

class hiprtTest : public ::testing::Test
{
  public:
	void SetUp();
	void TearDown() { checkOro( oroCtxDestroy( m_oroCtx ) ); }

  protected:
	void buildBvh( hiprtGeometryBuildInput& buildInput );

	void buildEmbreeBvh(
		RTCDevice embreeDevice, std::vector<RTCBuildPrimitive>& embreePrims, void* geomData, hiprtBvhNodeList& nodeList );

	void buildEmbreeGeometryBvh(
		RTCDevice embreeDevice, const float3* vertices, const uint32_t* indices, hiprtGeometryBuildInput& buildInput );

	void buildEmbreeSceneBvh(
		RTCDevice						  embreeDevice,
		const std::vector<Aabb>&		  geomBoxes,
		const std::vector<hiprtFrameSRT>& frames,
		hiprtSceneBuildInput&			  buildInput );

	bool readSourceCode(
		const std::filesystem::path&					  srcPath,
		std::string&									  src,
		std::optional<std::vector<std::filesystem::path>> includes = std::nullopt );

	void validateAndWriteImage(
		const std::filesystem::path& imgPath, uint8_t* data, std::optional<std::filesystem::path> refFilename = std::nullopt );

	void writeImage( const std::filesystem::path& imgPath, uint32_t width, uint32_t height, uint8_t* data );

	void waitForCompletion( oroStream stream = 0 ) { checkOro( oroStreamSynchronize( stream ) ); }

	hiprtError buildTraceKernels(
		hiprtContext								 ctxt,
		const std::filesystem::path&				 srcPath,
		std::vector<const char*>					 functionNames,
		std::vector<hiprtApiFunction>&				 functionsOut,
		std::optional<std::vector<const char*>>		 opts		  = std::nullopt,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		uint32_t									 numGeomTypes = 0u,
		uint32_t									 numRayTypes  = 1u );

	hiprtError buildTraceKernel(
		hiprtContext								 ctxt,
		const std::filesystem::path&				 srcPath,
		const std::string&							 functionName,
		oroFunction&								 functionOut,
		std::optional<std::vector<const char*>>		 opts		  = std::nullopt,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		uint32_t									 numGeomTypes = 0u,
		uint32_t									 numRayTypes  = 1u );

	hiprtError buildTraceKernelsFromBitcode(
		hiprtContext								 ctxt,
		const std::filesystem::path&				 srcPath,
		std::vector<const char*>					 functionNames,
		std::vector<hiprtApiFunction>&				 functionsOut,
		std::optional<std::vector<const char*>>		 opts		  = std::nullopt,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		uint32_t									 numGeomTypes = 0u,
		uint32_t									 numRayTypes  = 1u );

	hiprtError buildTraceKernelFromBitcode(
		hiprtContext								 ctxt,
		const std::filesystem::path&				 srcPath,
		const std::string&							 functionName,
		oroFunction&								 functionOut,
		std::optional<std::vector<const char*>>		 opts		  = std::nullopt,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		uint32_t									 numGeomTypes = 0u,
		uint32_t									 numRayTypes  = 1u );

	void launchKernel( oroFunction func, uint32_t nx, uint32_t ny, void** args, uint32_t sharedMemoryBytes = 0 );
	void launchKernel(
		oroFunction func, uint32_t nx, uint32_t ny, uint32_t tx, uint32_t ty, void** args, uint32_t sharedMemoryBytes = 0 );

	template <typename T>
	void malloc( T*& ptr, size_t n )
	{
		checkOro( oroMalloc( reinterpret_cast<oroDeviceptr*>( &ptr ), sizeof( T ) * n ) );
	}

	void free( void* ptr ) { checkOro( oroFree( reinterpret_cast<oroDeviceptr>( ptr ) ) ); }

	void memset( void* ptr, int val, size_t n ) { checkOro( oroMemsetD8( reinterpret_cast<oroDeviceptr>( ptr ), val, n ) ); }

	template <typename T>
	void copyHtoD( T* dst, const T* src, size_t n )
	{
		checkOro( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( dst ), const_cast<T*>( src ), sizeof( T ) * n ) );
	}

	template <typename T>
	void copyDtoH( T* dst, T* src, size_t n )
	{
		checkOro( oroMemcpyDtoH( dst, reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n ) );
	}

	template <typename T>
	void copyDtoD( T* dst, T* src, size_t n )
	{
		checkOro(
			oroMemcpyDtoD( reinterpret_cast<oroDeviceptr>( dst ), reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n ) );
	}

	template <typename T>
	void copyHtoDAsync( T* dst, const T* src, size_t n, oroStream stream )
	{
		checkOro( oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( dst ), const_cast<T*>( src ), sizeof( T ) * n, stream ) );
	}

	template <typename T>
	void copyDtoHAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		checkOro( oroMemcpyDtoHAsync( dst, reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n, stream ) );
	}

	template <typename T>
	void copyDtoDAsync( T* dst, T* src, size_t n, oroStream stream )
	{
		checkOro( oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( dst ), reinterpret_cast<oroDeviceptr>( src ), sizeof( T ) * n, stream ) );
	}

  protected:
	hiprtContextCreationInput m_ctxtInput;
	oroCtx					  m_oroCtx;
	oroDevice				  m_oroDevice;
};

enum class TestCasesType
{
	TestCornellBox,
	TestBistro,
	TestHairball
};

class ObjTestCases : public hiprtTest
{
  public:
	struct SceneData
	{
		uint32_t* m_bufMaterialIndices;
		uint32_t* m_bufMatIdsPerInstance; // count of material ids per instance use to calculate offset in material Idx buffer
										  // for instance
		Material*				   m_bufMaterials;
		float3*					   m_vertices;
		uint32_t*				   m_vertexOffsets;
		float3*					   m_normals;
		uint32_t*				   m_normalOffsets;
		uint32_t*				   m_indices;
		uint32_t*				   m_indexOffsets;
		Light*					   m_lights;
		uint32_t*				   m_numOfLights;
		hiprtScene				   m_scene;
		std::vector<hiprtGeometry> m_geometries;
		std::vector<hiprtInstance> m_instances;
		std::vector<void*>		   m_garbageCollector;
		hiprtContext			   m_ctx;
	};

	template <TestCasesType T>
	Camera createCamera()
	{
		Camera camera;

		if constexpr ( T == TestCasesType::TestCornellBox )
		{
			camera.m_translation = { 0.0f, 2.5f, 5.8f };
			camera.m_rotation	 = { 0.0f, 0.0f, 1.0f, 0.0f };
			camera.m_fov		 = 45.0f * hiprt::Pi / 180.f;
		}
		else if constexpr ( T == TestCasesType::TestBistro )
		{
			camera.m_translation = { -1200.0f, 254.0f, -260.0f };
			camera.m_rotation	 = { 0.0f, 1.0f, 0.0f, -1.4f };
			camera.m_fov		 = 60.0f * hiprt::Pi / 180.f;
		}
		else if constexpr ( T == TestCasesType::TestHairball )
		{
			camera.m_translation = { 0.0f, 0.0f, 15.0f };
			camera.m_rotation	 = { 0.0f, 0.0f, 1.0f, 0.0f };
			camera.m_fov		 = 45.0f * hiprt::Pi / 180.f;
		}
		else
		{
			ASSERT( 0 );
		}

		return camera;
	}

	void createScene(
		SceneData&					 scene,
		const std::filesystem::path& filename,
		bool						 enableRayMask = false,
		std::optional<hiprtFrameSRT> frame		   = std::nullopt,
		hiprtBuildFlags				 bvhBuildFlag  = hiprtBuildFlagBitPreferFastBuild,
		bool						 time		   = false );

	void setupScene(
		Camera&						 camera,
		const std::filesystem::path& filename,
		bool						 enableRayMask = false,
		std::optional<hiprtFrameSRT> frame		   = std::nullopt,
		hiprtBuildFlags				 bvhBuildFlag  = hiprtBuildFlagBitPreferFastBuild,
		bool						 time		   = false );

	void deleteScene( SceneData& scene );

	void TearDown()
	{
		for ( auto p : m_scene.m_garbageCollector )
			free( p );
		hiprtTest::TearDown();
	}

	void render(
		std::optional<std::filesystem::path> imgPath,
		const std::filesystem::path&		 kernelPath,
		const std::string&					 funcName	 = "PrimaryRayKernel",
		std::optional<std::filesystem::path> refFilename = std::nullopt,
		bool								 time		 = false,
		float								 aoRadius	 = 0.0f );

  public:
	SceneData m_scene;
	Camera	  m_camera;
};

class PerformanceTestCases : public ObjTestCases
{
};
