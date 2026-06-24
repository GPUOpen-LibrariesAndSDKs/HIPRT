#include "HybridMixedTracingTest.h"

#include <hiprt/hiprt_cpu.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

extern const char* g_hip_paths[];
extern const char* g_hiprtc_paths[];

namespace
{
std::filesystem::path getRootDir()
{
	const char* env = std::getenv( "HIPRT_PATH" );
	return env != nullptr ? std::filesystem::path( env ) : std::filesystem::path( ".." );
}

bool readKernelSource( const std::filesystem::path& srcPath, std::string& sourceCode )
{
	std::ifstream f( srcPath );
	if ( !f.is_open() )
		return false;
	sourceCode.assign( std::istreambuf_iterator<char>( f ), std::istreambuf_iterator<char>() );
	return true;
}

oroFunction buildHybridRowsKernel( hiprtContext context )
{
	const auto		kernelPath = getRootDir() / "test/kernels/HybridRowsKernel.h";
	std::string		sourceCode;
	const std::string moduleName = kernelPath.string();
	if ( !readKernelSource( kernelPath, sourceCode ) )
		return nullptr;

	const char* funcName = "HybridRowsKernel";

	std::vector<std::string> optionStorage;
	optionStorage.push_back( "-I" + getRootDir().string() );
	if ( oroGetCurAPI( 0 ) == ORO_API_HIP )
		optionStorage.push_back( "-ffast-math" );
	else
		optionStorage.push_back( "--use_fast_math" );

	std::vector<const char*> options;
	options.reserve( optionStorage.size() );
	for ( const auto& opt : optionStorage )
		options.push_back( opt.c_str() );

	hiprtApiFunction funcApi = nullptr;
	if ( hiprtBuildTraceKernels(
			 context, 1, &funcName, sourceCode.c_str(), moduleName.c_str(), 0, nullptr, nullptr,
			 static_cast<uint32_t>( options.size() ), options.data(), 0, 0, nullptr, &funcApi, nullptr,
			 true ) != hiprtSuccess )
		return nullptr;

	return *reinterpret_cast<oroFunction*>( &funcApi );
}

void launchHybridRowsKernel(
	oroFunction func, hiprtGeometry geom, uint8_t* image,
	uint32_t width, uint32_t height, uint32_t yBegin, uint32_t yEnd, uint8_t hitValue )
{
	const uint32_t tx	= 8;
	const uint32_t ty	= 8;
	const uint32_t nbx	= ( width + tx - 1 ) / tx;
	const uint32_t nby	= ( height + ty - 1 ) / ty;
	void*		   args[] = {
		&geom, &image, &width, &height, &yBegin, &yEnd, &hitValue,
	};
	const oroError err = oroModuleLaunchKernel( func, nbx, nby, 1, tx, ty, 1, 0, 0, args, 0 );
	if ( err != oroSuccess )
		FAIL() << "oroModuleLaunchKernel failed";
}

void renderCpuRows(
	hiprtContext context, hiprtGeometry geom, uint8_t* image,
	uint32_t width, uint32_t height, uint32_t yBegin, uint32_t yEnd, uint8_t hitValue )
{
	for ( uint32_t y = yBegin; y < yEnd; ++y )
	{
		for ( uint32_t x = 0; x < width; ++x )
		{
			hiprtRay ray{};
			ray.origin	  = { x / static_cast<float>( width ), y / static_cast<float>( height ), -1.0f };
			ray.direction = { 0.0f, 0.0f, 1.0f };
			ray.minT	  = 0.0f;
			ray.maxT	  = 1e30f;

			hiprtHit hit{};
			hiprtGeomTraversalClosestCPU::traceBatch( context, geom, &ray, &hit, 1 );

			const uint32_t index	 = x + y * width;
			const uint8_t  shade	 = hit.hasHit() ? hitValue : 0;
			image[index * 4 + 0]	 = shade;
			image[index * 4 + 1]	 = shade;
			image[index * 4 + 2]	 = shade;
			image[index * 4 + 3]	 = 255;
		}
	}
}
} // namespace

TEST_F( HybridMixedTracingTest, SplitRowsGpuTopCpuBottom )
{
	constexpr uint32_t width	= 32;
	constexpr uint32_t height	= 32;
	constexpr uint32_t splitY	= height / 2;
	constexpr uint8_t  gpuHit	= 200;
	constexpr uint8_t  cpuHit	= 100;

	std::vector<hiprtFloat3>	vertices;
	std::vector<uint32_t>		indices;
	hiprtGeometryBuildInput		in{};
	hiprtGeometry				geom = buildSampleTriangle( vertices, indices, in );
	ASSERT_TRUE( geom != nullptr );

	uint8_t* image = nullptr;
	ASSERT_EQ(
		oroMallocManaged( reinterpret_cast<oroDeviceptr*>( &image ), width * height * 4, oroMemAttachGlobal ),
		oroSuccess );
	m_managedAllocs.push_back( image );
	std::memset( image, 0, width * height * 4 );

	oroFunction kernel = buildHybridRowsKernel( m_context );
	ASSERT_TRUE( kernel != nullptr ) << "Failed to compile HybridRowsKernel.";

	launchHybridRowsKernel( kernel, geom, image, width, height, 0, splitY, gpuHit );
	ASSERT_EQ( oroDeviceSynchronize(), oroSuccess );

	renderCpuRows( m_context, geom, image, width, height, splitY, height, cpuHit );

	const auto pixel = [&]( uint32_t x, uint32_t y ) -> uint8_t { return image[( x + y * width ) * 4]; };

	EXPECT_EQ( pixel( 8, 8 ), gpuHit ) << "GPU top row should hit triangle.";
	EXPECT_EQ( pixel( 8, 24 ), cpuHit ) << "CPU bottom row should hit triangle via mirror.";
	EXPECT_EQ( pixel( 31, 31 ), 0 ) << "Corner should miss on both backends.";

	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}
