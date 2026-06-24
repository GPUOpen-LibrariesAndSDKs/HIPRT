#include "HybridSceneTest.h"

#include <hiprt/hiprt_cpu.h>

#include <string>

extern const char* g_hip_paths[];
extern const char* g_hiprtc_paths[];

namespace
{
void assertOroSuccess( oroError err )
{
	if ( err != oroSuccess )
	{
		const char* msg = nullptr;
		oroGetErrorString( err, &msg );
		FAIL() << "Orochi error: " << ( msg != nullptr ? msg : "unknown" );
	}
}
} // namespace

void HybridSceneTest::freeManagedAllocs()
{
	for ( void* ptr : m_managedAllocs )
		assertOroSuccess( oroFree( ptr ) );
	m_managedAllocs.clear();
}

void HybridSceneTest::SetUp()
{
	oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0, g_hip_paths, g_hiprtc_paths );
	assertOroSuccess( oroInit( 0 ) );
	assertOroSuccess( oroDeviceGet( &m_oroDevice, 0 ) );
	assertOroSuccess( oroCtxCreate( &m_oroCtx, 0, m_oroDevice ) );
	assertOroSuccess( oroCtxSetCurrent( m_oroCtx ) );

	oroDeviceProp props{};
	assertOroSuccess( oroGetDeviceProperties( &props, m_oroDevice ) );

	hiprtContextCreationInput ctxInput{};
	ctxInput.numCpuThreads = 1;
	ctxInput.ctxt			 = oroGetRawCtx( m_oroCtx );
	ctxInput.device		 = oroGetRawDevice( m_oroDevice );
	ctxInput.deviceType	 = ( std::string( props.name ).find( "NVIDIA" ) != std::string::npos )
							   ? static_cast<hiprtDeviceType>( hiprtDeviceNVIDIA | hiprtDeviceCPU )
							   : static_cast<hiprtDeviceType>( hiprtDeviceAMD | hiprtDeviceCPU );

	ASSERT_EQ( hiprtCreateContext( HIPRT_API_VERSION, ctxInput, m_context ), hiprtSuccess )
		<< "Failed to create hybrid context.";
	ASSERT_TRUE( m_context != nullptr ) << "Context pointer is null after creation.";
}

void HybridSceneTest::TearDown()
{
	if ( m_context != nullptr )
		EXPECT_EQ( hiprtDestroyContext( m_context ), hiprtSuccess )
			<< "Failed to cleanly destroy hybrid context.";
	m_context = nullptr;

	freeManagedAllocs();

	if ( m_oroCtx != nullptr )
		assertOroSuccess( oroCtxDestroy( m_oroCtx ) );
	m_oroCtx = nullptr;
}

hiprtGeometry HybridSceneTest::buildTriangleGeometry(
	std::vector<hiprtFloat3>& vertices,
	std::vector<uint32_t>&	  indices )
{
	hiprtFloat3* hostVertices = nullptr;
	uint32_t*	 hostIndices  = nullptr;
	assertOroSuccess( oroMallocManaged(
		reinterpret_cast<oroDeviceptr*>( &hostVertices ), 3 * sizeof( hiprtFloat3 ), oroMemAttachGlobal ) );
	assertOroSuccess( oroMallocManaged(
		reinterpret_cast<oroDeviceptr*>( &hostIndices ), 3 * sizeof( uint32_t ), oroMemAttachGlobal ) );
	m_managedAllocs.push_back( hostVertices );
	m_managedAllocs.push_back( hostIndices );

	hostVertices[0] = { 0.0f, 0.0f, 0.0f };
	hostVertices[1] = { 1.0f, 0.0f, 0.0f };
	hostVertices[2] = { 0.0f, 1.0f, 0.0f };
	hostIndices[0]	= 0;
	hostIndices[1]	= 1;
	hostIndices[2]	= 2;

	vertices = { hostVertices[0], hostVertices[1], hostVertices[2] };
	indices	 = { hostIndices[0], hostIndices[1], hostIndices[2] };

	hiprtTriangleMeshPrimitive mesh{};
	mesh.vertices		 = hostVertices;
	mesh.vertexCount	 = 3;
	mesh.vertexStride	 = sizeof( hiprtFloat3 );
	mesh.triangleIndices = hostIndices;
	mesh.triangleCount	 = 1;
	mesh.triangleStride	 = 3 * sizeof( uint32_t );

	hiprtGeometryBuildInput in{};
	in.type					  = hiprtPrimitiveTypeTriangleMesh;
	in.primitive.triangleMesh = mesh;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtGeometry geom = nullptr;
	EXPECT_EQ( hiprtCreateGeometry( m_context, in, opts, geom ), hiprtSuccess )
		<< "hiprtCreateGeometry failed.";
	EXPECT_TRUE( geom != nullptr ) << "Returned geometry handle is null.";
	EXPECT_EQ(
		hiprtBuildGeometry(
			m_context, hiprtBuildOperationBuild, in, opts, allocGeometryBuildTemp( in, opts ), nullptr, geom ),
		hiprtSuccess )
		<< "hiprtBuildGeometry failed.";
	return geom;
}

void HybridSceneTest::stageSceneBuildInput(
	hiprtSceneBuildInput&				in,
	const std::vector<hiprtInstance>&	instances,
	const std::vector<hiprtFrameSRT>& frames )
{
	ASSERT_FALSE( instances.empty() );
	ASSERT_EQ( instances.size(), frames.size() );

	hiprtInstance* hostInstances = nullptr;
	hiprtFrameSRT* hostFrames  = nullptr;
	assertOroSuccess( oroMallocManaged(
		reinterpret_cast<oroDeviceptr*>( &hostInstances ),
		instances.size() * sizeof( hiprtInstance ),
		oroMemAttachGlobal ) );
	assertOroSuccess( oroMallocManaged(
		reinterpret_cast<oroDeviceptr*>( &hostFrames ),
		frames.size() * sizeof( hiprtFrameSRT ),
		oroMemAttachGlobal ) );
	m_managedAllocs.push_back( hostInstances );
	m_managedAllocs.push_back( hostFrames );

	for ( size_t i = 0; i < instances.size(); ++i )
		hostInstances[i] = instances[i];
	for ( size_t i = 0; i < frames.size(); ++i )
		hostFrames[i] = frames[i];

	in.instances	  = hostInstances;
	in.instanceCount  = static_cast<uint32_t>( instances.size() );
	in.instanceFrames = hostFrames;
	in.frameCount	  = static_cast<uint32_t>( frames.size() );
	in.frameType	  = hiprtFrameTypeSRT;
}

hiprtDevicePtr HybridSceneTest::allocGeometryBuildTemp(
	const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions& buildOptions )
{
	size_t		   tempSize = 0;
	hiprtDevicePtr temp	  = nullptr;
	if ( hiprtGetGeometryBuildTemporaryBufferSize( m_context, buildInput, buildOptions, tempSize ) != hiprtSuccess )
		return nullptr;
	if ( tempSize == 0 )
		return nullptr;

	assertOroSuccess( oroMalloc( reinterpret_cast<oroDeviceptr*>( &temp ), tempSize ) );
	m_managedAllocs.push_back( reinterpret_cast<void*>( temp ) );
	return temp;
}

hiprtDevicePtr HybridSceneTest::allocSceneBuildTemp(
	const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions& buildOptions )
{
	size_t		   tempSize = 0;
	hiprtDevicePtr temp	  = nullptr;
	if ( hiprtGetSceneBuildTemporaryBufferSize( m_context, buildInput, buildOptions, tempSize ) != hiprtSuccess )
		return nullptr;
	if ( tempSize == 0 )
		return nullptr;

	assertOroSuccess( oroMalloc( reinterpret_cast<oroDeviceptr*>( &temp ), tempSize ) );
	m_managedAllocs.push_back( reinterpret_cast<void*>( temp ) );
	return temp;
}

namespace
{
hiprtFrameSRT makeSrt( float tx, float ty, float tz )
{
	hiprtFrameSRT f{};
	f.rotation	  = { 0.0f, 0.0f, 0.0f, 0.0f };
	f.scale		  = { 1.0f, 1.0f, 1.0f };
	f.translation = { tx, ty, tz };
	f.time		  = 0.0f;
	return f;
}

bool traceScene(
	hiprtContext ctx, hiprtScene scene, float ox, float oy, float oz, float dx, float dy, float dz )
{
	hiprtRay ray{};
	ray.origin	  = { ox, oy, oz };
	ray.minT	  = 0.0f;
	ray.direction = { dx, dy, dz };
	ray.maxT	  = 1e30f;

	hiprtHit hit{};
	hiprtGeomTraversalClosestCPU::traceBatch( ctx, scene, &ray, &hit, 1 );
	return hit.instanceID != hiprtInvalidValue;
}
} // namespace

TEST_F( HybridSceneTest, CreateAndDestroyScene )
{
	hiprtSceneBuildInput in{};
	in.instanceCount = 0;
	in.frameType	 = hiprtFrameTypeSRT;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess )
		<< "hiprtCreateScene failed on hybrid context.";
	ASSERT_TRUE( scene != nullptr ) << "Returned scene handle is null.";
	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess )
		<< "hiprtDestroyScene failed to release GPU/CPU mirror.";
}

TEST_F( HybridSceneTest, BuildSceneWithSingleInstanceIsHittableViaCpuMirror )
{
	std::vector<hiprtFloat3> vertices;
	std::vector<uint32_t>	 indices;
	hiprtGeometry			 geom = buildTriangleGeometry( vertices, indices );
	ASSERT_TRUE( geom != nullptr );

	std::vector<hiprtInstance> instances( 1 );
	instances[0].type	  = hiprtInstanceTypeGeometry;
	instances[0].geometry = geom;

	std::vector<hiprtFrameSRT> frames = { makeSrt( 0.0f, 0.0f, 0.0f ) };

	hiprtSceneBuildInput in{};
	stageSceneBuildInput( in, instances, frames );

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess );
	ASSERT_EQ(
		hiprtBuildScene(
			m_context, hiprtBuildOperationBuild, in, opts, allocSceneBuildTemp( in, opts ), nullptr, scene ),
		hiprtSuccess );

	EXPECT_TRUE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "CPU traceBatch on GPU scene handle should hit the Embree mirror.";
	EXPECT_FALSE( traceScene( m_context, scene, 5.0f, 5.0f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Ray outside instance should miss.";

	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess );
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}

TEST_F( HybridSceneTest, BuildSceneWithTranslatedInstance )
{
	std::vector<hiprtFloat3> vertices;
	std::vector<uint32_t>	 indices;
	hiprtGeometry			 geom = buildTriangleGeometry( vertices, indices );
	ASSERT_TRUE( geom != nullptr );

	std::vector<hiprtInstance> instances( 1 );
	instances[0].type	  = hiprtInstanceTypeGeometry;
	instances[0].geometry = geom;

	std::vector<hiprtFrameSRT> frames = { makeSrt( 3.0f, 0.0f, 0.0f ) };

	hiprtSceneBuildInput in{};
	stageSceneBuildInput( in, instances, frames );

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess );
	ASSERT_EQ(
		hiprtBuildScene(
			m_context, hiprtBuildOperationBuild, in, opts, allocSceneBuildTemp( in, opts ), nullptr, scene ),
		hiprtSuccess );

	EXPECT_TRUE( traceScene( m_context, scene, 3.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Translated instance should be hit at x=3.25 via CPU mirror.";
	EXPECT_FALSE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Original position should miss after +3X translation.";

	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess );
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}

TEST_F( HybridSceneTest, UpdateSceneRebuildsCpuMirror )
{
	std::vector<hiprtFloat3> vertices;
	std::vector<uint32_t>	 indices;
	hiprtGeometry			 geom = buildTriangleGeometry( vertices, indices );
	ASSERT_TRUE( geom != nullptr );

	std::vector<hiprtInstance> instances( 1 );
	instances[0].type	  = hiprtInstanceTypeGeometry;
	instances[0].geometry = geom;

	std::vector<hiprtFrameSRT> frames = { makeSrt( 0.0f, 0.0f, 0.0f ) };

	hiprtSceneBuildInput in{};
	stageSceneBuildInput( in, instances, frames );

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess );
	ASSERT_EQ(
		hiprtBuildScene(
			m_context, hiprtBuildOperationBuild, in, opts, allocSceneBuildTemp( in, opts ), nullptr, scene ),
		hiprtSuccess );

	EXPECT_TRUE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Before update hit at x=0.25.";

	auto* stagedFrames = reinterpret_cast<hiprtFrameSRT*>( in.instanceFrames );
	stagedFrames[0]	 = makeSrt( 3.0f, 0.0f, 0.0f );
	ASSERT_EQ(
		hiprtBuildScene(
			m_context, hiprtBuildOperationUpdate, in, opts, allocSceneBuildTemp( in, opts ), nullptr, scene ),
		hiprtSuccess )
		<< "hiprtBuildScene (Update) failed.";

	EXPECT_FALSE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "After update old position should miss.";
	EXPECT_TRUE( traceScene( m_context, scene, 3.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "After update new position x=3.25 should hit.";

	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess );
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}
