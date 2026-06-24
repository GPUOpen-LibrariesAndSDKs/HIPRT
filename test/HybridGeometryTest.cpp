#include "HybridGeometryTest.h"

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

void HybridGeometryTest::freeManagedAllocs()
{
	for ( void* ptr : m_managedAllocs )
		assertOroSuccess( oroFree( ptr ) );
	m_managedAllocs.clear();
}

void HybridGeometryTest::SetUp()
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

void HybridGeometryTest::TearDown()
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

hiprtDevicePtr HybridGeometryTest::allocGeometryBuildTemp(
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

hiprtGeometry HybridGeometryTest::buildSampleTriangle(
	std::vector<hiprtFloat3>&	vertices,
	std::vector<uint32_t>&		indices,
	hiprtGeometryBuildInput&	geomInputOut )
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

	geomInputOut						 = {};
	geomInputOut.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInputOut.primitive.triangleMesh = mesh;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtGeometry geom = nullptr;
	EXPECT_EQ( hiprtCreateGeometry( m_context, geomInputOut, opts, geom ), hiprtSuccess )
		<< "hiprtCreateGeometry failed.";
	EXPECT_TRUE( geom != nullptr ) << "Returned geometry handle is null.";
	EXPECT_EQ(
		hiprtBuildGeometry(
			m_context, hiprtBuildOperationBuild, geomInputOut, opts,
			allocGeometryBuildTemp( geomInputOut, opts ), nullptr, geom ),
		hiprtSuccess )
		<< "hiprtBuildGeometry (Build) failed.";
	return geom;
}

namespace
{
bool traceHit(
	hiprtContext ctx, hiprtGeometry geom, float ox, float oy, float oz, float dx, float dy, float dz )
{
	hiprtRay ray{};
	ray.origin	  = { ox, oy, oz };
	ray.minT	  = 0.0f;
	ray.direction = { dx, dy, dz };
	ray.maxT	  = 1e30f;

	hiprtHit hit{};
	hiprtGeomTraversalClosestCPU::traceBatch( ctx, geom, &ray, &hit, 1 );
	return hit.hasHit();
}
} // namespace

TEST_F( HybridGeometryTest, CreateAndDestroyGeometry )
{
	hiprtFloat3* vertices = nullptr;
	uint32_t*	 indices	= nullptr;
	assertOroSuccess( oroMallocManaged(
		reinterpret_cast<oroDeviceptr*>( &vertices ), 3 * sizeof( hiprtFloat3 ), oroMemAttachGlobal ) );
	assertOroSuccess( oroMallocManaged(
		reinterpret_cast<oroDeviceptr*>( &indices ), 3 * sizeof( uint32_t ), oroMemAttachGlobal ) );
	m_managedAllocs.push_back( vertices );
	m_managedAllocs.push_back( indices );

	vertices[0] = { 0.0f, 0.0f, 0.0f };
	vertices[1] = { 1.0f, 0.0f, 0.0f };
	vertices[2] = { 0.0f, 1.0f, 0.0f };
	indices[0]	= 0;
	indices[1]	= 1;
	indices[2]	= 2;

	hiprtTriangleMeshPrimitive mesh{};
	mesh.vertices		 = vertices;
	mesh.vertexCount	 = 3;
	mesh.vertexStride	 = sizeof( hiprtFloat3 );
	mesh.triangleIndices = indices;
	mesh.triangleCount	 = 1;
	mesh.triangleStride	 = 3 * sizeof( uint32_t );

	hiprtGeometryBuildInput in{};
	in.type					  = hiprtPrimitiveTypeTriangleMesh;
	in.primitive.triangleMesh = mesh;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtGeometry geom = nullptr;
	ASSERT_EQ( hiprtCreateGeometry( m_context, in, opts, geom ), hiprtSuccess )
		<< "hiprtCreateGeometry failed on hybrid context.";
	ASSERT_TRUE( geom != nullptr ) << "Returned geometry handle is null.";
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess )
		<< "hiprtDestroyGeometry failed to release GPU/CPU mirror.";
}

TEST_F( HybridGeometryTest, BuildGeometryProducesHittableBvhViaCpuMirror )
{
	std::vector<hiprtFloat3>	vertices;
	std::vector<uint32_t>		indices;
	hiprtGeometryBuildInput		in{};
	hiprtGeometry				geom = buildSampleTriangle( vertices, indices, in );
	ASSERT_TRUE( geom != nullptr );

	EXPECT_TRUE( traceHit( m_context, geom, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "CPU traceBatch on GPU handle should hit the Embree mirror.";
	EXPECT_FALSE( traceHit( m_context, geom, 2.0f, 2.0f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Ray outside triangle should miss.";

	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}

TEST_F( HybridGeometryTest, UpdateGeometryRebuildsCpuMirror )
{
	std::vector<hiprtFloat3>	vertices;
	std::vector<uint32_t>		indices;
	hiprtGeometryBuildInput		in{};
	hiprtGeometry				geom = buildSampleTriangle( vertices, indices, in );
	ASSERT_TRUE( geom != nullptr );

	EXPECT_FALSE( traceHit( m_context, geom, 0.1f, 1.5f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Before update CPU mirror should not cover y=1.5.";

	auto* meshVertices = reinterpret_cast<hiprtFloat3*>( in.primitive.triangleMesh.vertices );
	meshVertices[2]	   = { 0.0f, 2.0f, 0.0f };
	vertices[2]		   = meshVertices[2];

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	ASSERT_EQ(
		hiprtBuildGeometry(
			m_context, hiprtBuildOperationUpdate, in, opts, allocGeometryBuildTemp( in, opts ), nullptr, geom ),
		hiprtSuccess )
		<< "hiprtBuildGeometry (Update) failed to rebuild hybrid mirrors.";

	EXPECT_TRUE( traceHit( m_context, geom, 0.1f, 1.5f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "After update CPU mirror should cover y=1.5.";

	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}
