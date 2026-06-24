#include "CpuSceneTest.h"

#include <hiprt/hiprt_cpu.h>

void CpuSceneTest::SetUp()
{
	hiprtContextCreationInput ctxInput{};
	ctxInput.deviceType	   = hiprtDeviceCPU;
	ctxInput.numCpuThreads = 1;
	ASSERT_EQ( hiprtCreateContext( HIPRT_API_VERSION, ctxInput, m_context ), hiprtSuccess )
		<< "Failed to create CPU context.";
	ASSERT_TRUE( m_context != nullptr ) << "Context pointer is null after creation.";
}

void CpuSceneTest::TearDown()
{
	if ( m_context != nullptr )
		EXPECT_EQ( hiprtDestroyContext( m_context ), hiprtSuccess )
			<< "Failed to cleanly destroy CPU context.";
}

hiprtGeometry CpuSceneTest::buildTriangleGeometry(
	std::vector<hiprtFloat3>& vertices,
	std::vector<uint32_t>&	  indices )
{
	vertices = {
		{ 0.0f, 0.0f, 0.0f },
		{ 1.0f, 0.0f, 0.0f },
		{ 0.0f, 1.0f, 0.0f } };
	indices = { 0, 1, 2 };

	hiprtTriangleMeshPrimitive mesh{};
	mesh.vertices		 = vertices.data();
	mesh.vertexCount	 = 3;
	mesh.vertexStride	 = sizeof( hiprtFloat3 );
	mesh.triangleIndices = indices.data();
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
		hiprtBuildGeometry( m_context, hiprtBuildOperationBuild, in, opts, nullptr, nullptr, geom ),
		hiprtSuccess )
		<< "hiprtBuildGeometry failed.";
	return geom;
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

TEST_F( CpuSceneTest, CreateAndDestroyScene )
{
	hiprtSceneBuildInput in{};
	in.instanceCount = 0;
	in.frameType	 = hiprtFrameTypeSRT;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess )
		<< "hiprtCreateScene failed on CPU context.";
	ASSERT_TRUE( scene != nullptr ) << "Returned scene handle is null.";
	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess )
		<< "hiprtDestroyScene failed to release Embree scene memory.";
}

TEST_F( CpuSceneTest, BuildSceneWithSingleInstanceIsHittable )
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
	in.instances	  = instances.data();
	in.instanceCount  = 1;
	in.instanceFrames = frames.data();
	in.frameCount	  = 1;
	in.frameType	  = hiprtFrameTypeSRT;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess );
	ASSERT_EQ(
		hiprtBuildScene( m_context, hiprtBuildOperationBuild, in, opts, nullptr, nullptr, scene ),
		hiprtSuccess );

	EXPECT_TRUE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Promien w obszar instancji powinien trafic.";
	EXPECT_FALSE( traceScene( m_context, scene, 5.0f, 5.0f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Promien poza instancja nie powinien trafic.";

	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess );
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}

TEST_F( CpuSceneTest, BuildSceneWithTranslatedInstance )
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
	in.instances	  = instances.data();
	in.instanceCount  = 1;
	in.instanceFrames = frames.data();
	in.frameCount	  = 1;
	in.frameType	  = hiprtFrameTypeSRT;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess );
	ASSERT_EQ(
		hiprtBuildScene( m_context, hiprtBuildOperationBuild, in, opts, nullptr, nullptr, scene ),
		hiprtSuccess );

	EXPECT_TRUE( traceScene( m_context, scene, 3.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Po translacji +3X instancja powinna byc trafiona w x=3.25.";
	EXPECT_FALSE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Oryginalna pozycja x=0.25 nie powinna byc juz trafiona.";

	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess );
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}

TEST_F( CpuSceneTest, UpdateSceneRebuildsInstances )
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
	in.instances	  = instances.data();
	in.instanceCount  = 1;
	in.instanceFrames = frames.data();
	in.frameCount	  = 1;
	in.frameType	  = hiprtFrameTypeSRT;

	hiprtBuildOptions opts{};
	opts.buildFlags = hiprtBuildFlagBitPreferFastBuild;

	hiprtScene scene = nullptr;
	ASSERT_EQ( hiprtCreateScene( m_context, in, opts, scene ), hiprtSuccess );
	ASSERT_EQ(
		hiprtBuildScene( m_context, hiprtBuildOperationBuild, in, opts, nullptr, nullptr, scene ),
		hiprtSuccess );

	EXPECT_TRUE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Przed update trafienie w x=0.25.";

	frames[0]		  = makeSrt( 3.0f, 0.0f, 0.0f );
	in.instanceFrames = frames.data();
	ASSERT_EQ(
		hiprtBuildScene( m_context, hiprtBuildOperationUpdate, in, opts, nullptr, nullptr, scene ),
		hiprtSuccess )
		<< "hiprtBuildScene (Update) failed.";

	EXPECT_FALSE( traceScene( m_context, scene, 0.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Po update instancja przesunieta, stara pozycja pusta.";
	EXPECT_TRUE( traceScene( m_context, scene, 3.25f, 0.25f, -1.0f, 0.0f, 0.0f, 1.0f ) )
		<< "Po update trafienie w nowej pozycji x=3.25.";

	ASSERT_EQ( hiprtDestroyScene( m_context, scene ), hiprtSuccess );
	ASSERT_EQ( hiprtDestroyGeometry( m_context, geom ), hiprtSuccess );
}
