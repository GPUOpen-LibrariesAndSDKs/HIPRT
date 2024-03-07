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

#include <test/hiprtewTest.h>
#include <test/CornellBox.h>
#include <numeric>

#define CHECK_ORO( error ) ( checkOro( error, __FILE__, __LINE__ ) )
void checkOro( oroError res, const char* file, uint32_t line );

#define CHECK_HIPRT( error ) ( checkHiprt( error, __FILE__, __LINE__ ) )
void checkHiprt( hiprtError res, const char* file, uint32_t line );

#define CHECK_ORORTC( error ) ( checkOrortc( error, __FILE__, __LINE__ ) )
void checkOrortc( orortcResult res, const char* file, uint32_t line );

void checkOro( oroError res, const char* file, uint32_t line )
{
	if ( res != oroSuccess )
	{
		const char* msg;
		oroGetErrorString( res, &msg );
		std::cerr << "Orochi error: '" << msg << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkOrortc( orortcResult res, const char* file, uint32_t line )
{
	if ( res != ORORTC_SUCCESS )
	{
		std::cerr << "ORORTC error: '" << orortcGetErrorString( res ) << "' [ " << res << " ] on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkHiprt( hiprtError res, const char* file, uint32_t line )
{
	if ( res != hiprtSuccess )
	{
		std::cerr << "HIPRT error: '" << res << "' on line " << line << " "
				  << " in '" << file << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

TEST_F( hiprtewTest, HiprtEwTest )
{
	hiprtContext ctxt;
	CHECK_HIPRT( hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, ctxt ) );

	hiprtTriangleMeshPrimitive mesh;
	mesh.triangleCount	= CornellBoxTriangleCount;
	mesh.triangleStride = sizeof( int3 );
	malloc( reinterpret_cast<int3*&>( mesh.triangleIndices ), mesh.triangleCount );
	std::array<uint32_t, 3 * CornellBoxTriangleCount> idx;
	std::iota( idx.begin(), idx.end(), 0 );
	copyHtoD( reinterpret_cast<int3*>( mesh.triangleIndices ), reinterpret_cast<int3*>( idx.data() ), mesh.triangleCount );

	mesh.vertexCount  = 3 * mesh.triangleCount;
	mesh.vertexStride = sizeof( float3 );
	malloc( reinterpret_cast<float3*&>( mesh.vertices ), mesh.vertexCount );
	copyHtoD( reinterpret_cast<float3*>( mesh.vertices ), const_cast<float3*>( cornellBoxVertices.data() ), mesh.vertexCount );

	hiprtGeometryBuildInput geomInput;
	geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
	geomInput.primitive.triangleMesh = mesh;

	size_t			  geomTempSize;
	hiprtDevicePtr	  geomTemp;
	hiprtBuildOptions options;
	options.buildFlags = hiprtBuildFlagBitPreferFastBuild;
	CHECK_HIPRT( hiprtGetGeometryBuildTemporaryBufferSize( ctxt, geomInput, options, geomTempSize ) );
	malloc( reinterpret_cast<uint8_t*&>( geomTemp ), geomTempSize );

	hiprtGeometry geom;
	CHECK_HIPRT( hiprtCreateGeometry( ctxt, geomInput, options, geom ) );
	CHECK_HIPRT( hiprtBuildGeometry( ctxt, hiprtBuildOperationBuild, geomInput, options, geomTemp, 0, geom ) );

	float3 aabbMin, aabbMax;
	CHECK_HIPRT( hiprtExportGeometryAabb( ctxt, geom, aabbMin, aabbMax ) );

	printf(
		"Geometry bounding box: [%f %f %f] [%f %f %f]\n", aabbMin.x, aabbMin.y, aabbMin.z, aabbMax.x, aabbMax.y, aabbMax.z );

	free( mesh.triangleIndices );
	free( mesh.vertices );
	free( geomTemp );
	CHECK_HIPRT( hiprtDestroyGeometry( ctxt, geom ) );
	CHECK_HIPRT( hiprtDestroyContext( ctxt ) );
}

void hiprtewTest::SetUp()
{
	oroInitialize( (oroApi)( ORO_API_HIP | ORO_API_CUDA ), 0 );

	CHECK_ORO( oroInit( 0 ) );
	CHECK_ORO( oroDeviceGet( &m_oroDevice, 0 ) );
	CHECK_ORO( oroCtxCreate( &m_oroCtx, 0, m_oroDevice ) );

	oroDeviceProp props;
	CHECK_ORO( oroGetDeviceProperties( &props, m_oroDevice ) );
	std::cout << "Executing on '" << props.name << "'" << std::endl;

	if ( std::string( props.name ).find( "NVIDIA" ) != std::string::npos )
		m_ctxtInput.deviceType = hiprtDeviceNVIDIA;
	else
		m_ctxtInput.deviceType = hiprtDeviceAMD;
	m_ctxtInput.ctxt   = oroGetRawCtx( m_oroCtx );
	m_ctxtInput.device = oroGetRawDevice( m_oroDevice );

	int result;
	hiprtewInit( &result );
	ASSERT( result == HIPRTEW_SUCCESS );
	hiprtSetLogLevel( hiprtLogLevelError | hiprtLogLevelWarn );
}

int main( int argc, const char* argv[] )
{
	::testing::InitGoogleTest( &argc, const_cast<char**>( argv ) );

	return RUN_ALL_TESTS();
}