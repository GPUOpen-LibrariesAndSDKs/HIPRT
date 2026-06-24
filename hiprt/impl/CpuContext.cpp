#include "CpuContext.h"
#include <hiprt/impl/CpuDataRegistry.h>

#include <hiprt/impl/Quaternion.h>

#include <iostream>
#include <stdexcept>

namespace hiprt
{
namespace
{
[[noreturn]] void throwNotImplemented( const char* what )
{
	std::cerr << "[CpuContext] not implemented: " << what << std::endl;
	throw std::runtime_error( what );
}

void srtFrameToMatrix3x4( const hiprtFrameSRT& f, float ( &m )[3][4] )
{
	const float4 q = qtFromAxisAngle( f.rotation );
	float		 Q[3][3];
	qtToRotationMatrix( q, Q );
	const float s[3] = { f.scale.x, f.scale.y, f.scale.z };
	for ( int i = 0; i < 3; ++i )
	{
		m[i][0] = Q[i][0] * s[0];
		m[i][1] = Q[i][1] * s[1];
		m[i][2] = Q[i][2] * s[2];
	}
	m[0][3] = f.translation.x;
	m[1][3] = f.translation.y;
	m[2][3] = f.translation.z;
}
} // namespace

CpuContext::CpuContext( const hiprtContextCreationInput& input )
{
	std::string config;
	if ( input.numCpuThreads != 0 )
		config = "threads=" + std::to_string( input.numCpuThreads );

	m_rtcDevice = rtcNewDevice( config.empty() ? nullptr : config.c_str() );

	if ( m_rtcDevice == nullptr )
	{
		const RTCError err = rtcGetDeviceError( nullptr );
		std::cerr << "[CpuContext] rtcNewDevice failed, RTCError=" << static_cast<int>( err ) << std::endl;
		throw std::runtime_error( "rtcNewDevice failed" );
	}

	std::cerr << "[CpuContext] constructed (numCpuThreads=" << input.numCpuThreads
			  << ( input.numCpuThreads == 0 ? " -> all available)" : ")" ) << std::endl;
}

CpuContext::~CpuContext()
{
	if ( m_rtcDevice != nullptr )
	{
		rtcReleaseDevice( m_rtcDevice );
		m_rtcDevice = nullptr;
	}
}

std::vector<hiprtGeometry> CpuContext::createGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions )
{
	std::vector<hiprtGeometry> geometries;
	geometries.reserve( buildInputs.size() );

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		CpuGeometryData* data = new CpuGeometryData();
		data->rtcDevice		  = m_rtcDevice;
		data->rtcScene		  = rtcNewScene( m_rtcDevice );

		if ( data->rtcScene == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			delete data;
			std::cerr << "[CpuContext] rtcNewScene failed at index " << i
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewScene failed" );
		}

		hiprtGeometry handle = reinterpret_cast<hiprtGeometry>( data );
		registerCpuGeom( handle, data );
		geometries.push_back( handle );
	}

	return geometries;
}

void CpuContext::destroyGeometries( const std::vector<hiprtGeometry>& geometries )
{
	for ( hiprtGeometry geometry : geometries )
	{
		CpuGeometryData* data = reinterpret_cast<CpuGeometryData*>( geometry );
		if ( data == nullptr ) continue;

		unregisterCpuGeom( geometry );

		if ( data->rtcScene != nullptr )
		{
			rtcReleaseScene( data->rtcScene );
			data->rtcScene = nullptr;
		}

		delete data;
	}
}

void CpuContext::buildGeometryEntry( CpuGeometryData* data, const hiprtGeometryBuildInput& input, size_t index )
{
	if ( data == nullptr || data->rtcScene == nullptr )
		throw std::runtime_error( "CpuContext::buildGeometryEntry: invalid geometry handle" );

	if ( input.type == hiprtPrimitiveTypeTriangleMesh )
	{
		const hiprtTriangleMeshPrimitive& mesh = input.primitive.triangleMesh;

		RTCGeometry geom = rtcNewGeometry( m_rtcDevice, RTC_GEOMETRY_TYPE_TRIANGLE );
		if ( geom == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			std::cerr << "[CpuContext] rtcNewGeometry failed at index " << index
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewGeometry failed" );
		}

		rtcSetSharedGeometryBuffer(
			geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3,
			mesh.vertices, 0, mesh.vertexStride, mesh.vertexCount );

		if ( mesh.triangleIndices != nullptr )
		{
			rtcSetSharedGeometryBuffer(
				geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
				mesh.triangleIndices, 0, mesh.triangleStride, mesh.triangleCount );
		}
		else
		{
			const uint32_t triCount = mesh.triangleCount != 0 ? mesh.triangleCount : ( mesh.vertexCount / 3 );
			uint32_t*	   indices	= static_cast<uint32_t*>( rtcSetNewGeometryBuffer(
				 geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, 3 * sizeof( uint32_t ), triCount ) );
			if ( indices == nullptr )
			{
				const RTCError err = rtcGetDeviceError( m_rtcDevice );
				rtcReleaseGeometry( geom );
				std::cerr << "[CpuContext] rtcSetNewGeometryBuffer failed at index " << index
						  << ", RTCError=" << static_cast<int>( err ) << std::endl;
				throw std::runtime_error( "rtcSetNewGeometryBuffer failed" );
			}
			for ( uint32_t t = 0; t < triCount; ++t )
			{
				indices[3 * t + 0] = 3 * t + 0;
				indices[3 * t + 1] = 3 * t + 1;
				indices[3 * t + 2] = 3 * t + 2;
			}
		}

		rtcCommitGeometry( geom );
		rtcAttachGeometry( data->rtcScene, geom );
		rtcReleaseGeometry( geom );
	}
	else if ( input.type == hiprtPrimitiveTypeAABBList )
	{
		std::cerr << "[CpuContext] Warning: AABB geometry build is not implemented yet." << std::endl;
	}
	else
	{
		throw std::runtime_error( "CpuContext::buildGeometryEntry: unknown hiprtPrimitiveType" );
	}

	rtcCommitScene( data->rtcScene );
}

void CpuContext::buildGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	(void)buildOptions;
	(void)temporaryBuffer;
	(void)stream;

	for ( size_t i = 0; i < buildInputs.size(); ++i )
		buildGeometryEntry( getCpuGeom( reinterpret_cast<hiprtGeometry>( buffers[i] ) ), buildInputs[i], i );
}

void CpuContext::updateGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	(void)buildOptions;
	(void)temporaryBuffer;
	(void)stream;

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		CpuGeometryData* data = getCpuGeom( reinterpret_cast<hiprtGeometry>( buffers[i] ) );
		if ( data == nullptr )
			throw std::runtime_error( "CpuContext::updateGeometries: invalid geometry handle" );

		if ( data->rtcScene != nullptr )
			rtcReleaseScene( data->rtcScene );

		data->rtcScene = rtcNewScene( m_rtcDevice );
		if ( data->rtcScene == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			std::cerr << "[CpuContext] rtcNewScene failed at index " << i
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewScene failed" );
		}

		buildGeometryEntry( data, buildInputs[i], i );
	}
}

size_t CpuContext::getGeometriesBuildTempBufferSize( const std::vector<hiprtGeometryBuildInput>&, const hiprtBuildOptions )
{
	throwNotImplemented( "CpuContext::getGeometriesBuildTempBufferSize is not implemented yet." );
}

std::vector<hiprtGeometry> CpuContext::compactGeometries( const std::vector<hiprtGeometry>&, oroStream )
{
	throwNotImplemented( "CpuContext::compactGeometries is not implemented yet." );
}

std::vector<hiprtScene> CpuContext::createScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions )
{
	std::vector<hiprtScene> scenes;
	scenes.reserve( buildInputs.size() );

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		CpuSceneData* data = new CpuSceneData();
		data->rtcDevice	   = m_rtcDevice;
		data->rtcScene	   = rtcNewScene( m_rtcDevice );

		if ( data->rtcScene == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			delete data;
			std::cerr << "[CpuContext] rtcNewScene failed at index " << i
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewScene failed" );
		}

		hiprtScene handle = reinterpret_cast<hiprtScene>( data );
		registerCpuScene( handle, data );
		scenes.push_back( handle );
	}
	return scenes;
}

void CpuContext::destroyScenes( const std::vector<hiprtScene>& scenes )
{
	for ( hiprtScene scene : scenes )
	{
		CpuSceneData* data = getCpuScene( scene );
		if ( data == nullptr ) continue;

		unregisterCpuScene( scene );

		if ( data->rtcScene != nullptr )
		{
			rtcReleaseScene( data->rtcScene );
			data->rtcScene = nullptr;
		}
		delete data;
	}
}

void CpuContext::buildSceneEntry( CpuSceneData* data, const hiprtSceneBuildInput& input, size_t index )
{
	if ( data == nullptr || data->rtcScene == nullptr )
		throw std::runtime_error( "CpuContext::buildSceneEntry: invalid scene handle" );

	const hiprtInstance* instances =
		reinterpret_cast<const hiprtInstance*>( input.instances );
	const hiprtTransformHeader* headers =
		reinterpret_cast<const hiprtTransformHeader*>( input.instanceTransformHeaders );

	if ( instances == nullptr && input.instanceCount != 0 )
		throw std::runtime_error( "CpuContext::buildSceneEntry: null instances array" );

	for ( uint32_t i = 0; i < input.instanceCount; ++i )
	{
		const hiprtInstance& inst = instances[i];

		RTCScene child = nullptr;
		if ( inst.type == hiprtInstanceTypeGeometry )
		{
			CpuGeometryData* g = getCpuGeom( inst.geometry );
			child			   = ( g != nullptr ) ? g->rtcScene : nullptr;
		}
		else
		{
			CpuSceneData* s = getCpuScene( inst.scene );
			child			= ( s != nullptr ) ? s->rtcScene : nullptr;
		}

		if ( child == nullptr )
			throw std::runtime_error( "CpuContext::buildSceneEntry: instance has null child scene" );

		RTCGeometry instGeom = rtcNewGeometry( m_rtcDevice, RTC_GEOMETRY_TYPE_INSTANCE );
		if ( instGeom == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			std::cerr << "[CpuContext] rtcNewGeometry(INSTANCE) failed at scene " << index
					  << " instance " << i << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewGeometry(INSTANCE) failed" );
		}

		rtcSetGeometryInstancedScene( instGeom, child );

		float xfm[3][4];
		if ( input.instanceFrames == nullptr )
		{
			xfm[0][0] = 1.0f; xfm[0][1] = 0.0f; xfm[0][2] = 0.0f; xfm[0][3] = 0.0f;
			xfm[1][0] = 0.0f; xfm[1][1] = 1.0f; xfm[1][2] = 0.0f; xfm[1][3] = 0.0f;
			xfm[2][0] = 0.0f; xfm[2][1] = 0.0f; xfm[2][2] = 1.0f; xfm[2][3] = 0.0f;
		}
		else
		{
			const uint32_t frameIndex = ( headers != nullptr ) ? headers[i].frameIndex : i;

			if ( input.frameType == hiprtFrameTypeMatrix )
			{
				const hiprtFrameMatrix* mats =
					reinterpret_cast<const hiprtFrameMatrix*>( input.instanceFrames );
				for ( int r = 0; r < 3; ++r )
					for ( int c = 0; c < 4; ++c )
						xfm[r][c] = mats[frameIndex].matrix[r][c];
			}
			else
			{
				const hiprtFrameSRT* srts =
					reinterpret_cast<const hiprtFrameSRT*>( input.instanceFrames );
				srtFrameToMatrix3x4( srts[frameIndex], xfm );
			}
		}

		rtcSetGeometryTransform( instGeom, 0, RTC_FORMAT_FLOAT3X4_ROW_MAJOR, &xfm[0][0] );
		rtcCommitGeometry( instGeom );
		rtcAttachGeometry( data->rtcScene, instGeom );
		rtcReleaseGeometry( instGeom );
	}

	rtcCommitScene( data->rtcScene );
}

void CpuContext::buildScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	oroStream								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	(void)buildOptions;
	(void)temporaryBuffer;
	(void)stream;

	for ( size_t i = 0; i < buildInputs.size(); ++i )
		buildSceneEntry( getCpuScene( reinterpret_cast<hiprtScene>( buffers[i] ) ), buildInputs[i], i );
}

void CpuContext::updateScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	oroStream								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	(void)buildOptions;
	(void)temporaryBuffer;
	(void)stream;

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		CpuSceneData* data = getCpuScene( reinterpret_cast<hiprtScene>( buffers[i] ) );
		if ( data == nullptr )
			throw std::runtime_error( "CpuContext::updateScenes: invalid scene handle" );

		if ( data->rtcScene != nullptr )
			rtcReleaseScene( data->rtcScene );

		data->rtcScene = rtcNewScene( m_rtcDevice );
		if ( data->rtcScene == nullptr )
		{
			const RTCError err = rtcGetDeviceError( m_rtcDevice );
			std::cerr << "[CpuContext] rtcNewScene failed at index " << i
					  << ", RTCError=" << static_cast<int>( err ) << std::endl;
			throw std::runtime_error( "rtcNewScene failed" );
		}

		buildSceneEntry( data, buildInputs[i], i );
	}
}

size_t CpuContext::getScenesBuildTempBufferSize( const std::vector<hiprtSceneBuildInput>&, const hiprtBuildOptions )
{
	throwNotImplemented( "CpuContext::getScenesBuildTempBufferSize is not implemented yet." );
}

std::vector<hiprtScene> CpuContext::compactScenes( const std::vector<hiprtScene>&, oroStream )
{
	throwNotImplemented( "CpuContext::compactScenes is not implemented yet." );
}

hiprtFuncTable CpuContext::createFuncTable( uint32_t, uint32_t )
{
	throwNotImplemented( "CpuContext::createFuncTable is not implemented yet." );
}

void CpuContext::setFuncTable( hiprtFuncTable, uint32_t, uint32_t, hiprtFuncDataSet )
{
	throwNotImplemented( "CpuContext::setFuncTable is not implemented yet." );
}

void CpuContext::destroyFuncTable( hiprtFuncTable ) { throwNotImplemented( "CpuContext::destroyFuncTable is not implemented yet." ); }

void CpuContext::createGlobalStackBuffer( const hiprtGlobalStackBufferInput&, hiprtGlobalStackBuffer& )
{
	throwNotImplemented( "CpuContext::createGlobalStackBuffer is not implemented yet." );
}

void CpuContext::destroyGlobalStackBuffer( hiprtGlobalStackBuffer ) { throwNotImplemented( "CpuContext::destroyGlobalStackBuffer is not implemented yet." ); }

void CpuContext::saveGeometry( hiprtGeometry, const std::string& ) { throwNotImplemented( "CpuContext::saveGeometry is not implemented yet." ); }

hiprtGeometry CpuContext::loadGeometry( const std::string& ) { throwNotImplemented( "CpuContext::loadGeometry is not implemented yet." ); }

void CpuContext::saveScene( hiprtScene, const std::string& ) { throwNotImplemented( "CpuContext::saveScene is not implemented yet." ); }

hiprtScene CpuContext::loadScene( const std::string& ) { throwNotImplemented( "CpuContext::loadScene is not implemented yet." ); }

void CpuContext::exportGeometryAabb( hiprtGeometry, float3&, float3& )
{
	throwNotImplemented( "CpuContext::exportGeometryAabb is not implemented yet." );
}

void CpuContext::exportSceneAabb( hiprtScene, float3&, float3& ) { throwNotImplemented( "CpuContext::exportSceneAabb is not implemented yet." ); }

void CpuContext::buildKernels(
	const std::vector<const char*>&,
	const std::string&,
	const std::filesystem::path&,
	std::vector<const char*>&,
	std::vector<const char*>&,
	std::vector<const char*>&,
	uint32_t,
	uint32_t,
	const std::vector<hiprtFuncNameSet>&,
	std::vector<oroFunction>&,
	oroModule&,
	bool )
{
	throwNotImplemented( "CpuContext::buildKernels is not implemented yet." );
}

void CpuContext::buildKernelsFromBitcode(
	const std::vector<const char*>&,
	const std::filesystem::path&,
	const std::string_view,
	uint32_t,
	uint32_t,
	const std::vector<hiprtFuncNameSet>&,
	std::vector<oroFunction>&,
	bool )
{
	throwNotImplemented( "CpuContext::buildKernelsFromBitcode is not implemented yet." );
}

} // namespace hiprt
