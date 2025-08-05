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

#include <hiprt/impl/BvhCommon.h>
#include <hiprt/impl/BvhImporter.h>
#include <hiprt/impl/BatchBuilder.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/LbvhBuilder.h>
#include <hiprt/impl/PlocBuilder.h>
#include <hiprt/impl/SbvhBuilder.h>
#include <hiprt/impl/Transform.h>

namespace hiprt
{
Context::Context( const hiprtContextCreationInput& input )
{
	oroApi api = ( input.deviceType == hiprtDeviceAMD ) ? ORO_API_HIP : ORO_API_CUDA;
	oroCtxCreateFromRaw( &m_ctxt, api, input.ctxt );
	m_device = oroSetRawDevice( api, input.device );
}

Context::~Context()
{
	m_oroutils.unloadKernelCache();
	oroCtxCreateFromRawDestroy( m_ctxt );
}

std::vector<hiprtGeometry>
Context::createGeometries( const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	size_t				size = 0;
	std::vector<size_t> sizes( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( batchBuild( buildInputs[i], buildOptions ) )
		{
			logInfo( "BatchBuild::createGeometry\n" );
			sizes[i] = BatchBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::createGeometry\n" );
			sizes[i] = BvhImporter::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::createGeometry\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::createGeometry\n" );
			sizes[i] = SbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::createGeometry\n" );
			sizes[i] = PlocBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::createGeometry used instead\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
	}

	oroDeviceptr buffer;
	checkOro( oroMalloc( &buffer, size ) );

	std::vector<hiprtGeometry> geometries( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		geometries[i] = reinterpret_cast<hiprtGeometry>( buffer );
		buffer		  = static_cast<uint8_t*>( buffer ) + sizes[i];
	}

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<oroDeviceptr>( geometries.front() ), size }] = static_cast<uint32_t>( geometries.size() );

	return geometries;
}

void Context::destroyGeometries( const std::vector<hiprtGeometry> geometries )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	for ( hiprtGeometry geometry : geometries )
	{
		auto head = std::find_if(
			m_poolHeads.begin(), m_poolHeads.end(), [&]( const std::pair<std::pair<oroDeviceptr, size_t>, uint32_t>& h ) {
				return reinterpret_cast<hiprtGeometry>( h.first.first ) <= geometry &&
					   reinterpret_cast<uint8_t*>( geometry ) < reinterpret_cast<uint8_t*>( h.first.first ) + h.first.second;
			} );

		if ( head != m_poolHeads.end() )
		{
			if ( --head->second == 0 )
			{
				checkOro( oroFree( head->first.first ) );
				logInfo( "Geometry pool deallocated\n" );
				m_poolHeads.erase( head );
			}
		}
		else
		{
			logWarn( "Trying to destroy a geometry not allocated in this context!\n" );
		}
	}
}

void Context::buildGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	std::vector<hiprtGeometryBuildInput> batchInputs;
	std::vector<hiprtDevicePtr>			 batchBuffers;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( batchBuild( buildInputs[i], buildOptions ) )
		{
			batchInputs.push_back( buildInputs[i] );
			batchBuffers.push_back( buffers[i] );
		}
	}

	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::buildGeometry\n" );
		BatchBuilder::build( *this, batchInputs, buildOptions, temporaryBuffer, stream, batchBuffers );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !batchBuild( buildInputs[i], buildOptions ) )
		{
			if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::buildGeometry\n" );
				BvhImporter::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::buildGeometry\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::buildGeometry\n" );
				SbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::buildGeometry\n" );
				PlocBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::buildGeometry used instead\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
		}
	}
}

void Context::updateGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::updateGeometry\n" );
			BvhImporter::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::updateGeometry\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::updateGeometry\n" );
			SbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::updateGeometry\n" );
			PlocBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::updateGeometry used instead\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
	}
}

size_t Context::getGeometriesBuildTempBufferSize(
	const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	std::vector<hiprtGeometryBuildInput> batchInputs;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( batchBuild( buildInputs[i], buildOptions ) ) batchInputs.push_back( buildInputs[i] );
	}

	size_t size = 0;
	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::getGeometryBuildTempBufferSize\n" );
		size = BatchBuilder::getTemporaryBufferSize( *this, batchInputs, buildOptions );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !batchBuild( buildInputs[i], buildOptions ) )
		{
			if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, BvhImporter::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, SbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::getGeometryBuildTempBufferSize\n" );
				size = std::max( size, PlocBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::getGeometryBuildTempBufferSize used instead\n" );
				size = std::max( size, LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions ) );
			}
		}
	}

	return size;
}

std::vector<hiprtGeometry> Context::compactGeometries( const std::vector<hiprtGeometry>& geometriesIn, oroStream stream )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	size_t				size = 0;
	std::vector<size_t> sizes( geometriesIn.size() );
	for ( size_t i = 0; i < geometriesIn.size(); ++i )
	{
		GeomHeader header;
		checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( geometriesIn[i] ), sizeof( GeomHeader ) ) );
		const size_t primNodeSize = header.m_geomType & 1 ? getTriangleNodeSize() : sizeof( CustomNode );
		sizes[i] =
			getGeometryStorageBufferSize( header.m_primNodeCount, header.m_boxNodeCount, primNodeSize, getBoxNodeSize() );
		size += sizes[i];
	}

	oroDeviceptr buffer;
	checkOro( oroMalloc( &buffer, size ) );

	std::vector<hiprtGeometry> geometriesOut( geometriesIn.size() );
	for ( size_t i = 0; i < geometriesIn.size(); ++i )
	{
		GeomHeader header;
		checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( geometriesIn[i] ), sizeof( GeomHeader ) ) );
		const size_t primNodeSize = header.m_geomType & 1 ? getTriangleNodeSize() : sizeof( CustomNode );
		const size_t boxNodeSize  = getBoxNodeSize();

		geometriesOut[i] = reinterpret_cast<hiprtGeometry>( buffer );
		MemoryArena storageMemoryArena( geometriesOut[i], sizes[i], DefaultAlignment );
		storageMemoryArena.allocate<GeomHeader>();
		void* boxNodes	= storageMemoryArena.allocate<uint8_t>( boxNodeSize * header.m_boxNodeCount );
		void* primNodes = storageMemoryArena.allocate<uint8_t>( primNodeSize * header.m_primNodeCount );

		checkOro( oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( boxNodes ),
			reinterpret_cast<oroDeviceptr>( header.m_boxNodes ),
			boxNodeSize * header.m_boxNodeCount,
			stream ) );

		checkOro( oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( primNodes ),
			reinterpret_cast<oroDeviceptr>( header.m_primNodes ),
			primNodeSize * header.m_primNodeCount,
			stream ) );

		header.m_boxNodes  = boxNodes;
		header.m_primNodes = primNodes;
		checkOro(
			oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( geometriesOut[i] ), &header, sizeof( GeomHeader ), stream ) );

		buffer = static_cast<uint8_t*>( buffer ) + sizes[i];
	}

	{
		std::lock_guard<std::mutex> lockMutex( m_poolMutex );
		m_poolHeads[{ reinterpret_cast<oroDeviceptr>( geometriesOut.front() ), size }] =
			static_cast<uint32_t>( geometriesOut.size() );
	}

	checkOro( oroStreamSynchronize( stream ) );
	destroyGeometries( geometriesIn );

	return geometriesOut;
}

std::vector<hiprtScene>
Context::createScenes( const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	size_t				size = 0;
	std::vector<size_t> sizes( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( batchBuild( buildInputs[i], buildOptions ) )
		{
			logInfo( "BatchBuild::createScene\n" );
			sizes[i] = BatchBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::createScene\n" );
			sizes[i] = BvhImporter::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::createScene\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::createScene\n" );
			sizes[i] = SbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::createScene\n" );
			sizes[i] = PlocBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::createScene used instead\n" );
			sizes[i] = LbvhBuilder::getStorageBufferSize( *this, buildInputs[i], buildOptions );
			size += sizes[i];
		}
	}

	oroDeviceptr buffer;
	checkOro( oroMalloc( &buffer, size ) );

	std::vector<hiprtScene> scenes( buildInputs.size() );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		scenes[i] = reinterpret_cast<hiprtScene>( buffer );
		buffer	  = static_cast<uint8_t*>( buffer ) + sizes[i];
	}

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<oroDeviceptr>( scenes.front() ), size }] = static_cast<uint32_t>( scenes.size() );

	return scenes;
}

void Context::destroyScenes( const std::vector<hiprtScene> scenes )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	for ( hiprtScene scene : scenes )
	{
		auto head = std::find_if(
			m_poolHeads.begin(), m_poolHeads.end(), [&]( const std::pair<std::pair<oroDeviceptr, size_t>, uint32_t>& h ) {
				return reinterpret_cast<hiprtScene>( h.first.first ) <= scene &&
					   reinterpret_cast<uint8_t*>( scene ) < reinterpret_cast<uint8_t*>( h.first.first ) + h.first.second;
			} );

		if ( head != m_poolHeads.end() )
		{
			if ( --head->second == 0 )
			{
				checkOro( oroFree( head->first.first ) );
				logInfo( "Scene pool deallocated\n" );
				m_poolHeads.erase( head );
			}
		}
		else
		{
			logWarn( "Trying to destroy a scene not allocated in this context!\n" );
		}
	}
}

void Context::buildScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	oroStream								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	std::vector<hiprtSceneBuildInput> batchInputs;
	std::vector<hiprtDevicePtr>		  batchBuffers;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( batchBuild( buildInputs[i], buildOptions ) )
		{
			batchInputs.push_back( buildInputs[i] );
			batchBuffers.push_back( buffers[i] );
		}
	}

	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::buildScene\n" );
		BatchBuilder::build( *this, batchInputs, buildOptions, temporaryBuffer, stream, batchBuffers );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !batchBuild( buildInputs[i], buildOptions ) )
		{
			if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::buildScene\n" );
				BvhImporter::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::buildScene\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::buildScene\n" );
				SbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::buildScene\n" );
				PlocBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::buildScene used instead\n" );
				LbvhBuilder::build( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
			}
		}
	}
}

void Context::updateScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	oroStream								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
		{
			logInfo( "CustomBvhImport::updateScene\n" );
			BvhImporter::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
		{
			logInfo( "FastBuild::updateScene\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
		{
			logInfo( "HighQualityBuild::updateScene\n" );
			SbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
		{
			logInfo( "BalancedBuild::updateScene\n" );
			PlocBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
		else
		{
			logWarn( "Unknow build option => FastBuild::updateScene used instead\n" );
			LbvhBuilder::update( *this, buildInputs[i], buildOptions, temporaryBuffer, stream, buffers[i] );
		}
	}
}

size_t Context::getScenesBuildTempBufferSize(
	const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	std::vector<hiprtSceneBuildInput> batchInputs;
	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( InstanceIDBits < 32 && buildInputs[i].instanceCount >= ( 1u << InstanceIDBits ) )
		{
			std::string msg = Utility::format(
				"The number of instances is %u is greater than the maximum supported number of instances (%u)",
				buildInputs[i].instanceCount,
				1u << InstanceIDBits );
			throw std::runtime_error( msg );
		}

		if ( batchBuild( buildInputs[i], buildOptions ) ) batchInputs.push_back( buildInputs[i] );
	}

	size_t size = 0;
	if ( !batchInputs.empty() )
	{
		logInfo( "BatchBuild::getSceneBuildTempBufferSize\n" );
		size = BatchBuilder::getTemporaryBufferSize( *this, batchInputs, buildOptions );
	}

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		if ( !batchBuild( buildInputs[i], buildOptions ) )
		{

			if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitCustomBvhImport )
			{
				logInfo( "CustomBvhImport::getSceneBuildTempBufferSize\n" );
				size += BvhImporter::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferFastBuild )
			{
				logInfo( "FastBuild::getSceneBuildTempBufferSize\n" );
				size += LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferHighQualityBuild )
			{
				logInfo( "HighQualityBuild::getSceneBuildTempBufferSize\n" );
				size += SbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else if ( ( buildOptions.buildFlags & 3 ) == hiprtBuildFlagBitPreferBalancedBuild )
			{
				logInfo( "BalancedBuild::getSceneBuildTempBufferSize\n" );
				size += PlocBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
			else
			{
				logWarn( "Unknow build option => FastBuild::getSceneBuildTempBufferSize used instead\n" );
				size += LbvhBuilder::getTemporaryBufferSize( *this, buildInputs[i], buildOptions );
			}
		}
	}

	return size;
}

std::vector<hiprtScene> Context::compactScenes( const std::vector<hiprtScene>& scenesIn, oroStream stream )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	size_t				size = 0;
	std::vector<size_t> sizes( scenesIn.size() );
	for ( size_t i = 0; i < scenesIn.size(); ++i )
	{
		SceneHeader header;
		checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( scenesIn[i] ), sizeof( SceneHeader ) ) );
		sizes[i] = getSceneStorageBufferSize(
			header.m_primCount,
			header.m_primNodeCount,
			header.m_boxNodeCount,
			getInstanceNodeSize(),
			getBoxNodeSize(),
			header.m_frameCount );
		size += sizes[i];
	}

	oroDeviceptr buffer;
	checkOro( oroMalloc( &buffer, size ) );

	std::vector<hiprtScene> scenesOut( scenesIn.size() );
	for ( size_t i = 0; i < scenesIn.size(); ++i )
	{
		SceneHeader header;
		checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( scenesIn[i] ), sizeof( SceneHeader ) ) );

		scenesOut[i] = reinterpret_cast<hiprtScene>( buffer );
		MemoryArena storageMemoryArena( scenesOut[i], sizes[i], DefaultAlignment );
		storageMemoryArena.allocate<SceneHeader>();
		void*	  boxNodes	= storageMemoryArena.allocate<uint8_t>( getBoxNodeSize() * header.m_boxNodeCount );
		void*	  primNodes = storageMemoryArena.allocate<uint8_t>( getInstanceNodeSize() * header.m_primNodeCount );
		Instance* instances = storageMemoryArena.allocate<Instance>( header.m_primCount );
		Frame*	  frames	= storageMemoryArena.allocate<Frame>( header.m_frameCount );

		checkOro( oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( boxNodes ),
			reinterpret_cast<oroDeviceptr>( header.m_boxNodes ),
			getBoxNodeSize() * header.m_boxNodeCount,
			stream ) );

		checkOro( oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( primNodes ),
			reinterpret_cast<oroDeviceptr>( header.m_primNodes ),
			getInstanceNodeSize() * header.m_primNodeCount,
			stream ) );

		checkOro( oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( instances ),
			reinterpret_cast<oroDeviceptr>( header.m_instances ),
			sizeof( hiprtTransformHeader ) * header.m_primCount,
			stream ) );

		checkOro( oroMemcpyDtoDAsync(
			reinterpret_cast<oroDeviceptr>( frames ),
			reinterpret_cast<oroDeviceptr>( header.m_frames ),
			sizeof( Frame ) * header.m_frameCount,
			stream ) );

		header.m_boxNodes  = boxNodes;
		header.m_primNodes = primNodes;
		header.m_instances = instances;
		header.m_frames	   = frames;
		checkOro(
			oroMemcpyHtoDAsync( reinterpret_cast<oroDeviceptr>( scenesOut[i] ), &header, sizeof( SceneHeader ), stream ) );

		buffer = static_cast<uint8_t*>( buffer ) + sizes[i];
	}

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<oroDeviceptr>( scenesOut.front() ), size }] = static_cast<uint32_t>( scenesOut.size() );

	checkOro( oroStreamSynchronize( stream ) );
	destroyScenes( scenesOut );

	return scenesOut;
}

hiprtFuncTable Context::createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	uint8_t* ptr = nullptr;
	checkOro( oroMalloc(
		reinterpret_cast<oroDeviceptr*>( &ptr ),
		sizeof( hiprtFuncTableHeader ) + numGeomTypes * numRayTypes * sizeof( hiprtFuncDataSet ) ) );
	checkOro( oroMemsetD8(
		reinterpret_cast<oroDeviceptr>( ptr ),
		0,
		sizeof( hiprtFuncTableHeader ) + numGeomTypes * numRayTypes * sizeof( hiprtFuncDataSet ) ) );

	hiprtFuncTableHeader header{
		numGeomTypes, numRayTypes, reinterpret_cast<hiprtFuncDataSet*>( ptr + sizeof( hiprtFuncTableHeader ) ) };
	checkOro( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( ptr ), &header, sizeof( hiprtFuncTableHeader ) ) );

	return reinterpret_cast<hiprtFuncTable>( ptr );
}

void Context::setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	hiprtFuncTableHeader header;
	checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( funcTable ), sizeof( hiprtFuncTableHeader ) ) );

	uint32_t index = header.numGeomTypes * rayType + geomType;
	checkOro(
		oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( &header.funcDataSets[index] ), &set, sizeof( hiprtFuncDataSet ) ) );
}

void Context::destroyFuncTable( hiprtFuncTable funcTable )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroFree( reinterpret_cast<oroDeviceptr>( funcTable ) ) );
}

void Context::createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	const size_t stackEntrySize =
		input.entryType == hiprtStackEntryTypeInstance ? sizeof( hiprtInstanceStackEntry ) : sizeof( uint32_t );
	if ( input.type == hiprtStackTypeDynamic )
	{
		oroDeviceProp prop;
		checkOro( oroGetDeviceProperties( &prop, m_device ) );
		const uint32_t maxThreadsPerMultiProcessor =
			prop.maxThreadsPerMultiProcessor <= 0 ? 2048u : prop.maxThreadsPerMultiProcessor;
		const uint32_t		   stackCount  = prop.multiProcessorCount * maxThreadsPerMultiProcessor;
		const uint32_t		   activeWarps = stackCount / prop.warpSize;
		size_t				   size		   = activeWarps * sizeof( uint32_t ) + stackCount * input.stackSize * stackEntrySize;
		hiprtGlobalStackBuffer stackBuffer{ input.stackSize, stackCount, nullptr };
		checkOro( oroMalloc( reinterpret_cast<oroDeviceptr*>( &stackBuffer.stackData ), size ) );
		checkOro( oroMemsetD8( reinterpret_cast<oroDeviceptr>( stackBuffer.stackData ), 0, sizeof( uint32_t ) * stackCount ) );
		stackBufferOut = stackBuffer;
	}
	else
	{
		size_t				   size = input.stackSize * input.threadCount * stackEntrySize;
		hiprtGlobalStackBuffer stackBuffer{ input.stackSize, input.threadCount, nullptr };
		checkOro( oroMalloc( reinterpret_cast<oroDeviceptr*>( &stackBuffer.stackData ), size ) );
		stackBufferOut = stackBuffer;
	}
}

void Context::destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroFree( reinterpret_cast<oroDeviceptr>( stackBuffer.stackData ) ) );
}

void Context::saveGeometry( hiprtGeometry inGeometry, const std::string& filename )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	size_t size = 0;
	{
		GeomHeader header;
		checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( inGeometry ), sizeof( GeomHeader ) ) );
		size = header.m_size;
	}

	std::vector<uint8_t> buffer( size );
	checkOro( oroMemcpyDtoH( buffer.data(), reinterpret_cast<oroDeviceptr>( inGeometry ), size ) );

	GeomHeader header;
	std::memcpy( &header, buffer.data(), sizeof( GeomHeader ) );
	std::uintptr_t offset = reinterpret_cast<std::uintptr_t>( inGeometry );
	header.m_boxNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_boxNodes ) - offset );
	header.m_primNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_primNodes ) - offset );
	std::memcpy( buffer.data(), &header, sizeof( GeomHeader ) );

	std::ofstream file( filename, std::ios::out | std::ios::binary );
	file.write( reinterpret_cast<char*>( buffer.data() ), header.m_size );
}

hiprtGeometry Context::loadGeometry( const std::string& filename )
{
	std::ifstream file( filename, std::ios::in | std::ios::binary );

	size_t size = 0;
	{
		GeomHeader header;
		file.read( reinterpret_cast<char*>( &header ), sizeof( GeomHeader ) );
		size = header.m_size;
		if ( header.m_rtip != getRtip() && ( getRtip() >= 31 || header.m_rtip >= 31 ) )
		{
			std::string msg = Utility::format(
				"RTIP of the loaded geometry (%u) is not compatible with the RTIP of the current context (%u).",
				header.m_rtip,
				getRtip() );
			throw std::runtime_error( msg );
		}
	}

	std::vector<uint8_t> buffer( size );
	file.clear();
	file.seekg( 0, std::ios::beg );
	file.read( reinterpret_cast<char*>( buffer.data() ), size );

	hiprtGeometry geometry;
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroMalloc( reinterpret_cast<oroDeviceptr*>( &geometry ), size ) );

	GeomHeader header;
	std::memcpy( &header, buffer.data(), sizeof( GeomHeader ) );
	std::uintptr_t offset = reinterpret_cast<std::uintptr_t>( geometry );
	header.m_boxNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_boxNodes ) + offset );
	header.m_primNodes	  = reinterpret_cast<void*>( reinterpret_cast<std::uintptr_t>( header.m_primNodes ) + offset );
	std::memcpy( buffer.data(), &header, sizeof( GeomHeader ) );

	checkOro( oroMemcpyHtoD( reinterpret_cast<oroDeviceptr>( geometry ), buffer.data(), header.m_size ) );

	std::lock_guard<std::mutex> lockMutex( m_poolMutex );
	m_poolHeads[{ reinterpret_cast<oroDeviceptr>( geometry ), header.m_size }] = 1u;

	return geometry;
}

void Context::saveScene( [[maybe_unused]] hiprtScene inScene, [[maybe_unused]] const std::string& filename )
{
	throw std::runtime_error( "Not implemented" );
}

hiprtScene Context::loadScene( [[maybe_unused]] const std::string& filename ) { throw std::runtime_error( "Not implemented" ); }

void Context::exportGeometryAabb( hiprtGeometry inGeometry, float3& outAabbMin, float3& outAabbMax )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	GeomHeader header;
	checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( inGeometry ), sizeof( GeomHeader ) ) );

	constexpr uint32_t Alignment = alignof( Box8Node ) > alignof( Box4Node ) ? alignof( Box8Node ) : alignof( Box4Node );
	constexpr uint32_t Size		 = sizeof( Box8Node ) > sizeof( Box4Node ) ? sizeof( Box8Node ) : sizeof( Box4Node );
	alignas( Alignment ) uint8_t root[Size];
	checkOro( oroMemcpyDtoH( root, reinterpret_cast<oroDeviceptr>( header.m_boxNodes ), getBoxNodeSize() ) );

	Aabb box   = getRtip() >= 31 ? reinterpret_cast<Box8Node*>( root )->aabb() : reinterpret_cast<Box4Node*>( root )->aabb();
	outAabbMin = box.m_min;
	outAabbMax = box.m_max;
}

void Context::exportSceneAabb( hiprtScene inScene, float3& outAabbMin, float3& outAabbMax )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );

	SceneHeader header;
	checkOro( oroMemcpyDtoH( &header, reinterpret_cast<oroDeviceptr>( inScene ), sizeof( SceneHeader ) ) );

	constexpr uint32_t Alignment = alignof( Box8Node ) > alignof( Box4Node ) ? alignof( Box8Node ) : alignof( Box4Node );
	constexpr uint32_t Size		 = sizeof( Box8Node ) > sizeof( Box4Node ) ? sizeof( Box8Node ) : sizeof( Box4Node );
	alignas( Alignment ) uint8_t root[Size];
	checkOro( oroMemcpyDtoH( root, reinterpret_cast<oroDeviceptr>( header.m_boxNodes ), getBoxNodeSize() ) );

	Aabb box   = getRtip() >= 31 ? reinterpret_cast<Box8Node*>( root )->aabb() : reinterpret_cast<Box4Node*>( root )->aabb();
	outAabbMin = box.m_min;
	outAabbMax = box.m_max;
}

void Context::buildKernels(
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
	bool								 cache )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	m_compiler.buildKernels(
		*this,
		funcNames,
		src,
		moduleName,
		headers,
		includeNames,
		options,
		numGeomTypes,
		numRayTypes,
		funcNameSets,
		functions,
		module,
		true,
		cache );
}

void Context::buildKernelsFromBitcode(
	const std::vector<const char*>&		 funcNames,
	const std::filesystem::path&		 moduleName,
	const std::string_view				 bitcodeBinary,
	uint32_t							 numGeomTypes,
	uint32_t							 numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	std::vector<oroFunction>&			 functions,
	bool								 cache )
{
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	m_compiler.buildKernelsFromBitcode(
		*this, funcNames, moduleName, bitcodeBinary, numGeomTypes, numRayTypes, funcNameSets, functions, cache );
}

void Context::setCacheDir( const std::filesystem::path& path ) { m_compiler.setCacheDir( path ); }

uint32_t Context::getSMCount() const
{
	int smCount;
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroDeviceGetAttribute( &smCount, oroDeviceAttributeMultiprocessorCount, m_device ) );
	return smCount;
}

uint32_t Context::getMaxBlockSize() const
{
	oroDeviceProp prop;
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroGetDeviceProperties( &prop, m_device ) );
	return prop.maxThreadsPerBlock;
}

uint32_t Context::getMaxGridSize() const
{
	oroDeviceProp prop;
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroGetDeviceProperties( &prop, m_device ) );
	return prop.maxGridSize[0];
}

std::string Context::getDeviceName() const
{
	oroDeviceProp prop;
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroGetDeviceProperties( &prop, m_device ) );
	return std::string( prop.name );
}

std::string Context::getGcnArchName() const
{
	oroDeviceProp prop;
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroGetDeviceProperties( &prop, m_device ) );
	return std::string( prop.gcnArchName );
}

std::string Context::getDriverVersion() const
{
	int driverVersion;
	checkOro( oroCtxSetCurrent( m_ctxt ) );
	checkOro( oroDriverGetVersion( &driverVersion ) );
	return std::to_string( driverVersion );
}

uint32_t Context::getRtip() const
{
	std::string deviceName = getDeviceName();
	std::string archName   = getGcnArchName();

	uint32_t archNumber = 0;
	if ( archName.substr( 0, 3 ) == "gfx" )
	{
		std::string numberPart = archName.substr( 3 );
		archNumber			   = std::stoi( numberPart );
	}

	uint32_t rtip = 0; // 0 means no hwi
	if ( deviceName.find( "NVIDIA" ) == std::string::npos )
	{
		if ( archNumber >= 1030 ) rtip = 11;
		if ( archNumber >= 1100 ) rtip = 20;
		if ( archNumber >= 1200 )
		{
			int driverVersion = 0;
			checkOro( oroDriverGetVersion( &driverVersion ) );

			bool driverRtip31 = false;
#if defined( __WINDOWS__ )
			if ( driverVersion >= 60400000 ) // on windows: rocm >= 6.4
#else
			if ( driverVersion >= 70000000 ) // on linux: rocm >= 7.0
#endif
				driverRtip31 = true;

			const bool rtcRtip31 = m_compiler.isRtip31Supported();
#if defined( __WINDOWS__ )
			if ( driverRtip31 && !rtcRtip31 )
				logWarn( "The driver supports RTIP 3.1 but HIPRTC DLLs are of an older version; use HIPRTC DLLs 6.4+ to fully "
						 "utilize "
						 "HW ray tracing features\n" );
#else
			if ( !driverRtip31 )
				logWarn( "HW supports RTIP 3.1 but the driver is of an older version; use driver ROCm 6.4+ (Win) or 7.0+ "
						 "(Linux) to fully "
						 "utilize HW ray tracing features\n" );
#endif

			if ( rtcRtip31 )
				rtip = 31;
			else
				rtip = 20;
		}
	}

	return rtip;
}

uint32_t Context::getBranchingFactor() const
{
	if ( getRtip() >= 31 ) return 8;
	return 4;
}

uint32_t Context::getWarpSize() const
{
	std::string deviceName = getDeviceName();
	std::string archName   = getGcnArchName();

	uint32_t archNumber = 0;
	if ( archName.substr( 0, 3 ) == "gfx" )
	{
		std::string numberPart = archName.substr( 3 );
		archNumber			   = std::stoi( numberPart );
	}

	uint32_t warpSize = 32;
	if ( deviceName.find( "NVIDIA" ) == std::string::npos )
	{
		if ( archNumber < 1030 ) warpSize = 64;
	}

	return warpSize;
}

size_t Context::getTriangleNodeSize() const
{
	if ( getRtip() >= 31 ) return sizeof( TrianglePacketNode );
	return sizeof( TrianglePairNode );
}

size_t Context::getBoxNodeSize() const
{
	if ( getRtip() >= 31 ) return sizeof( Box8Node );
	return sizeof( Box4Node );
}

size_t Context::getInstanceNodeSize() const
{
	if ( getRtip() >= 31 ) return sizeof( HwInstanceNode );
	return sizeof( UserInstanceNode );
}
} // namespace hiprt
