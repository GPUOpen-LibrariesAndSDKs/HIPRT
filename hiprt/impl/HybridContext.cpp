#include <hiprt/impl/HybridContext.h>
#include <hiprt/impl/CpuDataRegistry.h>

#include <Orochi/Orochi.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace hiprt
{
namespace
{
void syncGpu( oroStream stream )
{
	const oroError err = ( stream != nullptr ) ? oroStreamSynchronize( stream ) : oroDeviceSynchronize();
	if ( err != oroSuccess )
		throw std::runtime_error( "HybridContext: GPU synchronization failed before CPU mirror build" );
}

[[noreturn]] void throwHybridNotImplemented( const char* what )
{
	std::cerr << "[HybridContext] not implemented: " << what << std::endl;
	throw std::runtime_error( what );
}

#ifndef NDEBUG
// Debug: warn if Embree input buffer is device-only memory.
void warnIfDeviceOnly( const void* ptr, const char* what )
{
	if ( ptr == nullptr ) return;
	oroPointerAttribute_t attr{};
	if ( oroPointerGetAttributes( &attr, const_cast<void*>( ptr ) ) == oroSuccess && attr.type == oroMemoryTypeDevice )
	{
		std::cerr << "[HybridContext] WARNING: " << what
				  << " points to device-only memory; Embree (CPU) cannot read it. "
				  << "Use oroMallocManaged / host-visible memory for CPU/hybrid mode.\n";
	}
}
#else
inline void warnIfDeviceOnly( const void*, const char* ) {}
#endif
} // namespace

// Construction

HybridContext::HybridContext( const hiprtContextCreationInput& input )
	: m_gpu( input ), m_cpu( input )
{
	std::cerr << "[HybridContext] constructed - GPU BVH + Embree CPU mirror active.\n";
}

HybridContext::~HybridContext() {}

oroFunction HybridContext::getBatchKernel( HybridBatchKernel kind )
{
	const int idx = static_cast<int>( kind );

	std::lock_guard<std::mutex> lock( m_batchKernelMutex );
	if ( m_batchKernels[idx] != nullptr )
		return m_batchKernels[idx];

	struct KernelDef
	{
		const char* include;
		const char* func;
	};
	static const KernelDef defs[static_cast<int>( HybridBatchKernel::Count )] = {
		{ "#include <hiprt/kernels/HybridSceneTraceBatchKernel.h>\n", "HybridSceneTraceBatchKernel" },
		{ "#include <hiprt/kernels/HybridGeomTraceBatchKernel.h>\n", "HybridGeomTraceBatchKernel" },
		{ "#include <hiprt/kernels/HybridSceneAnyHitBatchKernel.h>\n", "HybridSceneAnyHitBatchKernel" },
		{ "#include <hiprt/kernels/HybridGeomAnyHitBatchKernel.h>\n", "HybridGeomAnyHitBatchKernel" },
	};

	std::vector<const char*>	  funcNames = { defs[idx].func };
	std::string					  src		= defs[idx].include;
	std::vector<const char*>	  headers;
	std::vector<const char*>	  includeNames;
	std::vector<const char*>	  options;
	std::vector<hiprtFuncNameSet> funcNameSets;
	std::vector<oroFunction>	  functions;
	oroModule					  module = nullptr;

	m_gpu.buildKernels(
		funcNames, src, defs[idx].func, headers, includeNames, options, 0, 0, funcNameSets, functions, module, true );

	if ( functions.empty() || functions[0] == nullptr )
		throw std::runtime_error( "HybridContext: failed to compile internal batch kernel" );

	m_batchKernels[idx] = functions[0];
	return m_batchKernels[idx];
}

// Geometry

std::vector<hiprtGeometry> HybridContext::createGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	std::vector<hiprtGeometry> gpuHandles = m_gpu.createGeometries( buildInputs, buildOptions );
	std::vector<hiprtGeometry> cpuHandles = m_cpu.createGeometries( buildInputs, buildOptions );

	for ( size_t i = 0; i < gpuHandles.size(); ++i )
	{
		auto* cpuData = reinterpret_cast<CpuGeometryData*>( cpuHandles[i] );
		m_gpuToCpuGeom[gpuHandles[i]] = cpuHandles[i];
		registerCpuGeom( gpuHandles[i], cpuData );
	}

	return gpuHandles;
}

void HybridContext::destroyGeometries( const std::vector<hiprtGeometry>& gpuGeometries )
{
	std::vector<hiprtGeometry> cpuHandles;
	cpuHandles.reserve( gpuGeometries.size() );

	for ( hiprtGeometry g : gpuGeometries )
	{
		auto it = m_gpuToCpuGeom.find( g );
		if ( it != m_gpuToCpuGeom.end() )
		{
			cpuHandles.push_back( it->second );
			unregisterCpuGeom( g );
			m_gpuToCpuGeom.erase( it );
		}
	}

	m_gpu.destroyGeometries( gpuGeometries );
	if ( !cpuHandles.empty() )
		m_cpu.destroyGeometries( cpuHandles );
}

void HybridContext::dispatchCpuGeomBuild(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const std::vector<hiprtDevicePtr>&			gpuBuffers,
	bool										isUpdate )
{
	// Map GPU buffers to CPU buffers.
	std::vector<hiprtDevicePtr> cpuBuffers;
	cpuBuffers.reserve( gpuBuffers.size() );
	for ( auto gp : gpuBuffers )
	{
		auto it = m_gpuToCpuGeom.find( reinterpret_cast<hiprtGeometry>( gp ) );
		if ( it == m_gpuToCpuGeom.end() )
			throw std::runtime_error( "HybridContext: GPU geometry buffer not found in map" );
		cpuBuffers.push_back( reinterpret_cast<hiprtDevicePtr>( it->second ) );
	}

	// Debug: check host visibility of primitive data.
	for ( const auto& in : buildInputs )
	{
		if ( in.type == hiprtPrimitiveTypeTriangleMesh )
		{
			warnIfDeviceOnly( in.primitive.triangleMesh.vertices, "geometry vertices" );
			warnIfDeviceOnly( in.primitive.triangleMesh.triangleIndices, "geometry triangleIndices" );
		}
		else if ( in.type == hiprtPrimitiveTypeAABBList )
		{
			warnIfDeviceOnly( in.primitive.aabbList.aabbs, "geometry aabbs" );
		}
	}

	if ( isUpdate )
		m_cpu.updateGeometries( buildInputs, {}, nullptr, nullptr, cpuBuffers );
	else
		m_cpu.buildGeometries( buildInputs, {}, nullptr, nullptr, cpuBuffers );
}

void HybridContext::buildGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	m_gpu.buildGeometries( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
	syncGpu( stream );
	dispatchCpuGeomBuild( buildInputs, buffers, false );
}

void HybridContext::updateGeometries(
	const std::vector<hiprtGeometryBuildInput>& buildInputs,
	const hiprtBuildOptions						buildOptions,
	hiprtDevicePtr								temporaryBuffer,
	oroStream									stream,
	std::vector<hiprtDevicePtr>&				buffers )
{
	m_gpu.updateGeometries( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
	syncGpu( stream );
	dispatchCpuGeomBuild( buildInputs, buffers, true );
}

size_t HybridContext::getGeometriesBuildTempBufferSize(
	const std::vector<hiprtGeometryBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	return m_gpu.getGeometriesBuildTempBufferSize( buildInputs, buildOptions );
}

std::vector<hiprtGeometry> HybridContext::compactGeometries(
	const std::vector<hiprtGeometry>& /*geometries*/, oroStream /*stream*/ )
{
	throwHybridNotImplemented( "compactGeometries is not supported in hybrid mode (CPU mirror rebuild required)" );
}

// Scene

std::vector<hiprtScene> HybridContext::createScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	std::vector<hiprtScene> gpuHandles = m_gpu.createScenes( buildInputs, buildOptions );
	std::vector<hiprtScene> cpuHandles = m_cpu.createScenes( buildInputs, buildOptions );

	for ( size_t i = 0; i < gpuHandles.size(); ++i )
	{
		auto* cpuData = reinterpret_cast<CpuSceneData*>( cpuHandles[i] );
		m_gpuToCpuScene[gpuHandles[i]] = cpuHandles[i];
		registerCpuScene( gpuHandles[i], cpuData );
	}

	return gpuHandles;
}

void HybridContext::destroyScenes( const std::vector<hiprtScene>& gpuScenes )
{
	std::vector<hiprtScene> cpuHandles;
	cpuHandles.reserve( gpuScenes.size() );

	for ( hiprtScene s : gpuScenes )
	{
		auto it = m_gpuToCpuScene.find( s );
		if ( it != m_gpuToCpuScene.end() )
		{
			cpuHandles.push_back( it->second );
			unregisterCpuScene( s );
			m_gpuToCpuScene.erase( it );
		}
	}

	m_gpu.destroyScenes( gpuScenes );
	if ( !cpuHandles.empty() )
		m_cpu.destroyScenes( cpuHandles );
}

void HybridContext::dispatchCpuSceneBuild(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const std::vector<hiprtDevicePtr>&		 gpuBuffers,
	bool									 isUpdate )
{
	// Remap GPU instance handles to CPU handles.
	std::vector<hiprtDevicePtr>				 cpuBuffers;
	std::vector<hiprtSceneBuildInput>		 cpuInputs;
	std::vector<std::vector<hiprtInstance>>  remappedInstances( buildInputs.size() );

	cpuBuffers.reserve( gpuBuffers.size() );
	cpuInputs.reserve( buildInputs.size() );

	for ( size_t i = 0; i < buildInputs.size(); ++i )
	{
		auto it = m_gpuToCpuScene.find( reinterpret_cast<hiprtScene>( gpuBuffers[i] ) );
		if ( it == m_gpuToCpuScene.end() )
			throw std::runtime_error( "HybridContext: GPU scene buffer not found in map" );
		cpuBuffers.push_back( reinterpret_cast<hiprtDevicePtr>( it->second ) );

		hiprtSceneBuildInput cpuIn = buildInputs[i];

		warnIfDeviceOnly( buildInputs[i].instances, "scene instances" );
		warnIfDeviceOnly( buildInputs[i].instanceTransformHeaders, "scene instanceTransformHeaders" );
		warnIfDeviceOnly( buildInputs[i].instanceFrames, "scene instanceFrames" );

		if ( buildInputs[i].instanceCount > 0 && buildInputs[i].instances != nullptr )
		{
			const auto* srcInst = reinterpret_cast<const hiprtInstance*>( buildInputs[i].instances );
			auto&		dst		 = remappedInstances[i];
			dst.resize( buildInputs[i].instanceCount );

			for ( uint32_t j = 0; j < buildInputs[i].instanceCount; ++j )
			{
				dst[j] = srcInst[j];
				if ( srcInst[j].type == hiprtInstanceTypeGeometry )
				{
					auto gIt = m_gpuToCpuGeom.find( srcInst[j].geometry );
					if ( gIt != m_gpuToCpuGeom.end() )
						dst[j].geometry = gIt->second;
				}
				else
				{
					auto sIt = m_gpuToCpuScene.find( srcInst[j].scene );
					if ( sIt != m_gpuToCpuScene.end() )
						dst[j].scene = sIt->second;
				}
			}
			cpuIn.instances = dst.data();
		}

		cpuInputs.push_back( cpuIn );
	}

	if ( isUpdate )
		m_cpu.updateScenes( cpuInputs, {}, nullptr, nullptr, cpuBuffers );
	else
		m_cpu.buildScenes( cpuInputs, {}, nullptr, nullptr, cpuBuffers );
}

void HybridContext::buildScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	oroStream								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	m_gpu.buildScenes( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
	syncGpu( stream );
	dispatchCpuSceneBuild( buildInputs, buffers, false );
}

void HybridContext::updateScenes(
	const std::vector<hiprtSceneBuildInput>& buildInputs,
	const hiprtBuildOptions					 buildOptions,
	hiprtDevicePtr							 temporaryBuffer,
	oroStream								 stream,
	std::vector<hiprtDevicePtr>&			 buffers )
{
	m_gpu.updateScenes( buildInputs, buildOptions, temporaryBuffer, stream, buffers );
	syncGpu( stream );
	dispatchCpuSceneBuild( buildInputs, buffers, true );
}

size_t HybridContext::getScenesBuildTempBufferSize(
	const std::vector<hiprtSceneBuildInput>& buildInputs, const hiprtBuildOptions buildOptions )
{
	return m_gpu.getScenesBuildTempBufferSize( buildInputs, buildOptions );
}

std::vector<hiprtScene> HybridContext::compactScenes(
	const std::vector<hiprtScene>& /*scenes*/, oroStream /*stream*/ )
{
	throwHybridNotImplemented( "compactScenes is not supported in hybrid mode (CPU mirror rebuild required)" );
}

// GPU delegators

hiprtFuncTable HybridContext::createFuncTable( uint32_t numGeomTypes, uint32_t numRayTypes )
{
	return m_gpu.createFuncTable( numGeomTypes, numRayTypes );
}

void HybridContext::setFuncTable( hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set )
{
	m_gpu.setFuncTable( funcTable, geomType, rayType, set );
}

void HybridContext::destroyFuncTable( hiprtFuncTable funcTable ) { m_gpu.destroyFuncTable( funcTable ); }

void HybridContext::createGlobalStackBuffer( const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut )
{
	m_gpu.createGlobalStackBuffer( input, stackBufferOut );
}

void HybridContext::destroyGlobalStackBuffer( hiprtGlobalStackBuffer stackBuffer )
{
	m_gpu.destroyGlobalStackBuffer( stackBuffer );
}

void HybridContext::saveGeometry( hiprtGeometry inGeometry, const std::string& filename )
{
	m_gpu.saveGeometry( inGeometry, filename );
}

hiprtGeometry HybridContext::loadGeometry( const std::string& /*filename*/ )
{
	throwHybridNotImplemented( "loadGeometry is not supported in hybrid mode (CPU mirror rebuild required)" );
}

void HybridContext::saveScene( hiprtScene inScene, const std::string& filename )
{
	m_gpu.saveScene( inScene, filename );
}

hiprtScene HybridContext::loadScene( const std::string& /*filename*/ )
{
	throwHybridNotImplemented( "loadScene is not supported in hybrid mode (CPU mirror rebuild required)" );
}

void HybridContext::exportGeometryAabb( hiprtGeometry inGeometry, float3& outAabbMin, float3& outAabbMax )
{
	m_gpu.exportGeometryAabb( inGeometry, outAabbMin, outAabbMax );
}

void HybridContext::exportSceneAabb( hiprtScene inScene, float3& outAabbMin, float3& outAabbMax )
{
	m_gpu.exportSceneAabb( inScene, outAabbMin, outAabbMax );
}

void HybridContext::buildKernels(
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
	m_gpu.buildKernels(
		funcNames, src, moduleName, headers, includeNames, options,
		numGeomTypes, numRayTypes, funcNameSets, functions, module, cache );
}

void HybridContext::buildKernelsFromBitcode(
	const std::vector<const char*>&		 funcNames,
	const std::filesystem::path&		 moduleName,
	const std::string_view				 bitcodeBinary,
	uint32_t							 numGeomTypes,
	uint32_t							 numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	std::vector<oroFunction>&			 functions,
	bool								 cache )
{
	m_gpu.buildKernelsFromBitcode( funcNames, moduleName, bitcodeBinary, numGeomTypes, numRayTypes, funcNameSets, functions, cache );
}

void HybridContext::setCacheDir( const std::filesystem::path& path ) { m_gpu.setCacheDir( path ); }

void HybridContext::setLogLevel( hiprtLogLevel level )
{
	m_gpu.setLogLevel( level );
}

// CPU data accessors

CpuGeometryData* HybridContext::getCpuGeom( hiprtGeometry geometry )
{
	auto it = m_gpuToCpuGeom.find( geometry );
	if ( it == m_gpuToCpuGeom.end() ) return nullptr;
	return reinterpret_cast<CpuGeometryData*>( it->second );
}

CpuSceneData* HybridContext::getCpuScene( hiprtScene scene )
{
	auto it = m_gpuToCpuScene.find( scene );
	if ( it == m_gpuToCpuScene.end() ) return nullptr;
	return reinterpret_cast<CpuSceneData*>( it->second );
}

} // namespace hiprt
