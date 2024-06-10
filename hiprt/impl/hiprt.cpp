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

#include <hiprt/hiprt.h>
#include <hiprt/hiprt_libpath.h>
#include <hiprt/impl/Error.h>
#include <hiprt/impl/Context.h>
#include <hiprt/impl/Geometry.h>
#include <hiprt/impl/Utility.h>
#include <hiprt/impl/Logger.h>

using namespace hiprt;

hiprtError hiprtCreateContext( uint32_t hiprtApiVersion, const hiprtContextCreationInput& input, hiprtContext& contextOut )
{
	oroInitialize( ( input.deviceType == hiprtDeviceAMD ) ? ORO_API_HIP : ORO_API_CUDA, 0, g_hip_paths, g_hiprtc_paths );
	if ( hiprtApiVersion != HIPRT_API_VERSION ) return hiprtErrorInvalidApiVersion;
	Context* ctxt = new Context( input );
	contextOut	  = reinterpret_cast<hiprtContext>( ctxt );
	return hiprtSuccess;
}

hiprtError hiprtDestroyContext( hiprtContext context )
{
	if ( !context ) return hiprtErrorInvalidParameter;
	delete reinterpret_cast<Context*>( context );
	return hiprtSuccess;
}

hiprtError hiprtCreateGeometry(
	hiprtContext				   context,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtGeometry&				   geometryOut )
{
	hiprtGeometry* geometryAddr = &geometryOut;
	return hiprtCreateGeometries( context, 1, &buildInput, buildOptions, &geometryAddr );
}

hiprtError hiprtCreateGeometries(
	hiprtContext				   context,
	uint32_t					   numGeometries,
	const hiprtGeometryBuildInput* buildInputsIn,
	const hiprtBuildOptions		   buildOptions,
	hiprtGeometry**				   geometriesOut )
{
	if ( !context || numGeometries == 0 || buildInputsIn == nullptr || geometriesOut == nullptr )
		return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtGeometryBuildInput> buildInputs;
	for ( uint32_t i = 0; i < numGeometries; ++i )
	{
		if ( geometriesOut[i] == nullptr ) return hiprtErrorInvalidParameter;
		buildInputs.push_back( buildInputsIn[i] );
	}

	try
	{
		std::vector<hiprtGeometry> geometries =
			reinterpret_cast<Context*>( context )->createGeometries( buildInputs, buildOptions );
		for ( uint32_t i = 0; i < numGeometries; ++i )
			*geometriesOut[i] = geometries[i];
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtDestroyGeometry( hiprtContext context, hiprtGeometry geometry )
{
	return hiprtDestroyGeometries( context, 1, &geometry );
}

hiprtError hiprtDestroyGeometries( hiprtContext context, uint32_t numGeometries, hiprtGeometry* geometriesIn )
{
	if ( !context || numGeometries == 0 || geometriesIn == nullptr ) return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtGeometry> geometries;
	for ( uint32_t i = 0; i < numGeometries; ++i )
	{
		if ( !geometriesIn[i] ) return hiprtErrorInvalidParameter;
		geometries.push_back( geometriesIn[i] );
	}

	try
	{
		reinterpret_cast<Context*>( context )->destroyGeometries( geometries );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtBuildGeometry(
	hiprtContext				   context,
	hiprtBuildOperation			   buildOperation,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	hiprtApiStream				   stream,
	hiprtGeometry				   geometryOut )
{
	return hiprtBuildGeometries( context, buildOperation, 1, &buildInput, buildOptions, temporaryBuffer, stream, &geometryOut );
}

hiprtError hiprtBuildGeometries(
	hiprtContext				   context,
	hiprtBuildOperation			   buildOperation,
	uint32_t					   numGeometries,
	const hiprtGeometryBuildInput* buildInputsIn,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	hiprtApiStream				   stream,
	hiprtGeometry*				   geometriesOut )
{
	if ( !context || numGeometries == 0 || buildInputsIn == nullptr || geometriesOut == nullptr )
		return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtDevicePtr>			 buffers;
	std::vector<hiprtGeometryBuildInput> buildInputs;
	for ( uint32_t i = 0; i < numGeometries; ++i )
	{
		if ( !geometriesOut[i] ) return hiprtErrorInvalidParameter;
		buffers.push_back( geometriesOut[i] );
		buildInputs.push_back( buildInputsIn[i] );
	}

	try
	{
		switch ( buildOperation )
		{
		case hiprtBuildOperationBuild: {
			reinterpret_cast<Context*>( context )->buildGeometries(
				buildInputs, buildOptions, temporaryBuffer, reinterpret_cast<oroStream>( stream ), buffers );
			break;
		}
		case hiprtBuildOperationUpdate: {
			reinterpret_cast<Context*>( context )->updateGeometries(
				buildInputs, buildOptions, temporaryBuffer, reinterpret_cast<oroStream>( stream ), buffers );
			break;
		}
		}
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}

	return hiprtSuccess;
}

hiprtError hiprtGetGeometryBuildTemporaryBufferSize(
	hiprtContext context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions, size_t& sizeOut )
{
	return hiprtGetGeometriesBuildTemporaryBufferSize( context, 1, &buildInput, buildOptions, sizeOut );
}

hiprtError hiprtGetGeometriesBuildTemporaryBufferSize(
	hiprtContext				   context,
	uint32_t					   numGeometries,
	const hiprtGeometryBuildInput* buildInputsIn,
	const hiprtBuildOptions		   buildOptions,
	size_t&						   sizeOut )
{
	if ( !context || numGeometries == 0 || buildInputsIn == nullptr ) return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtGeometryBuildInput> buildInputs;
	for ( uint32_t i = 0; i < numGeometries; ++i )
		buildInputs.push_back( buildInputsIn[i] );

	try
	{
		sizeOut = reinterpret_cast<Context*>( context )->getGeometriesBuildTempBufferSize( buildInputs, buildOptions );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}

	return hiprtSuccess;
}

hiprtError
hiprtCompactGeometry( hiprtContext context, hiprtApiStream stream, hiprtGeometry geometryIn, hiprtGeometry& geometryOut )
{
	hiprtGeometry* geometryAddr = &geometryOut;
	return hiprtCompactGeometries( context, 1, stream, &geometryIn, &geometryAddr );
}

hiprtError hiprtCompactGeometries(
	hiprtContext	context,
	uint32_t		numGeometries,
	hiprtApiStream	stream,
	hiprtGeometry*	geometriesIn,
	hiprtGeometry** geometriesOut )
{
	if ( !context || numGeometries == 0 || geometriesIn == nullptr || geometriesOut == nullptr )
		return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtGeometry> geometries;
	for ( uint32_t i = 0; i < numGeometries; ++i )
	{
		if ( geometriesIn[i] == nullptr ) return hiprtErrorInvalidParameter;
		geometries.push_back( geometriesIn[i] );
	}

	try
	{
		std::vector<hiprtGeometry> compactedGeometries =
			reinterpret_cast<Context*>( context )->compactGeometries( geometries, reinterpret_cast<oroStream>( stream ) );
		for ( uint32_t i = 0; i < numGeometries; ++i )
			*geometriesOut[i] = compactedGeometries[i];
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtCreateScene(
	hiprtContext context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions, hiprtScene& sceneOut )
{
	hiprtScene* sceneAddr = &sceneOut;
	return hiprtCreateScenes( context, 1, &buildInput, buildOptions, &sceneAddr );
}

hiprtError hiprtCreateScenes(
	hiprtContext				context,
	uint32_t					numScenes,
	const hiprtSceneBuildInput* buildInputsIn,
	const hiprtBuildOptions		buildOptions,
	hiprtScene**				scenesOut )
{
	if ( !context || numScenes == 0 || buildInputsIn == nullptr || scenesOut == nullptr ) return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtSceneBuildInput> buildInputs;
	for ( uint32_t i = 0; i < numScenes; ++i )
	{
		if ( scenesOut[i] == nullptr ) return hiprtErrorInvalidParameter;
		buildInputs.push_back( buildInputsIn[i] );
	}

	try
	{
		std::vector<hiprtScene> scenes = reinterpret_cast<Context*>( context )->createScenes( buildInputs, buildOptions );
		for ( uint32_t i = 0; i < numScenes; ++i )
			*scenesOut[i] = scenes[i];
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtDestroyScene( hiprtContext context, hiprtScene scene ) { return hiprtDestroyScenes( context, 1, &scene ); }

hiprtError hiprtDestroyScenes( hiprtContext context, uint32_t numScenes, hiprtScene* scenesIn )
{
	if ( !context || numScenes == 0 || scenesIn == nullptr ) return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtScene> scenes;
	for ( uint32_t i = 0; i < numScenes; ++i )
	{
		if ( !scenesIn[i] ) return hiprtErrorInvalidParameter;
		scenes.push_back( scenesIn[i] );
	}

	try
	{
		reinterpret_cast<Context*>( context )->destroyScenes( scenes );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtBuildScene(
	hiprtContext				context,
	hiprtBuildOperation			buildOperation,
	const hiprtSceneBuildInput& buildInput,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	hiprtApiStream				stream,
	hiprtScene					sceneOut )
{
	return hiprtBuildScenes( context, buildOperation, 1, &buildInput, buildOptions, temporaryBuffer, stream, &sceneOut );
}

hiprtError hiprtBuildScenes(
	hiprtContext				context,
	hiprtBuildOperation			buildOperation,
	uint32_t					numScenes,
	const hiprtSceneBuildInput* buildInputsIn,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	hiprtApiStream				stream,
	hiprtScene*					scenesOut )
{
	if ( !context || numScenes == 0 || buildInputsIn == nullptr || scenesOut == nullptr ) return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtDevicePtr>		  buffers;
	std::vector<hiprtSceneBuildInput> buildInputs;
	for ( uint32_t i = 0; i < numScenes; ++i )
	{
		if ( !scenesOut[i] ) return hiprtErrorInvalidParameter;
		buffers.push_back( scenesOut[i] );
		buildInputs.push_back( buildInputsIn[i] );
	}

	try
	{
		switch ( buildOperation )
		{
		case hiprtBuildOperationBuild: {
			reinterpret_cast<Context*>( context )->buildScenes(
				buildInputs, buildOptions, temporaryBuffer, reinterpret_cast<oroStream>( stream ), buffers );
			break;
		}
		case hiprtBuildOperationUpdate: {
			reinterpret_cast<Context*>( context )->updateScenes(
				buildInputs, buildOptions, temporaryBuffer, reinterpret_cast<oroStream>( stream ), buffers );
			break;
		}
		}
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}

	return hiprtSuccess;
}

hiprtError hiprtGetSceneBuildTemporaryBufferSize(
	hiprtContext context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions, size_t& sizeOut )
{
	return hiprtGetScenesBuildTemporaryBufferSize( context, 1, &buildInput, buildOptions, sizeOut );
}

hiprtError hiprtGetScenesBuildTemporaryBufferSize(
	hiprtContext				context,
	uint32_t					numScenes,
	const hiprtSceneBuildInput* buildInputsIn,
	const hiprtBuildOptions		buildOptions,
	size_t&						sizeOut )
{
	if ( !context || numScenes == 0 || buildInputsIn == nullptr ) return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtSceneBuildInput> buildInputs;
	for ( uint32_t i = 0; i < numScenes; ++i )
		buildInputs.push_back( buildInputsIn[i] );

	try
	{
		sizeOut = reinterpret_cast<Context*>( context )->getScenesBuildTempBufferSize( buildInputs, buildOptions );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}

	return hiprtSuccess;
}

hiprtError hiprtCompactScene( hiprtContext context, hiprtApiStream stream, hiprtScene sceneIn, hiprtScene& sceneOut )
{
	hiprtScene* sceneAddr = &sceneOut;
	return hiprtCompactScenes( context, 1, stream, &sceneIn, &sceneAddr );
}

hiprtError hiprtCompactScenes(
	hiprtContext context, uint32_t numScenes, hiprtApiStream stream, hiprtScene* scenesIn, hiprtScene** scenesOut )
{
	if ( !context || numScenes == 0 || scenesIn == nullptr || scenesOut == nullptr ) return hiprtErrorInvalidParameter;

	// TODO: use std::span after we switch to c++20
	std::vector<hiprtScene> scenes;
	for ( uint32_t i = 0; i < numScenes; ++i )
	{
		if ( scenesIn[i] == nullptr ) return hiprtErrorInvalidParameter;
		scenes.push_back( scenesIn[i] );
	}

	try
	{
		std::vector<hiprtScene> compactedScenes =
			reinterpret_cast<Context*>( context )->compactScenes( scenes, reinterpret_cast<oroStream>( stream ) );
		for ( uint32_t i = 0; i < numScenes; ++i )
			*scenesOut[i] = compactedScenes[i];
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError
hiprtCreateFuncTable( hiprtContext context, uint32_t numGeomTypes, uint32_t numRayTypes, hiprtFuncTable& funcTableOut )
{
	if ( !context ) return hiprtErrorInvalidParameter;
	try
	{
		funcTableOut = reinterpret_cast<Context*>( context )->createFuncTable( numGeomTypes, numRayTypes );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError
hiprtSetFuncTable( hiprtContext context, hiprtFuncTable funcTable, uint32_t geomType, uint32_t rayType, hiprtFuncDataSet set )
{
	if ( !context || !funcTable ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->setFuncTable( funcTable, geomType, rayType, set );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtDestroyFuncTable( hiprtContext context, hiprtFuncTable funcTable )
{
	if ( !context || !funcTable ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->destroyFuncTable( funcTable );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtCreateGlobalStackBuffer(
	hiprtContext context, const hiprtGlobalStackBufferInput& input, hiprtGlobalStackBuffer& stackBufferOut )
{
	if ( !context ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->createGlobalStackBuffer( input, stackBufferOut );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtDestroyGlobalStackBuffer( hiprtContext context, hiprtGlobalStackBuffer stackBuffer )
{
	if ( !context ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->destroyGlobalStackBuffer( stackBuffer );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtSaveGeometry( hiprtContext context, hiprtGeometry geometry, const char* filename )
{
	if ( !context || !geometry || !filename ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->saveGeometry( geometry, filename );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtLoadGeometry( hiprtContext context, hiprtGeometry& geometryOut, const char* filename )
{
	if ( !context || !filename ) return hiprtErrorInvalidParameter;
	try
	{
		geometryOut = reinterpret_cast<Context*>( context )->loadGeometry( filename );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtSaveScene( hiprtContext context, hiprtScene scene, const char* filename )
{
	if ( !context || !scene || !filename ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->saveScene( scene, filename );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtLoadScene( hiprtContext context, hiprtScene& sceneOut, const char* filename )
{
	if ( !context || !sceneOut || !filename ) return hiprtErrorInvalidParameter;
	try
	{
		sceneOut = reinterpret_cast<Context*>( context )->loadScene( filename );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError
hiprtExportGeometryAabb( hiprtContext context, hiprtGeometry geometry, hiprtFloat3& aabbMinOut, hiprtFloat3& aabbMaxOut )
{
	if ( !context || !geometry ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->exportGeometryAabb( geometry, aabbMinOut, aabbMaxOut );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtExportSceneAabb( hiprtContext context, hiprtScene scene, hiprtFloat3& aabbMinOut, hiprtFloat3& aabbMaxOut )
{
	if ( !context || !scene ) return hiprtErrorInvalidParameter;
	try
	{
		reinterpret_cast<Context*>( context )->exportSceneAabb( scene, aabbMinOut, aabbMaxOut );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}
	return hiprtSuccess;
}

hiprtError hiprtBuildTraceKernels(
	hiprtContext	  context,
	uint32_t		  numFunctions,
	const char**	  funcNamesIn,
	const char*		  src,
	const char*		  moduleName,
	uint32_t		  numHeaders,
	const char**	  headersIn,
	const char**	  includeNamesIn,
	uint32_t		  numOptions,
	const char**	  optionsIn,
	uint32_t		  numGeomTypes,
	uint32_t		  numRayTypes,
	hiprtFuncNameSet* funcNameSetsIn,
	hiprtApiFunction* functionsOut,
	hiprtApiModule*	  moduleOut,
	bool			  cache )
{
	if ( !context || numFunctions == 0 || funcNamesIn == nullptr || functionsOut == nullptr || moduleName == nullptr ||
		 src == nullptr )
		return hiprtErrorInvalidParameter;

	try
	{
		// TODO: use std::span after we switch to c++20
		std::vector<const char*> options;
		for ( uint32_t i = 0; i < numOptions; ++i )
			options.push_back( optionsIn[i] );

		std::vector<const char*> funcNames;
		for ( uint32_t i = 0; i < numFunctions; ++i )
			funcNames.push_back( funcNamesIn[i] );

		std::vector<const char*> headers;
		std::vector<const char*> includeNames;
		for ( uint32_t i = 0; i < numHeaders; ++i )
		{
			headers.push_back( headersIn[i] );
			includeNames.push_back( includeNamesIn[i] );
		}

		std::vector<hiprtFuncNameSet> funcNameSets;
		if ( funcNameSetsIn != nullptr )
		{
			for ( uint32_t i = 0; i < numGeomTypes * numRayTypes; ++i )
				funcNameSets.push_back( funcNameSetsIn[i] );
		}

		std::vector<oroFunction> functions;
		oroModule				 module = nullptr;
		reinterpret_cast<Context*>( context )->buildKernels(
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
			cache );

		for ( uint32_t i = 0; i < numFunctions; ++i )
			functionsOut[i] = reinterpret_cast<hiprtApiFunction>( functions[i] );

		if ( moduleOut != nullptr ) *moduleOut = reinterpret_cast<hiprtApiModule>( module );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}

	return hiprtSuccess;
}

hiprtError hiprtBuildTraceKernelsFromBitcode(
	hiprtContext	  context,
	uint32_t		  numFunctions,
	const char**	  functionNames,
	const char*		  moduleName,
	const char*		  bitcodeBinary,
	size_t			  bitcodeBinarySize,
	uint32_t		  numGeomTypes,
	uint32_t		  numRayTypes,
	hiprtFuncNameSet* functionNameSets,
	hiprtApiFunction* functionsOut,
	bool			  cache )
{
	if ( !context || numFunctions == 0 || functionNames == nullptr || functionsOut == nullptr || moduleName == nullptr ||
		 bitcodeBinary == nullptr || bitcodeBinarySize == 0 )
		return hiprtErrorInvalidParameter;

	try
	{
		// TODO: use std::span after we switch to c++20
		std::vector<const char*> funcNames;
		for ( uint32_t i = 0; i < numFunctions; ++i )
			funcNames.push_back( functionNames[i] );

		std::vector<hiprtFuncNameSet> funcNameSets;
		if ( functionNameSets != nullptr )
		{
			for ( uint32_t i = 0; i < numGeomTypes * numRayTypes; ++i )
				funcNameSets.push_back( functionNameSets[i] );
		}
		std::string_view		 binary( bitcodeBinary, bitcodeBinarySize );
		std::vector<oroFunction> functions;
		reinterpret_cast<Context*>( context )->buildKernelsFromBitcode(
			funcNames, moduleName, binary, numGeomTypes, numRayTypes, funcNameSets, functions, cache );

		for ( uint32_t i = 0; i < numFunctions; ++i )
			functionsOut[i] = reinterpret_cast<hiprtApiFunction>( functions[i] );
	}
	catch ( std::exception& e )
	{
		logError( e.what() );
		return hiprtErrorInternal;
	}

	return hiprtSuccess;
}

void hiprtSetCacheDirPath( hiprtContext context, const char* path )
{
	reinterpret_cast<Context*>( context )->setCacheDir( path );
}

void hiprtSetLogLevel( hiprtLogLevel level ) { Logger::getInstance().setLevel( level ); }
