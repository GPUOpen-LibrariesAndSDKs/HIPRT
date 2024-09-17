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
#include <hiprt/impl/Compiler.h>
#include <hiprt/impl/Error.h>
#include <hiprt/impl/Utility.h>
#include <hiprt/impl/Context.h>
#if defined( HIPRT_ENCRYPT )
#include <contrib/easy-encryption/encrypt.h>
#endif
#include <regex>
#if defined( HIPRT_BAKE_KERNEL_GENERATED )
#include <hiprt/cache/Kernels.h>
#include <hiprt/cache/KernelArgs.h>
#endif

namespace
{
#if defined( HIPRT_BITCODE_LINKING )
constexpr auto UseBitcode = true;
#else
constexpr auto UseBitcode			= false;
#endif

#if defined( HIPRT_LOAD_FROM_STRING )
constexpr auto UseBakedCode = true;
#else
constexpr auto UseBakedCode			= false;
#endif

#if defined( HIPRT_BAKE_KERNEL_GENERATED )
constexpr auto BakedCodeIsGenerated = true;
#else
constexpr auto BakedCodeIsGenerated = false;
#endif
HIPRT_STATIC_ASSERT( !UseBakedCode || BakedCodeIsGenerated );
} // namespace

namespace hiprt
{
Compiler::~Compiler()
{
	for ( auto& module : m_moduleCache )
		checkOro( oroModuleUnload( module.second ) );
}

Kernel Compiler::getKernel(
	Context&					 context,
	const std::filesystem::path& moduleName,
	const std::string&			 funcName,
	std::vector<const char*>&	 options,
	uint32_t					 numHeaders,
	const char**				 headersIn,
	const char**				 includeNamesIn )
{
	std::lock_guard<std::mutex> lock( m_kernelMutex );

	std::string cacheName  = moduleName.string() + funcName;
	auto		cacheEntry = m_kernelCache.find( cacheName );
	if ( cacheEntry != m_kernelCache.end() ) return cacheEntry->second;

	oroFunction function;
	if constexpr ( UseBitcode )
	{
		function = getFunctionFromPrecompiledBinary( funcName );
	}
	else
	{
		std::vector<const char*>	  funcNames = { funcName.c_str() };
		std::vector<const char*>	  headers;
		std::vector<const char*>	  includeNames;
		std::vector<hiprtFuncNameSet> funcNameSets;
		std::vector<oroFunction>	  functions;
		oroModule					  module;

		if ( numHeaders == 0 )
		{
			std::string src = readSourceCode( moduleName );
			buildKernels(
				context,
				funcNames,
				src,
				moduleName,
				headers,
				includeNames,
				options,
				0,
				0,
				funcNameSets,
				functions,
				module,
				false,
				true );
		}
		else
		{
			std::vector<std::string> headerData( numHeaders - 1 );
			for ( uint32_t i = 0; i < numHeaders - 1; ++i )
			{
				includeNames.push_back( includeNamesIn[i] );
				headerData[i] = decryptSourceCode( headersIn[i] );
				headers.push_back( headerData[i].c_str() );
			}

			std::string src = decryptSourceCode( headersIn[numHeaders - 1] );
			buildKernels(
				context,
				funcNames,
				src,
				moduleName,
				headers,
				includeNames,
				options,
				0,
				0,
				funcNameSets,
				functions,
				module,
				false,
				true );
		}
		function = functions.back();
	}

	Kernel kernel( function );
	m_kernelCache[cacheName] = kernel;
	return kernel;
}

void Compiler::buildProgram(
	Context&							 context,
	const std::vector<const char*>&		 funcNames,
	const std::string&					 src,
	const std::filesystem::path&		 moduleName,
	std::vector<const char*>&			 headers,
	std::vector<const char*>&			 includeNames,
	std::vector<const char*>&			 options,
	uint32_t							 numGeomTypes,
	uint32_t							 numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	orortcProgram&						 progOut )
{
	checkOrortc( orortcCreateProgram(
		&progOut,
		src.c_str(),
		moduleName.string().c_str(),
		static_cast<int>( headers.size() ),
		headers.data(),
		includeNames.data() ) );

	for ( size_t i = 0; i < funcNames.size(); ++i )
		checkOrortc( orortcAddNameExpression( progOut, funcNames[i] ) );

	orortcResult e = orortcCompileProgram( progOut, static_cast<int>( options.size() ), options.data() );
	if ( e != ORORTC_SUCCESS )
	{
		size_t logSize;
		checkOrortc( orortcGetProgramLogSize( progOut, &logSize ) );

		if ( logSize )
		{
			std::string log( logSize, '\0' );
			checkOrortc( orortcGetProgramLog( progOut, &log[0] ) );
			std::cout << log << '\n';
			throw std::runtime_error( "Runtime compilation failed" );
		}
	}
}

void Compiler::buildKernels(
	Context&							 context,
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
	bool								 extended,
	bool								 cache )
{
	if ( !std::filesystem::exists( m_cacheDirectory ) && !std::filesystem::create_directory( m_cacheDirectory ) )
		throw std::runtime_error( "Cannot create cache directory" );

	std::lock_guard<std::mutex> lock( m_moduleMutex );
	auto						cacheEntry = m_moduleCache.find( moduleName.string() );
	if ( cacheEntry != m_moduleCache.end() )
	{
		module = cacheEntry->second;
	}
	else
	{
		std::string cacheName = getCacheFilename( context, src, moduleName, options, funcNameSets, numGeomTypes, numRayTypes );
		bool		upToDate  = isCachedFileUpToDate( m_cacheDirectory / cacheName, moduleName );

		orortcProgram prog;
		std::string	  binary;
		if ( upToDate && cache )
		{
			binary = loadCacheFileToBinary( cacheName, context.getDeviceName() );
		}
		else
		{
			std::vector<std::string> headerData;
			std::string				 extSrc = src;
			if ( extended )
			{
				if constexpr ( UseBitcode && !BakedCodeIsGenerated )
				{
					throw std::runtime_error( "Not compiled with the baked kernel code support" );
				}
				if constexpr ( BakedCodeIsGenerated )
				{
					extSrc = "#include <hiprt_device_impl.h>\n";
					addCustomFuncsSwitchCase( extSrc, funcNameSets, numGeomTypes, numRayTypes );
					extSrc += "\n" + src;

					const uint32_t numHeaders = sizeof( GET_ARGS( hiprt_device_impl ) ) / sizeof( void* );
					headerData.resize( numHeaders );
					for ( uint32_t i = 0; i < numHeaders - 1; i++ )
					{
						auto includeName =
							std::find_if( includeNames.begin(), includeNames.end(), [&]( const std::string& rhs ) {
								return GET_INC( hiprt_device_impl )[i] == rhs;
							} );
						if ( includeName != includeNames.end() ) continue;
						includeNames.push_back( GET_INC( hiprt_device_impl )[i] );
						headerData[i] = decryptSourceCode( GET_ARGS( hiprt_device_impl )[i] );
						headers.push_back( headerData[i].c_str() );
					}
					includeNames.push_back( "hiprt_device_impl.h" );
					headerData[numHeaders - 1] = decryptSourceCode( GET_ARGS( hiprt_device_impl )[numHeaders - 1] );
					headers.push_back( headerData[numHeaders - 1].c_str() );
				}
				else
				{
					extSrc = "#include <hiprt/impl/hiprt_device_impl.h>\n";
					addCustomFuncsSwitchCase( extSrc, funcNameSets, numGeomTypes, numRayTypes );
					extSrc += "\n" + src;
				}
			}

			std::vector<const char*> opts		 = options;
			std::string				 includePath = ( "-I" + Utility::getEnvVariable( "HIPRT_PATH" ) + "/" );
			opts.push_back( includePath.c_str() );
			addCommonOpts( context, opts );

			buildProgram(
				context,
				funcNames,
				extSrc,
				moduleName,
				headers,
				includeNames,
				opts,
				numGeomTypes,
				numRayTypes,
				funcNameSets,
				prog );

			size_t binarySize = 0;
			checkOrortc( orortcGetCodeSize( prog, &binarySize ) );
			binary.resize( binarySize );
			checkOrortc( orortcGetCode( prog, binary.data() ) );

			if ( cache ) cacheBinaryToFile( binary, cacheName, context.getDeviceName() );
			checkOrortc( orortcDestroyProgram( &prog ) );
		}

		checkOro( oroModuleLoadData( &module, binary.data() ) );
		m_moduleCache[moduleName.string()] = module;
	}

	for ( size_t i = 0; i < funcNames.size(); ++i )
	{
		oroFunction func;
		checkOro( oroModuleGetFunction( &func, module, funcNames[i] ) );
		functions.push_back( func );
	}
}

void Compiler::buildKernelsFromBitcode(
	Context&							 context,
	const std::vector<const char*>&		 funcNames,
	const std::filesystem::path&		 moduleName,
	const std::string_view				 bitcodeBinary,
	uint32_t							 numGeomTypes,
	uint32_t							 numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	std::vector<oroFunction>&			 functions,
	bool								 cache )
{
	if constexpr ( UseBitcode )
	{
		if ( !std::filesystem::exists( m_cacheDirectory ) && !std::filesystem::create_directory( m_cacheDirectory ) )
			throw std::runtime_error( "Cannot create cache directory" );

		std::lock_guard<std::mutex> lock( m_moduleMutex );
		auto						cacheEntry = m_moduleCache.find( moduleName.string() );
		oroModule					module;
		if ( cacheEntry != m_moduleCache.end() )
		{
			module = cacheEntry->second;
		}
		else
		{
			std::string cacheName = getCacheFilename(
				context,
				std::to_string( bitcodeBinary.size() ),
				moduleName,
				std::nullopt,
				funcNameSets,
				numGeomTypes,
				numRayTypes );
			bool upToDate = isCachedFileUpToDate( m_cacheDirectory / cacheName, moduleName );

			std::string binary;
			if ( upToDate && cache )
			{
				binary = loadCacheFileToBinary( cacheName, context.getDeviceName() );
			}
			else
			{
				std::string customFuncBitcodeBinary =
					buildFunctionTableBitcode( context, numGeomTypes, numRayTypes, funcNameSets );

				const uint32_t	 JITOptCount = 6u;
				orortcLinkState	 rtcLinkState;
				orortcJIT_option options[JITOptCount];
				void*			 optionVals[JITOptCount];
				float			 wallTime;

				constexpr uint32_t LogSize = 8192u;
				char			   errorLog[LogSize];
				char			   infoLog[LogSize];

				options[0]	  = ORORTC_JIT_WALL_TIME;
				optionVals[0] = reinterpret_cast<void*>( &wallTime );

				options[1]	  = ORORTC_JIT_INFO_LOG_BUFFER;
				optionVals[1] = infoLog;

				options[2]	  = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
				optionVals[2] = reinterpret_cast<void*>( static_cast<uintptr_t>( LogSize ) );

				options[3]	  = ORORTC_JIT_ERROR_LOG_BUFFER;
				optionVals[3] = errorLog;

				options[4]	  = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
				optionVals[4] = reinterpret_cast<void*>( static_cast<uintptr_t>( LogSize ) );

				options[5]	  = ORORTC_JIT_LOG_VERBOSE;
				optionVals[5] = reinterpret_cast<void*>( static_cast<uintptr_t>( 1 ) );

				bool					 amd		= oroGetCurAPI( 0 ) == ORO_API_HIP;
				std::filesystem::path	 bcPath		= getBitcodePath( amd );
				const orortcJITInputType typeBc		= amd ? ORORTC_JIT_INPUT_LLVM_BUNDLED_BITCODE : ORORTC_JIT_INPUT_FATBINARY;
				const orortcJITInputType typeUserBc = amd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_PTX;

				void* binaryPtr;
				checkOrortc( orortcLinkCreate( JITOptCount, options, optionVals, &rtcLinkState ) );

				orortcResult res = orortcLinkAddFile( rtcLinkState, typeBc, bcPath.string().c_str(), 0, 0, 0 );
				if ( res != ORORTC_SUCCESS )
				{
					// add some verbose to help debugging missing file.
					std::cout << "orortcLinkAddFile FAILED (error=" << res << ") loading file: " << bcPath.string().c_str()
							  << std::endl;
				}
				checkOrortc( res );

				checkOrortc( orortcLinkAddData(
					rtcLinkState, typeUserBc, const_cast<char*>( bitcodeBinary.data() ), bitcodeBinary.size(), 0, 0, 0, 0 ) );
				checkOrortc( orortcLinkAddData(
					rtcLinkState, typeUserBc, customFuncBitcodeBinary.data(), customFuncBitcodeBinary.size(), 0, 0, 0, 0 ) );

				size_t binarySize = 0;
				checkOrortc( orortcLinkComplete( rtcLinkState, &binaryPtr, &binarySize ) );
				binary = std::string( reinterpret_cast<char*>( binaryPtr ), binarySize );

				if ( cache ) cacheBinaryToFile( binary, cacheName, context.getDeviceName() );

				checkOrortc( orortcLinkDestroy( rtcLinkState ) );
			}

			checkOro( oroModuleLoadData( &module, binary.data() ) );
			m_moduleCache[moduleName.string()] = module;
		}

		for ( size_t i = 0; i < funcNames.size(); ++i )
		{
			oroFunction func;
			checkOro( oroModuleGetFunction( &func, module, funcNames[i] ) );
			functions.push_back( func );
		}
	}
	else
	{
		throw std::runtime_error( "Not compiled with the bitcode linking support" );
	}
}

void Compiler::setCacheDir( const std::filesystem::path& cacheDirectory )
{
	if ( !cacheDirectory.empty() ) m_cacheDirectory = cacheDirectory;
}

std::string Compiler::kernelNameSufix( const std::string& traits )
{
	const std::string delimiter = "::";
	std::string		  result	= traits.substr( traits.find_last_of( delimiter ) + 1 );
	result						= std::regex_replace( result, std::regex( ">| " ), "" );
	result						= std::regex_replace( result, std::regex( "<|," ), "_" );
	return result;
}

std::string
Compiler::readSourceCode( const std::filesystem::path& path, std::optional<std::vector<std::filesystem::path>> includes )
{
	std::string	  src;
	std::ifstream file( path );
	if ( !file.is_open() )
	{
		std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
		throw std::runtime_error( msg );
	}
	size_t sizeFile;
	file.seekg( 0, std::ifstream::end );
	size_t size = sizeFile = static_cast<size_t>( file.tellg() );
	file.seekg( 0, std::ifstream::beg );
	if ( includes )
	{
		std::string line;
		while ( std::getline( file, line ) )
		{
			if ( line.find( "#include" ) != std::string::npos )
			{
				size_t		pa	= line.find( "<" );
				size_t		pb	= line.find( ">" );
				std::string buf = line.substr( pa + 1, pb - pa - 1 );
				includes.value().push_back( buf );
				src += line + '\n';
			}
			src += line + '\n';
		}
	}
	else
	{
		src.resize( size, ' ' );
		file.read( &src[0], size );
	}
	return src;
}

void Compiler::addCommonOpts( Context& context, std::vector<const char*>& opts )
{
	if ( context.getDeviceName().find( "NVIDIA" ) != std::string::npos )
		opts.push_back( "--use_fast_math" );
	else
		opts.push_back( "-ffast-math" );

	if ( context.enableHwi() ) opts.push_back( "-D__USE_HWI__" );

	opts.push_back( "-D__USE_HIP__" );
	opts.push_back( "-std=c++17" );
}

void Compiler::addCustomFuncsSwitchCase(
	std::string&								 extSrc,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::string intersectFuncDef =
		"HIPRT_DEVICE bool intersectFunc( uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, "
		"const hiprtRay& ray, void* payload, hiprtHit& hit )\n{\n\tconst uint32_t index = tableHeader.numGeomTypes * rayType + "
		"geomType;\n\t[[maybe_unused]] const void* data = tableHeader.funcDataSets[index].intersectFuncData;\n\tswitch ( index "
		") \n\t{\n";
	std::string filterFuncDef =
		"HIPRT_DEVICE bool filterFunc( uint32_t geomType, uint32_t rayType, const hiprtFuncTableHeader& tableHeader, const "
		"hiprtRay& ray, void* payload, const hiprtHit& hit )\n{\n\tconst uint32_t index = tableHeader.numGeomTypes * rayType + "
		"geomType;\n\t[[maybe_unused]] const void* data = tableHeader.funcDataSets[index].filterFuncData;\n\tswitch ( index ) "
		"\n\t{\n";
	std::string funcDecls;
	if ( funcNameSets )
	{
		for ( uint32_t i = 0; i < numRayTypes; ++i )
		{
			for ( uint32_t j = 0; j < numGeomTypes; ++j )
			{
				uint32_t k = numGeomTypes * i + j;
				if ( funcNameSets.value()[k].intersectFuncName != nullptr )
				{
					const std::string intersectFuncName = funcNameSets.value()[k].intersectFuncName;
					if ( !intersectFuncName.empty() )
					{
						funcDecls += "__device__ bool " + intersectFuncName +
									 "( const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit );\n";
						intersectFuncDef += "\t\tcase " + std::to_string( k ) + ": { return " + intersectFuncName +
											"( ray, data, payload, hit ); }\n";
					}
				}
				if ( funcNameSets.value()[k].filterFuncName != nullptr )
				{
					const std::string filterFuncName = funcNameSets.value()[k].filterFuncName;
					if ( !filterFuncName.empty() )
					{
						funcDecls += "__device__ bool " + filterFuncName +
									 "( const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit );\n";
						filterFuncDef += "\t\tcase " + std::to_string( k ) + ": { return " + filterFuncName +
										 "( ray, data, payload, hit ); }\n";
					}
				}
			}
		}
	}
	intersectFuncDef += "\t\t default: { return false; }\n\t}\n}\n";
	filterFuncDef += "\t\t default: { return false; }\n\t}\n}\n";
	extSrc += "\n" + funcDecls + "\n" + intersectFuncDef + "\n" + filterFuncDef;
}

std::filesystem::path Compiler::getBitcodePath( bool amd )
{
	std::string hipSdkVersion = "_" + std::string( HIP_VERSION_STR );
	std::string filename	  = "hiprt" + std::string( HIPRT_VERSION_STR );

	if ( amd ) filename += hipSdkVersion;

	if ( amd )
#if !defined( __GNUC__ )
		filename += "_amd_lib_win.bc";
#else
		filename += "_amd_lib_linux.bc";
#endif
	else
		filename += "_nv_lib.fatbin";
	return Utility::getCurrentDir() / std::filesystem::path( filename );
}

std::filesystem::path Compiler::getFatbinPath( bool amd )
{
	std::string hipSdkVersion = "_" + std::string( HIP_VERSION_STR );
	std::string filename	  = "hiprt" + std::string( HIPRT_VERSION_STR );
	if ( amd ) filename += hipSdkVersion;

	if ( amd )
		filename += "_amd.hipfb";
	else
		filename += "_nv.fatbin";
	return Utility::getCurrentDir() / std::filesystem::path( filename );
}

bool Compiler::isCachedFileUpToDate( const std::filesystem::path& cachedFile, const std::filesystem::path& moduleName )
{
	if ( !std::filesystem::exists( cachedFile ) ) return false;
	if ( !std::filesystem::exists( moduleName ) ) return true;
	return std::filesystem::last_write_time( moduleName ) < std::filesystem::last_write_time( cachedFile );
}

std::string Compiler::decryptSourceCode( const std::string& srcIn )
{
#if defined( HIPRT_ENCRYPT )
	std::lock_guard<std::mutex> lock( m_decryptMutex );
	std::string					src;
	if ( m_decryptCache.find( srcIn ) != m_decryptCache.end() )
	{
		src = m_decryptCache[srcIn];
	}
	else
	{
		std::string deryptKeyStr( DecryptKey );
		src					  = srcIn;
		src					  = decrypt( src, deryptKeyStr );
		m_decryptCache[srcIn] = src;
	}
	return src;
#else
	return srcIn;
#endif
}

std::string Compiler::getCacheFilename(
	Context&									 context,
	const std::string&							 src,
	const std::filesystem::path&				 moduleName,
	std::optional<std::vector<const char*>>		 options,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::string driverVersion = context.getDriverVersion();
	std::string deviceName	  = context.getDeviceName();
	deviceName				  = deviceName.substr( 0, deviceName.find( ":" ) );

	std::string moduleHash = moduleName.string() + src;
	moduleHash			   = Utility::format( "%08x", Utility::hashString( moduleHash ) );

	std::string optionHash = moduleName.string();
	if ( funcNameSets )
	{
		for ( uint32_t i = 0; i < numRayTypes; ++i )
		{
			for ( uint32_t j = 0; j < numGeomTypes; ++j )
			{
				uint32_t k = numGeomTypes * i + j;
				if ( funcNameSets.value()[k].intersectFuncName != nullptr )
					optionHash += funcNameSets.value()[k].intersectFuncName;
				if ( funcNameSets.value()[k].filterFuncName != nullptr ) optionHash += funcNameSets.value()[k].filterFuncName;
			}
		}
	}

	if ( options )
	{
		optionHash.append( "\n" );
		for ( const auto& option : options.value() )
			optionHash += option + std::string( "\n" );
	}
	optionHash = Utility::format( "%08x", Utility::hashString( optionHash ) );

	return moduleHash + "-" + optionHash + ".v." + deviceName + "." + driverVersion + "_" +
		   std::to_string( 8 * sizeof( void* ) ) + ".bin";
}

std::string Compiler::loadCacheFileToBinary( const std::string& cacheName, const std::string& deviceName )
{
	long long checksumValue = 0;
	{
		std::filesystem::path path = m_cacheDirectory / ( cacheName + ".check" );
		std::ifstream		  file( path, std::ios::in | std::ios::binary );
		if ( !file.is_open() )
		{
			std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		file.read( reinterpret_cast<char*>( &checksumValue ), sizeof( long long ) );
	}
	if ( checksumValue == 0 ) throw std::runtime_error( "Checksum is zero" );

	std::string binary;
	{
		std::filesystem::path path = m_cacheDirectory / cacheName;
		std::ifstream		  file( path, std::ios::in | std::ios::binary | std::ios::ate );
		if ( !file.is_open() )
		{
			std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		size_t binarySize = file.tellg();
		file.clear();
		file.seekg( 0, std::ios::beg );
		binary.resize( binarySize );
		file.read( binary.data(), binary.size() );
	}

	long long hash = Utility::hashString( binary );
	if ( hash != checksumValue )
	{
		std::string msg = Utility::format( "Checksum doesn't match %llx : %llx", hash, checksumValue );
		throw std::runtime_error( msg );
	}

	if constexpr ( !UseBitcode )
	{
		if ( deviceName.find( "NVIDIA" ) != std::string::npos )
		{
			std::lock_guard<std::mutex> lockMutex( m_binMutex );
			if ( m_binCache.find( cacheName ) != m_binCache.end() )
			{
				binary = m_binCache[cacheName];
			}
			else
			{
#if defined( HIPRT_ENCRYPT )
				std::string deryptKeyStr( DecryptKey );
				binary = decrypt( binary, deryptKeyStr );
#endif
				m_binCache[cacheName] = binary;
			}
		}
	}

	return binary;
}

void Compiler::cacheBinaryToFile( const std::string& binaryIn, const std::string& cacheName, const std::string& deviceName )
{
	std::string binary = binaryIn;
#if defined( HIPRT_ENCRYPT )
	if constexpr ( !UseBitcode )
	{
		std::string deryptKeyStr( DecryptKey );
		if ( deviceName.find( "NVIDIA" ) != std::string::npos ) binary = encrypt( binary, deryptKeyStr );
	}
#endif

	{
		std::filesystem::path path = m_cacheDirectory / cacheName;
		std::ofstream		  file( path, std::ios::out | std::ios::binary );
		if ( !file.is_open() )
		{
			std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		file.write( binary.data(), binary.size() );
	}

	long long hash = Utility::hashString( binary );
	{
		std::filesystem::path path = m_cacheDirectory / ( cacheName + ".check" );
		std::ofstream		  file( path, std::ios::out | std::ios::binary );
		if ( !file.is_open() )
		{
			std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		file.write( reinterpret_cast<char*>( &hash ), sizeof( long long ) );
	}
}

oroFunction Compiler::getFunctionFromPrecompiledBinary( const std::string& funcName )
{
	bool						amd	 = oroGetCurAPI( 0 ) == ORO_API_HIP;
	const std::filesystem::path path = getFatbinPath( amd );

	std::lock_guard<std::mutex> lock( m_moduleMutex );
	auto						cacheEntry = m_moduleCache.find( path.string() );
	oroModule					module;
	if ( cacheEntry != m_moduleCache.end() )
	{
		module = cacheEntry->second;
	}
	else
	{
		std::ifstream file( path, std::ios::binary | std::ios::in );
		if ( !file.is_open() )
		{
			std::string msg = Utility::format( "Unable to open '%s'\n", path.string().c_str() );
			throw std::runtime_error( msg );
		}

		size_t sizeFile;
		file.seekg( 0, std::fstream::end );
		size_t size = sizeFile = static_cast<size_t>( file.tellg() );

		std::vector<char> binary;
		binary.resize( size );
		file.seekg( 0, std::fstream::beg );
		file.read( binary.data(), size );

		checkOro( oroModuleLoadData( &module, binary.data() ) );
		m_moduleCache[path.string()] = module;
	}

	oroFunction function;
	checkOro( oroModuleGetFunction( &function, module, funcName.c_str() ) );

	return function;
}

std::string Compiler::buildFunctionTableBitcode(
	Context& context, uint32_t numGeomTypes, uint32_t numRayTypes, const std::vector<hiprtFuncNameSet>& funcNameSets )
{
	if constexpr ( BakedCodeIsGenerated )
	{
		bool amd = oroGetCurAPI( 0 ) == ORO_API_HIP;

		std::vector<const char*> options;
		std::string				 includePath = ( "-I" + Utility::getEnvVariable( "HIPRT_PATH" ) + "/" );
		options.push_back( includePath.c_str() );
		addCommonOpts( context, options );

		if ( amd )
		{
			options.push_back( "-fgpu-rdc" );
			options.push_back( "-Xclang" );
			options.push_back( "-mno-constructor-aliases" );
		}
		else
		{
			options.push_back( "--device-c" );
			options.push_back( "-arch=compute_60" );
		}

		const uint32_t			 numHeaders = sizeof( GET_ARGS( hiprt_device ) ) / sizeof( void* );
		std::vector<const char*> includeNames;
		std::vector<const char*> headers;
		std::vector<std::string> headerData( numHeaders );
		for ( uint32_t i = 0; i < numHeaders - 1; ++i )
		{
			includeNames.push_back( GET_INC( hiprt_device )[i] );
			headerData[i] = decryptSourceCode( GET_ARGS( hiprt_device )[i] );
			headers.push_back( headerData[i].c_str() );
		}
		includeNames.push_back( "hiprt_device.h" );
		headerData[numHeaders - 1] = decryptSourceCode( GET_ARGS( hiprt_device )[numHeaders - 1] );
		headers.push_back( headerData[numHeaders - 1].c_str() );

		std::string src = "#include <hiprt_device.h>\n";
		addCustomFuncsSwitchCase( src, funcNameSets, numGeomTypes, numRayTypes );

		std::vector<const char*> funcNames;
		orortcProgram			 prog;

		buildProgram(
			context,
			funcNames,
			src,
			std::string(),
			headers,
			includeNames,
			options,
			numGeomTypes,
			numRayTypes,
			funcNameSets,
			prog );

		size_t size = 0;
		if ( amd )
			checkOrortc( orortcGetBitcodeSize( prog, &size ) );
		else
			checkOrortc( orortcGetCodeSize( prog, &size ) );

		std::string binary;
		binary.resize( size );
		if ( amd )
			checkOrortc( orortcGetBitcode( prog, binary.data() ) );
		else
			checkOrortc( orortcGetCode( prog, binary.data() ) );
		checkOrortc( orortcDestroyProgram( &prog ) );
		return binary;
	}
	else
	{
		throw std::runtime_error( "Not compiled with the baked kernel code support" );
	}
}
} // namespace hiprt
