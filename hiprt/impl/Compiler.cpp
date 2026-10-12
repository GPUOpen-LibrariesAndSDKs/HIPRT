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
#include <regex>
#if defined( HIPRT_BAKE_KERNEL_GENERATED )
#include <hiprt/cache/Kernels.h>
#include <hiprt/cache/KernelArgs.h>
#endif

// HIPRT_BAKE_COMPILED_KERNEL means we are using a precompiled kernel that is embedded as a buffer inside the executable.
#if defined( HIPRT_BAKE_COMPILED_KERNEL )
constexpr auto UseBakedCompiledKernel = true;
#include "hiprt/impl/bvh_build_array.h"
#else
constexpr auto       UseBakedCompiledKernel = false;
const unsigned char* bvh_build_array_h      = nullptr; // if bvh_build_array.h is not used, declare a nullptr precompiled data.
const size_t         bvh_build_array_h_size = 0;
const size_t         bvh_build_array_h_size_uncompressed = 0;
const bool           bvh_build_array_h_isCompressed      = false;
#endif

namespace
{
#if defined( HIPRT_BITCODE_LINKING )
constexpr auto UseBitcode = true;
#else
constexpr auto       UseBitcode                          = false;
#endif

#if defined( HIPRT_LOAD_FROM_STRING )
constexpr auto UseBakedCode = true;
#else
constexpr auto       UseBakedCode                        = false;
#endif

#if defined( HIPRT_BAKE_KERNEL_GENERATED )
constexpr auto BakedCodeIsGenerated = true;
#else
constexpr auto       BakedCodeIsGenerated                = false;
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

void Compiler::init()
{
	if ( UseBitcode || UseBakedCompiledKernel || hiprtcCreateProgram == nullptr || hiprtcCompileProgram == nullptr ||
	     hiprtcDestroyProgram == nullptr )
	{
		// If we use the precompiled bitcode, we won't check RTIP 3.1 support through HIPRTC.
		// Or, if the HIP Run Time Compiler is not loaded (e.g. hiprtc0604.dll) , we can't check RTIP 3.1 support.

		// We'll assume it's supported, and Context::getRtip is supposed to make extra checks to be sure it's actually
		// supported.
		m_rtip31Support = true;
	}
	else
	{
		// check the RTIP 3.1 support of the RTC

		const std::string src =
			"__global__ void rtcTest() { __builtin_amdgcn_image_bvh8_intersect_ray(0, 0.0f, 0xff, { 0.0f, 0.0f, 0.0f "
			"}, { 1.0f, 0.0f, 0.0f }, 0, { 0, 0, 0, 0 }, nullptr, nullptr ); }";

		orortcProgram prog;
		checkOrortc( orortcCreateProgram( &prog, src.c_str(), "", 0, nullptr, nullptr ) );

		m_rtip31Support = false;

		if ( orortcCompileProgram( prog, 0, nullptr ) == ORORTC_SUCCESS )
		{
			m_rtip31Support = true;
			checkOrortc( orortcDestroyProgram( &prog ) );
		}
	}
}

Kernel Compiler::getKernel(
	const Context&                  context,
	const std::filesystem::path&    moduleName,
	const std::string&              funcName,
	const std::vector<const char*>& options,
	const uint32_t                  numHeaders,
	const char**                    headersIn,
	const char**                    includeNamesIn )
{
	std::lock_guard<std::mutex> lock( m_kernelMutex );

	const std::string cacheName  = moduleName.string() + funcName;
	auto              cacheEntry = m_kernelCache.find( cacheName );
	if ( cacheEntry != m_kernelCache.end() ) return cacheEntry->second;

	oroFunction function;

	// if we use the precompiled bitcode as file or baked as binary, we use 'getFunctionFromPrecompiledBinary'
	if constexpr ( UseBitcode || UseBakedCompiledKernel )
	{
		function = getFunctionFromPrecompiledBinary( funcName );
	}
	else
	{
		std::vector<const char*>      funcNames = { funcName.c_str() };
		std::vector<const char*>      headers;
		std::vector<const char*>      includeNames;
		std::vector<hiprtFuncNameSet> funcNameSets;
		std::vector<oroFunction>      functions;
		oroModule                     module;

		if ( numHeaders == 0 )
		{
			const std::string src = readSourceCode( moduleName );
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
				headerData[i] = headersIn[i];
				headers.push_back( headerData[i].c_str() );
			}

			const std::string src = headersIn[numHeaders - 1];
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
	const std::vector<const char*>& funcNames,
	const std::string&              src,
	const std::filesystem::path&    moduleName,
	std::vector<const char*>&       headers,
	std::vector<const char*>&       includeNames,
	const std::vector<const char*>& options,
	orortcProgram&                  progOut )
{
	// Only the file name, not the full path. This name is not merely cosmetic: comgr stages the
	// source into a temporary input directory under it, so diagnostics come out of paths like
	// /tmp/comgr-XXXXXX/input/BvhBuilderKernels.h. Handing it an absolute Windows path makes that
	// staging path invalid and comgr then fails silently, returning 6 with an empty log, which is
	// expensive to diagnose. Includes do not depend on this: they are resolved from the header
	// arrays below and from the -I that the caller adds.
	const std::string programName = moduleName.filename().string();
	checkOrortc( orortcCreateProgram(
		&progOut, src.c_str(), programName.c_str(), static_cast<int>( headers.size() ), headers.data(), includeNames.data() ) );

	for ( size_t i = 0; i < funcNames.size(); ++i )
		checkOrortc( orortcAddNameExpression( progOut, funcNames[i] ) );

	orortcResult e =
		orortcCompileProgram( progOut, static_cast<int>( options.size() ), std::vector<const char*>( options ).data() );
	if ( e != ORORTC_SUCCESS )
	{
		size_t logSize;
		checkOrortc( orortcGetProgramLogSize( progOut, &logSize ) );

		if ( logSize > 0 )
		{
			std::string log( logSize, '\0' );
			checkOrortc( orortcGetProgramLog( progOut, &log[0] ) );
			throw std::runtime_error( "Runtime compilation failed:\n" + log );
		}
	}
}

void Compiler::buildKernels(
	const Context&                       context,
	const std::vector<const char*>&      funcNames,
	const std::string&                   src,
	const std::filesystem::path&         moduleName,
	std::vector<const char*>&            headers,
	std::vector<const char*>&            includeNames,
	const std::vector<const char*>&      options,
	const uint32_t                       numGeomTypes,
	const uint32_t                       numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	std::vector<oroFunction>&            functions,
	oroModule&                           module,
	const bool                           extended,
	const bool                           cache )
{
	if ( !std::filesystem::exists( m_cacheDirectory ) && !std::filesystem::create_directory( m_cacheDirectory ) )
		throw std::runtime_error( "Cannot create cache directory" );

	std::lock_guard<std::mutex> lock( m_moduleMutex );
	auto                        cacheEntry = m_moduleCache.find( moduleName.string() );
	if ( cacheEntry != m_moduleCache.end() )
	{
		module = cacheEntry->second;
	}
	else
	{
		const std::string cacheName =
			getCacheFilename( context, src, moduleName, options, funcNameSets, numGeomTypes, numRayTypes );
		const bool upToDate = isCachedFileUpToDate( m_cacheDirectory / cacheName, moduleName );

		orortcProgram prog;
		std::string   binary;
		if ( upToDate && cache )
		{
			binary = loadCacheFileToBinary( cacheName, context.getDeviceName() );
		}
		else
		{
			std::vector<std::string> headerData;
			std::string              extSrc = src;
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
						headerData[i] = GET_ARGS( hiprt_device_impl )[i];
						headers.push_back( headerData[i].c_str() );
					}
					includeNames.push_back( "hiprt_device_impl.h" );
					headerData[numHeaders - 1] = GET_ARGS( hiprt_device_impl )[numHeaders - 1];
					headers.push_back( headerData[numHeaders - 1].c_str() );
				}
				else
				{
					extSrc = "#include <hiprt/impl/hiprt_device_impl.h>\n";
					addCustomFuncsSwitchCase( extSrc, funcNameSets, numGeomTypes, numRayTypes );
					extSrc += "\n" + src;
				}
			}

			std::vector<const char*> opts        = options;
			std::string              includePath = "-I" + Utility::getRootDir().string();
			opts.push_back( includePath.c_str() );
			addCommonOpts( context, opts, extended );

			buildProgram( funcNames, extSrc, moduleName, headers, includeNames, opts, prog );

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
	const Context&                       context,
	const std::vector<const char*>&      funcNames,
	const std::filesystem::path&         moduleName,
	const std::string_view               bitcodeBinary,
	const uint32_t                       numGeomTypes,
	const uint32_t                       numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets,
	std::vector<oroFunction>&            functions,
	const bool                           cache )
{
	if constexpr ( UseBitcode )
	{
		if ( !std::filesystem::exists( m_cacheDirectory ) && !std::filesystem::create_directory( m_cacheDirectory ) )
			throw std::runtime_error( "Cannot create cache directory" );

		std::lock_guard<std::mutex> lock( m_moduleMutex );
		auto                        cacheEntry = m_moduleCache.find( moduleName.string() );
		oroModule                   module;
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

				const uint32_t   JITOptCount = 6u;
				orortcLinkState  rtcLinkState;
				orortcJIT_option options[JITOptCount];
				void*            optionVals[JITOptCount];
				float            wallTime;

				constexpr uint32_t LogSize = 8192u;
				char               errorLog[LogSize];
				char               infoLog[LogSize];

				options[0]    = ORORTC_JIT_WALL_TIME;
				optionVals[0] = reinterpret_cast<void*>( &wallTime );

				options[1]    = ORORTC_JIT_INFO_LOG_BUFFER;
				optionVals[1] = infoLog;

				options[2]    = ORORTC_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
				optionVals[2] = reinterpret_cast<void*>( static_cast<uintptr_t>( LogSize ) );

				options[3]    = ORORTC_JIT_ERROR_LOG_BUFFER;
				optionVals[3] = errorLog;

				options[4]    = ORORTC_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
				optionVals[4] = reinterpret_cast<void*>( static_cast<uintptr_t>( LogSize ) );

				options[5]    = ORORTC_JIT_LOG_VERBOSE;
				optionVals[5] = reinterpret_cast<void*>( static_cast<uintptr_t>( 1 ) );

				bool                     amd        = oroGetCurAPI( 0 ) == ORO_API_HIP;
				std::filesystem::path    bcPath     = getBitcodePath( amd );
				const orortcJITInputType typeBc     = amd ? ORORTC_JIT_INPUT_LLVM_BUNDLED_BITCODE : ORORTC_JIT_INPUT_FATBINARY;
				const orortcJITInputType typeUserBc = amd ? ORORTC_JIT_INPUT_LLVM_BITCODE : ORORTC_JIT_INPUT_PTX;

				void* binaryPtr;
				checkOrortc( orortcLinkCreate( JITOptCount, options, optionVals, &rtcLinkState ) );

				orortcResult res = orortcLinkAddFile( rtcLinkState, typeBc, bcPath.string().c_str(), 0, 0, 0 );
				if ( res != ORORTC_SUCCESS )
				{
					std::string msg = Utility::format(
						"Orortc error: '%s' [ %d ] linking file '%s'\n",
						orortcGetErrorString( res ),
						static_cast<uint32_t>( res ),
						bcPath.string().c_str() );
					throw std::runtime_error( msg );
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
	std::string_view view( traits );

	// Find last "::"
	std::size_t last = view.rfind( "::" );
	if ( last != std::string::npos )
	{
		std::size_t      prev = view.rfind( "::", last - 1 );
		std::string_view lastNs =
			( prev == std::string::npos ) ? view.substr( 0, last ) : view.substr( prev + 2, last - ( prev + 2 ) );

		if ( lastNs == "hiprt" )
		{
			view = view.substr( last + 2 );
		}
		else if ( prev != std::string::npos )
		{
			view = view.substr( prev + 2 );
		}
	}

	std::string result;
	result.reserve( view.size() );

	for ( std::size_t i = 0; i < view.size(); ++i )
	{
		char c = view[i];

		// Collapse "::" → "_"
		if ( c == ':' && i + 1 < view.size() && view[i + 1] == ':' )
		{
			result.push_back( '_' );
			++i; // skip second ':'
			continue;
		}

		switch ( c )
		{
		case '<':
		case ',':
			result.push_back( '_' );
			break;
		case '>':
		case ' ':
			break;
		default:
			result.push_back( c );
		}
	}

	return result;
}

std::string
Compiler::readSourceCode( const std::filesystem::path& path, std::optional<std::vector<std::filesystem::path>> includes )
{
	std::string   src;
	std::ifstream file( path );
	if ( !file.is_open() )
	{
		const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
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
				const size_t      pa  = line.find( "<" );
				const size_t      pb  = line.find( ">" );
				const std::string buf = line.substr( pa + 1, pb - pa - 1 );
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

void Compiler::addCommonOpts( const Context& context, std::vector<const char*>& opts, const bool extended )
{
	if ( !extended ) // do not use fast math for trace kernel (the fast math option can be passed from the application)
	{
		if ( context.getDeviceName().find( "NVIDIA" ) != std::string::npos )
			opts.push_back( "--use_fast_math" );
		else
			opts.push_back( "-ffast-math" );
	}

	if ( !std::any_of( opts.begin(), opts.end(), []( const char* s ) { return s && std::strstr( s, "std=c++" ) != nullptr; } ) )
	{
		opts.push_back( "-std=c++17" );
	}

	const uint32_t rtip = context.getRtip();
	if ( rtip > 0 )
	{
		m_rtipStr = "-DHIPRT_RTIP=" + std::to_string( rtip );
		opts.push_back( m_rtipStr.c_str() );
	}

	// Name the target explicitly rather than letting the runtime compiler infer it from whichever
	// context happens to be current. Inference works when the context is set, which is why this has
	// gone unnoticed, but it leaves the produced code object depending on ambient state, and a
	// toolchain whose default target set excludes the running device then yields a module that fails
	// to load with hipErrorInvalidImage. Skipped for NVIDIA, where the option does not apply, and
	// skipped if the caller already named a target.
	if ( context.getDeviceName().find( "NVIDIA" ) == std::string::npos &&
	     !std::any_of( opts.begin(), opts.end(), []( const char* s ) {
			 return s && ( std::strstr( s, "offload-arch" ) != nullptr || std::strstr( s, "mcpu" ) != nullptr );
		 } ) )
	{
		const std::string gcnArchName = context.getGcnArchName();
		if ( !gcnArchName.empty() )
		{
			m_offloadArchStr = "--offload-arch=" + gcnArchName;
			opts.push_back( m_offloadArchStr.c_str() );
		}
	}

	opts.push_back( "-D__USE_HIP__" );
}

std::filesystem::path Compiler::getBitcodePath( const bool amd )
{
	const std::string hipSdkVersion = "_" + std::string( HIP_VERSION_STR );
	std::string       filename      = "hiprt" + std::string( HIPRT_VERSION_STR );

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

std::filesystem::path Compiler::getFatbinPath( const bool amd )
{
	const std::string hipSdkVersion = "_" + std::string( HIP_VERSION_STR );
	std::string       filename      = "hiprt" + std::string( HIPRT_VERSION_STR );
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

void Compiler::addCustomFuncsSwitchCase(
	std::string&                                 extSrc,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	const uint32_t                               numGeomTypes,
	const uint32_t                               numRayTypes )
{
	std::string intersectFuncDef =
		"HIPRT_DEVICE bool intersectFunc( const uint32_t geomType, const uint32_t rayType, const hiprtFuncTableHeader& "
		"tableHeader, "
		"const hiprtRay& ray, void* payload, hiprtHit& hit )\n{\n\tconst uint32_t index = tableHeader.numGeomTypes * rayType + "
		"geomType;\n\t[[maybe_unused]] const void* data = tableHeader.funcDataSets[index].intersectFuncData;\n\tswitch ( index "
		") \n\t{\n";
	std::string filterFuncDef =
		"HIPRT_DEVICE bool filterFunc( const uint32_t geomType, const uint32_t rayType, const hiprtFuncTableHeader& "
		"tableHeader, const "
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
				const uint32_t k = numGeomTypes * i + j;
				if ( funcNameSets.value()[k].intersectFuncName != nullptr )
				{
					const std::string intersectFuncName = funcNameSets.value()[k].intersectFuncName;
					if ( !intersectFuncName.empty() )
					{
						funcDecls += "HIPRT_DEVICE bool " + intersectFuncName +
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
						funcDecls += "HIPRT_DEVICE bool " + filterFuncName +
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

std::string Compiler::getCacheFilename(
	const Context&                               context,
	const std::string&                           src,
	const std::filesystem::path&                 moduleName,
	std::optional<std::vector<const char*>>      options,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	const uint32_t                               numGeomTypes,
	const uint32_t                               numRayTypes )
{
	const std::string driverVersion = context.getDriverVersion();
	const std::string moduleHash    = Utility::format( "%08x", Utility::hashString( moduleName.string() + src ) );

	std::string deviceName = context.getDeviceName();
	deviceName             = deviceName.substr( 0, deviceName.find( ":" ) );

	// The device name ends up as a path component below, and some devices report a name that is not a
	// legal file name: "AMD Instinct MI250X / MI250", for instance, would send the cache write into a
	// directory that does not exist. Replace anything unsafe rather than assume the name is clean.
	// Windows rejects the ASCII control characters as well as the punctuation, so cover both.
	for ( char& c : deviceName )
	{
		const unsigned char u = static_cast<unsigned char>( c );
		if ( u <= 0x1F || u == 0x7F || c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' ||
		     c == '>' || c == '|' )
			c = '_';
	}

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
		std::ifstream         file( path, std::ios::in | std::ios::binary );
		if ( !file.is_open() )
		{
			const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
			throw std::runtime_error( msg );
		}
		file.read( reinterpret_cast<char*>( &checksumValue ), sizeof( long long ) );
	}
	if ( checksumValue == 0 ) throw std::runtime_error( "Checksum is zero" );

	std::string binary;
	{
		std::filesystem::path path = m_cacheDirectory / cacheName;
		std::ifstream         file( path, std::ios::in | std::ios::binary | std::ios::ate );
		if ( !file.is_open() )
		{
			const std::string msg = Utility::format( "Unable to open '%s'", path.string().c_str() );
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
				m_binCache[cacheName] = binary;
			}
		}
	}

	return binary;
}

void Compiler::cacheBinaryToFile( const std::string& binaryIn, const std::string& cacheName, const std::string& deviceName )
{
	std::string binary = binaryIn;

	{
		std::filesystem::path path = m_cacheDirectory / cacheName;
		std::ofstream         file( path, std::ios::out | std::ios::binary );
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
		std::ofstream         file( path, std::ios::out | std::ios::binary );
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
	bool                        amd  = oroGetCurAPI( 0 ) == ORO_API_HIP;
	const std::filesystem::path path = getFatbinPath( amd );

	std::lock_guard<std::mutex> lock( m_moduleMutex );
	auto                        cacheEntry = m_moduleCache.find( path.string() );
	oroModule                   module     = nullptr;
	if ( cacheEntry != m_moduleCache.end() )
	{
		module = cacheEntry->second;
	}
	else
	{
		std::ifstream file( path, std::ios::binary | std::ios::in );
		if ( !file.is_open() )
		{
			// Note: even if 'HIPRT_BAKE_COMPILED_KERNEL' is enable, if the file exists, it overrides the embedded precompiled
			// kernel.
			if constexpr ( UseBakedCompiledKernel )
			{
				std::vector<unsigned char> binary;
				OrochiUtils::HandlePrecompiled(
					binary,
					bvh_build_array_h,
					bvh_build_array_h_size,
					bvh_build_array_h_isCompressed ? std::optional<size_t>{ bvh_build_array_h_size_uncompressed }
												   : std::nullopt );
				checkOro( oroModuleLoadData( &module, binary.data() ) );
			}
			else
			{
				std::string msg = Utility::format( "Unable to open '%s'\n", path.string().c_str() );
				throw std::runtime_error( msg );
			}
		}
		else
		{

			size_t sizeFile;
			file.seekg( 0, std::fstream::end );
			size_t size = sizeFile = static_cast<size_t>( file.tellg() );

			std::vector<char> binary;
			binary.resize( size );
			file.seekg( 0, std::fstream::beg );
			file.read( binary.data(), size );

			checkOro( oroModuleLoadData( &module, binary.data() ) );
		}

		m_moduleCache[path.string()] = module;
	}

	oroFunction function;
	checkOro( oroModuleGetFunction( &function, module, funcName.c_str() ) );

	return function;
}

std::string Compiler::buildFunctionTableBitcode(
	const Context&                       context,
	const uint32_t                       numGeomTypes,
	const uint32_t                       numRayTypes,
	const std::vector<hiprtFuncNameSet>& funcNameSets )
{
	if constexpr ( BakedCodeIsGenerated )
	{
		bool amd = oroGetCurAPI( 0 ) == ORO_API_HIP;

		std::vector<const char*> options;
		std::string              includePath = "-I" + Utility::getRootDir().string();
		options.push_back( includePath.c_str() );
		addCommonOpts( context, options, false );

		if ( amd )
		{
			options.push_back( "-fgpu-rdc" ); // relocatable device code
			options.push_back( "-Xclang" );
			options.push_back( "-mno-constructor-aliases" );
		}
		else
		{
			options.push_back( "--device-c" ); // relocatable device code
		}

		const uint32_t           numHeaders = sizeof( GET_ARGS( hiprt_device ) ) / sizeof( void* );
		std::vector<const char*> includeNames;
		std::vector<const char*> headers;
		std::vector<std::string> headerData( numHeaders );
		for ( uint32_t i = 0; i < numHeaders - 1; ++i )
		{
			includeNames.push_back( GET_INC( hiprt_device )[i] );
			headerData[i] = GET_ARGS( hiprt_device )[i];
			headers.push_back( headerData[i].c_str() );
		}
		includeNames.push_back( "hiprt_device.h" );
		headerData[numHeaders - 1] = GET_ARGS( hiprt_device )[numHeaders - 1];
		headers.push_back( headerData[numHeaders - 1].c_str() );

		std::string src = "#include <hiprt_device.h>\n";
		addCustomFuncsSwitchCase( src, funcNameSets, numGeomTypes, numRayTypes );

		std::vector<const char*> funcNames;
		orortcProgram            prog;

		buildProgram( funcNames, src, std::string(), headers, includeNames, options, prog );

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
