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

#pragma once
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/Kernel.h>
#include <filesystem>
#include <optional>
#include <mutex>

namespace hiprt
{
class Context;
class Compiler
{
  public:
	static constexpr std::string_view DecryptKey = "20220318";

	~Compiler();

	Kernel getKernel(
		Context&					 context,
		const std::filesystem::path& moduleName,
		const std::string&			 funcName,
		std::vector<const char*>&	 options,
		uint32_t					 numHeaders	  = 0,
		const char**				 headers	  = nullptr,
		const char**				 includeNames = nullptr );

	void buildProgram(
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
		orortcProgram&						 progOut );

	void buildKernels(
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
		bool								 cache );

	void buildKernelsFromBitcode(
		Context&							 context,
		const std::vector<const char*>&		 funcNames,
		const std::filesystem::path&		 moduleName,
		const std::string_view				 bitcodeBinary,
		uint32_t							 numGeomTypes,
		uint32_t							 numRayTypes,
		const std::vector<hiprtFuncNameSet>& funcNameSets,
		std::vector<oroFunction>&			 functions,
		bool								 cache );

	void setCacheDir( const std::filesystem::path& path );

	static std::string kernelNameSufix( const std::string& traits );

  private:
	static std::string readSourceCode(
		const std::filesystem::path& path, std::optional<std::vector<std::filesystem::path>> includes = std::nullopt );

	static void addCommonOpts( Context& context, std::vector<const char*>& opts );
	static void addCustomFuncsSwitchCase(
		std::string&								 extSrc,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		uint32_t									 numGeomTypes = 0,
		uint32_t									 numRayTypes  = 1 );

	static std::filesystem::path getBitcodePath( bool amd );
	static std::filesystem::path getFatbinPath( bool amd );

	static bool isCachedFileUpToDate( const std::filesystem::path& cachedFile, const std::filesystem::path& moduleName );

	std::string decryptSourceCode( const std::string& src );

	std::string getCacheFilename(
		Context&									 context,
		const std::string&							 src,
		const std::filesystem::path&				 moduleName,
		std::optional<std::vector<const char*>>		 options	  = std::nullopt,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		uint32_t									 numGeomTypes = 0,
		uint32_t									 numRayTypes  = 1 );

	std::string loadCacheFileToBinary( const std::string& cacheName, const std::string& deviceName );

	void cacheBinaryToFile( const std::string& binary, const std::string& cacheName, const std::string& deviceName );

	oroFunction getFunctionFromPrecompiledBinary( const std::string& funcName );

	std::string buildFunctionTableBitcode(
		Context& context, uint32_t numGeomTypes, uint32_t numRayTypes, const std::vector<hiprtFuncNameSet>& funcNameSets );

	std::filesystem::path m_cacheDirectory = "cache";

	std::mutex					  m_kernelMutex;
	std::map<std::string, Kernel> m_kernelCache;

	std::mutex						 m_moduleMutex;
	std::map<std::string, oroModule> m_moduleCache;

	std::mutex						   m_decryptMutex;
	std::map<std::string, std::string> m_decryptCache;

	std::mutex						   m_binMutex;
	std::map<std::string, std::string> m_binCache;
};
} // namespace hiprt
