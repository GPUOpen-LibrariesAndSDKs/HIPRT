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
	~Compiler();

	void init();

	[[nodiscard]] Kernel getKernel(
		const Context&                  context,
		const std::filesystem::path&    moduleName,
		const std::string&              funcName,
		const std::vector<const char*>& options,
		const uint32_t                  numHeaders   = 0,
		const char**                    headers      = nullptr,
		const char**                    includeNames = nullptr );

	void buildProgram(
		const std::vector<const char*>& funcNames,
		const std::string&              src,
		const std::filesystem::path&    moduleName,
		std::vector<const char*>&       headers,
		std::vector<const char*>&       includeNames,
		const std::vector<const char*>& options,
		orortcProgram&                  progOut );

	void buildKernels(
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
		const bool                           cache );

	void buildKernelsFromBitcode(
		const Context&                       context,
		const std::vector<const char*>&      funcNames,
		const std::filesystem::path&         moduleName,
		const std::string_view               bitcodeBinary,
		const uint32_t                       numGeomTypes,
		const uint32_t                       numRayTypes,
		const std::vector<hiprtFuncNameSet>& funcNameSets,
		std::vector<oroFunction>&            functions,
		const bool                           cache );

	void setCacheDir( const std::filesystem::path& path );

	[[nodiscard]] bool isRtip31Supported() const { return m_rtip31Support; }

	[[nodiscard]] static std::string kernelNameSufix( const std::string& traits );

  private:
	[[nodiscard]] static std::string readSourceCode(
		const std::filesystem::path& path, std::optional<std::vector<std::filesystem::path>> includes = std::nullopt );

	[[nodiscard]] static std::filesystem::path getBitcodePath( const bool amd );
	[[nodiscard]] static std::filesystem::path getFatbinPath( const bool amd );

	[[nodiscard]] static bool
	isCachedFileUpToDate( const std::filesystem::path& cachedFile, const std::filesystem::path& moduleName );

	void addCustomFuncsSwitchCase(
		std::string&                                 extSrc,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		const uint32_t                               numGeomTypes = 0,
		const uint32_t                               numRayTypes  = 1 );

	void addCommonOpts( const Context& context, std::vector<const char*>& opts, const bool extended );

	[[nodiscard]] std::string getCacheFilename(
		const Context&                               context,
		const std::string&                           src,
		const std::filesystem::path&                 moduleName,
		std::optional<std::vector<const char*>>      options      = std::nullopt,
		std::optional<std::vector<hiprtFuncNameSet>> funcNameSets = std::nullopt,
		const uint32_t                               numGeomTypes = 0,
		const uint32_t                               numRayTypes  = 1 );

	[[nodiscard]] std::string loadCacheFileToBinary( const std::string& cacheName, const std::string& deviceName );

	void cacheBinaryToFile( const std::string& binary, const std::string& cacheName, const std::string& deviceName );

	[[nodiscard]] oroFunction getFunctionFromPrecompiledBinary( const std::string& funcName );

	[[nodiscard]] std::string buildFunctionTableBitcode(
		const Context&                       context,
		const uint32_t                       numGeomTypes,
		const uint32_t                       numRayTypes,
		const std::vector<hiprtFuncNameSet>& funcNameSets );

	std::filesystem::path m_cacheDirectory = "cache";

	bool m_rtip31Support = false;

	std::string m_rtipStr;
	std::string m_offloadArchStr;

	std::mutex                    m_kernelMutex;
	std::map<std::string, Kernel> m_kernelCache;

	std::mutex                       m_moduleMutex;
	std::map<std::string, oroModule> m_moduleCache;

	std::mutex                         m_binMutex;
	std::map<std::string, std::string> m_binCache;
};
} // namespace hiprt
