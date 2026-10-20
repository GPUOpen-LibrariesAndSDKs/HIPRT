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

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

//
// Define the library path that HIPRT will use.
// Order matters: the first library file of the list to exist will be the one loaded.
//

#ifdef _WIN32

inline const char* g_hip_paths[] = {
	"amdhip64_7.dll",
	"amdhip64_6.dll",
	"amdhip64.dll", // <- hip '5.x' DLL.
	nullptr };

inline const char* g_hiprtc_paths[] = { "hiprtc0720.dll",  "hiprtc0719.dll",  "hiprtc0718.dll",
                                        "hiprtc0717.dll",  "hiprtc0716.dll",  "hiprtc0715.dll",
                                        "hiprtc07014.dll", "hiprtc07013.dll", "hiprtc07012.dll",
                                        "hiprtc07011.dll", "hiprtc07010.dll", "hiprtc0709.dll",
                                        "hiprtc0708.dll",  "hiprtc0707.dll",  "hiprtc0706.dll",
                                        "hiprtc0705.dll",  "hiprtc0704.dll",  "hiprtc0703.dll",
                                        "hiprtc0702.dll",  "hiprtc0701.dll",  "hiprtc0700.dll",
                                        "hiprtc0605.dll",  "hiprtc0604.dll",  "hiprtc0603.dll",
                                        "hiprtc0602.dll",  "hiprtc0601.dll",  "hiprtc0600.dll",
                                        "hiprtc0507.dll",  "hiprtc0506.dll",  "hiprtc0505.dll",
                                        "hiprtc0504.dll",  "hiprtc0503.dll",  nullptr };
#elif defined( __APPLE__ )

inline const char** g_hip_paths    = nullptr;
inline const char** g_hiprtc_paths = nullptr;
#else

inline const char* g_hip_paths[] = {

	// first, we try with the generic symbolic link.
	"libamdhip64.so",
	"/opt/rocm/lib/libamdhip64.so",
	"/opt/rocm/hip/lib/libamdhip64.so",

	// .. if it doesn't exist, we try the specific versions
	"libamdhip64.so.7",
	"/opt/rocm/lib/libamdhip64.so.7",
	"/opt/rocm/hip/lib/libamdhip64.so.7",

	"libamdhip64.so.6",
	"/opt/rocm/lib/libamdhip64.so.6",
	"/opt/rocm/hip/lib/libamdhip64.so.6",

	"libamdhip64.so.5",
	"/opt/rocm/lib/libamdhip64.so.5",
	"/opt/rocm/hip/lib/libamdhip64.so.5",

	nullptr };

inline const char* g_hiprtc_paths[] = {

	// Let the dynamic loader honor its search path before probing conventional installs.
	"libhiprtc.so",
	"/opt/rocm/hip/lib/libhiprtc.so",
	"/opt/rocm/lib/libhiprtc.so",

	// .. if it doesn't exist, we try the specific versions
	"/opt/rocm/hip/lib/libhiprtc.so.7",
	"/opt/rocm/lib/libhiprtc.so.7",
	"libhiprtc.so.7",

	"/opt/rocm/hip/lib/libhiprtc.so.6",
	"/opt/rocm/lib/libhiprtc.so.6",
	"libhiprtc.so.6",

	"/opt/rocm/hip/lib/libhiprtc.so.5",
	"/opt/rocm/lib/libhiprtc.so.5",
	"libhiprtc.so.5",

	nullptr };
#endif

//
// The lists above name most libraries by file name only. Loading by bare name follows the platform's
// default search order, and on Windows that finds the display driver's copy in System32 before the
// ROCm SDK the application was built against. Mixing the two fails in confusing ways far from the
// load itself.
//
// So, for each library named in a list, offer SDK relative candidates first, derived from HIP_PATH
// and ROCM_PATH, one per layout the list itself probes. The candidates go at the head of that
// library's run of entries, which means they precede the hard coded directory entries for the same
// library as well as the bare name: pointing us at an SDK explicitly should outrank a guess at
// /opt/rocm. Entries keep their order relative to one another, so the version preference the list
// encodes still holds, and if neither variable is set the result is exactly the original list.
//

namespace hiprt_libpath
{
#if defined( _WIN32 )
constexpr char        PathSeparator = '\\';
constexpr const char* LibSubDirs[]  = { "bin" };
#else
constexpr char      PathSeparator  = '/';
// Mirror the layouts the hard coded lists probe: an install may carry the library under only one of
// them, and for hiprtc the lists actually prefer hip/lib first.
constexpr const char* LibSubDirs[] = { "lib", "hip/lib" };
#endif

[[nodiscard]] inline const char* baseName( const char* path )
{
	// Scan forward rather than taking the later of strrchr('/') and strrchr('\\'): either of those
	// can be null, and ordering a null pointer against a real one is undefined.
	const char* base = path;
	for ( const char* c = path; *c != '\0'; ++c )
	{
		if ( *c == '/' || *c == '\\' ) base = c + 1;
	}
	return base;
}

// The layout an entry sits in, i.e. the longest LibSubDirs element its directory ends on, or null
// when the entry is a bare name. Longest wins because "/opt/rocm/hip/lib" ends on both "lib" and
// "hip/lib" and only the latter describes it.
[[nodiscard]] inline const char* matchedSubDir( const char* entry )
{
	const char*  base   = baseName( entry );
	const size_t dirLen = static_cast<size_t>( base - entry );
	if ( dirLen == 0 ) return nullptr;

	const size_t end     = dirLen - 1; // directory without its trailing separator
	const char*  best    = nullptr;
	size_t       bestLen = 0;

	for ( const char* subDir : LibSubDirs )
	{
		const size_t len = std::strlen( subDir );
		if ( len > end || len <= bestLen ) continue;

		bool match = true;
		for ( size_t i = 0; i < len && match; ++i )
		{
			char a = entry[end - len + i];
			char b = subDir[i];
			if ( a == '\\' ) a = '/';
			if ( b == '\\' ) b = '/';
			match = ( a == b );
		}

		// Must start on a component boundary, so "mylib" does not count as "lib".
		if ( match && ( end == len || entry[end - len - 1] == '/' || entry[end - len - 1] == '\\' ) )
		{
			best    = subDir;
			bestLen = len;
		}
	}
	return best;
}

[[nodiscard]] inline bool contains( const std::vector<const char*>& v, const char* s )
{
	for ( const char* e : v )
		if ( e == s ) return true;
	return false;
}

// The layouts, ordered the way this particular list probes them. g_hip_paths tries lib before
// hip/lib and g_hiprtc_paths the other way round, so a single fixed order would contradict one of
// them and change which copy is reached first when an SDK populates both. Anything the list never
// mentions is appended in declaration order.
[[nodiscard]] inline std::vector<const char*> preferredSubDirs( const char** paths )
{
	std::vector<const char*> ordered;
	for ( const char** p = paths; *p != nullptr; ++p )
	{
		const char* subDir = matchedSubDir( *p );
		if ( subDir != nullptr && !contains( ordered, subDir ) ) ordered.push_back( subDir );
	}
	for ( const char* subDir : LibSubDirs )
		if ( !contains( ordered, subDir ) ) ordered.push_back( subDir );
	return ordered;
}

inline void addEnvCandidates(
	std::vector<std::string>& out, const char* envVar, const char* fileName, const std::vector<const char*>& subDirs )
{
	const char* root = std::getenv( envVar );
	if ( root == nullptr || root[0] == '\0' ) return;

	std::string base = root;

	// The variable may or may not end in a separator, and may use either one on Windows. Trim
	// whatever is there and append our own, then normalise, so we never emit something like
	// C:/rocm/bin\amdhip64.dll.
	while ( !base.empty() && ( base.back() == '/' || base.back() == '\\' ) )
		base.pop_back();
	if ( base.empty() ) return;

	for ( const char* subDir : subDirs )
	{
		std::string path = base;
		path += PathSeparator;
		path += subDir;
		path += PathSeparator;
		path += fileName;

		for ( char& c : path )
			if ( c == '/' || c == '\\' ) c = PathSeparator;

		out.push_back( path );
	}
}

// The input list with SDK relative candidates inserted at the head of each library's run of entries,
// a run being the consecutive entries sharing a base name. The candidates therefore come before the
// hard coded directory entries for that library, not just before the bare name. Runs stay in their
// original order, so the version preference the list encodes is preserved. Initialised once, in
// place, by the static in the accessors below: that is thread safe under the usual guarantee for
// function local statics, so no lock is needed here.
struct PathTable
{
	explicit PathTable( const char** paths )
	{
		if ( paths == nullptr ) return;

		size_t originalCount = 0;
		for ( const char** p = paths; *p != nullptr; ++p )
			++originalCount;

		// First pass: materialise every candidate, so the strings stop moving and c_str() below
		// stays valid, recording how many candidates belong to each entry.
		std::vector<size_t> candidateCounts;
		candidateCounts.reserve( originalCount );
		const std::vector<const char*> subDirs = preferredSubDirs( paths );

		const char* previousBase = nullptr;
		for ( const char** p = paths; *p != nullptr; ++p )
		{
			const size_t before = m_storage.size();

			// One run of candidates per library, emitted at the head of that library's run of
			// entries. Keying on the base name rather than on "is it bare" matters: in the hiprtc
			// list the bare name comes last, so hanging the candidates off it would let a hard coded
			// /opt/rocm guess outrank the SDK the caller explicitly pointed us at.
			const char* currentBase = baseName( *p );
			if ( previousBase == nullptr || std::strcmp( previousBase, currentBase ) != 0 )
			{
				addEnvCandidates( m_storage, "HIP_PATH", currentBase, subDirs );
				addEnvCandidates( m_storage, "ROCM_PATH", currentBase, subDirs );
			}
			previousBase = currentBase;

			candidateCounts.push_back( m_storage.size() - before );
		}

		// Second pass: emit in the original order, each library's candidates at the head of its own
		// run. Hoisting every candidate to the front instead would reorder the list against itself,
		// so that a late entry's candidate, say libamdhip64.so.5, would be tried before an early
		// one, say /opt/rocm/lib/libamdhip64.so, inverting the version preference.
		m_table.reserve( m_storage.size() + originalCount + 1 );
		size_t nextCandidate = 0;
		size_t index         = 0;
		for ( const char** p = paths; *p != nullptr; ++p, ++index )
		{
			for ( size_t i = 0; i < candidateCounts[index]; ++i )
				m_table.push_back( m_storage[nextCandidate++].c_str() );
			m_table.push_back( *p );
		}
		m_table.push_back( nullptr );
	}

	// Deliberately not const. oroInitialize takes const char**, and on a const object data() would
	// give const char* const*, so a const overload could only get there through a const_cast. Every
	// caller holds a non const PathTable, so leave the cast out and keep the accessor honest.
	[[nodiscard]] const char** get() { return m_table.empty() ? nullptr : m_table.data(); }

  private:
	std::vector<std::string> m_storage;
	std::vector<const char*> m_table;
};
} // namespace hiprt_libpath

// Prefer these over g_hip_paths / g_hiprtc_paths directly.
[[nodiscard]] inline const char** hiprtGetHipPaths()
{
	static hiprt_libpath::PathTable table( g_hip_paths );
	return table.get();
}

[[nodiscard]] inline const char** hiprtGetHiprtcPaths()
{
	static hiprt_libpath::PathTable table( g_hiprtc_paths );
	return table.get();
}
