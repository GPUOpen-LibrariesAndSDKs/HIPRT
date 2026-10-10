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

// Host side tests for the library search path resolution in hiprt_libpath.h. No device needed.
//
// These exercise hiprt_libpath::PathTable rather than hiprtGetHipPaths(), because the accessors
// cache in a function local static and so read the environment exactly once per process, which is
// what we want at run time but leaves nothing to vary in a test. PathTable reads it on construction,
// so each case can set the environment and build a fresh one.

#include <gtest/gtest.h>
#include <hiprt/hiprt_libpath.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

namespace
{
void setEnvVar( const char* name, const char* value )
{
#if defined( _WIN32 )
	_putenv_s( name, value != nullptr ? value : "" );
#else
	if ( value != nullptr )
		setenv( name, value, 1 );
	else
		unsetenv( name );
#endif
}

// Starts each case from a known empty state and puts the environment back afterwards. Restoring
// matters: every hiprtT*.cpp lands in one unittest process, so leaving HIP_PATH or ROCM_PATH cleared
// would follow the other tests around, and on the platform this whole change is about those
// variables are the thing that finds the library at all.
class ScopedEnv
{
  public:
	ScopedEnv()
	{
		save( "HIP_PATH", m_hipPath, m_hadHipPath );
		save( "ROCM_PATH", m_rocmPath, m_hadRocmPath );
		clear();
	}

	~ScopedEnv()
	{
		setEnvVar( "HIP_PATH", m_hadHipPath ? m_hipPath.c_str() : nullptr );
		setEnvVar( "ROCM_PATH", m_hadRocmPath ? m_rocmPath.c_str() : nullptr );
	}

	ScopedEnv( const ScopedEnv& )            = delete;
	ScopedEnv& operator=( const ScopedEnv& ) = delete;

	void clear()
	{
		setEnvVar( "HIP_PATH", nullptr );
		setEnvVar( "ROCM_PATH", nullptr );
	}

  private:
	static void save( const char* name, std::string& value, bool& present )
	{
		const char* current = std::getenv( name );
		present             = current != nullptr;
		if ( present ) value = current;
	}

	std::string m_hipPath;
	std::string m_rocmPath;
	bool        m_hadHipPath  = false;
	bool        m_hadRocmPath = false;
};

std::vector<std::string> resolve( const char** paths )
{
	hiprt_libpath::PathTable table( paths );
	std::vector<std::string> out;
	for ( const char** p = table.get(); p != nullptr && *p != nullptr; ++p )
		out.push_back( *p );
	return out;
}

std::vector<std::string> original( const char** paths )
{
	std::vector<std::string> out;
	for ( const char** p = paths; p != nullptr && *p != nullptr; ++p )
		out.push_back( *p );
	return out;
}

// gtest 1.6 cannot print a std::vector<std::string>: under MSVC its universal printer picks an
// ambiguous operator<<. Compare a joined rendering instead, which also reads better on failure.
std::string join( const std::vector<std::string>& items )
{
	std::string out;
	for ( const std::string& item : items )
	{
		out += item;
		out += '\n';
	}
	return out;
}

// True when needle appears in haystack in order, i.e. haystack never reorders it.
bool isSubsequence( const std::vector<std::string>& needle, const std::vector<std::string>& haystack )
{
	size_t i = 0;
	for ( const std::string& h : haystack )
	{
		if ( i < needle.size() && h == needle[i] ) ++i;
	}
	return i == needle.size();
}

#if defined( _WIN32 )
constexpr char Sep      = '\\';
constexpr char OtherSep = '/';
#else
constexpr char Sep = '/';
constexpr char OtherSep = '\\';
#endif
} // namespace

// With neither variable set the resolved list must be exactly what was passed in. This is the
// property that keeps the change invisible to anyone not opting in.
TEST( LibPath, UnsetEnvironmentLeavesListsUntouched )
{
	ScopedEnv env;
	EXPECT_EQ( join( resolve( g_hip_paths ) ), join( original( g_hip_paths ) ) );
	EXPECT_EQ( join( resolve( g_hiprtc_paths ) ), join( original( g_hiprtc_paths ) ) );
}

TEST( LibPath, NullListResolvesToNull )
{
	ScopedEnv                env;
	hiprt_libpath::PathTable table( nullptr );
	EXPECT_TRUE( table.get() == nullptr );
}

// Every original entry must survive, in its original relative order.
TEST( LibPath, OriginalEntriesArePreservedInOrder )
{
	ScopedEnv env;
	setEnvVar( "ROCM_PATH", "/sdk/rocm" );

	for ( const char** paths : { g_hip_paths, g_hiprtc_paths } )
	{
		if ( paths == nullptr ) continue;
		const std::vector<std::string> resolved = resolve( paths );
		EXPECT_TRUE( isSubsequence( original( paths ), resolved ) ) << "the resolver reordered or dropped an original entry";
	}
}

// The candidates for a library have to come before every original entry naming that same library,
// not merely before the bare name. In g_hiprtc_paths the bare name is last, so hanging them off it
// would let a hard coded /opt/rocm entry outrank the SDK the caller pointed us at.
TEST( LibPath, SdkCandidatesPrecedeTheirLibrarysEntries )
{
	ScopedEnv env;
	setEnvVar( "ROCM_PATH", "/sdk/rocm" );

	for ( const char** paths : { g_hip_paths, g_hiprtc_paths } )
	{
		if ( paths == nullptr ) continue;
		const std::vector<std::string> resolved = resolve( paths );

		for ( size_t i = 0; i < resolved.size(); ++i )
		{
			if ( resolved[i].find( "/sdk/rocm" ) == std::string::npos &&
			     resolved[i].find( "\\sdk\\rocm" ) == std::string::npos )
				continue;

			const std::string base = hiprt_libpath::baseName( resolved[i].c_str() );

			// Nothing before this candidate may name the same library, other than another candidate.
			for ( size_t j = 0; j < i; ++j )
			{
				const bool sameLibrary   = ( base == hiprt_libpath::baseName( resolved[j].c_str() ) );
				const bool alsoCandidate = resolved[j].find( "sdk" ) != std::string::npos;
				EXPECT_TRUE( !sameLibrary || alsoCandidate )
					<< "'" << resolved[j] << "' precedes the SDK candidate '" << resolved[i] << "'";
			}
		}
	}
}

// One candidate per layout the hard coded list itself probes, so an install that only populates one
// of them is still reachable.
TEST( LibPath, EveryProbedLayoutGetsACandidate )
{
	ScopedEnv env;
	setEnvVar( "ROCM_PATH", "/sdk/rocm" );

	const std::vector<std::string> resolved = resolve( g_hiprtc_paths );
	if ( resolved.empty() ) return; // no library list on this platform

	size_t candidates = 0;
	for ( const std::string& entry : resolved )
		if ( entry.find( "sdk" ) != std::string::npos ) ++candidates;

	EXPECT_EQ( candidates, resolved.size() - original( g_hiprtc_paths ).size() );
	EXPECT_TRUE( candidates > 0 );
}

// A variable with a trailing separator, or one written with the other platform's separator, must
// still produce a well formed path: no doubled and no mixed separators.
TEST( LibPath, SeparatorsAreNormalisedAndNotDoubled )
{
	const char* roots[] = { "/sdk/rocm", "/sdk/rocm/", "\\sdk\\rocm", "\\sdk\\rocm\\" };

	for ( const char* root : roots )
	{
		ScopedEnv env;
		setEnvVar( "ROCM_PATH", root );

		for ( const std::string& entry : resolve( g_hip_paths ) )
		{
			if ( entry.find( "sdk" ) == std::string::npos ) continue;

			EXPECT_TRUE( entry.find( OtherSep ) == std::string::npos )
				<< "mixed separators in '" << entry << "' for root '" << root << "'";

			const std::string doubled = std::string( 2, Sep );
			EXPECT_TRUE( entry.find( doubled ) == std::string::npos )
				<< "doubled separator in '" << entry << "' for root '" << root << "'";
		}
	}
}

// An empty or separator only variable must be ignored rather than turned into a path rooted at /.
TEST( LibPath, DegenerateRootsAreIgnored )
{
	for ( const char* root : { "", "/", "\\" } )
	{
		ScopedEnv env;
		setEnvVar( "ROCM_PATH", root );
		EXPECT_EQ( join( resolve( g_hip_paths ) ), join( original( g_hip_paths ) ) )
			<< "root '" << root << "' should be ignored";
	}
}

// The helper has to leave the environment exactly as it found it, otherwise these cases quietly
// reconfigure every later test in the process.
TEST( LibPath, ScopedEnvRestoresWhatItFound )
{
	setEnvVar( "ROCM_PATH", "/sentinel/rocm" );
	setEnvVar( "HIP_PATH", nullptr );
	{
		ScopedEnv env;
		EXPECT_TRUE( std::getenv( "ROCM_PATH" ) == nullptr );
		setEnvVar( "ROCM_PATH", "/something/else" );
		setEnvVar( "HIP_PATH", "/also/set" );
	}
	const char* rocm = std::getenv( "ROCM_PATH" );
	ASSERT_TRUE( rocm != nullptr );
	EXPECT_STREQ( rocm, "/sentinel/rocm" );
	EXPECT_TRUE( std::getenv( "HIP_PATH" ) == nullptr );

	setEnvVar( "ROCM_PATH", nullptr );
}

TEST( LibPath, BaseNameHandlesBothSeparatorsAndEdgeCases )
{
	EXPECT_STREQ( hiprt_libpath::baseName( "libamdhip64.so" ), "libamdhip64.so" );
	EXPECT_STREQ( hiprt_libpath::baseName( "/opt/rocm/lib/libamdhip64.so" ), "libamdhip64.so" );
	EXPECT_STREQ( hiprt_libpath::baseName( "C:\\rocm\\bin\\amdhip64.dll" ), "amdhip64.dll" );
	EXPECT_STREQ( hiprt_libpath::baseName( "C:/rocm\\bin/amdhip64.dll" ), "amdhip64.dll" );
	EXPECT_STREQ( hiprt_libpath::baseName( "trailing/" ), "" );
	EXPECT_STREQ( hiprt_libpath::baseName( "" ), "" );
}

// HIP_PATH and ROCM_PATH both contribute, HIP_PATH first.
// The candidates must probe layouts in the order the list itself does. g_hip_paths tries lib before
// hip/lib, g_hiprtc_paths the other way round, so one fixed order would contradict one of them.
TEST( LibPath, CandidateLayoutOrderMirrorsTheList )
{
	ScopedEnv env;
	setEnvVar( "ROCM_PATH", "/sdk/rocm" );

	for ( const char** paths : { g_hip_paths, g_hiprtc_paths } )
	{
		if ( paths == nullptr ) continue;

		// The order the hard coded entries themselves establish.
		std::vector<std::string> listOrder;
		for ( const char** p = paths; *p != nullptr; ++p )
		{
			const char* subDir = hiprt_libpath::matchedSubDir( *p );
			if ( subDir == nullptr ) continue;
			if ( std::find( listOrder.begin(), listOrder.end(), subDir ) == listOrder.end() ) listOrder.push_back( subDir );
		}
		if ( listOrder.size() < 2 ) continue; // single layout, nothing to order

		// The order the candidates for the first library use.
		const std::vector<std::string> resolved = resolve( paths );
		std::vector<std::string>       candidateOrder;
		std::string                    firstBase;
		for ( const std::string& entry : resolved )
		{
			if ( entry.find( "sdk" ) == std::string::npos ) continue;
			const std::string base = hiprt_libpath::baseName( entry.c_str() );
			if ( firstBase.empty() ) firstBase = base;
			if ( base != firstBase ) break;
			const char* subDir = hiprt_libpath::matchedSubDir( entry.c_str() );
			if ( subDir != nullptr ) candidateOrder.push_back( subDir );
		}

		EXPECT_EQ( join( candidateOrder ), join( listOrder ) ) << "candidate layout order contradicts the list";
	}
}

TEST( LibPath, BothVariablesContributeWithHipPathFirst )
{
	ScopedEnv env;
	setEnvVar( "HIP_PATH", "/sdk/hip" );
	setEnvVar( "ROCM_PATH", "/sdk/rocm" );

	const std::vector<std::string> resolved = resolve( g_hip_paths );
	if ( resolved.empty() ) return; // no library list on this platform

	size_t firstHip = std::string::npos, firstRocm = std::string::npos;
	for ( size_t i = 0; i < resolved.size(); ++i )
	{
		const bool hip  = resolved[i].find( "hip" ) != std::string::npos && resolved[i].find( "sdk" ) != std::string::npos;
		const bool rocm = resolved[i].find( "rocm" ) != std::string::npos && resolved[i].find( "sdk" ) != std::string::npos;
		if ( hip && firstHip == std::string::npos ) firstHip = i;
		if ( rocm && firstRocm == std::string::npos ) firstRocm = i;
	}

	ASSERT_TRUE( firstHip != std::string::npos );
	ASSERT_TRUE( firstRocm != std::string::npos );
	EXPECT_LT( firstHip, firstRocm );
}
