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

#if ( defined( __CUDACC__ ) || defined( __HIPCC__ ) )
#define __KERNELCC__
#endif

#if ( defined( __CUDACC_RTC__ ) || defined( __HIPCC_RTC__ ) )
#define __KERNELCC_RTC__
#endif

#if !defined( __KERNELCC__ )
#include <algorithm>
#include <cfloat>
#include <cstring>
#include <cmath>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <type_traits>
#define __host__
#define __device__
#endif

#if !defined( __KERNELCC_RTC__ )
#include <cstdint>
#endif

#if !defined( __KERNELCC__ )
#if defined( _MSC_VER )
#define HIPRT_ASSERT( cond ) \
	if ( !( cond ) )         \
	{                        \
		__debugbreak();      \
	}
#elif defined( __GNUC__ )
#include <signal.h>
#define HIPRT_ASSERT( cond ) \
	if ( !( cond ) )         \
	{                        \
		raise( SIGTRAP );    \
	}
#else
#define HIPRT_ASSERT( cond )
#endif
#else
#define HIPRT_ASSERT( cond )
#endif

#define HIPRT_STATIC_ASSERT( cond ) static_assert( ( cond ), "" )

#ifdef __KERNELCC__
#define HIPRT_INLINE __forceinline__
#define HIPRT_CONST __constant__
#else
#define HIPRT_INLINE inline
#define HIPRT_CONST const
#endif

#define HIPRT_HOST __host__
#define HIPRT_DEVICE __device__
#define HIPRT_HOST_DEVICE __host__ __device__

#if defined( HIPRT_BAKE_KERNEL_GENERATED )
#define GET_ARGS( X ) ( hip::X##Args )
#define GET_INC( X ) ( hip::X##Includes )
#else
#define GET_ARGS( X ) static_cast<const char**>( nullptr )
#define GET_INC( X ) static_cast<const char**>( nullptr )
#endif

#if defined( HIPRT_LOAD_FROM_STRING )
#define GET_ARG_LIST( X ) sizeof( GET_ARGS( X ) ) / sizeof( void* ), GET_ARGS( X ), GET_INC( X )
#else
#define GET_ARG_LIST( X ) 0, 0, 0
#endif

#if defined( __KERNELCC_RTC__ )
#if defined( __CUDACC_RTC__ ) || HIP_VERSION_MAJOR < 7
using int8_t   = char;
using uint8_t  = unsigned char;
using int16_t  = short;
using uint16_t = unsigned short;
using int32_t  = int;
using uint32_t = unsigned int;
using int64_t  = long long;
using uint64_t = unsigned long long;
#else
using int8_t					   = __hip_internal::int8_t;
using uint8_t					   = __hip_internal::uint8_t;
using int16_t					   = __hip_internal::int16_t;
using uint16_t					   = __hip_internal::uint16_t;
using int32_t					   = __hip_internal::int32_t;
using uint32_t					   = __hip_internal::uint32_t;
using int64_t					   = __hip_internal::int64_t;
using uint64_t					   = __hip_internal::uint64_t;
#endif
#endif

HIPRT_STATIC_ASSERT( sizeof( int8_t ) == 1 );
HIPRT_STATIC_ASSERT( sizeof( int16_t ) == 2 );
HIPRT_STATIC_ASSERT( sizeof( int32_t ) == 4 );
HIPRT_STATIC_ASSERT( sizeof( int64_t ) == 8 );
HIPRT_STATIC_ASSERT( sizeof( uint8_t ) == 1 );
HIPRT_STATIC_ASSERT( sizeof( uint16_t ) == 2 );
HIPRT_STATIC_ASSERT( sizeof( uint32_t ) == 4 );
HIPRT_STATIC_ASSERT( sizeof( uint64_t ) == 8 );
HIPRT_STATIC_ASSERT( alignof( int8_t ) == 1 );
HIPRT_STATIC_ASSERT( alignof( int16_t ) == 2 );
HIPRT_STATIC_ASSERT( alignof( int32_t ) == 4 );
HIPRT_STATIC_ASSERT( alignof( int64_t ) == 8 );
HIPRT_STATIC_ASSERT( alignof( uint8_t ) == 1 );
HIPRT_STATIC_ASSERT( alignof( uint16_t ) == 2 );
HIPRT_STATIC_ASSERT( alignof( uint32_t ) == 4 );
HIPRT_STATIC_ASSERT( alignof( uint64_t ) == 8 );

HIPRT_STATIC_ASSERT( sizeof( double ) == 8 );
HIPRT_STATIC_ASSERT( sizeof( float ) == 4 );
HIPRT_STATIC_ASSERT( alignof( double ) == 8 );
HIPRT_STATIC_ASSERT( alignof( float ) == 4 );

HIPRT_STATIC_ASSERT( sizeof( unsigned long long int ) == 8 );
HIPRT_STATIC_ASSERT( sizeof( long long int ) == 8 );
HIPRT_STATIC_ASSERT( sizeof( unsigned int ) == 4 );
HIPRT_STATIC_ASSERT( sizeof( int ) == 4 );
HIPRT_STATIC_ASSERT( sizeof( unsigned short int ) == 2 );
HIPRT_STATIC_ASSERT( sizeof( short int ) == 2 );
HIPRT_STATIC_ASSERT( sizeof( unsigned char ) == 1 );
HIPRT_STATIC_ASSERT( sizeof( char ) == 1 );
HIPRT_STATIC_ASSERT( alignof( unsigned long long int ) == 8 );
HIPRT_STATIC_ASSERT( alignof( long long int ) == 8 );
HIPRT_STATIC_ASSERT( alignof( unsigned int ) == 4 );
HIPRT_STATIC_ASSERT( alignof( int ) == 4 );
HIPRT_STATIC_ASSERT( alignof( unsigned short int ) == 2 );
HIPRT_STATIC_ASSERT( alignof( short int ) == 2 );
HIPRT_STATIC_ASSERT( alignof( unsigned char ) == 1 );
HIPRT_STATIC_ASSERT( alignof( char ) == 1 );

namespace hiprt
{
constexpr float Pi	   = 3.14159265358979323846f;
constexpr float TwoPi  = 2.0f * Pi;
constexpr float FltMin = 1.175494351e-38f;
constexpr float FltMax = 3.402823466e+38f;
constexpr int	IntMin = -2147483647 - 1;
constexpr int	IntMax = 2147483647;

constexpr uint32_t InvalidValue				 = ~0u;
constexpr uint32_t FullRayMask				 = ~0u;
constexpr uint32_t MaxBatchBuildMaxPrimCount = 512u;
constexpr uint32_t MaxInstanceLevels		 = 4u;
constexpr uint32_t DefaultAlignment			 = 64u;
constexpr uint32_t NoRotationIndex			 = 127u;
constexpr uint32_t InstanceIDBits			 = 24u;

#ifdef __KERNELCC__
#ifndef HIPRT_RTIP
#if __gfx1200__ || __gfx1201__
// to enable full hardware support, we need either the Rocm 6.4+ on Windows, or the Rocm 7+ on Linux
#if ( HIP_VERSION_MAJOR >= 7 ) || \
	( defined( HIPCC_OS_WINDOWS ) && ( ( HIP_VERSION_MAJOR > 6 ) || ( HIP_VERSION_MAJOR == 6 && HIP_VERSION_MINOR >= 4 ) ) )
#define HIPRT_RTIP 31
#else
#define HIPRT_RTIP 20
#warning \
	"HW supports RTIP 3.1 but the compiler is of an older version; build with ROCm 6.4+ (Win) or 7.0+ (Linux) to fully utilize HW ray tracing features"
#endif
#elif __gfx1100__ || __gfx1101__ || __gfx1102__ || __gfx1103__ || __gfx1150__ || __gfx1151__ || __gfx1152__ || __gfx1153__
#define HIPRT_RTIP 20
#elif __gfx1030__ || __gfx1031__ || __gfx1032__ || __gfx1033__ || __gfx1034__ || __gfx1035__ || __gfx1036__
#define HIPRT_RTIP 11
#else
#define HIPRT_RTIP 0
#endif
#endif

#if __gfx900__ || __gfx902__ || __gfx904__ || __gfx906__ || __gfx908__ || __gfx909__ || __gfx90a__ || __gfx90c__ || \
	__gfx940__ || __gfx941__ || __gfx942__
constexpr uint32_t WarpSize = 64;
#else
constexpr uint32_t WarpSize		   = 32;
#endif

constexpr uint32_t Rtip = HIPRT_RTIP;

#if HIPRT_RTIP >= 31
constexpr uint32_t BranchingFactor = 8;
#else
constexpr uint32_t BranchingFactor = 4;
#endif
#endif

HIPRT_HOST_DEVICE HIPRT_INLINE float as_float( uint32_t value )
{
#if defined( __KERNELCC__ )
	return __uint_as_float( value );
#else
	return *reinterpret_cast<float*>( &value );
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE int as_int( float value )
{
#if defined( __KERNELCC__ )
	return __float_as_int( value );
#else
	return *reinterpret_cast<int*>( &value );
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t as_uint( float value )
{
#if defined( __KERNELCC__ )
	return __float_as_uint( value );
#else
	return *reinterpret_cast<uint32_t*>( &value );
#endif
}

HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t clz( uint32_t value )
{
#if defined( __KERNELCC__ )
	return __clz( value );
#else
	uint32_t count = 0;
	for ( uint32_t mask = 1u << 31; mask && !( value & mask ); mask >>= 1 )
		++count;
	return value == 0 ? 32 : count;
#endif
}

#ifdef __KERNELCC__
template <typename T>
HIPRT_INLINE HIPRT_DEVICE T shfl( T var, int srcLane )
{
#ifdef __CUDACC__
	return __shfl_sync( __activemask(), var, srcLane );
#else
	return __shfl( var, srcLane );
#endif
}

template <typename T>
HIPRT_INLINE HIPRT_DEVICE T shfl_up( T var, int srcLane )
{
#ifdef __CUDACC__
	return __shfl_up_sync( __activemask(), var, srcLane );
#else
	return __shfl_up( var, srcLane );
#endif
}

template <typename T>
HIPRT_INLINE HIPRT_DEVICE T shfl_down( T var, int srcLane )
{
#ifdef __CUDACC__
	return __shfl_down_sync( __activemask(), var, srcLane );
#else
	return __shfl_down( var, srcLane );
#endif
}

template <typename T>
HIPRT_INLINE HIPRT_DEVICE T shfl_xor( T var, int srcLane )
{
#ifdef __CUDACC__
	return __shfl_xor_sync( __activemask(), var, srcLane );
#else
	return __shfl_xor( var, srcLane );
#endif
}

HIPRT_INLINE HIPRT_DEVICE uint64_t ballot( int predicate )
{
#ifdef __CUDACC__
	return static_cast<uint64_t>( __ballot_sync( __activemask(), predicate ) );
#else
	return __ballot( predicate );
#endif
}

HIPRT_INLINE HIPRT_DEVICE uint32_t any( int predicate )
{
#ifdef __CUDACC__
	return __any_sync( __activemask(), predicate );
#else
	return __any( predicate );
#endif
}
HIPRT_INLINE HIPRT_DEVICE uint32_t all( int predicate )
{
#ifdef __CUDACC__
	return __all_sync( __activemask(), predicate );
#else
	return __all( predicate );
#endif
}
#endif

template <typename T, typename U>
constexpr HIPRT_HOST_DEVICE T RoundUp( T value, U factor )
{
	return ( value + factor - 1 ) / factor * factor;
}

template <typename T, typename U>
constexpr HIPRT_HOST_DEVICE T DivideRoundUp( T value, U factor )
{
	return ( value + factor - 1 ) / factor;
}

template <typename T>
constexpr HIPRT_HOST_DEVICE T Log2( T n )
{
	return n <= 1 ? 0 : 1 + Log2( ( n + 1 ) / 2 );
}

#if !defined( __KERNELCC_RTC__ ) || defined( __CUDACC_RTC__ ) || HIP_VERSION_MAJOR < 7
template <class T, class U>
struct is_same
{
	enum
	{
		value = 0
	};
};

template <class T>
struct is_same<T, T>
{
	enum
	{
		value = 1
	};
};

template <bool B, class T, class F>
struct conditional
{
	using type = T;
};

template <class T, class F>
struct conditional<false, T, F>
{
	using type = F;
};
#else
template <class T, class U>
using is_same = __hip_internal::is_same<T, U>;
template <bool B, class T, class F>
using conditional = __hip_internal::conditional<B, T, F>;
#endif

template <class T>
struct remove_reference
{
	using type = T;
};

template <class T>
struct remove_reference<T&>
{
	using type = T;
};

template <class T>
struct remove_reference<T&&>
{
	using type = T;
};

template <class Ty, Ty Val>
struct integral_constant
{
	static constexpr Ty value = Val;
	using value_type		  = Ty;
	using type				  = integral_constant;

	HIPRT_DEVICE constexpr							operator value_type() const noexcept { return value; }
	[[nodiscard]] HIPRT_DEVICE constexpr value_type operator()() const noexcept { return value; }
};

template <bool _Val>
using bool_constant = integral_constant<bool, _Val>;

using true_type	 = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_lvalue_reference : false_type
{
};

template <class T>
struct is_lvalue_reference<T&> : true_type
{
};

template <class T>
HIPRT_DEVICE constexpr typename remove_reference<T>::type&& move( T&& t ) noexcept
{
	return static_cast<typename remove_reference<T>::type&&>( t );
}

template <class T>
HIPRT_DEVICE constexpr T&& forward( typename remove_reference<T>::type& t ) noexcept
{
	return static_cast<T&&>( t );
}

template <class T>
HIPRT_DEVICE constexpr T&& forward( typename remove_reference<T>::type&& t ) noexcept
{
	HIPRT_STATIC_ASSERT( !is_lvalue_reference<T>::value );
	return static_cast<T&&>( t );
}

template <class T>
struct alignment_of : integral_constant<size_t, alignof( T )>
{
};

template <size_t Size, uint32_t Align>
struct aligned_storage
{
	struct type
	{
		alignas( Align ) uint8_t data[Size];
	};
};

#if !defined( __KERNELCC__ )
template <typename T>
struct Traits
{
	static const std::string TYPE_NAME;
};

#define DECLARE_TYPE_TRAITS( name ) \
	template <>                     \
	const std::string Traits<name>::TYPE_NAME = #name;
#endif
} // namespace hiprt

template <typename T, size_t Size, uint32_t Align>
class hiprtPimpl
{
	typename hiprt::aligned_storage<Size, Align>::type data;

  public:
	template <size_t T_size>
	HIPRT_DEVICE static constexpr void PimplSizeCheck()
	{
		HIPRT_STATIC_ASSERT( T_size == Size );
	};

	HIPRT_DEVICE static constexpr void PimplPtrCheck()
	{
		PimplSizeCheck<sizeof( T )>();
		HIPRT_STATIC_ASSERT( alignof( T ) >= hiprt::alignment_of<T>::value );
	}

	template <typename... Args>
	HIPRT_DEVICE hiprtPimpl( Args&&... args )
	{
		PimplPtrCheck();
		new ( &data ) T( hiprt::forward<Args>( args )... );
	}

	HIPRT_DEVICE hiprtPimpl( hiprtPimpl const& o )
	{
		PimplPtrCheck();
		new ( &data ) T( *o );
	}

	HIPRT_DEVICE hiprtPimpl( hiprtPimpl& o )
	{
		PimplPtrCheck();
		new ( &data ) T( *o );
	}

	HIPRT_DEVICE hiprtPimpl( hiprtPimpl&& o )
	{
		PimplPtrCheck();
		new ( &data ) T( hiprt::move( *o ) );
	}

	HIPRT_DEVICE ~hiprtPimpl() {}

	HIPRT_DEVICE hiprtPimpl& operator=( hiprtPimpl const& o )
	{
		PimplPtrCheck();
		**this = *o;
		return *this;
	}

	HIPRT_DEVICE hiprtPimpl& operator=( hiprtPimpl&& o )
	{
		PimplPtrCheck();
		**this = hiprt::move( *o );
		return *this;
	}

	HIPRT_DEVICE T& operator*()
	{
		PimplPtrCheck();
		return *reinterpret_cast<T*>( &data );
	}

	HIPRT_DEVICE T const& operator*() const
	{
		PimplPtrCheck();
		return *reinterpret_cast<T const*>( &data );
	}

	HIPRT_DEVICE T* operator->()
	{
		PimplPtrCheck();
		return &**this;
	}

	HIPRT_DEVICE T const* operator->() const
	{
		PimplPtrCheck();
		return &**this;
	}
};

enum TraversalObjSize
{
	SizePrivateStack			   = 260,
	SizeGlobalStack				   = 48,
	SizePrivateInstanceStack	   = 160,
	SizeGlobalInstanceStack		   = 48,
	SizeGeomTraversalCustomStack   = 128,
	SizeSceneTraversalCustomStack  = 176,
	SizeGeomTraversalPrivateStack  = 400,
	SizeSceneTraversalPrivateStack = 608,
};

enum TraversalObjAlignment
{
	AlignmentPrivateStack				= 4,
	AlignmentGlobalStack				= 8,
	AlignmentPrivateInstanceStack		= 16,
	AlignmentGlobalInstanceStack		= 8,
	AlignmentGeomTraversalCustomStack	= 16,
	AlignmentSceneTraversalCustomStack	= 16,
	AlignmentGeomTraversalPrivateStack	= 16,
	AlignmentSceneTraversalPrivateStack = 16
};
