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
#include <hiprt/hiprt_common.h>
using namespace hiprt;

HIPRT_DEVICE Aabb shflAabb( const Aabb& box, uint32_t srcLane )
{
	Aabb b;
	b.m_min.x = __shfl( box.m_min.x, srcLane );
	b.m_min.y = __shfl( box.m_min.y, srcLane );
	b.m_min.z = __shfl( box.m_min.z, srcLane );
	b.m_max.x = __shfl( box.m_max.x, srcLane );
	b.m_max.y = __shfl( box.m_max.y, srcLane );
	b.m_max.z = __shfl( box.m_max.z, srcLane );
	return b;
}

template <typename T>
HIPRT_DEVICE T warpMin( T warpVal )
{
	T warpValue = __shfl_xor( warpVal, 1 );
	warpVal		= min( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 2 );
	warpVal		= min( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 4 );
	warpVal		= min( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 8 );
	warpVal		= min( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 16 );
	warpVal		= min( warpVal, warpValue );
	if constexpr ( WarpSize == 64 )
	{
		warpValue = __shfl_xor( warpVal, 32 );
		warpVal	  = min( warpVal, warpValue );
	}
	warpVal = __shfl( warpVal, WarpSize - 1 );
	return warpVal;
}

template <typename T>
HIPRT_DEVICE T warpMax( T warpVal )
{
	T warpValue = __shfl_xor( warpVal, 1 );
	warpVal		= max( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 2 );
	warpVal		= max( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 4 );
	warpVal		= max( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 8 );
	warpVal		= max( warpVal, warpValue );
	warpValue	= __shfl_xor( warpVal, 16 );
	warpVal		= max( warpVal, warpValue );
	if constexpr ( WarpSize == 64 )
	{
		warpValue = __shfl_xor( warpVal, 32 );
		warpVal	  = max( warpVal, warpValue );
	}
	warpVal = __shfl( warpVal, WarpSize - 1 );
	return warpVal;
}

template <typename T>
HIPRT_DEVICE T warpSum( T warpVal )
{
	T warpValue = __shfl_xor( warpVal, 1 );
	warpVal += warpValue;
	warpValue = __shfl_xor( warpVal, 2 );
	warpVal += warpValue;
	warpValue = __shfl_xor( warpVal, 4 );
	warpVal += warpValue;
	warpValue = __shfl_xor( warpVal, 8 );
	warpVal += warpValue;
	warpValue = __shfl_xor( warpVal, 16 );
	warpVal += warpValue;
	if constexpr ( WarpSize == 64 )
	{
		warpValue = __shfl_xor( warpVal, 32 );
		warpVal += warpValue;
	}
	warpVal = __shfl( warpVal, WarpSize - 1 );
	return warpVal;
}

HIPRT_DEVICE Aabb warpUnion( Aabb warpVal )
{
	const uint32_t laneIndex = threadIdx.x & ( WarpSize - 1 );
	Aabb		   warpValue = shflAabb( warpVal, laneIndex ^ 1 );
	warpVal.grow( warpValue );
	warpValue = shflAabb( warpVal, laneIndex ^ 2 );
	warpVal.grow( warpValue );
	warpValue = shflAabb( warpVal, laneIndex ^ 4 );
	warpVal.grow( warpValue );
	warpValue = shflAabb( warpVal, laneIndex ^ 8 );
	warpVal.grow( warpValue );
	warpValue = shflAabb( warpVal, laneIndex ^ 16 );
	warpVal.grow( warpValue );
	if constexpr ( WarpSize == 64 )
	{
		warpValue = shflAabb( warpVal, laneIndex ^ 32 );
		warpVal.grow( warpValue );
	}
	warpVal = shflAabb( warpVal, WarpSize - 1 );
	return warpVal;
}

template <typename T>
HIPRT_DEVICE T warpScan( T warpVal )
{
	const uint32_t laneIndex = threadIdx.x & ( WarpSize - 1 );
	T			   warpValue = __shfl_up( warpVal, 1 );
	if ( laneIndex >= 1 ) warpVal += warpValue;
	warpValue = __shfl_up( warpVal, 2 );
	if ( laneIndex >= 2 ) warpVal += warpValue;
	warpValue = __shfl_up( warpVal, 4 );
	if ( laneIndex >= 4 ) warpVal += warpValue;
	warpValue = __shfl_up( warpVal, 8 );
	if ( laneIndex >= 8 ) warpVal += warpValue;
	warpValue = __shfl_up( warpVal, 16 );
	if ( laneIndex >= 16 ) warpVal += warpValue;
	if constexpr ( WarpSize == 64 )
	{
		warpValue = __shfl_up( warpVal, 32 );
		if ( laneIndex >= 32 ) warpVal += warpValue;
	}
	return warpVal;
}

template <typename T>
HIPRT_DEVICE T warpOffset( T warpVal, T* counter )
{
	const uint32_t laneIndex  = threadIdx.x & ( WarpSize - 1 );
	T			   warpSum	  = warpScan( warpVal );
	T			   warpOffset = static_cast<T>( 0 );
	if ( laneIndex == WarpSize - 1 ) warpOffset = atomicAdd( counter, warpSum );
	warpSum -= warpVal;
	warpOffset = __shfl( warpOffset, WarpSize - 1 );
	return warpOffset + warpSum;
}

template <typename T>
HIPRT_DEVICE T warpOffset( bool warpVal, T* counter )
{
	const uint32_t laneIndex  = threadIdx.x & ( WarpSize - 1 );
	const uint64_t warpBallot = __ballot( warpVal );
	const T		   warpCount  = __popcll( warpBallot );
	const T		   warpSum	  = __popcll( warpBallot & ( ( 1ull << laneIndex ) - 1ull ) );
	T			   warpOffset;
	if ( laneIndex == __ffsll( static_cast<unsigned long long>( warpBallot ) ) - 1 )
		warpOffset = atomicAdd( counter, warpCount );
	warpOffset = __shfl( warpOffset, __ffsll( static_cast<unsigned long long>( warpBallot ) ) - 1 );
	return warpOffset + warpSum;
}

template <typename T>
HIPRT_DEVICE T blockMin( T blockVal, T* blockCache )
{
	const uint32_t laneIndex	 = threadIdx.x & ( WarpSize - 1 );
	const uint32_t warpIndex	 = threadIdx.x >> Log2( WarpSize );
	const uint32_t warpsPerBlock = DivideRoundUp( static_cast<uint32_t>( blockDim.x ), WarpSize );

	blockVal = warpMin( blockVal );
	if ( laneIndex == 0 ) blockCache[warpIndex] = blockVal;

	for ( uint32_t i = 1; i < warpsPerBlock; i <<= 1 )
	{
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock )
			blockCache[threadIdx.x] = min( blockCache[threadIdx.x], blockCache[threadIdx.x ^ i] );
	}
	__syncthreads();
	return blockCache[0];
}

template <typename T>
HIPRT_DEVICE T blockMax( T blockVal, T* blockCache )
{
	const uint32_t laneIndex	 = threadIdx.x & ( WarpSize - 1 );
	const uint32_t warpIndex	 = threadIdx.x >> Log2( WarpSize );
	const uint32_t warpsPerBlock = DivideRoundUp( static_cast<uint32_t>( blockDim.x ), WarpSize );

	blockVal = warpMax( blockVal );
	if ( laneIndex == 0 ) blockCache[warpIndex] = blockVal;

	for ( uint32_t i = 1; i < warpsPerBlock; i <<= 1 )
	{
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock )
			blockCache[threadIdx.x] = max( blockCache[threadIdx.x], blockCache[threadIdx.x ^ i] );
	}
	__syncthreads();
	return blockCache[0];
}

template <typename T>
HIPRT_DEVICE T blockSum( T blockVal, T* blockCache )
{
	const uint32_t laneIndex	 = threadIdx.x & ( WarpSize - 1 );
	const uint32_t warpIndex	 = threadIdx.x >> Log2( WarpSize );
	const uint32_t warpsPerBlock = DivideRoundUp( static_cast<uint32_t>( blockDim.x ), WarpSize );

	blockVal = warpSum( blockVal );
	if ( laneIndex == 0 ) blockCache[warpIndex] = blockVal;

	for ( uint32_t i = 1; i < warpsPerBlock; i <<= 1 )
	{
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock ) blockCache[threadIdx.x] += blockCache[threadIdx.x ^ i];
	}
	__syncthreads();
	return blockCache[0];
}

HIPRT_DEVICE Aabb blockUnion( Aabb blockVal, Aabb* blockCache )
{
	const uint32_t laneIndex	 = threadIdx.x & ( WarpSize - 1 );
	const uint32_t warpIndex	 = threadIdx.x >> Log2( WarpSize );
	const uint32_t warpsPerBlock = DivideRoundUp( static_cast<uint32_t>( blockDim.x ), WarpSize );

	blockVal = warpUnion( blockVal );
	if ( laneIndex == 0 ) blockCache[warpIndex] = blockVal;

	for ( uint32_t i = 1; i < warpsPerBlock; i <<= 1 )
	{
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock ) blockCache[threadIdx.x].grow( blockCache[threadIdx.x ^ i] );
	}
	__syncthreads();
	return blockCache[0];
}

template <typename T>
HIPRT_DEVICE T blockScan( T blockVal, T* blockCache )
{
	const uint32_t laneIndex	 = threadIdx.x & ( WarpSize - 1 );
	const uint32_t warpIndex	 = threadIdx.x >> Log2( WarpSize );
	const uint32_t warpsPerBlock = DivideRoundUp( static_cast<uint32_t>( blockDim.x ), WarpSize );

	T blockValue = blockVal;
	T warpSum	 = warpScan( blockValue );

	if ( laneIndex == WarpSize - 1 ) blockCache[warpIndex] = warpSum;

	__syncthreads();
	if ( threadIdx.x < warpsPerBlock ) blockValue = blockCache[threadIdx.x];

	for ( uint32_t i = 1; i < warpsPerBlock; i <<= 1 )
	{
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock && threadIdx.x >= i ) blockValue += blockCache[threadIdx.x - i];
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock ) blockCache[threadIdx.x] = blockValue;
	}

	__syncthreads();
	if ( laneIndex == WarpSize - 1 ) blockCache[warpIndex] -= warpSum;

	__syncthreads();
	return blockCache[warpIndex] + warpSum;
}

template <typename T>
HIPRT_DEVICE T blockScan( bool blockVal, T* blockCache )
{
	const uint32_t laneIndex	 = threadIdx.x & ( WarpSize - 1 );
	const uint32_t warpIndex	 = threadIdx.x >> Log2( WarpSize );
	const uint32_t warpsPerBlock = DivideRoundUp( static_cast<uint32_t>( blockDim.x ), WarpSize );

	T			   blockValue = blockVal;
	const uint64_t warpBallot = __ballot( blockVal );
	const T		   warpCount  = __popcll( warpBallot );
	const T		   warpSum	  = __popcll( warpBallot & ( ( 1ull << laneIndex ) - 1ull ) );

	if ( laneIndex == 0 ) blockCache[warpIndex] = warpCount;

	__syncthreads();
	if ( threadIdx.x < warpsPerBlock ) blockValue = blockCache[threadIdx.x];

	for ( uint32_t i = 1; i < warpsPerBlock; i <<= 1 )
	{
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock && threadIdx.x >= i ) blockValue += blockCache[threadIdx.x - i];
		__syncthreads();
		if ( threadIdx.x < warpsPerBlock ) blockCache[threadIdx.x] = blockValue;
	}

	__syncthreads();
	return blockCache[warpIndex] + warpSum - warpCount + static_cast<T>( blockVal );
}

HIPRT_DEVICE HIPRT_INLINE void SyncWarp()
{
#if defined( __CUDACC__ )
	__syncwarp();
#endif
}
