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

namespace hiprt
{
HIPRT_DEVICE HIPRT_INLINE uint32_t expandBits2D( uint32_t v )
{
	v &= 0x0000ffff;					 /* w = ---- ---- ---- ---- fedc ba98 7654 3210 */
	v = ( v ^ ( v << 8 ) ) & 0x00ff00ff; /* w = ---- ---- fedc ba98 ---- ---- 7654 3210 */
	v = ( v ^ ( v << 4 ) ) & 0x0f0f0f0f; /* w = ---- fedc ---- ba98 ---- 7654 ---- 3210 */
	v = ( v ^ ( v << 2 ) ) & 0x33333333; /* w = --fe --dc --ba --98 --76 --54 --32 --10 */
	v = ( v ^ ( v << 1 ) ) & 0x55555555; /* w = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0 */
	return v;
}

HIPRT_DEVICE HIPRT_INLINE uint32_t expandBits3D( uint32_t v )
{
	v = ( v * 0x00010001u ) & 0xFF0000FFu;
	v = ( v * 0x00000101u ) & 0x0F00F00Fu;
	v = ( v * 0x00000011u ) & 0xC30C30C3u;
	v = ( v * 0x00000005u ) & 0x49249249u;
	return v;
}

HIPRT_DEVICE HIPRT_INLINE uint32_t computeMortonCode( float3 normalizedPos )
{
	float	 x	= min( max( normalizedPos.x * 1024.0f, 0.0f ), 1023.0f );
	float	 y	= min( max( normalizedPos.y * 1024.0f, 0.0f ), 1023.0f );
	float	 z	= min( max( normalizedPos.z * 1024.0f, 0.0f ), 1023.0f );
	uint32_t xx = expandBits3D( uint32_t( x ) );
	uint32_t yy = expandBits3D( uint32_t( y ) );
	uint32_t zz = expandBits3D( uint32_t( z ) );
	return xx * 4 + yy * 2 + zz;
}

HIPRT_DEVICE HIPRT_INLINE uint32_t computeExtendedMortonCode( float3 normalizedPos, float3 sceneExtent )
{
	const uint32_t numMortonBits = 30;
	int3		   numBits		 = make_int3( 0 );

	int3 numPrebits;
	int3 startAxis;

	// Find the largest start axis and how many prebits are needed between largest and two other axes
	if ( sceneExtent.x < sceneExtent.y )
	{
		if ( sceneExtent.x < sceneExtent.z )
		{
			if ( sceneExtent.y < sceneExtent.z )
			{
				// z, y, x
				startAxis.x	 = 2;
				numPrebits.x = log2( sceneExtent.z / sceneExtent.y );

				startAxis.y	 = 1;
				numPrebits.y = log2( sceneExtent.y / sceneExtent.x );

				startAxis.z	 = 0;
				numPrebits.z = log2( sceneExtent.z / sceneExtent.x );
			}
			else
			{
				// y, z, x
				startAxis.x	 = 1;
				numPrebits.x = log2( sceneExtent.y / sceneExtent.z );

				startAxis.y	 = 2;
				numPrebits.y = log2( sceneExtent.z / sceneExtent.x );

				startAxis.z	 = 0;
				numPrebits.z = log2( sceneExtent.y / sceneExtent.x );
			}
		}
		else
		{
			// y, x, z
			startAxis.x	 = 1;
			numPrebits.x = log2( sceneExtent.y / sceneExtent.x );

			startAxis.y	 = 0;
			numPrebits.y = log2( sceneExtent.x / sceneExtent.z );

			startAxis.z	 = 2;
			numPrebits.z = log2( sceneExtent.y / sceneExtent.z );
		}
	}
	else
	{
		if ( sceneExtent.y < sceneExtent.z )
		{
			if ( sceneExtent.x < sceneExtent.z )
			{
				// z, x, y
				startAxis.x	 = 2;
				numPrebits.x = log2( sceneExtent.z / sceneExtent.x );

				startAxis.y	 = 0;
				numPrebits.y = log2( sceneExtent.x / sceneExtent.y );

				startAxis.z	 = 1;
				numPrebits.z = log2( sceneExtent.z / sceneExtent.y );
			}
			else
			{
				// x, z, y
				startAxis.x	 = 0;
				numPrebits.x = log2( sceneExtent.x / sceneExtent.z );

				startAxis.y	 = 2;
				numPrebits.y = log2( sceneExtent.z / sceneExtent.y );

				startAxis.z	 = 1;
				numPrebits.z = log2( sceneExtent.x / sceneExtent.y );
			}
		}
		else
		{
			// x, y, z
			startAxis.x	 = 0;
			numPrebits.x = log2( sceneExtent.x / sceneExtent.y );

			startAxis.y	 = 1;
			numPrebits.y = log2( sceneExtent.y / sceneExtent.z );

			startAxis.z	 = 2;
			numPrebits.z = log2( sceneExtent.x / sceneExtent.z );
		}
	}

	// say x > y > z
	// prebits[0] = 3
	// prebits[1] = 2
	// if swap == 1
	// xxx xy xy x yxz yxz ...
	// if swap == 0
	// xxx xy xy xyz xyz ...
	int swap = numPrebits.z - ( numPrebits.x + numPrebits.y );

	numPrebits.x = min( numPrebits.x, numMortonBits );
	numPrebits.y = min( numPrebits.y * 2, numMortonBits - numPrebits.x ) / 2;

	int numPrebitsSum = numPrebits.x + numPrebits.y * 2;

	if ( numPrebitsSum != numMortonBits )
		numPrebitsSum += swap;
	else
		swap = 0;

	// The scene might be 2D so check for the smallest axis
	numBits.z = ( ptr( sceneExtent )[startAxis.z] != 0 ) ? max( 0, ( numMortonBits - numPrebitsSum ) / 3 ) : 0;

	if ( swap > 0 )
	{
		numBits.x = max( 0, ( numMortonBits - numBits.z - numPrebitsSum ) / 2 + numPrebits.y + numPrebits.x + 1 );
		numBits.y = numMortonBits - numBits.x - numBits.z;
	}
	else
	{
		numBits.y = max( 0, ( numMortonBits - numBits.z - numPrebitsSum ) / 2 + numPrebits.y );
		numBits.x = numMortonBits - numBits.y - numBits.z;
	}

	uint32_t mortonCode = 0;
	int3	 axisCode;

	// Based on the number of bits, calculate each code per axis
	axisCode.x =
		min( uint32_t( max( ptr( normalizedPos )[startAxis.x] * ( 1u << numBits.x ), 0.0f ) ), ( 1u << numBits.x ) - 1 );
	axisCode.y =
		min( uint32_t( max( ptr( normalizedPos )[startAxis.y] * ( 1u << numBits.y ), 0.0f ) ), ( 1u << numBits.y ) - 1 );
	axisCode.z =
		min( uint32_t( max( ptr( normalizedPos )[startAxis.z] * ( 1u << numBits.z ), 0.0f ) ), ( 1u << numBits.z ) - 1 );

	uint32_t delta0 = 0;
	uint32_t delta1 = 0;

	// if there are prebits, set them in the morton code:
	// if swap == 1
	// [xxx xy xy x] yxz yxz ...
	// if swap == 0
	// [xxx xy xy xyz] xyz ...
	if ( numPrebitsSum > 0 )
	{
		numBits.x -= numPrebits.x;
		mortonCode = axisCode.x & ( ( ( 1U << numPrebits.x ) - 1 ) << numBits.x );
		mortonCode >>= numBits.x;

		mortonCode <<= numPrebits.y * 2;
		numBits.x -= numPrebits.y;
		numBits.y -= numPrebits.y;
		uint32_t temp0 = axisCode.x & ( ( ( 1u << numPrebits.y ) - 1 ) << numBits.x );
		temp0 >>= numBits.x;
		temp0 = expandBits2D( temp0 );

		uint32_t temp1 = axisCode.y & ( ( ( 1u << numPrebits.y ) - 1 ) << numBits.y );
		temp1 >>= numBits.y;
		temp1 = expandBits2D( temp1 );

		mortonCode |= temp0 * 2 + temp1;

		if ( swap > 0 )
		{
			mortonCode <<= 1;
			numBits.x -= 1;
			uint32_t temp = axisCode.x & ( 1U << numBits.x );
			temp >>= numBits.x;
			mortonCode |= temp;
		}

		mortonCode <<= numBits.x + numBits.y + numBits.z;

		axisCode.x &= ( ( 1u << numBits.x ) - 1 );
		axisCode.y &= ( ( 1u << numBits.y ) - 1 );

		if ( swap > 0 )
		{
			delta0 = ( numBits.y - numBits.x );
			axisCode.x <<= delta0;

			delta1 = ( numBits.y - numBits.z );
			axisCode.z <<= delta1;
		}
		else
		{
			delta0 = ( numBits.x - numBits.y );
			axisCode.y <<= delta0;

			delta1 = ( numBits.x - numBits.z );
			axisCode.z <<= delta1;
		}
	}

	// 2D case, just use xy xy xy...
	if ( numBits.z == 0 )
	{
		axisCode.x = expandBits2D( axisCode.x );
		axisCode.y = expandBits2D( axisCode.y );
		mortonCode |= axisCode.x * 2 + axisCode.y;
	}
	else // 3D case, just use if swap == 0 xyz xyz xyz..., if swap == 1 yxz yxz yxz...
	{
		axisCode.x = ( axisCode.x > 0 ) ? expandBits3D( axisCode.x ) : 0;
		axisCode.y = ( axisCode.y > 0 ) ? expandBits3D( axisCode.y ) : 0;
		axisCode.z = ( axisCode.z > 0 ) ? expandBits3D( axisCode.z ) : 0;

		if ( swap > 0 )
			mortonCode |= ( axisCode.y * 4 + axisCode.x * 2 + axisCode.z ) >> ( delta0 + delta1 );
		else
			mortonCode |= ( axisCode.x * 4 + axisCode.y * 2 + axisCode.z ) >> ( delta0 + delta1 );
	}

	return mortonCode;
}

HIPRT_DEVICE uint64_t findHighestDifferentBit( int i, int j, int n, const uint32_t* sortedMortonCodeKeys )
{
	if ( j < 0 || j >= n ) return ~0ull;
	const uint64_t a = ( static_cast<uint64_t>( sortedMortonCodeKeys[i] ) << 32ull ) | i;
	const uint64_t b = ( static_cast<uint64_t>( sortedMortonCodeKeys[j] ) << 32ull ) | j;
	return a ^ b;
}
} // namespace hiprt