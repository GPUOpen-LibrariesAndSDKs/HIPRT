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
#include <hiprt/hiprt_math.h>
#include <hiprt/impl/Transform.h>

namespace hiprt
{
static constexpr uint32_t RotationCount = 88;

HIPRT_CONST static uint32_t EncodedRotations[RotationCount][9] = {
	{ 25, 0, 0, 0, 22, 43, 0, 11, 22 },		{ 25, 0, 0, 0, 22, 11, 0, 43, 22 },		{ 25, 0, 0, 0, 17, 49, 0, 17, 17 },
	{ 25, 0, 0, 0, 0, 57, 0, 25, 0 },		{ 22, 0, 11, 0, 25, 0, 43, 0, 22 },		{ 22, 0, 43, 0, 25, 0, 11, 0, 22 },
	{ 17, 0, 17, 0, 25, 0, 49, 0, 17 },		{ 22, 43, 0, 11, 22, 0, 0, 0, 25 },		{ 22, 11, 0, 43, 22, 0, 0, 0, 25 },
	{ 17, 49, 0, 17, 17, 0, 0, 0, 25 },		{ 22, 38, 6, 6, 24, 1, 38, 1, 24 },		{ 22, 6, 38, 38, 24, 1, 6, 1, 24 },
	{ 17, 44, 12, 12, 20, 2, 44, 2, 20 },	{ 17, 12, 44, 44, 20, 2, 12, 2, 20 },	{ 11, 47, 15, 15, 16, 7, 47, 7, 16 },
	{ 11, 15, 47, 47, 16, 7, 15, 7, 16 },	{ 0, 49, 17, 17, 12, 12, 49, 12, 12 },	{ 0, 17, 49, 49, 12, 12, 17, 12, 12 },
	{ 22, 38, 38, 6, 24, 33, 6, 33, 24 },	{ 22, 6, 6, 38, 24, 33, 38, 33, 24 },	{ 17, 44, 44, 12, 20, 34, 12, 34, 20 },
	{ 17, 12, 12, 44, 20, 34, 44, 34, 20 }, { 11, 47, 47, 15, 16, 39, 15, 39, 16 }, { 11, 15, 15, 47, 16, 39, 47, 39, 16 },
	{ 0, 49, 49, 17, 12, 44, 17, 44, 12 },	{ 0, 17, 17, 49, 12, 44, 49, 44, 12 },	{ 24, 38, 1, 6, 22, 38, 1, 6, 24 },
	{ 24, 6, 1, 38, 22, 6, 1, 38, 24 },		{ 20, 44, 2, 12, 17, 44, 2, 12, 20 },	{ 20, 12, 2, 44, 17, 12, 2, 44, 20 },
	{ 16, 47, 7, 15, 11, 47, 7, 15, 16 },	{ 16, 15, 7, 47, 11, 15, 7, 47, 16 },	{ 12, 49, 12, 17, 0, 49, 12, 17, 12 },
	{ 12, 17, 12, 49, 0, 17, 12, 49, 12 },	{ 24, 6, 33, 38, 22, 38, 33, 6, 24 },	{ 24, 38, 33, 6, 22, 6, 33, 38, 24 },
	{ 20, 12, 34, 44, 17, 44, 34, 12, 20 }, { 20, 44, 34, 12, 17, 12, 34, 44, 20 }, { 16, 15, 39, 47, 11, 47, 39, 15, 16 },
	{ 16, 47, 39, 15, 11, 15, 39, 47, 16 }, { 12, 17, 44, 49, 0, 49, 44, 17, 12 },	{ 12, 49, 44, 17, 0, 17, 44, 49, 12 },
	{ 24, 1, 6, 1, 24, 38, 38, 6, 22 },		{ 24, 1, 38, 1, 24, 6, 6, 38, 22 },		{ 20, 2, 12, 2, 20, 44, 44, 12, 17 },
	{ 20, 2, 44, 2, 20, 12, 12, 44, 17 },	{ 16, 7, 15, 7, 16, 47, 47, 15, 11 },	{ 16, 7, 47, 7, 16, 15, 15, 47, 11 },
	{ 12, 12, 17, 12, 12, 49, 49, 17, 0 },	{ 24, 33, 6, 33, 24, 6, 38, 38, 22 },	{ 24, 33, 38, 33, 24, 38, 6, 6, 22 },
	{ 20, 34, 12, 34, 20, 12, 44, 44, 17 }, { 20, 34, 44, 34, 20, 44, 12, 12, 17 }, { 16, 39, 15, 39, 16, 15, 47, 47, 11 },
	{ 16, 39, 47, 39, 16, 47, 15, 15, 11 }, { 12, 44, 17, 44, 12, 17, 49, 49, 0 },	{ 23, 35, 5, 5, 23, 35, 35, 5, 23 },
	{ 23, 5, 35, 35, 23, 5, 5, 35, 23 },	{ 19, 40, 13, 13, 19, 40, 40, 13, 19 }, { 19, 13, 40, 40, 19, 13, 13, 40, 19 },
	{ 14, 41, 18, 18, 14, 41, 41, 18, 14 }, { 14, 18, 41, 41, 14, 18, 18, 41, 14 }, { 10, 36, 21, 21, 10, 36, 36, 21, 10 },
	{ 10, 21, 36, 36, 10, 21, 21, 36, 10 }, { 23, 37, 3, 3, 23, 5, 37, 35, 23 },	{ 23, 3, 37, 37, 23, 35, 3, 5, 23 },
	{ 19, 45, 8, 8, 19, 13, 45, 40, 19 },	{ 19, 8, 45, 45, 19, 40, 8, 13, 19 },	{ 14, 50, 9, 9, 14, 18, 50, 41, 14 },
	{ 14, 9, 50, 50, 14, 41, 9, 18, 14 },	{ 10, 53, 4, 4, 10, 21, 53, 36, 10 },	{ 10, 4, 53, 53, 10, 36, 4, 21, 10 },
	{ 23, 37, 35, 3, 23, 37, 5, 3, 23 },	{ 23, 3, 5, 37, 23, 3, 35, 37, 23 },	{ 19, 45, 40, 8, 19, 45, 13, 8, 19 },
	{ 19, 8, 13, 45, 19, 8, 40, 45, 19 },	{ 14, 50, 41, 9, 14, 50, 18, 9, 14 },	{ 14, 9, 18, 50, 14, 9, 41, 50, 14 },
	{ 10, 53, 36, 4, 10, 53, 21, 4, 10 },	{ 10, 4, 21, 53, 10, 4, 36, 53, 10 },	{ 23, 35, 37, 5, 23, 3, 3, 37, 23 },
	{ 23, 5, 3, 35, 23, 37, 37, 3, 23 },	{ 19, 40, 45, 13, 19, 8, 8, 45, 19 },	{ 19, 13, 8, 40, 19, 45, 45, 8, 19 },
	{ 14, 41, 50, 18, 14, 9, 9, 50, 14 },	{ 14, 18, 9, 41, 14, 50, 50, 9, 14 },	{ 10, 36, 53, 21, 10, 4, 4, 53, 10 },
	{ 10, 21, 4, 36, 10, 53, 53, 4, 10 } };

HIPRT_CONST static uint32_t IndexToFloat[26] = {
	0,			0x3d1be50c, 0x3e15f61a, 0x3e484336, 0x3e79df93, 0x3e7c3a3a, 0x3e8a8bd4, 0x3e9e0875, 0x3e9f0938,
	0x3ea7bf1b, 0x3eaaaaab, 0x3ec3ef15, 0x3f000000, 0x3f01814f, 0x3f16a507, 0x3f273d75, 0x3f30fbc5, 0x3f3504f3,
	0x3f3d3a87, 0x3f4e034d, 0x3f5a827a, 0x3f692290, 0x3f6c835e, 0x3f73023f, 0x3f7641af, 0x3f800000 };

HIPRT_CONST static uint32_t MatrixIndexToId[RotationCount + 1] = {
	0,	1,	2,	6,	8,	9,	10, 16, 17, 18, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,	 39,  40,  41,	42, 43,
	44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69,	 70,  72,  73,	74, 75,
	76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 127 };

HIPRT_HOST_DEVICE HIPRT_INLINE static float
getRotationMatrixEntry( const uint32_t matrixIndex, const uint32_t row, const uint32_t col )
{
	const uint32_t entryIndex = EncodedRotations[matrixIndex][3 * row + col];
	const uint32_t out		  = IndexToFloat[entryIndex & 0x1f] | ( ( entryIndex >> 5 ) << 31 );
	return as_float( out );
}

HIPRT_HOST_DEVICE HIPRT_INLINE static MatrixFrame getRotationMatrix( const uint32_t matrixIndex )
{
	MatrixFrame m{};
	for ( uint32_t i = 0; i < 3; ++i )
	{
		for ( uint32_t j = 0; j < 3; ++j )
		{
			m.m_matrix[i][j] = getRotationMatrixEntry( matrixIndex, i, j );
		}
	}
	return m;
}

class Obb
{
  public:
	HIPRT_HOST_DEVICE Obb( const uint32_t matrixIndex ) : m_matrixIndex( matrixIndex ) { m_box.reset(); }

	HIPRT_HOST_DEVICE Obb& grow( const float3& point )
	{
		if ( m_matrixIndex < RotationCount )
		{
			const MatrixFrame m = getRotationMatrix( m_matrixIndex );
			float3			  p{};
			p.x = dot( { m.m_matrix[0][0], m.m_matrix[0][1], m.m_matrix[0][2] }, point );
			p.y = dot( { m.m_matrix[1][0], m.m_matrix[1][1], m.m_matrix[1][2] }, point );
			p.z = dot( { m.m_matrix[2][0], m.m_matrix[2][1], m.m_matrix[2][2] }, point );
			m_box.grow( p );
		}
		else
		{
			m_box.grow( point );
		}

		return *this;
	}

	HIPRT_HOST_DEVICE Obb& grow( const Aabb& aabb )
	{
		grow( aabb.m_min );
		grow( { aabb.m_min.x, aabb.m_min.y, aabb.m_max.z } );
		grow( { aabb.m_min.x, aabb.m_max.y, aabb.m_min.z } );
		grow( { aabb.m_min.x, aabb.m_max.y, aabb.m_max.z } );
		grow( { aabb.m_max.x, aabb.m_min.y, aabb.m_min.z } );
		grow( { aabb.m_max.x, aabb.m_min.y, aabb.m_max.z } );
		grow( { aabb.m_max.x, aabb.m_max.y, aabb.m_min.z } );
		grow( aabb.m_max );
		return *this;
	}

	HIPRT_HOST_DEVICE Aabb aabb() const { return m_box; }

	HIPRT_HOST_DEVICE float area() const { return aabb().area(); }

	HIPRT_HOST_DEVICE bool valid() const { return m_box.valid(); }

	Aabb	 m_box;
	uint32_t m_matrixIndex;
};

class Kdop
{
  public:
	HIPRT_HOST_DEVICE Kdop() { reset(); }

	HIPRT_HOST_DEVICE Kdop( const Kdop& kdop )
	{
		for ( uint32_t i = 0; i <= RotationCount; ++i )
			m_boxes[i] = kdop.m_boxes[i];
	}

	HIPRT_HOST_DEVICE Kdop( const Aabb& aabb )
	{
		reset();
		grow( aabb );
	}

	HIPRT_HOST_DEVICE void reset( void )
	{
		for ( uint32_t i = 0; i <= RotationCount; ++i )
			m_boxes[i].reset();
	}

	HIPRT_HOST_DEVICE Kdop& grow( const Kdop& kdop )
	{
		for ( uint32_t i = 0; i <= RotationCount; ++i )
			m_boxes[i].grow( kdop.m_boxes[i] );
		return *this;
	}

	HIPRT_HOST_DEVICE Kdop& grow( const Aabb& aabb )
	{
		grow( aabb.m_min );
		grow( { aabb.m_min.x, aabb.m_min.y, aabb.m_max.z } );
		grow( { aabb.m_min.x, aabb.m_max.y, aabb.m_min.z } );
		grow( { aabb.m_min.x, aabb.m_max.y, aabb.m_max.z } );
		grow( { aabb.m_max.x, aabb.m_min.y, aabb.m_min.z } );
		grow( { aabb.m_max.x, aabb.m_min.y, aabb.m_max.z } );
		grow( { aabb.m_max.x, aabb.m_max.y, aabb.m_min.z } );
		grow( aabb.m_max );
		return *this;
	}

	HIPRT_HOST_DEVICE Kdop& grow( const float3& point )
	{
		m_boxes[RotationCount].grow( point );
		for ( uint32_t i = 0; i < RotationCount; ++i )
		{
			const MatrixFrame m = getRotationMatrix( i );

			float3 p{};
			p.x = dot( { m.m_matrix[0][0], m.m_matrix[0][1], m.m_matrix[0][2] }, point );
			p.y = dot( { m.m_matrix[1][0], m.m_matrix[1][1], m.m_matrix[1][2] }, point );
			p.z = dot( { m.m_matrix[2][0], m.m_matrix[2][1], m.m_matrix[2][2] }, point );

			m_boxes[i].grow( p );
		}

		return *this;
	}

	HIPRT_HOST_DEVICE Obb obb( const uint32_t matrixIndex ) const
	{
		Obb obb( matrixIndex );
		obb.m_box = m_boxes[matrixIndex];
		return obb;
	}

	HIPRT_HOST_DEVICE uint32_t minMatrixIndex() const
	{
		float	 minArea  = FltMax;
		uint32_t minIndex = InvalidValue;
		for ( int32_t i = RotationCount; i >= 0; --i )
		{
			const float area = m_boxes[i].area();
			if ( minArea > area )
			{
				minArea	 = area;
				minIndex = i;
			}
		}
		return minIndex;
	}

	// aabb is valid => all other frames are valid as well
	HIPRT_HOST_DEVICE bool valid() const { return m_boxes[RotationCount].valid(); }

  public:
	Aabb m_boxes[RotationCount + 1];
};
} // namespace hiprt
