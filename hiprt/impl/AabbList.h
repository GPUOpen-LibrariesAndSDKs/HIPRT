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
#include <hiprt/impl/Aabb.h>

namespace hiprt
{
class AabbList
{
  public:
	HIPRT_HOST_DEVICE AabbList( const hiprtAABBListPrimitive& list )
		: m_aabbCount( list.aabbCount ), m_aabbStride( list.aabbStride )
	{
		m_aabbs = reinterpret_cast<const uint8_t*>( list.aabbs );
	}

	HIPRT_HOST_DEVICE Aabb fetchAabb( uint32_t index ) const
	{
		const uint32_t halfStride = ( m_aabbStride >> 1 );
		const float*   boxMinPtr  = reinterpret_cast<const float*>( m_aabbs + index * m_aabbStride + 0 * halfStride );
		const float*   boxMaxPtr  = reinterpret_cast<const float*>( m_aabbs + index * m_aabbStride + 1 * halfStride );
		Aabb		   box;
		box.m_min = { boxMinPtr[0], boxMinPtr[1], boxMinPtr[2] };
		box.m_max = { boxMaxPtr[0], boxMaxPtr[1], boxMaxPtr[2] };
		return box;
	}

	HIPRT_HOST_DEVICE float3 fetchCenter( uint32_t index ) const { return fetchAabb( index ).center(); }

	HIPRT_HOST_DEVICE uint32_t getCount() const { return m_aabbCount; }

  private:
	const uint8_t* m_aabbs;
	uint32_t	   m_aabbCount;
	uint32_t	   m_aabbStride;
};
} // namespace hiprt
