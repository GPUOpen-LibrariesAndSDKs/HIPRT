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
#include <hiprt/impl/Aabb.h>

namespace hiprt
{
struct alignas( 32 ) Bin
{
	HIPRT_HOST_DEVICE float cost() { return m_box.area() * m_counter; }

	HIPRT_HOST_DEVICE float leftCost() { return m_box.area() * m_enter; }

	HIPRT_HOST_DEVICE float rightCost() { return m_box.area() * m_exit; }

	HIPRT_HOST_DEVICE void reset()
	{
		m_box.reset();
		m_enter = 0;
		m_exit	= 0;
	}

	HIPRT_HOST_DEVICE void include( const Bin& bin )
	{
		m_box.grow( bin.m_box );
		m_enter += bin.m_enter;
		m_exit += bin.m_exit;
	}

	Aabb	 m_box;
	uint32_t m_enter;
	union
	{
		uint32_t m_counter;
		uint32_t m_exit;
	};
};
HIPRT_STATIC_ASSERT( sizeof( Bin ) == 32 );

struct Split
{
	HIPRT_HOST_DEVICE void
	setSplitInfo( uint8_t splitAxis, uint32_t splitIndex, bool leftLeaf, bool rightLeaf, bool spatialSplit )
	{
		m_splitIndex   = splitIndex;
		m_splitAxis	   = splitAxis;
		m_spatialSplit = spatialSplit;
		m_leftLeaf	   = leftLeaf;
		m_rightLeaf	   = rightLeaf;
	}

	uint32_t m_splitIndex : 27;
	uint32_t m_splitAxis : 2;
	uint32_t m_spatialSplit : 1;
	uint32_t m_leftLeaf : 1;
	uint32_t m_rightLeaf : 1;
};
HIPRT_STATIC_ASSERT( sizeof( Split ) == 4 );

struct alignas( 64 ) Task
{
	Task() = default;

	HIPRT_HOST_DEVICE Task( const Aabb& box, uint32_t refOffset = InvalidValue ) : m_box( box ), m_refOffset( refOffset ) {}

	Aabb m_box;
	Aabb m_box0;
	Aabb m_box1;

	Split m_split{};
	float m_cost{};

	union
	{
		uint32_t m_counter0 = 0;
		uint32_t m_taskOffset;
	};
	union
	{
		uint32_t m_counter1 = 0;
		uint32_t m_refOffset;
	};
};
HIPRT_STATIC_ASSERT( sizeof( Task ) == 128 );

} // namespace hiprt
