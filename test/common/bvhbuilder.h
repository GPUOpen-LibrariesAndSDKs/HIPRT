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
#include <algorithm>
#if defined( USE_HIPRTEW )
#include <hiprt/hiprtew.h>
#else
#include <hiprt/hiprt.h>
#endif
#include <queue>
#include <cfloat>

struct Aabb
{
	Aabb() { reset(); }

	Aabb( const float3& p ) : m_min( p ), m_max( p ) {}

	Aabb( const float3& mi, const float3& ma ) : m_min( mi ), m_max( ma ) {}

	Aabb( const Aabb& rhs, const Aabb& lhs )
	{
		m_min.x = fminf( lhs.m_min.x, rhs.m_min.x );
		m_min.y = fminf( lhs.m_min.y, rhs.m_min.y );
		m_min.z = fminf( lhs.m_min.z, rhs.m_min.z );
		m_max.x = fmaxf( lhs.m_max.x, rhs.m_max.x );
		m_max.y = fmaxf( lhs.m_max.y, rhs.m_max.y );
		m_max.z = fmaxf( lhs.m_max.z, rhs.m_max.z );
	}

	void reset( void )
	{
		m_min = { FLT_MAX, FLT_MAX, FLT_MAX };
		m_max = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
	}

	Aabb& grow( const float3& p )
	{
		m_min.x = fminf( m_min.x, p.x );
		m_min.y = fminf( m_min.y, p.y );
		m_min.z = fminf( m_min.z, p.z );
		m_max.x = fmaxf( m_max.x, p.x );
		m_max.y = fmaxf( m_max.y, p.y );
		m_max.z = fmaxf( m_max.z, p.z );
		return *this;
	}

	Aabb& grow( const Aabb& rhs )
	{
		m_min.x = fminf( m_min.x, rhs.m_min.x );
		m_min.y = fminf( m_min.y, rhs.m_min.y );
		m_min.z = fminf( m_min.z, rhs.m_min.z );
		m_max.x = fmaxf( m_max.x, rhs.m_max.x );
		m_max.y = fmaxf( m_max.y, rhs.m_max.y );
		m_max.z = fmaxf( m_max.z, rhs.m_max.z );
		return *this;
	}

	float3 center() const
	{
		float3 c;
		c.x = ( m_max.x + m_min.x ) * 0.5f;
		c.y = ( m_max.y + m_min.y ) * 0.5f;
		c.z = ( m_max.z + m_min.z ) * 0.5f;
		return c;
	}

	float3 extent() const
	{
		float3 e;
		e.x = m_max.x - m_min.x;
		e.y = m_max.y - m_min.y;
		e.z = m_max.z - m_min.z;
		return e;
	}

	float area() const
	{
		float3 ext = extent();
		return 2 * ( ext.x * ext.y + ext.x * ext.z + ext.y * ext.z );
	}

	bool valid( void ) { return m_min.x <= m_max.x && m_min.y <= m_max.y && m_min.z <= m_max.z; }

	void intersect( const Aabb& box )
	{
		m_min.x = fmaxf( m_min.x, box.m_min.x );
		m_min.y = fmaxf( m_min.y, box.m_min.y );
		m_min.z = fmaxf( m_min.z, box.m_min.z );
		m_max.x = fminf( m_max.x, box.m_max.x );
		m_max.y = fminf( m_max.y, box.m_max.y );
		m_max.z = fminf( m_max.z, box.m_max.z );
	}

	float3 m_min;
	float3 m_max;
};

struct QueueEntry
{
	uint32_t m_nodeIndex;
	uint32_t m_begin;
	uint32_t m_end;
	Aabb	 m_box;
	QueueEntry( uint32_t nodeIndex, uint32_t begin, uint32_t end, const Aabb& box )
		: m_nodeIndex( nodeIndex ), m_begin( begin ), m_end( end ), m_box( box )
	{
	}
};

class BvhBuilder
{
  public:
	BvhBuilder( void )						   = delete;
	BvhBuilder& operator=( const BvhBuilder& ) = delete;

	static void build( uint32_t nPrims, const std::vector<Aabb>& primBoxes, std::vector<hiprtInternalNode>& nodes );
};

void BvhBuilder::build( uint32_t nPrims, const std::vector<Aabb>& primBoxes, std::vector<hiprtInternalNode>& nodes )
{
	ASSERT( nPrims >= 2 );
	std::vector<Aabb>	  rightBoxes( nPrims );
	std::vector<uint32_t> tmpIndices( nPrims );
	std::vector<uint32_t> leftIndices( nPrims );

	std::vector<uint32_t> indices[3];
	for ( uint32_t k = 0; k < 3; ++k )
	{
		indices[k].resize( nPrims );
		for ( uint32_t i = 0; i < nPrims; ++i )
			indices[k][i] = i;
		std::sort( indices[k].begin(), indices[k].end(), [&]( uint32_t a, uint32_t b ) {
			float3 ca = primBoxes[a].center();
			float3 cb = primBoxes[b].center();
			return reinterpret_cast<float*>( &ca )[k] > reinterpret_cast<float*>( &cb )[k];
		} );
	}

	Aabb box;
	for ( uint32_t i = 0; i < nPrims; ++i )
		box.grow( primBoxes[i] );

	std::queue<QueueEntry> queue;
	queue.push( QueueEntry( 0u, 0u, nPrims, box ) );
	nodes.push_back( hiprtInternalNode() );
	while ( !queue.empty() )
	{
		uint32_t nodeIndex = queue.front().m_nodeIndex;
		uint32_t begin	   = queue.front().m_begin;
		uint32_t end	   = queue.front().m_end;
		Aabb	 box	   = queue.front().m_box;
		queue.pop();

		float	 minCost  = FLT_MAX;
		uint32_t minAxis  = 0;
		uint32_t minIndex = 0;
		Aabb	 minLeftBox, minRightBox;
		for ( uint32_t k = 0; k < 3; ++k )
		{
			rightBoxes[end - 1] = primBoxes[indices[k][end - 1]];
			for ( int32_t i = end - 2; i >= static_cast<int32_t>( begin ); --i )
				rightBoxes[i] = Aabb( primBoxes[indices[k][i]], rightBoxes[i + 1] );

			Aabb leftBox, rightBox;
			for ( uint32_t i = begin; i < end - 1; ++i )
			{
				uint32_t leftCount	= ( i + 1 ) - begin;
				uint32_t rightCount = end - ( i + 1 );
				leftBox.grow( primBoxes[indices[k][i]] );
				rightBox   = rightBoxes[i + 1];
				float cost = leftBox.area() * leftCount + rightBox.area() * rightCount;
				if ( cost < minCost )
				{
					minCost		= cost;
					minIndex	= i + 1;
					minAxis		= k;
					minLeftBox	= leftBox;
					minRightBox = rightBox;
				}
				assert( leftBox.area() <= box.area() );
				assert( rightBox.area() <= box.area() );
			}
		}

		assert( minIndex > begin );
		assert( end > minIndex );

		std::memset( leftIndices.data(), 0, nPrims * sizeof( uint32_t ) );
		for ( uint32_t i = begin; i < minIndex; ++i )
		{
			uint32_t index	   = indices[minAxis][i];
			leftIndices[index] = 1;
		}

		for ( uint32_t j = 0; j < 3; ++j )
		{
			if ( j != minAxis )
			{
				uint32_t k = begin;
				uint32_t l = minIndex;
				for ( uint32_t i = begin; i < end; ++i )
				{
					uint32_t index = indices[j][i];
					if ( leftIndices[indices[j][i]] )
						tmpIndices[k++] = index;
					else
						tmpIndices[l++] = index;
				}
				assert( k == minIndex );
				assert( l == end );
				std::memcpy( &indices[j][begin], &tmpIndices[begin], ( end - begin ) * sizeof( uint32_t ) );
			}
		}

		nodes[nodeIndex].aabbMin = min( minLeftBox.m_min, minRightBox.m_min );
		nodes[nodeIndex].aabbMax = max( minLeftBox.m_max, minRightBox.m_max );
		if ( minIndex - begin == 1 )
		{
			nodes[nodeIndex].childIndices[0]   = indices[minAxis][begin];
			nodes[nodeIndex].childNodeTypes[0] = hiprtBvhNodeTypeLeaf;
		}
		else
		{
			nodes[nodeIndex].childIndices[0]   = static_cast<uint32_t>( nodes.size() );
			nodes[nodeIndex].childNodeTypes[0] = hiprtBvhNodeTypeInternal;
			queue.push( QueueEntry( nodes[nodeIndex].childIndices[0], begin, minIndex, minLeftBox ) );
			nodes.push_back( hiprtInternalNode() );
		}

		if ( end - minIndex == 1 )
		{
			nodes[nodeIndex].childIndices[1]   = indices[minAxis][minIndex];
			nodes[nodeIndex].childNodeTypes[1] = hiprtBvhNodeTypeLeaf;
		}
		else
		{
			nodes[nodeIndex].childIndices[1]   = static_cast<uint32_t>( nodes.size() );
			nodes[nodeIndex].childNodeTypes[1] = hiprtBvhNodeTypeInternal;
			queue.push( QueueEntry( nodes[nodeIndex].childIndices[1], minIndex, end, minRightBox ) );
			nodes.push_back( hiprtInternalNode() );
		}
	}
}
