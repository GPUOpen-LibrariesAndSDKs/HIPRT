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
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Transform.h>

namespace hiprt
{
template <typename ApiFrame>
class InstanceList
{
  public:
	static constexpr uint32_t StackSize = 2;

	HIPRT_HOST_DEVICE InstanceList( const hiprtSceneBuildInput& input )
		: m_instanceCount( input.instanceCount ), m_frameCount( input.frameCount )
	{
		m_instances		   = reinterpret_cast<hiprtInstance*>( input.instances );
		m_transformHeaders = reinterpret_cast<hiprtTransformHeader*>( input.instanceTransformHeaders );
		m_apiFrames		   = reinterpret_cast<ApiFrame*>( input.instanceFrames );
		m_masks			   = reinterpret_cast<uint32_t*>( input.instanceMasks );
	}

	HIPRT_HOST_DEVICE Aabb fetchAabb( uint32_t index ) const
	{
		hiprtTransformHeader header = fetchTransformHeader( index );
		Transform			 t( m_frames, header.frameIndex, header.frameCount );
		hiprtInstance		 instance = fetchInstance( index );
		const BoxNode*		 boxNodes = instance.type == hiprtInstanceTypeScene
											? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes
											: reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes;

		if constexpr ( StackSize == 0 )
		{
			Aabb aabb = boxNodes->aabb();
			return t.motionBounds( aabb );
		}
		else
		{
			uint32_t stack[StackSize == 0 ? 1 : StackSize];
			uint32_t stackTop = 0;

			Aabb aabb;
			stack[stackTop++] = RootIndex;
			while ( stackTop > 0 )
			{
				uint32_t	   nodeAddr = getNodeAddr( stack[--stackTop] );
				const BoxNode& node		= boxNodes[nodeAddr];

				if ( node.m_childIndex0 != InvalidValue )
				{
					if ( stackTop < StackSize && node.getChildType( 0 ) == BoxType )
						stack[stackTop++] = node.m_childIndex0;
					else
						aabb.grow( t.motionBounds( node.m_box0 ) );
				}

				if ( node.m_childIndex1 != InvalidValue )
				{
					if ( stackTop < StackSize && node.getChildType( 1 ) == BoxType )
						stack[stackTop++] = node.m_childIndex1;
					else
						aabb.grow( t.motionBounds( node.m_box1 ) );
				}

				if ( node.m_childIndex2 != InvalidValue )
				{
					if ( stackTop < StackSize && node.getChildType( 2 ) == BoxType )
						stack[stackTop++] = node.m_childIndex2;
					else
						aabb.grow( t.motionBounds( node.m_box2 ) );
				}

				if ( node.m_childIndex3 != InvalidValue )
				{
					if ( stackTop < StackSize && node.getChildType( 3 ) == BoxType )
						stack[stackTop++] = node.m_childIndex3;
					else
						aabb.grow( t.motionBounds( node.m_box3 ) );
				}
			}

			return aabb;
		}
	}

	HIPRT_HOST_DEVICE float3 fetchCenter( uint32_t index ) const { return fetchAabb( index ).center(); }

	HIPRT_HOST_DEVICE uint32_t fetchMask( uint32_t index ) const
	{
		if ( m_masks == nullptr ) return FullRayMask;
		return m_masks[index];
	}

	HIPRT_HOST_DEVICE hiprtInstance fetchInstance( uint32_t index ) const { return m_instances[index]; }

	HIPRT_HOST_DEVICE hiprtTransformHeader fetchTransformHeader( uint32_t index ) const
	{
		if ( m_transformHeaders == nullptr ) return hiprtTransformHeader{ index, 1 };
		return m_transformHeaders[index];
	}

	HIPRT_HOST_DEVICE Frame fetchFrame( uint32_t index ) const
	{
		if ( m_frameCount == 0 || m_apiFrames == nullptr || m_frames == nullptr ) return Frame();
		return m_frames[index];
	}

	HIPRT_HOST_DEVICE void convertFrame( uint32_t index )
	{
		if ( m_frameCount > 0 && m_apiFrames != nullptr && m_frames != nullptr ) m_frames[index] = m_apiFrames[index].convert();
	}

	HIPRT_HOST_DEVICE bool copyInvTransformMatrix( uint32_t index, float ( &matrix )[3][4] ) const
	{
		Frame		frame		   = fetchFrame( index );
		MatrixFrame invMatrixFrame = MatrixFrame::getMatrixFrameInv( frame );
		memcpy( &matrix[0][0], &invMatrixFrame.m_matrix[0][0], sizeof( float ) * 12 );
		return frame.identity();
	}

	HIPRT_HOST_DEVICE uint32_t getCount() const { return m_instanceCount; }

	HIPRT_HOST_DEVICE uint32_t getFrameCount() const { return m_frameCount; }

	HIPRT_HOST_DEVICE void setFrames( Frame* frames ) { m_frames = frames; }

  private:
	hiprtInstance*		  m_instances;
	hiprtTransformHeader* m_transformHeaders;
	Frame*				  m_frames = nullptr;
	ApiFrame*			  m_apiFrames;
	uint32_t*			  m_masks;
	uint32_t			  m_instanceCount;
	uint32_t			  m_frameCount;
};
} // namespace hiprt
