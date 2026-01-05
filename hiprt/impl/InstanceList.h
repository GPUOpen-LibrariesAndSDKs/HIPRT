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
#include <hiprt/impl/Header.h>

namespace hiprt
{
template <typename ApiFrame>
class InstanceList
{
  public:
	HIPRT_HOST_DEVICE InstanceList( const hiprtSceneBuildInput& input )
		: m_instanceCount( input.instanceCount ), m_frameCount( input.frameCount )
	{
		m_instances		   = reinterpret_cast<hiprtInstance*>( input.instances );
		m_transformHeaders = reinterpret_cast<hiprtTransformHeader*>( input.instanceTransformHeaders );
		m_apiFrames		   = reinterpret_cast<ApiFrame*>( input.instanceFrames );
		m_masks			   = reinterpret_cast<uint32_t*>( input.instanceMasks );
	}

#if defined( __KERNELCC__ )
	HIPRT_DEVICE InstanceNode fetchPrimNode( const uint32_t index ) const
	{
		const hiprtInstance		   instance	 = fetchInstance( index );
		const hiprtTransformHeader transform = fetchTransformHeader( index );
		const Frame				   frame	 = fetchFrame( transform.frameIndex );
		const uint32_t			   mask		 = fetchMask( index );

		InstanceNode instanceNode{};
		if constexpr ( is_same<InstanceNode, HwInstanceNode>::value )
		{
			const BoxNode root = instance.type == hiprtInstanceTypeScene
									 ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes[0]
									 : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes[0];

			Aabb	 childBoxes[BranchingFactor];
			uint32_t childIndices[BranchingFactor];
			uint32_t childCount = root.getChildCount();

			for ( uint32_t i = 0; i < childCount; ++i )
				childBoxes[i] = root.getChildBox( i );

			while ( childCount > HwInstanceNode::BranchingFactor )
			{
				float	 minArea = FltMax;
				uint32_t min_i	 = InvalidValue;
				uint32_t min_j	 = InvalidValue;

				for ( uint32_t i = 0; i < childCount; ++i )
				{
					for ( uint32_t j = i + 1; j < childCount; ++j )
					{
						const Aabb	box( childBoxes[i], childBoxes[j] );
						const float area = box.area();
						if ( minArea > area )
						{
							minArea = area;
							min_i	= i;
							min_j	= j;
						}
					}
				}

				childBoxes[min_i] = Aabb( childBoxes[min_i], childBoxes[min_j] );
				childBoxes[min_j] = childBoxes[--childCount];
			}

			for ( uint32_t i = 0; i < childCount; ++i )
				childIndices[i] = RootIndex;

			instanceNode.init( index, mask, frame, instance, transform, childCount, childIndices, childBoxes );
		}
		else
		{
			instanceNode.init( index, mask, frame, instance, transform, InvalidValue, nullptr, nullptr );
		}

		return instanceNode;
	}

	HIPRT_DEVICE Aabb fetchAabb( const uint32_t index ) const
	{
		const hiprtTransformHeader header = fetchTransformHeader( index );
		const Transform			   t( m_frames, header.frameIndex, header.frameCount );
		const hiprtInstance		   instance = fetchInstance( index );
		const BoxNode*			   boxNodes = instance.type == hiprtInstanceTypeScene
												  ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes
												  : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes;
		const BoxNode&			   root		= boxNodes[0];

		Aabb aabb;
		for ( uint32_t i = 0; i < root.getChildCount(); ++i )
		{
			const Aabb childBox = root.getChildBox( i );
			aabb.grow( t.motionBounds( childBox ) );
		}
		return aabb;
	}

	HIPRT_DEVICE float3 fetchCenter( const uint32_t index ) const { return fetchAabb( index ).center(); }

	HIPRT_DEVICE void split(
		const uint32_t index, const uint32_t axis, const float position, const Aabb& box, Aabb& leftBox, Aabb& rightBox ) const
	{
		const hiprtTransformHeader header = fetchTransformHeader( index );
		const Transform			   t( m_frames, header.frameIndex, header.frameCount );
		const hiprtInstance		   instance = fetchInstance( index );
		const BoxNode*			   boxNodes = instance.type == hiprtInstanceTypeScene
												  ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes
												  : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes;
		const BoxNode&			   root		= boxNodes[0];

		leftBox = rightBox = Aabb();
		for ( uint32_t i = 0; i < root.getChildCount(); ++i )
		{
			const Aabb	childBox = t.motionBounds( root.getChildBox( i ) );
			const float mn		 = ( &childBox.m_min.x )[axis];
			const float mx		 = ( &childBox.m_max.x )[axis];
			if ( position >= mx )
			{
				leftBox.grow( childBox );
			}
			else if ( position <= mn )
			{
				rightBox.grow( childBox );
			}
			else
			{
				Aabb leftChildBox				 = childBox;
				Aabb rightChildBox				 = childBox;
				( &leftChildBox.m_max.x )[axis]	 = position;
				( &rightChildBox.m_min.x )[axis] = position;
				leftBox.grow( leftChildBox );
				rightBox.grow( rightChildBox );
			}
		}

		( &leftBox.m_max.x )[axis]	= position;
		( &rightBox.m_min.x )[axis] = position;
		leftBox.intersect( box );
		rightBox.intersect( box );
	}

	HIPRT_DEVICE Obb fetchObb( const uint32_t index, const uint32_t matrixIndex, const Aabb& box ) const
	{
		const hiprtTransformHeader header = fetchTransformHeader( index );
		const Transform			   t( m_frames, header.frameIndex, header.frameCount );
		const hiprtInstance		   instance = fetchInstance( index );
		const BoxNode*			   boxNodes = instance.type == hiprtInstanceTypeScene
												  ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes
												  : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes;
		const BoxNode&			   root		= boxNodes[0];

		Obb obb( matrixIndex );
		for ( uint32_t i = 0; i < root.getChildCount(); ++i )
		{
			const Aabb childBox = t.motionBounds( root.getChildBox( i ) ).intersect( box );
			if ( childBox.valid() ) obb.grow( childBox );
		}

		if ( !obb.valid() ) obb.grow( box );

		return obb;
	}
#endif

	HIPRT_HOST_DEVICE uint32_t fetchMask( const uint32_t index ) const
	{
		if ( m_masks == nullptr ) return FullRayMask;
		return m_masks[index];
	}

	HIPRT_HOST_DEVICE hiprtInstance fetchInstance( const uint32_t index ) const { return m_instances[index]; }

	HIPRT_HOST_DEVICE hiprtTransformHeader fetchTransformHeader( uint32_t index ) const
	{
		if ( m_transformHeaders == nullptr ) return hiprtTransformHeader{ index, 1 };
		return m_transformHeaders[index];
	}

	HIPRT_HOST_DEVICE Frame fetchFrame( const uint32_t index ) const
	{
		if ( m_frameCount == 0 || m_apiFrames == nullptr || m_frames == nullptr ) return Frame();
		return m_frames[index];
	}

	HIPRT_HOST_DEVICE void convertFrame( const uint32_t index )
	{
		if ( m_frameCount > 0 && m_apiFrames != nullptr && m_frames != nullptr ) m_frames[index] = Frame( m_apiFrames[index] );
	}

	HIPRT_HOST_DEVICE bool computeInvTransformMatrix( const uint32_t index, float ( &matrix )[3][4] ) const
	{
		const Frame frame = fetchFrame( index );
		return hiprt::computeInvTransformMatrix( frame, matrix );
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
