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
	HIPRT_HOST_DEVICE HIPRT_INLINE explicit InstanceList( const hiprtSceneBuildInput& input )
		: m_instanceCount( input.instanceCount ), m_frameCount( input.frameCount )
	{
		m_instances        = reinterpret_cast<const hiprtInstance*>( input.instances );
		m_transformHeaders = reinterpret_cast<const hiprtTransformHeader*>( input.instanceTransformHeaders );
		m_apiFrames        = reinterpret_cast<const ApiFrame*>( input.instanceFrames );
		m_masks            = reinterpret_cast<const uint32_t*>( input.instanceMasks );
	}

#if defined( __KERNELCC__ )
	template <typename PrimitiveNode>
	[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE PrimitiveNode fetchPrimNode( const uint32_t index ) const
	{
		if constexpr ( is_same_v<PrimitiveNode, InstanceNode> )
		{
			const hiprtInstance        instance  = fetchInstance( index );
			const hiprtTransformHeader transform = fetchTransformHeader( index );
			const Frame                frame     = fetchFrame( transform.frameIndex );
			const uint32_t             mask      = fetchMask( index );

			InstanceNode   instanceNode{};
			const BoxNode& root = instance.type == hiprtInstanceTypeScene
			                          ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes[0]
			                          : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes[0];
			instanceNode.init( index, mask, frame, instance, transform, root );
			return instanceNode;
		}
		else
		{
			HIPRT_STATIC_ASSERT( (is_same_v<PrimitiveNode, InstanceNode>));
		}
	}

	[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE Aabb fetchAabb( const uint32_t index ) const
	{
		const hiprtTransformHeader header = fetchTransformHeader( index );
		const Transform            t( m_frames, header.frameIndex, header.frameCount );
		const hiprtInstance        instance = fetchInstance( index );
		const BoxNode*             boxNodes = instance.type == hiprtInstanceTypeScene
		                                          ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes
		                                          : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes;
		const BoxNode&             root     = boxNodes[0];

		Aabb aabb;
		for ( uint32_t i = 0; i < root.getChildCount(); ++i )
		{
			const Aabb childBox = root.getChildBox( i );
			aabb.grow( t.motionBounds( childBox ) );
		}
		return aabb;
	}

	[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE float3 fetchCenter( const uint32_t index ) const
	{
		return fetchAabb( index ).center();
	}

	HIPRT_DEVICE HIPRT_INLINE void split(
		const uint32_t index, const uint32_t axis, const float position, const Aabb& box, Aabb& leftBox, Aabb& rightBox ) const
	{
		const hiprtTransformHeader header = fetchTransformHeader( index );
		const Transform            t( m_frames, header.frameIndex, header.frameCount );
		const hiprtInstance        instance = fetchInstance( index );
		const BoxNode*             boxNodes = instance.type == hiprtInstanceTypeScene
		                                          ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes
		                                          : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes;
		const BoxNode&             root     = boxNodes[0];

		leftBox = rightBox = Aabb();
		for ( uint32_t i = 0; i < root.getChildCount(); ++i )
		{
			const Aabb  childBox = t.motionBounds( root.getChildBox( i ) );
			const float mn       = component( childBox.m_min, axis );
			const float mx       = component( childBox.m_max, axis );
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
				Aabb leftChildBox                      = childBox;
				Aabb rightChildBox                     = childBox;
				component( leftChildBox.m_max, axis )  = position;
				component( rightChildBox.m_min, axis ) = position;
				leftBox.grow( leftChildBox );
				rightBox.grow( rightChildBox );
			}
		}

		component( leftBox.m_max, axis )  = position;
		component( rightBox.m_min, axis ) = position;
		leftBox.intersect( box );
		rightBox.intersect( box );
	}

	[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE Obb
	fetchObb( const uint32_t index, const uint32_t matrixIndex, const Aabb& box ) const
	{
		const hiprtTransformHeader header = fetchTransformHeader( index );
		const Transform            t( m_frames, header.frameIndex, header.frameCount );
		const hiprtInstance        instance = fetchInstance( index );
		const BoxNode*             boxNodes = instance.type == hiprtInstanceTypeScene
		                                          ? reinterpret_cast<SceneHeader*>( instance.scene )->m_boxNodes
		                                          : reinterpret_cast<GeomHeader*>( instance.geometry )->m_boxNodes;
		const BoxNode&             root     = boxNodes[0];

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

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t fetchMask( const uint32_t index ) const
	{
		if ( m_masks == nullptr ) return FullRayMask;
		return m_masks[index];
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE hiprtInstance fetchInstance( const uint32_t index ) const
	{
		return m_instances[index];
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE hiprtTransformHeader fetchTransformHeader( uint32_t index ) const
	{
		if ( m_transformHeaders == nullptr ) return hiprtTransformHeader{ index, 1 };
		return m_transformHeaders[index];
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE Frame fetchFrame( const uint32_t index ) const
	{
		if ( m_frameCount == 0 || m_apiFrames == nullptr || m_frames == nullptr ) return Frame();
		return m_frames[index];
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE void convertFrame( const uint32_t index )
	{
		if ( m_frameCount > 0 && m_apiFrames != nullptr && m_frames != nullptr ) m_frames[index] = Frame( m_apiFrames[index] );
	}

	HIPRT_HOST_DEVICE HIPRT_INLINE bool computeInvTransformMatrix( const uint32_t index, float ( &matrix )[3][4] ) const
	{
		const Frame frame = fetchFrame( index );
		return hiprt::computeInvTransformMatrix( frame, matrix );
	}

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getCount() const { return m_instanceCount; }

	[[nodiscard]] HIPRT_HOST_DEVICE HIPRT_INLINE uint32_t getFrameCount() const { return m_frameCount; }

	HIPRT_HOST_DEVICE HIPRT_INLINE void setFrames( Frame* frames ) { m_frames = frames; }

  private:
	const hiprtInstance*        m_instances        = nullptr;
	const hiprtTransformHeader* m_transformHeaders = nullptr;
	const ApiFrame*             m_apiFrames        = nullptr;
	const uint32_t*             m_masks            = nullptr;
	Frame*                      m_frames           = nullptr;
	const uint32_t              m_instanceCount{};
	const uint32_t              m_frameCount{};
};
} // namespace hiprt
