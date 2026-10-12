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

#include <hiprt/hiprt_common.h>
#include <hiprt/hiprt_vec.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Header.h>
#include <hiprt/impl/Instance.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>
#include <hiprt/impl/Transform.h>
#include <hiprt/impl/Triangle.h>
#include <hiprt/hiprt_device.h>

#if HIPRT_RTIP >= 11
extern "C" __device__ float __ocml_native_recip_f32( float );
#endif

#if HIPRT_RTIP >= 31
using hip_float3 = float __attribute__( ( ext_vector_type( 3 ) ) );
#endif

// This is a hack as hiprtc doesnt include placement new and including <new> throws error
// So, manually defining it for non bitcode path as hipcc does include placement new.
// Also check if compiling via hipcc as hipcc would complain of redefinition
#if !defined( HIPRT_BITCODE_LINKING ) && defined( __HIPCC_RTC__ )
HIPRT_DEVICE void* operator new( size_t size, void* ptr ) noexcept { return ptr; };
HIPRT_DEVICE void* operator new[]( size_t size, void* ptr ) noexcept { return ptr; };
#endif

HIPRT_DEVICE bool intersectFunc(
	const uint32_t              geomType,
	const uint32_t              rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&             ray,
	void*                       payload,
	hiprtHit&                   hit );
HIPRT_DEVICE bool filterFunc(
	const uint32_t              geomType,
	const uint32_t              rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&             ray,
	void*                       payload,
	const hiprtHit&             hit );

namespace hiprt
{
enum
{
	Triangle0Processed = 1,
	Triangle1Processed = 2,
};

HIPRT_DEVICE HIPRT_INLINE float3 rcp( const float3& a )
{
#if HIPRT_RTIP >= 11
	return float3{ __ocml_native_recip_f32( a.x ), __ocml_native_recip_f32( a.y ), __ocml_native_recip_f32( a.z ) };
#else
	return 1.0f / a;
#endif
}

template <typename StackEntry, uint32_t StackSize>
HIPRT_DEVICE HIPRT_INLINE PrivateStack<StackEntry, StackSize>::PrivateStack() : m_top( 0u )
{
}

template <typename StackEntry, uint32_t StackSize>
HIPRT_DEVICE HIPRT_INLINE StackEntry PrivateStack<StackEntry, StackSize>::pop()
{
	return m_stackBuffer[--m_top];
}

template <typename StackEntry, uint32_t StackSize>
HIPRT_DEVICE HIPRT_INLINE void PrivateStack<StackEntry, StackSize>::push( const StackEntry val )
{
	m_stackBuffer[m_top++] = val;
}

template <typename StackEntry, uint32_t StackSize>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE bool PrivateStack<StackEntry, StackSize>::empty() const
{
	return m_top == 0u;
}

template <typename StackEntry, uint32_t StackSize>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE uint32_t PrivateStack<StackEntry, StackSize>::vacancy() const
{
	return StackSize - m_top;
}

template <typename StackEntry, uint32_t StackSize>
HIPRT_DEVICE HIPRT_INLINE void PrivateStack<StackEntry, StackSize>::reset()
{
	m_top = 0u;
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE GlobalStack<StackEntry, DynamicAssignment>::GlobalStack(
	hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer )
{
	const uint32_t threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
	const uint32_t warpIndex   = threadIndex >> LogStride;
	const uint32_t laneIndex   = threadIndex & ( Stride - 1 );

	const uint32_t sharedStackOffset = laneIndex + warpIndex * Stride * sharedStackBuffer.stackSize;
	m_sharedStackBuffer              = reinterpret_cast<StackEntry*>( sharedStackBuffer.stackData ) + sharedStackOffset;
	m_sharedStackSize                = sharedStackBuffer.stackSize;
	if constexpr ( DynamicAssignment )
	{
		const uint32_t warpsPerBlock    = hiprt::DivideRoundUp( blockDim.x * blockDim.y, Stride );
		const uint32_t activeWarps      = globalStackBuffer.stackCount >> LogStride;
		const uint32_t firstThreadIndex = __ffsll( static_cast<unsigned long long>( hiprt::ballot( true ) ) ) - 1;

		uint32_t  warpHash          = InvalidValue;
		uint32_t  warpHashCandidate = ( warpIndex + ( blockIdx.x + blockIdx.y * gridDim.x ) * warpsPerBlock ) % activeWarps;
		uint32_t* globalStackLocks  = reinterpret_cast<uint32_t*>( globalStackBuffer.stackData );
		while ( warpHash == InvalidValue )
		{
			if ( laneIndex == firstThreadIndex )
			{
				if ( atomicCAS( &globalStackLocks[warpHashCandidate], 0, 1 ) == 0 ) warpHash = warpHashCandidate;
			}
			warpHashCandidate = ( warpHashCandidate + 1 ) % activeWarps;
			warpHash          = shfl( warpHash, firstThreadIndex );
		}
		__threadfence();
		m_globalStackLock = &globalStackLocks[warpHash];

		const uint32_t globalStackOffset = activeWarps + laneIndex + ( warpHash << LogStride ) * globalStackBuffer.stackSize;
		m_globalStackBuffer              = reinterpret_cast<StackEntry*>( globalStackBuffer.stackData ) + globalStackOffset;
		m_globalStackSize                = globalStackBuffer.stackSize;
	}
	else
	{
		const uint32_t globalStackOffset =
			laneIndex + ( warpIndex * Stride + ( blockIdx.x + blockIdx.y * gridDim.x ) * ( blockDim.x * blockDim.y ) ) *
							globalStackBuffer.stackSize;
		m_globalStackBuffer = reinterpret_cast<StackEntry*>( globalStackBuffer.stackData ) + globalStackOffset;
		m_globalStackSize   = globalStackBuffer.stackSize;
	}
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE HIPRT_INLINE GlobalStack<StackEntry, DynamicAssignment>::~GlobalStack()
{
	if constexpr ( DynamicAssignment )
	{
		__threadfence();
		const uint32_t threadIndex      = threadIdx.x + threadIdx.y * blockDim.x;
		const uint32_t laneIndex        = threadIndex & ( Stride - 1 );
		const uint32_t firstThreadIndex = __ffsll( static_cast<unsigned long long>( hiprt::ballot( true ) ) ) - 1;
		if ( laneIndex == firstThreadIndex ) atomicExch( m_globalStackLock, 0 );
	}
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE HIPRT_INLINE StackEntry GlobalStack<StackEntry, DynamicAssignment>::pop()
{
	if ( m_sharedCount > 0 )
	{
		m_sharedCount--;
		if ( --m_sharedIndex < 0 ) m_sharedIndex += m_sharedStackSize;
		return m_sharedStackBuffer[m_sharedIndex << LogStride];
	}
	else
	{
		return m_globalStackBuffer[--m_globalIndex << LogStride];
	}
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE HIPRT_INLINE void GlobalStack<StackEntry, DynamicAssignment>::push( StackEntry val )
{
	if ( m_sharedCount < m_sharedStackSize )
	{
		m_sharedStackBuffer[m_sharedIndex++ << LogStride] = val;
		m_sharedCount++;
	}
	else
	{
		if ( m_sharedStackSize == 0 )
		{
			m_globalStackBuffer[m_globalIndex++ << LogStride] = val;
		}
		else
		{
			m_globalStackBuffer[m_globalIndex++ << LogStride] = m_sharedStackBuffer[m_sharedIndex << LogStride];
			m_sharedStackBuffer[m_sharedIndex++ << LogStride] = val;
		}
	}
	if ( m_sharedIndex >= m_sharedStackSize ) m_sharedIndex -= m_sharedStackSize;
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE HIPRT_INLINE uint32_t GlobalStack<StackEntry, DynamicAssignment>::vacancy() const
{
	return m_globalStackSize - m_globalIndex + m_sharedStackSize - m_sharedCount;
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE HIPRT_INLINE bool GlobalStack<StackEntry, DynamicAssignment>::empty() const
{
	return m_sharedCount == 0 && m_globalIndex == 0;
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE HIPRT_INLINE void GlobalStack<StackEntry, DynamicAssignment>::reset()
{
	m_globalIndex = 0;
	m_sharedIndex = 0;
	m_sharedCount = 0u;
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE TraversalBase<Stack, TraversalType>::TraversalBase(
	const hiprtRay&          ray,
	Stack&                   stack,
	const hiprtTraversalHint hint,
	void*                    payload,
	const hiprtFuncTable     funcTable,
	const uint32_t           rayType )
	: m_ray( ray ), m_stack( stack ), m_payload( payload ), m_nodeIndex( RootIndex ), m_rayType( rayType ), m_hint( hint )
{
	if ( funcTable != nullptr ) m_tableHeader = *reinterpret_cast<const hiprtFuncTableHeader*>( funcTable );
#if HIPRT_RTIP >= 11
	packDescriptor( static_cast<uint32_t>( hint ), Rtip >= 31 );
#endif
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE hiprtTraversalState TraversalBase<Stack, TraversalType>::getCurrentState() const
{
	return m_state;
}

#if HIPRT_RTIP >= 11
template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE void
TraversalBase<Stack, TraversalType>::packDescriptor( const uint32_t boxSortHeuristic, const bool compressed )
{
	uint64_t baseAddress          = 0x0;
	uint64_t size                 = 0x3ffffffffffull;
	uint32_t boxGrowUlp           = 0x6;
	uint32_t boxSortEnable        = 0x1;
	uint32_t triangleReturnMode   = 0x1;
	uint32_t type                 = 0x8;
	uint32_t compressFormatEnable = 0x0;
	uint32_t instanceEnable       = 0x0;
	uint32_t sortTrianglesFirst   = 0x0;
	uint32_t wideSortEnable       = 0x0;
	if ( compressed )
	{
		compressFormatEnable = 0x1;
		instanceEnable       = 0x1;
		sortTrianglesFirst   = 0x1;
		wideSortEnable       = 0x1;
	}
	m_descriptor.x = baseAddress & 0xffffffff;
	m_descriptor.y = ( baseAddress >> 32ull ) | ( boxSortEnable << 31u ) | ( boxGrowUlp << 23u ) |
	                 ( ( boxSortHeuristic & 0x3 ) << 21u ) | ( sortTrianglesFirst << 20u );
	m_descriptor.z = size & 0xffffffff;
	m_descriptor.w = ( size >> 32ull ) | ( triangleReturnMode << 24u ) | ( type << 28u ) | ( instanceEnable << 22u ) |
	                 ( wideSortEnable << 21u ) | ( compressFormatEnable << 19u );
}
#endif

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE bool TraversalBase<Stack, TraversalType>::testInternalNode(
	const hiprtRay& ray, [[maybe_unused]] const float3& invD, const BoxNode* nodes, uint32_t& nodeIndex )
{
#if HIPRT_RTIP < 11
	const BoxNode node = nodes[getNodeAddr( nodeIndex )];
	float2        s0   = node.m_childBoxes[0].intersect( ray.origin, invD, ray.maxT );
	float2        s1   = node.m_childBoxes[1].intersect( ray.origin, invD, ray.maxT );
	float2        s2   = node.m_childBoxes[2].intersect( ray.origin, invD, ray.maxT );
	float2        s3   = node.m_childBoxes[3].intersect( ray.origin, invD, ray.maxT );

	uint32_t result[4];
	result[0] = s0.x <= s0.y ? node.m_childIndices[0] : InvalidValue;
	result[1] = s1.x <= s1.y ? node.m_childIndices[1] : InvalidValue;
	result[2] = s2.x <= s2.y ? node.m_childIndices[2] : InvalidValue;
	result[3] = s3.x <= s3.y ? node.m_childIndices[3] : InvalidValue;

#define SORT( childIndexA, childIndexB, distA, distB )                                     \
	if ( ( childIndexB != InvalidValue && distB < distA ) || childIndexA == InvalidValue ) \
	{                                                                                      \
		const float    t0 = distA;                                                         \
		const uint32_t t1 = childIndexA;                                                   \
		childIndexA       = childIndexB;                                                   \
		distA             = distB;                                                         \
		childIndexB       = t1;                                                            \
		distB             = t0;                                                            \
	}

	SORT( result[0], result[2], s0.x, s2.x )
	SORT( result[1], result[3], s1.x, s3.x )
	SORT( result[0], result[1], s0.x, s1.x )
	SORT( result[2], result[3], s2.x, s3.x )
	SORT( result[1], result[2], s1.x, s2.x )
#undef SORT
#elif HIPRT_RTIP >= 31
	hip_float3 dummy0, dummy1;
	auto       result = __builtin_amdgcn_image_bvh8_intersect_ray(
        encodeBaseAddr( nodes ),
        ray.maxT,
        0xff,
        { ray.origin.x, ray.origin.y, ray.origin.z },
        { ray.direction.x, ray.direction.y, ray.direction.z },
        nodeIndex,
        { m_descriptor.x, m_descriptor.y, m_descriptor.z, m_descriptor.w },
        &dummy0,
        &dummy1 );
#else
	auto result = __builtin_amdgcn_image_bvh_intersect_ray_l(
		encodeBaseAddr( nodes, nodeIndex ),
		ray.maxT,
		{ ray.origin.x, ray.origin.y, ray.origin.z, 0.0f },
		{ ray.direction.x, ray.direction.y, ray.direction.z, 0.0f },
		{ invD.x, invD.y, invD.z, 0.0f },
		{ m_descriptor.x, m_descriptor.y, m_descriptor.z, m_descriptor.w } );
#endif

	if ( m_stack.vacancy() < BoxNode::BranchingFactor - 1 )
	{
		m_state = hiprtTraversalStateStackOverflow;
		return true;
	}

	HIPRT_UNROLL
	for ( uint32_t i = BoxNode::BranchingFactor - 1; i >= 1; --i )
		if ( result[i] != InvalidValue ) m_stack.push( result[i] );

	if ( result[0] != InvalidValue )
	{
		nodeIndex = result[0];
		return true;
	}

	return false;
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE bool TraversalBase<Stack, TraversalType>::testTriangleNode(
	const hiprtRay&                ray,
	[[maybe_unused]] const float3& invD,
	const TriangleNode*            nodes,
	const uint32_t                 geomType,
	uint32_t&                      leafIndex,
	[[maybe_unused]] uint32_t&     triangleMask,
	hiprtHit&                      hit )
{
	bool     hasHit   = false;
	uint32_t leafAddr = getNodeAddr( leafIndex );

	const bool useFilter = geomType != InvalidValue && m_tableHeader.funcDataSets != nullptr;

	if constexpr ( is_same_v<TriangleNode, rtip31::TrianglePacketNode> ) // RTIP 3.1
	{
		const rtip31::TrianglePacketNode* packetNodes  = reinterpret_cast<const rtip31::TrianglePacketNode*>( nodes );
		hiprtHit                          secondHit    = hit;
		uint32_t                          triPairIndex = typeToTriPairIndex( getNodeType( leafIndex ) );

		// testTrianglePair() deliberately leaves the normals unset: fetching one decodes a whole triangle out
		// of the compressed packet (a descriptor read plus three unaligned vertex reads), far too expensive to
		// pay for every candidate hit. Remember which triangle won instead and decode it once, at the end.
		// A user filter may inspect hit.normal, so in that case it is still materialized up front.
		uint32_t normalAddr = InvalidValue;
		uint32_t normalPair = 0u;
		uint32_t normalTri  = 0u;

		if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
		{
			while ( true )
			{
				const uint32_t pairAddr  = leafAddr;
				const uint32_t pairIndex = triPairIndex;

				bool     nodeEnd  = false;
				bool     rangeEnd = false;
				uint32_t hitMask =
					this->testTrianglePair( ray, packetNodes, leafAddr, triPairIndex, hit, secondHit, nodeEnd, rangeEnd );

				bool firstHasHit  = hitMask & 1;
				bool secondHasHit = hitMask & 2;
				if ( useFilter )
				{
					if ( firstHasHit ) hit.normal = packetNodes[pairAddr].getNormal( pairIndex, 0 );
					if ( secondHasHit ) secondHit.normal = packetNodes[pairAddr].getNormal( pairIndex, 1 );
					if ( firstHasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit ) )
						firstHasHit = false;
					if ( secondHasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, secondHit ) )
						secondHasHit = false;
				}

				if ( ( triangleMask & Triangle0Processed ) == 0 )
				{
					hasHit = firstHasHit;
					if ( hasHit )
					{
						normalAddr = pairAddr;
						normalPair = pairIndex;
						normalTri  = 0u;
					}
					triangleMask |= Triangle0Processed;
				}

				if ( !hasHit )
				{
					if ( secondHasHit )
					{
						hit.t      = secondHit.t;
						hit.primID = secondHit.primID;
						hit.uv     = secondHit.uv;
						// Only a filter materializes secondHit.normal; otherwise the winning
						// triangle is decoded once at the end of the function.
						if ( useFilter ) hit.normal = secondHit.normal;
						hasHit     = true;
						normalAddr = pairAddr;
						normalPair = pairIndex;
						normalTri  = 1u;
					}
					triangleMask |= Triangle1Processed;
				}

				if ( !secondHasHit || ( triangleMask & Triangle1Processed ) != 0 ) // !( secondHasHit && !secondProcessed )
				{
					triPairIndex++;
					triangleMask = 0;
					if ( nodeEnd )
					{
						triPairIndex = 0;
						leafAddr++;
					}

					if ( rangeEnd )
					{
						triangleMask = InvalidValue; // indicate range end by 'invalid value'
						break;
					}
				}

				if ( hasHit )
				{
					leafIndex = encodeNodeIndex( leafAddr, triPairIndexToType( triPairIndex ) );
					break;
				}
			}
		}
		else
		{
			hit.t             = ray.maxT;
			hiprtHit firstHit = hit;
			while ( true )
			{
				const uint32_t pairAddr  = leafAddr;
				const uint32_t pairIndex = triPairIndex;

				bool     nodeEnd  = false;
				bool     rangeEnd = false;
				uint32_t hitMask =
					this->testTrianglePair( ray, packetNodes, leafAddr, triPairIndex, firstHit, secondHit, nodeEnd, rangeEnd );

				bool firstHasHit  = hitMask & 1;
				bool secondHasHit = hitMask & 2;
				if ( useFilter )
				{
					if ( firstHasHit ) firstHit.normal = packetNodes[pairAddr].getNormal( pairIndex, 0 );
					if ( secondHasHit ) secondHit.normal = packetNodes[pairAddr].getNormal( pairIndex, 1 );
					if ( firstHasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, firstHit ) )
						firstHasHit = false;
					if ( secondHasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, secondHit ) )
						secondHasHit = false;
				}

				if ( firstHasHit && ( !hasHit || hit.t > firstHit.t ) )
				{
					hit.t      = firstHit.t;
					hit.primID = firstHit.primID;
					hit.uv     = firstHit.uv;
					// Only a filter materializes firstHit.normal; otherwise the winning
					// triangle is decoded once at the end of the function.
					if ( useFilter ) hit.normal = firstHit.normal;
					hasHit     = true;
					normalAddr = pairAddr;
					normalPair = pairIndex;
					normalTri  = 0u;
				}

				if ( secondHasHit && ( !hasHit || hit.t > secondHit.t ) )
				{
					hit.t      = secondHit.t;
					hit.primID = secondHit.primID;
					hit.uv     = secondHit.uv;
					// Only a filter materializes secondHit.normal; otherwise the winning
					// triangle is decoded once at the end of the function.
					if ( useFilter ) hit.normal = secondHit.normal;
					hasHit     = true;
					normalAddr = pairAddr;
					normalPair = pairIndex;
					normalTri  = 1u;
				}

				if ( rangeEnd ) break;

				triPairIndex++;
				if ( nodeEnd )
				{
					triPairIndex = 0;
					leafAddr++;
				}
			}
		}

		// Decode the winning triangle once. Under a user filter the normal is already in place.
		if ( hasHit && !useFilter ) hit.normal = packetNodes[normalAddr].getNormal( normalPair, normalTri );
	}
	else
	{
		const rtip11::TrianglePairNode* pairNodes = reinterpret_cast<const rtip11::TrianglePairNode*>( nodes );

		if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
		{
			if ( ( triangleMask & Triangle0Processed ) == 0 )
			{
				hasHit = this->testTriangle( ray, invD, pairNodes, leafAddr, TriangleType, hit );
				if ( useFilter && hasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit ) )
					hasHit = false;
				triangleMask |= Triangle0Processed;
			}

			if ( !hasHit )
			{
				hasHit = this->testTriangle( ray, invD, pairNodes, leafAddr, TriangleType + 1, hit );
				if ( useFilter && hasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit ) )
					hasHit = false;
				triangleMask |= Triangle1Processed;
			}

			if ( triangleMask & Triangle1Processed ) triangleMask = InvalidValue;
		}
		else
		{
			hasHit = this->testTriangle( ray, invD, pairNodes, leafAddr, TriangleType, hit );
			if ( useFilter && hasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit ) )
				hasHit = false;

			hiprtHit secondHit    = hit;
			bool     secondHasHit = this->testTriangle( ray, invD, pairNodes, leafAddr, TriangleType + 1, secondHit );
			if ( useFilter && secondHasHit && filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, secondHit ) )
				secondHasHit = false;

			if ( secondHasHit && ( !hasHit || hit.t > secondHit.t ) )
			{
				hit.t      = secondHit.t;
				hit.normal = secondHit.normal;
				hit.primID = secondHit.primID;
				hit.uv     = secondHit.uv;
				hasHit     = true;
			}
		}
	}
	return hasHit;
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE bool TraversalBase<Stack, TraversalType>::testTriangle(
	const hiprtRay&                 ray,
	[[maybe_unused]] const float3&  invD,
	const rtip11::TrianglePairNode* nodes,
	const uint32_t                  leafAddr,
	const uint32_t                  leafType,
	hiprtHit&                       hit )
{
	const rtip11::TrianglePairNode& node = nodes[leafAddr];
	if ( node.getPrimIndex( 0 ) == node.getPrimIndex( 1 ) && leafType == TriangleType + 1 ) return false;
	bool hasHit = false;
#if HIPRT_RTIP < 11
	hasHit =
		node.m_triPair.fetchTriangle( leafType & 1 ).intersect( ray, hit.uv, hit.t, node.m_flags >> ( ( leafType & 1 ) * 8 ) );
	if ( hasHit )
	{
		hit.primID = leafType & 1 ? node.getPrimIndex( 1 ) : node.getPrimIndex( 0 );
		hit.normal = node.getNormal( leafType & 1 );
	}
#else
	auto result = __builtin_amdgcn_image_bvh_intersect_ray_l(
		encodeBaseAddr( nodes, encodeNodeIndex( leafAddr, leafType ) ),
		ray.maxT,
		{ ray.origin.x, ray.origin.y, ray.origin.z, 0.0f },
		{ ray.direction.x, ray.direction.y, ray.direction.z, 0.0f },
		{ invD.x, invD.y, invD.z, 0.0f },
		{ m_descriptor.x, m_descriptor.y, m_descriptor.z, m_descriptor.w } );
	float invDenom = __ocml_native_recip_f32( __int_as_float( result[1] ) );
	float t        = __int_as_float( result[0] ) * invDenom;
	hasHit         = ray.minT <= t && t <= ray.maxT;
	if ( hasHit )
	{
		hit.t      = t;
		hit.uv.x   = __int_as_float( result[2] ) * invDenom;
		hit.uv.y   = __int_as_float( result[3] ) * invDenom;
		hit.primID = leafType & 1 ? node.getPrimIndex( 1 ) : node.getPrimIndex( 0 );
		hit.normal = node.getNormal( leafType & 1 );
	}
#endif
	return hasHit;
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE uint32_t TraversalBase<Stack, TraversalType>::testTrianglePair(
	[[maybe_unused]] const hiprtRay&                   ray,
	[[maybe_unused]] const rtip31::TrianglePacketNode* nodes,
	[[maybe_unused]] const uint32_t                    leafAddr,
	[[maybe_unused]] const uint32_t                    triPairIndex,
	[[maybe_unused]] hiprtHit&                         hit0,
	[[maybe_unused]] hiprtHit&                         hit1,
	bool&                                              nodeEnd,
	bool&                                              rangeEnd )
{
#if HIPRT_RTIP >= 31
	hip_float3 dummy0, dummy1;
	auto       result = __builtin_amdgcn_image_bvh8_intersect_ray(
        encodeBaseAddr( nodes ),
        ray.maxT,
        0xff,
        { ray.origin.x, ray.origin.y, ray.origin.z },
        { ray.direction.x, ray.direction.y, ray.direction.z },
        encodeNodeIndex( leafAddr, triPairIndexToType( triPairIndex ) ),
        { m_descriptor.x, m_descriptor.y, m_descriptor.z, m_descriptor.w },
        &dummy0,
        &dummy1 );

	uint32_t hitMask = 0;
	{
		float t      = __int_as_float( result[0] );
		bool  hasHit = ray.minT <= t && t <= ray.maxT;
		hitMask |= hasHit ? 1 : 0;
		if ( hasHit )
		{
			hit0.t      = t;
			hit0.uv.x   = __int_as_float( result[1] );
			hit0.uv.y   = __int_as_float( result[2] );
			hit0.primID = result[3] >> 1;
		}
	}

	{
		float t      = __int_as_float( result[4] );
		bool  hasHit = ray.minT <= t && t <= ray.maxT;
		hitMask |= hasHit ? 2 : 0;
		if ( hasHit )
		{
			hit1.t      = t;
			hit1.uv.x   = __int_as_float( result[5] );
			hit1.uv.y   = __int_as_float( result[6] );
			hit1.primID = result[7] >> 1;
		}
	}

	nodeEnd  = ( result[8] & 3 ) == 1;
	rangeEnd = ( result[8] & 3 ) == 3;

	return hitMask;
#else
	// Triangle packets only exist on RTIP 3.1, and testTriangleNode() discards this call on older
	// targets, so we never get here. Report both ends anyway: the callers loop until rangeEnd, so
	// returning 0 on its own would spin.
	nodeEnd  = true;
	rangeEnd = true;
	return 0;
#endif
}

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE GeomTraversal<Stack, PrimitiveNode, TraversalType>::GeomTraversal(
	const hiprtGeometry      geometry,
	const hiprtRay&          ray,
	Stack&                   stack,
	const hiprtTraversalHint hint,
	void*                    payload,
	hiprtFuncTable           funcTable,
	const uint32_t           rayType )
	: TraversalBase<Stack, TraversalType>( ray, stack, hint, payload, funcTable, rayType ), m_leafIndex( InvalidValue )
{
	const GeomHeader* geomHeader = reinterpret_cast<const GeomHeader*>( geometry );
	this->m_boxNodes             = geomHeader->m_boxNodes;
	m_primNodes                  = reinterpret_cast<PrimitiveNode*>( geomHeader->m_primNodes );
	m_geomType                   = geomHeader->m_geomType;
	this->m_stack.reset();
}

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE hiprtHit GeomTraversal<Stack, PrimitiveNode, TraversalType>::getNextHit()
{
	hiprtRay ray = this->m_ray;
	float3   invD;
	if constexpr ( Rtip < 31 ) invD = rcp( this->m_ray.direction );

	if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
	{
		if ( m_leafIndex == InvalidValue && isLeafNode( this->m_nodeIndex ) )
		{
			m_leafIndex       = this->m_nodeIndex;
			this->m_nodeIndex = this->m_stack.pop();
		}
	}
	hiprtHit result;

	if ( this->m_stack.empty() ) this->m_stack.push( InvalidValue );

	while ( this->m_nodeIndex != InvalidValue || m_leafIndex != InvalidValue )
	{
		while ( isInternalNode( this->m_nodeIndex ) )
		{
			if ( !this->testInternalNode( ray, invD, this->m_boxNodes, this->m_nodeIndex ) )
				this->m_nodeIndex = this->m_stack.pop();

			if ( this->m_state == hiprtTraversalStateStackOverflow ) return hiprtHit();

			if ( isLeafNode( this->m_nodeIndex ) && m_leafIndex == InvalidValue )
			{
				m_leafIndex       = this->m_nodeIndex;
				this->m_nodeIndex = this->m_stack.pop();
			}

			if ( !hiprt::any( m_leafIndex == InvalidValue ) ) break;
		}

		while ( m_leafIndex != InvalidValue )
		{
			hiprtHit hit;
			if ( testLeafNode( ray, invD, m_leafIndex, this->m_triangleMask, hit ) )
			{
				if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
				{
					if ( getNodeType( m_leafIndex ) == CustomType || this->m_triangleMask == InvalidValue )
					{
						this->m_triangleMask = 0;
						m_leafIndex          = InvalidValue;
					}
					this->m_state = hiprtTraversalStateHit;
					return hit;
				}
				else
				{
					result   = hit;
					ray.maxT = hit.t;
				}
			}

			this->m_triangleMask = 0;
			m_leafIndex          = InvalidValue;
			if ( isLeafNode( this->m_nodeIndex ) )
			{
				m_leafIndex       = this->m_nodeIndex;
				this->m_nodeIndex = this->m_stack.pop();
			}
		}
	}

	if ( this->m_state != hiprtTraversalStateStackOverflow ) this->m_state = hiprtTraversalStateFinished;

	return result;
}

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE bool GeomTraversal<Stack, PrimitiveNode, TraversalType>::testLeafNode(
	const hiprtRay&                ray,
	[[maybe_unused]] const float3& invD,
	uint32_t&                      leafIndex,
	[[maybe_unused]] uint32_t&     triangleMask,
	hiprtHit&                      hit )
{
	bool hasHit = false;
	if constexpr ( is_same_v<PrimitiveNode, TriangleNode> )
	{
		hasHit = this->testTriangleNode( ray, invD, m_primNodes, m_geomType, leafIndex, triangleMask, hit );
	}
	else
	{
		const bool useFilter = m_geomType != InvalidValue && this->m_tableHeader.funcDataSets != nullptr;
		hit.primID           = m_primNodes[getNodeAddr( leafIndex )].m_primIndex;
		hasHit = intersectFunc( m_geomType >> 1, this->m_rayType, this->m_tableHeader, ray, this->m_payload, hit );
		if ( useFilter && hasHit &&
		     filterFunc( m_geomType >> 1, this->m_rayType, this->m_tableHeader, ray, this->m_payload, hit ) )
			hasHit = false;
		if ( !hasHit ) hit.primID = InvalidValue;
	}
	return hasHit;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE SceneTraversal<Stack, InstanceStack, TraversalType>::SceneTraversal(
	const hiprtScene         scene,
	const hiprtRay&          ray,
	Stack&                   stack,
	InstanceStack&           instanceStack,
	const hiprtRayMask       mask,
	const hiprtTraversalHint hint,
	void*                    payload,
	hiprtFuncTable           funcTable,
	const uint32_t           rayType,
	const float              time )
	: TraversalBase<Stack, TraversalType>( ray, stack, hint, payload, funcTable, rayType ), m_instanceStack( instanceStack ),
	  m_mask( mask ), m_level( 0u ), m_time( time )
{
	const SceneHeader* sceneHeader = reinterpret_cast<const SceneHeader*>( scene );
	this->m_boxNodes               = sceneHeader->m_boxNodes;
	m_instanceNodes                = sceneHeader->m_primNodes;
	m_frames                       = sceneHeader->m_frames;
	this->m_stack.reset();
	m_instanceIndex = InvalidValue;
	instanceId()    = InvalidValue;
	if constexpr ( !is_same_v<InstanceStack, hiprtEmptyInstanceStack> )
	{
		m_instanceStack.reset();
		m_scene = sceneHeader;
	}
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE hiprtHit SceneTraversal<Stack, InstanceStack, TraversalType>::getNextHit()
{
	const BoxNode* nodes     = this->m_boxNodes;
	const void*    primNodes = nullptr;
	uint32_t       geomType  = InvalidValue;

	hiprtRay ray = this->m_ray;
	float3   invD;
	if constexpr ( Rtip < 31 )
	{
		if ( instanceId() == InvalidValue ) invD = rcp( this->m_ray.direction );
	}

	if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
	{
		if ( instanceId() != InvalidValue )
		{
			transformRay( m_instanceIndex, ray, invD );
			nodes     = m_instanceNodes[getNodeAddr( m_instanceIndex )].getHeader<GeomHeader>()->m_boxNodes;
			primNodes = m_instanceNodes[getNodeAddr( m_instanceIndex )].getHeader<GeomHeader>()->m_primNodes;
			geomType  = m_instanceNodes[getNodeAddr( m_instanceIndex )].getHeader<GeomHeader>()->m_geomType;
		}
	}

	hiprtHit result;

	if ( this->m_stack.empty() ) this->m_stack.push( InvalidValue );

	while ( this->m_nodeIndex != InvalidValue && this->m_state != hiprtTraversalStateStackOverflow )
	{
		if ( isInternalNode( this->m_nodeIndex ) )
		{
			if ( this->testInternalNode( ray, invD, nodes, this->m_nodeIndex ) ) continue;
		}
		else
		{
			if ( instanceId() != InvalidValue )
			{
				hiprtHit hit;
				if ( testLeafNode( primNodes, ray, invD, geomType, this->m_nodeIndex, this->m_triangleMask, hit ) )
				{
					if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
					{
						this->m_state = hiprtTraversalStateHit;
						if ( getNodeType( this->m_nodeIndex ) == CustomType || this->m_triangleMask == InvalidValue )
						{
							this->m_triangleMask = 0;
							this->m_nodeIndex    = this->m_stack.pop();

							while ( this->m_nodeIndex == InvalidValue && !this->m_stack.empty() )
							{
								if constexpr ( !is_same_v<InstanceStack, hiprtEmptyInstanceStack> )
								{
									if ( instanceId() == InvalidValue )
									{
										hiprtInstanceStackEntry instanceEntry = m_instanceStack.pop();
										this->m_ray                           = instanceEntry.ray;
										m_scene = reinterpret_cast<SceneHeader*>( instanceEntry.scene );
										m_level--;

										this->m_boxNodes = m_scene->m_boxNodes;
										m_instanceNodes  = m_scene->m_primNodes;
										m_frames         = m_scene->m_frames;
									}
								}

								instanceId()      = InvalidValue;
								this->m_nodeIndex = this->m_stack.pop();
							}
						}
						return hit;
					}
					else
					{
						ray.maxT = hit.t;
						result   = hit;
					}
				}
			}
			else
			{
				const uint32_t instanceAddr = getNodeAddr( this->m_nodeIndex );
				if ( ( m_instanceNodes[instanceAddr].m_mask & m_mask ) && transformRay( this->m_nodeIndex, ray, invD ) )
				{
					if ( this->m_stack.vacancy() < 1 )
					{
						this->m_state = hiprtTraversalStateStackOverflow;
						continue;
					}

					m_instanceIndex = this->m_nodeIndex;
					instanceId()    = m_instanceNodes[instanceAddr].m_primIndex;

					this->m_nodeIndex = RootIndex;
					this->m_stack.push( InvalidValue );

					if constexpr ( !is_same_v<InstanceStack, hiprtEmptyInstanceStack> )
					{
						if ( m_instanceNodes[instanceAddr].m_type == hiprtInstanceTypeScene )
						{
							m_instanceStack.push(
								{ this->m_ray, reinterpret_cast<hiprtScene>( const_cast<SceneHeader*>( m_scene ) ) } );
							this->m_ray = ray;
							m_scene     = m_instanceNodes[instanceAddr].getHeader<SceneHeader>();
							m_level++;
							instanceId() = InvalidValue;

							this->m_boxNodes = m_scene->m_boxNodes;
							m_instanceNodes  = m_scene->m_primNodes;
							m_frames         = m_scene->m_frames;

							nodes = this->m_boxNodes;
							continue;
						}
					}
					nodes     = m_instanceNodes[instanceAddr].getHeader<GeomHeader>()->m_boxNodes;
					primNodes = m_instanceNodes[instanceAddr].getHeader<GeomHeader>()->m_primNodes;
					geomType  = m_instanceNodes[instanceAddr].getHeader<GeomHeader>()->m_geomType;
					continue;
				}
			}
		}

		this->m_triangleMask = 0;
		this->m_nodeIndex    = this->m_stack.pop();
		while ( this->m_nodeIndex == InvalidValue && !this->m_stack.empty() )
		{
			if constexpr ( !is_same_v<InstanceStack, hiprtEmptyInstanceStack> )
			{
				if ( instanceId() == InvalidValue )
				{
					hiprtInstanceStackEntry instanceEntry = m_instanceStack.pop();
					this->m_ray                           = instanceEntry.ray;
					m_scene                               = reinterpret_cast<SceneHeader*>( instanceEntry.scene );
					m_level--;

					this->m_boxNodes = m_scene->m_boxNodes;
					m_instanceNodes  = m_scene->m_primNodes;
					m_frames         = m_scene->m_frames;
				}
			}
			instanceId()      = InvalidValue;
			this->m_nodeIndex = this->m_stack.pop();
			nodes             = this->m_boxNodes;
			restoreRay( ray, invD );
		}
	}

	if ( this->m_state != hiprtTraversalStateStackOverflow ) this->m_state = hiprtTraversalStateFinished;

	return result;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE const uint32_t& SceneTraversal<Stack, InstanceStack, TraversalType>::instanceId() const
{
	if constexpr ( !is_same_v<InstanceStack, hiprtEmptyInstanceStack> )
		return m_instanceIds[m_level];
	else
		return m_instanceId;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE uint32_t& SceneTraversal<Stack, InstanceStack, TraversalType>::instanceId()
{
	if constexpr ( !is_same_v<InstanceStack, hiprtEmptyInstanceStack> )
		return m_instanceIds[m_level];
	else
		return m_instanceId;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE bool SceneTraversal<Stack, InstanceStack, TraversalType>::transformRay(
	const uint32_t nodeIndex, hiprtRay& ray, [[maybe_unused]] float3& invD )
{
	const InstanceNode& instanceNode = m_instanceNodes[getNodeAddr( nodeIndex )];
	if ( instanceNode.m_identity == 0 )
	{
		if ( instanceNode.m_static != 0 )
		{
#if HIPRT_RTIP >= 31
			hip_float3 origin, direction;
			auto       result = __builtin_amdgcn_image_bvh8_intersect_ray(
                encodeBaseAddr( m_instanceNodes ),
                ray.maxT,
                0xff,
                { ray.origin.x, ray.origin.y, ray.origin.z },
                { ray.direction.x, ray.direction.y, ray.direction.z },
                nodeIndex,
                { this->m_descriptor.x, this->m_descriptor.y, this->m_descriptor.z, this->m_descriptor.w },
                &origin,
                &direction );

			if ( result[7] == InvalidValue ) return false;

			ray.origin    = { origin.x, origin.y, origin.z };
			ray.direction = { direction.x, direction.y, direction.z };
#else
			ray = instanceNode.transformRay( ray );
#endif
		}
		else
		{
			const Transform tr( m_frames, instanceNode.m_transform.frameIndex, instanceNode.m_transform.frameCount );
			ray = tr.transformRay( ray, m_time );
		}
		if constexpr ( Rtip < 31 ) invD = rcp( ray.direction );
	}

	return true;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE void
SceneTraversal<Stack, InstanceStack, TraversalType>::restoreRay( hiprtRay& ray, float3& invD ) const
{
	ray.origin    = this->m_ray.origin;
	ray.direction = this->m_ray.direction;
	if constexpr ( Rtip < 31 ) invD = rcp( this->m_ray.direction );
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE bool SceneTraversal<Stack, InstanceStack, TraversalType>::testLeafNode(
	const void*                    primNodes,
	const hiprtRay&                ray,
	[[maybe_unused]] const float3& invD,
	const uint32_t                 geomType,
	uint32_t&                      leafIndex,
	[[maybe_unused]] uint32_t&     triangleMask,
	hiprtHit&                      hit )
{
	bool hasHit = false;
	if constexpr ( !is_same_v<InstanceStack, hiprtEmptyInstanceStack> )
	{
		HIPRT_UNROLL
		for ( uint32_t i = 0; i < MaxInstanceLevels; ++i )
		{
			if ( i <= m_level )
				hit.instanceIDs[i] = m_instanceIds[i];
			else
				hit.instanceIDs[i] = InvalidValue;
		}
	}
	else
	{
		hit.instanceID = instanceId();
	}

	if ( geomType & 1 )
	{
		const TriangleNode* nodes = reinterpret_cast<const TriangleNode*>( primNodes );
		hasHit                    = this->testTriangleNode( ray, invD, nodes, geomType, leafIndex, triangleMask, hit );
	}
	else
	{
		const bool        useFilter = geomType != InvalidValue && this->m_tableHeader.funcDataSets != nullptr;
		const CustomNode* nodes     = reinterpret_cast<const CustomNode*>( primNodes );
		hit.primID                  = nodes[getNodeAddr( leafIndex )].m_primIndex;
		hasHit = intersectFunc( geomType >> 1, this->m_rayType, this->m_tableHeader, ray, this->m_payload, hit );
		if ( useFilter && hasHit &&
		     filterFunc( geomType >> 1, this->m_rayType, this->m_tableHeader, ray, this->m_payload, hit ) )
			hasHit = false;
		if ( !hasHit ) hit.primID = InvalidValue;
	}

	return hasHit;
}

template <typename PrimitiveNode, hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE GeomTraversalPrivateStack<PrimitiveNode, TraversalType>::GeomTraversalPrivateStack(
	hiprtGeometry            geom,
	const hiprtRay&          ray,
	const hiprtTraversalHint hint,
	void*                    payload,
	hiprtFuncTable           funcTable,
	const uint32_t           rayType )
	: m_traversal( geom, ray, this->m_stack, hint, payload, funcTable, rayType )
{
}

template <typename PrimitiveNode, hiprtTraversalType TraversalType>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE hiprtHit GeomTraversalPrivateStack<PrimitiveNode, TraversalType>::getNextHit()
{
	return m_traversal.getNextHit();
}

template <typename PrimitiveNode, hiprtTraversalType TraversalType>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE hiprtTraversalState
GeomTraversalPrivateStack<PrimitiveNode, TraversalType>::getCurrentState() const
{
	return m_traversal.getCurrentState();
}

template <hiprtTraversalType TraversalType>
HIPRT_DEVICE HIPRT_INLINE SceneTraversalPrivateStack<TraversalType>::SceneTraversalPrivateStack(
	hiprtScene               scene,
	const hiprtRay&          ray,
	const hiprtRayMask       mask,
	const hiprtTraversalHint hint,
	void*                    payload,
	hiprtFuncTable           funcTable,
	const uint32_t           rayType,
	const float              time )
	: m_traversal( scene, ray, this->m_stack, m_instanceStack, mask, hint, payload, funcTable, rayType, time )
{
}

template <hiprtTraversalType TraversalType>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE hiprtHit SceneTraversalPrivateStack<TraversalType>::getNextHit()
{
	return m_traversal.getNextHit();
}

template <hiprtTraversalType TraversalType>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE hiprtTraversalState SceneTraversalPrivateStack<TraversalType>::getCurrentState() const
{
	return m_traversal.getCurrentState();
}
} // namespace hiprt

// transformation getters for two-level BVHs
HIPRT_DEVICE hiprtFrameSRT hiprtGetObjectToWorldFrameSRT( const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );

	hiprt::SRTFrame srtFrame{};
#if defined( HIPRT_MATRIX_FRAME )
	hiprtFrameMatrix matrixFrame{};
	memcpy( matrixFrame.matrix, frame.m_matrix, sizeof( matrixFrame.matrix ) );
	srtFrame = hiprt::SRTFrame( matrixFrame );
#else
	srtFrame = frame;
#endif

	hiprtFrameSRT result{};
	result.rotation    = hiprt::qtToAxisAngle( srtFrame.m_rotation );
	result.scale       = srtFrame.m_scale;
	result.shear       = srtFrame.m_shear;
	result.translation = srtFrame.m_translation;
	result.time        = time;

	return result;
}

HIPRT_DEVICE hiprtFrameSRT hiprtGetWorldToObjectFrameSRT( const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );

	float matrixInv[3][4];
	hiprt::computeInvTransformMatrix( frame, matrixInv );

	hiprtFrameMatrix matrixFrame{};
	memcpy( matrixFrame.matrix, matrixInv, sizeof( matrixInv ) );

	const hiprt::SRTFrame invSrtFrame( matrixFrame );

	hiprtFrameSRT result{};
	result.rotation    = hiprt::qtToAxisAngle( invSrtFrame.m_rotation );
	result.scale       = invSrtFrame.m_scale;
	result.shear       = invSrtFrame.m_shear;
	result.translation = invSrtFrame.m_translation;
	result.time        = time;

	return result;
}

HIPRT_DEVICE hiprtFrameMatrix
hiprtGetObjectToWorldFrameMatrix( const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );

	hiprtFrameMatrix result{};
#if defined( HIPRT_MATRIX_FRAME )
	memcpy( result.matrix, frame.m_matrix, sizeof( result.matrix ) );
#else
	float Q[3][3];
	hiprt::qtToRotationMatrix( frame.m_rotation, Q );
	HIPRT_UNROLL
	for ( uint32_t i = 0; i < 3; ++i )
	{
		result.matrix[i][0] = Q[i][0] * frame.m_scale.x;
		result.matrix[i][1] = Q[i][1] * frame.m_scale.y + Q[i][0] * frame.m_shear.x;
		result.matrix[i][2] = Q[i][2] * frame.m_scale.z + Q[i][1] * frame.m_shear.z + Q[i][0] * frame.m_shear.y;
	}
	result.matrix[0][3] = frame.m_translation.x;
	result.matrix[1][3] = frame.m_translation.y;
	result.matrix[2][3] = frame.m_translation.z;
#endif
	result.time = time;

	return result;
}

HIPRT_DEVICE hiprtFrameMatrix
hiprtGetWorldToObjectFrameMatrix( const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );

	float matrixInv[3][4];
	hiprt::computeInvTransformMatrix( frame, matrixInv );

	hiprtFrameMatrix result{};
	memcpy( result.matrix, matrixInv, sizeof( result.matrix ) );
	result.time = time;
	return result;
}

// transformation getters for MLAS
HIPRT_DEVICE hiprtFrameSRT hiprtGetObjectToWorldFrameSRT(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprtFrameMatrix matrixFrame = hiprtGetObjectToWorldFrameMatrix( scene, instanceIDs, time );
	hiprt::SRTFrame        srtFrame    = hiprt::SRTFrame( matrixFrame );

	hiprtFrameSRT result{};
	result.rotation    = hiprt::qtToAxisAngle( srtFrame.m_rotation );
	result.scale       = srtFrame.m_scale;
	result.shear       = srtFrame.m_shear;
	result.translation = srtFrame.m_translation;
	result.time        = time;

	return result;
}

HIPRT_DEVICE hiprtFrameSRT hiprtGetWorldToObjectFrameSRT(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprtFrameMatrix matrixFrame = hiprtGetWorldToObjectFrameMatrix( scene, instanceIDs, time );
	hiprt::SRTFrame        srtFrame    = hiprt::SRTFrame( matrixFrame );

	hiprtFrameSRT result{};
	result.rotation    = hiprt::qtToAxisAngle( srtFrame.m_rotation );
	result.scale       = srtFrame.m_scale;
	result.shear       = srtFrame.m_shear;
	result.translation = srtFrame.m_translation;
	result.time        = time;

	return result;
}

HIPRT_DEVICE hiprtFrameMatrix hiprtGetObjectToWorldFrameMatrix(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );

	hiprt::MatrixFrame matrixFrame{}; // default constructor sets identity
	HIPRT_UNROLL
	for ( uint32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		const hiprt::Transform tr = hiprt::Transform(
			sceneHeader->m_frames,
			sceneHeader->m_instances[instanceIDs[i]].m_frameIndex,
			sceneHeader->m_instances[instanceIDs[i]].m_frameCount );
		const hiprt::Frame frame = tr.interpolateFrames( time );

		hiprt::MatrixFrame curMatrixFrame{};
#if defined( HIPRT_MATRIX_FRAME )
		curMatrixFrame = frame;
#else
		float Q[3][3];
		hiprt::qtToRotationMatrix( frame.m_rotation, Q );
		HIPRT_UNROLL
		for ( uint32_t j = 0; j < 3; ++j )
		{
			curMatrixFrame.m_matrix[j][0] = Q[j][0] * frame.m_scale.x;
			curMatrixFrame.m_matrix[j][1] = Q[j][1] * frame.m_scale.y + Q[j][0] * frame.m_shear.x;
			curMatrixFrame.m_matrix[j][2] = Q[j][2] * frame.m_scale.z + Q[j][1] * frame.m_shear.z + Q[j][0] * frame.m_shear.y;
		}
		curMatrixFrame.m_matrix[0][3] = frame.m_translation.x;
		curMatrixFrame.m_matrix[1][3] = frame.m_translation.y;
		curMatrixFrame.m_matrix[2][3] = frame.m_translation.z;
#endif
		matrixFrame = hiprt::MatrixFrame::multiply( matrixFrame, curMatrixFrame );

		if ( sceneHeader->m_instances[instanceIDs[i]].m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = sceneHeader->m_instances[instanceIDs[i]].m_scene;
	}

	hiprtFrameMatrix result{};
	memcpy( result.matrix, matrixFrame.m_matrix, sizeof( result.matrix ) );
	result.time = time;

	return result;
}

HIPRT_DEVICE hiprtFrameMatrix hiprtGetWorldToObjectFrameMatrix(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );

	hiprt::MatrixFrame matrixFrame{}; // default constructor sets identity
	HIPRT_UNROLL
	for ( uint32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		const hiprt::Transform tr = hiprt::Transform(
			sceneHeader->m_frames,
			sceneHeader->m_instances[instanceIDs[i]].m_frameIndex,
			sceneHeader->m_instances[instanceIDs[i]].m_frameCount );
		const hiprt::Frame frame = tr.interpolateFrames( time );

		hiprt::MatrixFrame curMatrixFrame{};
		hiprt::computeInvTransformMatrix( frame, curMatrixFrame.m_matrix );
		matrixFrame = hiprt::MatrixFrame::multiply( curMatrixFrame, matrixFrame );

		if ( sceneHeader->m_instances[instanceIDs[i]].m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = sceneHeader->m_instances[instanceIDs[i]].m_scene;
	}

	hiprtFrameMatrix result{};
	memcpy( result.matrix, matrixFrame.m_matrix, sizeof( result.matrix ) );
	result.time = time;

	return result;
}

// transformation of points and vectors between object space and world space
HIPRT_DEVICE float3
hiprtPointObjectToWorld( const float3& point, const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );
	return frame.transform( point );
}

HIPRT_DEVICE float3
hiprtPointWorldToObject( const float3& point, const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );
	return frame.invTransform( point );
}

HIPRT_DEVICE float3
hiprtVectorObjectToWorld( const float3& vector, const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );
	return frame.transformVector( vector );
}

HIPRT_DEVICE float3
hiprtVectorWorldToObject( const float3& vector, const hiprtScene scene, const uint32_t instanceID, const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	const hiprt::Transform    tr(
        sceneHeader->m_frames,
        sceneHeader->m_instances[instanceID].m_frameIndex,
        sceneHeader->m_instances[instanceID].m_frameCount );
	const hiprt::Frame frame = tr.interpolateFrames( time );
	return frame.invTransformVector( vector );
}

HIPRT_DEVICE float3 hiprtPointObjectToWorld(
	const float3& point, const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprt::SceneHeader* sceneHeaders[hiprtMaxInstanceLevels];
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	float3                    p           = point;

	uint32_t depth = 0;
	HIPRT_UNROLL
	for ( uint32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		sceneHeaders[i]      = sceneHeader;
		const auto& instance = sceneHeader->m_instances[instanceIDs[i]];
		++depth;
		if ( instance.m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = instance.m_scene;
	}

	HIPRT_UNROLL
	for ( uint32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		const int32_t j = static_cast<int32_t>( depth ) - 1 - static_cast<int32_t>( i );
		if ( j >= 0 )
		{
			sceneHeader                     = sceneHeaders[j];
			const auto&            instance = sceneHeader->m_instances[instanceIDs[j]];
			const hiprt::Transform tr = hiprt::Transform( sceneHeader->m_frames, instance.m_frameIndex, instance.m_frameCount );
			const hiprt::Frame     frame = tr.interpolateFrames( time );
			p                            = frame.transform( p );
		}
	}

	return p;
}

HIPRT_DEVICE float3 hiprtPointWorldToObject(
	const float3& point, const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	float3                    p           = point;

	HIPRT_UNROLL
	for ( int32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		const auto&            instance = sceneHeader->m_instances[instanceIDs[i]];
		const hiprt::Transform tr    = hiprt::Transform( sceneHeader->m_frames, instance.m_frameIndex, instance.m_frameCount );
		const hiprt::Frame     frame = tr.interpolateFrames( time );
		p                            = frame.invTransform( p );
		if ( instance.m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = instance.m_scene;
	}

	return p;
}

HIPRT_DEVICE float3 hiprtVectorObjectToWorld(
	const float3& vector, const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprt::SceneHeader* sceneHeaders[hiprtMaxInstanceLevels];
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	float3                    v           = vector;

	uint32_t depth = 0;
	HIPRT_UNROLL
	for ( uint32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		sceneHeaders[i]      = sceneHeader;
		const auto& instance = sceneHeader->m_instances[instanceIDs[i]];
		++depth;
		if ( instance.m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = instance.m_scene;
	}

	HIPRT_UNROLL
	for ( uint32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		const int32_t j = static_cast<int32_t>( depth ) - 1 - static_cast<int32_t>( i );
		if ( j >= 0 )
		{
			sceneHeader                     = sceneHeaders[j];
			const auto&            instance = sceneHeader->m_instances[instanceIDs[j]];
			const hiprt::Transform tr = hiprt::Transform( sceneHeader->m_frames, instance.m_frameIndex, instance.m_frameCount );
			const hiprt::Frame     frame = tr.interpolateFrames( time );
			v                            = frame.transformVector( v );
		}
	}

	return v;
}

HIPRT_DEVICE float3 hiprtVectorWorldToObject(
	const float3& vector, const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time )
{
	const hiprt::SceneHeader* sceneHeader = reinterpret_cast<const hiprt::SceneHeader*>( scene );
	float3                    v           = vector;

	HIPRT_UNROLL
	for ( int32_t i = 0; i < hiprtMaxInstanceLevels; ++i )
	{
		const auto&            instance = sceneHeader->m_instances[instanceIDs[i]];
		const hiprt::Transform tr    = hiprt::Transform( sceneHeader->m_frames, instance.m_frameIndex, instance.m_frameCount );
		const hiprt::Frame     frame = tr.interpolateFrames( time );
		v                            = frame.invTransformVector( v );
		if ( instance.m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = instance.m_scene;
	}

	return v;
}

// explicit template instatiation is required for bitcode linking
namespace hiprt
{
template class PrivateStack<uint32_t, 64u>;
template class GlobalStack<uint32_t, false>;
template class GlobalStack<uint32_t, true>;

template class PrivateStack<hiprtInstanceStackEntry, hiprtMaxInstanceLevels - 1u>;
template class GlobalStack<hiprtInstanceStackEntry, false>;
template class GlobalStack<hiprtInstanceStackEntry, true>;

template class GeomTraversal<hiprtPrivateStack, TriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversal<hiprtPrivateStack, TriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class GeomTraversal<hiprtPrivateStack, CustomNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversal<hiprtPrivateStack, CustomNode, hiprtTraversalTerminateAtAnyHit>;

template class GeomTraversal<hiprtGlobalStack, TriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversal<hiprtGlobalStack, TriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class GeomTraversal<hiprtGlobalStack, CustomNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversal<hiprtGlobalStack, CustomNode, hiprtTraversalTerminateAtAnyHit>;

template class GeomTraversal<hiprtDynamicStack, TriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversal<hiprtDynamicStack, TriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class GeomTraversal<hiprtDynamicStack, CustomNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversal<hiprtDynamicStack, CustomNode, hiprtTraversalTerminateAtAnyHit>;

template class GeomTraversalPrivateStack<TriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversalPrivateStack<TriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class GeomTraversalPrivateStack<CustomNode, hiprtTraversalTerminateAtClosestHit>;
template class GeomTraversalPrivateStack<CustomNode, hiprtTraversalTerminateAtAnyHit>;

template class SceneTraversal<hiprtPrivateStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtClosestHit>;
template class SceneTraversal<hiprtPrivateStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtAnyHit>;
template class SceneTraversal<hiprtPrivateStack, hiprtPrivateInstanceStack, hiprtTraversalTerminateAtClosestHit>;
template class SceneTraversal<hiprtPrivateStack, hiprtPrivateInstanceStack, hiprtTraversalTerminateAtAnyHit>;

template class SceneTraversal<hiprtGlobalStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtClosestHit>;
template class SceneTraversal<hiprtGlobalStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtAnyHit>;
template class SceneTraversal<hiprtGlobalStack, hiprtPrivateInstanceStack, hiprtTraversalTerminateAtClosestHit>;
template class SceneTraversal<hiprtGlobalStack, hiprtPrivateInstanceStack, hiprtTraversalTerminateAtAnyHit>;

template class SceneTraversal<hiprtDynamicStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtClosestHit>;
template class SceneTraversal<hiprtDynamicStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtAnyHit>;
template class SceneTraversal<hiprtDynamicStack, hiprtPrivateInstanceStack, hiprtTraversalTerminateAtClosestHit>;
template class SceneTraversal<hiprtDynamicStack, hiprtPrivateInstanceStack, hiprtTraversalTerminateAtAnyHit>;

template class SceneTraversalPrivateStack<hiprtTraversalTerminateAtClosestHit>;
template class SceneTraversalPrivateStack<hiprtTraversalTerminateAtAnyHit>;
} // namespace hiprt
