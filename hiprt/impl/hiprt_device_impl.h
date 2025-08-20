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

#if HIPRT_RTIP >= 20
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
	uint32_t					geomType,
	uint32_t					rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&				ray,
	void*						payload,
	hiprtHit&					hit );
HIPRT_DEVICE bool filterFunc(
	uint32_t					geomType,
	uint32_t					rayType,
	const hiprtFuncTableHeader& tableHeader,
	const hiprtRay&				ray,
	void*						payload,
	const hiprtHit&				hit );

namespace hiprt
{
enum
{
	Triangle0Processed = 1,
	Triangle1Processed = 2,
};

HIPRT_DEVICE HIPRT_INLINE float3 rcp( const float3 a )
{
#if defined( __USE_HWI__ )
	return float3{ __ocml_native_recip_f32( a.x ), __ocml_native_recip_f32( a.y ), __ocml_native_recip_f32( a.z ) };
#else
	return 1.0f / a;
#endif
}

template <typename StackEntry, uint32_t StackSize>
class PrivateStack
{
  public:
	HIPRT_DEVICE PrivateStack() : m_top( 0u ) {}

	HIPRT_DEVICE StackEntry pop() { return m_stackBuffer[--m_top]; }
	HIPRT_DEVICE void		push( StackEntry val ) { m_stackBuffer[m_top++] = val; }
	HIPRT_DEVICE bool		empty() const { return m_top == 0u; }
	HIPRT_DEVICE uint32_t	vacancy() const { return StackSize - m_top; }
	HIPRT_DEVICE void		reset() { m_top = 0u; }

  private:
	StackEntry m_stackBuffer[StackSize];
	uint32_t   m_top;
};

template <typename StackEntry, bool DynamicAssignment>
class GlobalStack
{
  public:
	static constexpr uint32_t Stride	= hiprt::WarpSize;
	static constexpr uint32_t LogStride = hiprt::Log2( Stride );

	HIPRT_DEVICE
	GlobalStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer );

	HIPRT_DEVICE ~GlobalStack();

	HIPRT_DEVICE StackEntry pop();
	HIPRT_DEVICE void		push( StackEntry val );
	HIPRT_DEVICE uint32_t	vacancy() const;
	HIPRT_DEVICE bool		empty() const;
	HIPRT_DEVICE void		reset();

  private:
	uint32_t*	m_globalStackLock;
	StackEntry* m_globalStackBuffer;
	StackEntry* m_sharedStackBuffer;
	uint32_t	m_globalStackSize;
	uint32_t	m_sharedStackSize;
	int32_t		m_globalIndex;
	int32_t		m_sharedIndex;
	uint32_t	m_sharedCount;
};

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE GlobalStack<StackEntry, DynamicAssignment>::GlobalStack(
	hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer )
{
	const uint32_t threadIndex = threadIdx.x + threadIdx.y * blockDim.x;
	const uint32_t warpIndex   = threadIndex >> LogStride;
	const uint32_t laneIndex   = threadIndex & ( Stride - 1 );

	const uint32_t sharedStackOffset = laneIndex + warpIndex * Stride * sharedStackBuffer.stackSize;
	m_sharedStackBuffer				 = reinterpret_cast<StackEntry*>( sharedStackBuffer.stackData ) + sharedStackOffset;
	m_sharedStackSize				 = sharedStackBuffer.stackSize;
	if constexpr ( DynamicAssignment )
	{
		const uint32_t warpsPerBlock	= hiprt::DivideRoundUp( blockDim.x * blockDim.y, Stride );
		const uint32_t activeWarps		= globalStackBuffer.stackCount >> LogStride;
		const uint32_t firstThreadIndex = __ffsll( static_cast<unsigned long long>( hiprt::ballot( true ) ) ) - 1;

		uint32_t  warpHash			= InvalidValue;
		uint32_t  warpHashCandidate = ( warpIndex + ( blockIdx.x + blockIdx.y * gridDim.x ) * warpsPerBlock ) % activeWarps;
		uint32_t* globalStackLocks	= reinterpret_cast<uint32_t*>( globalStackBuffer.stackData );
		while ( warpHash == InvalidValue )
		{
			if ( laneIndex == firstThreadIndex )
			{
				if ( atomicCAS( &globalStackLocks[warpHashCandidate], 0, 1 ) == 0 ) warpHash = warpHashCandidate;
			}
			warpHashCandidate = ( warpHashCandidate + 1 ) % activeWarps;
			warpHash		  = shfl( warpHash, firstThreadIndex );
		}
		__threadfence();
		m_globalStackLock = &globalStackLocks[warpHash];

		const uint32_t globalStackOffset = activeWarps + laneIndex + ( warpHash << LogStride ) * globalStackBuffer.stackSize;
		m_globalStackBuffer				 = reinterpret_cast<StackEntry*>( globalStackBuffer.stackData ) + globalStackOffset;
		m_globalStackSize				 = globalStackBuffer.stackSize;
	}
	else
	{
		const uint32_t globalStackOffset =
			laneIndex + ( warpIndex * Stride + ( blockIdx.x + blockIdx.y * gridDim.x ) * ( blockDim.x * blockDim.y ) ) *
							globalStackBuffer.stackSize;
		m_globalStackBuffer = reinterpret_cast<StackEntry*>( globalStackBuffer.stackData ) + globalStackOffset;
		m_globalStackSize	= globalStackBuffer.stackSize;
	}
}

template <typename StackEntry, bool DynamicAssignment>
HIPRT_DEVICE GlobalStack<StackEntry, DynamicAssignment>::~GlobalStack()
{
	if constexpr ( DynamicAssignment )
	{
		__threadfence();
		const uint32_t threadIndex		= threadIdx.x + threadIdx.y * blockDim.x;
		const uint32_t laneIndex		= threadIndex & ( Stride - 1 );
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
class TraversalBase
{
  public:
	HIPRT_DEVICE TraversalBase(
		const hiprtRay& ray, Stack& stack, hiprtTraversalHint hint, void* payload, hiprtFuncTable funcTable, uint32_t rayType )
		: m_ray( ray ), m_stack( stack ), m_payload( payload ), m_rayType( rayType ), m_nodeIndex( RootIndex ), m_hint( hint )
	{
		if ( funcTable != nullptr ) m_tableHeader = *reinterpret_cast<hiprtFuncTableHeader*>( funcTable );
#if HIPRT_RTIP >= 20
		packDescriptor( static_cast<uint32_t>( hint ), Rtip >= 31 );
#endif
	}

#if HIPRT_RTIP >= 20
	HIPRT_DEVICE void packDescriptor( uint32_t boxSortHeuristic = 0u, bool compressed = false );
#endif

	HIPRT_DEVICE hiprtTraversalState getCurrentState() { return m_state; }

	HIPRT_DEVICE bool testInternalNode( const hiprtRay& ray, const float3& invD, BoxNode* nodes, uint32_t& nodeIndex );

	HIPRT_DEVICE bool testTriangleNode(
		const hiprtRay& ray,
		const float3&	invD,
		TriangleNode*	nodes,
		uint32_t&		leafIndex,
		uint32_t&		triangleMask,
		hiprtHit&		hit );

	HIPRT_DEVICE bool testTriangle(
		const hiprtRay& ray, const float3& invD, TriangleNode* nodes, uint32_t leafAddr, uint32_t leafType, hiprtHit& hit );

	HIPRT_DEVICE uint32_t testTrianglePair(
		const hiprtRay& ray,
		TriangleNode*	nodes,
		uint32_t		leafAddr,
		uint32_t		triPairIndex,
		hiprtHit&		hit0,
		hiprtHit&		hit1,
		bool&			nodeEnd,
		bool&			rangeEnd );

  protected:
	hiprtRay			 m_ray;
	hiprtFuncTableHeader m_tableHeader = { 0, 1, nullptr };
	uint4				 m_descriptor;
	Stack&				 m_stack;
	BoxNode*			 m_boxNodes;
	void*				 m_payload;
	uint32_t			 m_nodeIndex;
	uint32_t			 m_rayType;
	uint32_t			 m_triangleMask = 0;
	hiprtTraversalState	 m_state		= hiprtTraversalStateInit;
	hiprtTraversalHint	 m_hint			= hiprtTraversalHintDefault;
};

#if HIPRT_RTIP >= 20
template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE void TraversalBase<Stack, TraversalType>::packDescriptor( uint32_t boxSortHeuristic, bool compressed )
{
	boxSortHeuristic &= 0x3;
	uint64_t baseAddress		  = 0x0;
	uint64_t size				  = 0x3ffffffffffull;
	uint32_t boxGrowUlp			  = 0x6;
	uint32_t boxSortEnable		  = 0x1;
	uint32_t triangleReturnMode	  = 0x1;
	uint32_t type				  = 0x8;
	uint32_t compressFormatEnable = 0x0;
	uint32_t instanceEnable		  = 0x0;
	uint32_t sortTrianglesFirst	  = 0x0;
	uint32_t wideSortEnable		  = 0x0;
	if ( compressed )
	{
		compressFormatEnable = 0x1;
		instanceEnable		 = 0x1;
		sortTrianglesFirst	 = 0x1;
		wideSortEnable		 = 0x1;
	}
	m_descriptor.x = baseAddress & 0xffffffff;
	m_descriptor.y = ( baseAddress >> 32ull ) | ( boxSortEnable << 31u ) | ( boxGrowUlp << 23u ) | ( boxSortHeuristic << 21u ) |
					 ( sortTrianglesFirst << 20u );
	m_descriptor.z = size & 0xffffffff;
	m_descriptor.w = ( size >> 32ull ) | ( triangleReturnMode << 24u ) | ( type << 28u ) | ( instanceEnable << 22u ) |
					 ( wideSortEnable << 21u ) | ( compressFormatEnable << 19u );
}
#endif

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE bool TraversalBase<Stack, TraversalType>::testInternalNode(
	const hiprtRay& ray, [[maybe_unused]] const float3& invD, BoxNode* nodes, uint32_t& nodeIndex )
{
#if HIPRT_RTIP < 20
	BoxNode node = nodes[getNodeAddr( nodeIndex )];
	float2	s0	 = node.m_box0.intersect( ray.origin, invD, ray.maxT );
	float2	s1	 = node.m_box1.intersect( ray.origin, invD, ray.maxT );
	float2	s2	 = node.m_box2.intersect( ray.origin, invD, ray.maxT );
	float2	s3	 = node.m_box3.intersect( ray.origin, invD, ray.maxT );

	uint32_t result[4];
	result[0] = s0.x <= s0.y ? node.m_childIndex0 : InvalidValue;
	result[1] = s1.x <= s1.y ? node.m_childIndex1 : InvalidValue;
	result[2] = s2.x <= s2.y ? node.m_childIndex2 : InvalidValue;
	result[3] = s3.x <= s3.y ? node.m_childIndex3 : InvalidValue;

#define SORT( childIndexA, childIndexB, distA, distB )                                     \
	if ( ( childIndexB != InvalidValue && distB < distA ) || childIndexA == InvalidValue ) \
	{                                                                                      \
		float	 t0 = distA;                                                               \
		uint32_t t1 = childIndexA;                                                         \
		childIndexA = childIndexB;                                                         \
		distA		= distB;                                                               \
		childIndexB = t1;                                                                  \
		distB		= t0;                                                                  \
	}

	SORT( result[0], result[2], s0.x, s2.x )
	SORT( result[1], result[3], s1.x, s3.x )
	SORT( result[0], result[1], s0.x, s1.x )
	SORT( result[2], result[3], s2.x, s3.x )
	SORT( result[1], result[2], s1.x, s2.x )
#undef SORT
#elif HIPRT_RTIP >= 31
	hip_float3 dummy0, dummy1;
	auto	   result = __builtin_amdgcn_image_bvh8_intersect_ray(
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

	if ( m_stack.vacancy() < BranchingFactor - 1 )
	{
		m_state = hiprtTraversalStateStackOverflow;
		return true;
	}

#pragma unroll
	for ( uint32_t i = BranchingFactor - 1; i >= 1; --i )
		if ( result[i] != InvalidValue ) m_stack.push( result[i] );

	if ( result[0] != InvalidValue )
	{
		nodeIndex = result[0];
		return true;
	}

	return false;
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE bool TraversalBase<Stack, TraversalType>::testTriangleNode(
	const hiprtRay&				   ray,
	[[maybe_unused]] const float3& invD,
	TriangleNode*				   nodes,
	uint32_t&					   leafIndex,
	[[maybe_unused]] uint32_t&	   triangleMask,
	hiprtHit&					   hit )
{
	bool	 hasHit	  = false;
	uint32_t leafAddr = getNodeAddr( leafIndex );

	if constexpr ( Rtip >= 31 )
	{
		hiprtHit secondHit;
		uint32_t triPairIndex = typeToTriPairIndex( getNodeType( leafIndex ) );

		if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
		{
			while ( true )
			{
				bool	 nodeEnd  = false;
				bool	 rangeEnd = false;
				uint32_t hitMask =
					this->testTrianglePair( ray, nodes, leafAddr, triPairIndex, hit, secondHit, nodeEnd, rangeEnd );

				bool firstHasHit  = hitMask & 1;
				bool secondHasHit = hitMask & 2;
				if ( ( triangleMask & Triangle0Processed ) == 0 )
				{
					hasHit = firstHasHit;
					triangleMask |= Triangle0Processed;
				}

				if ( !hasHit )
				{
					if ( secondHasHit )
					{
						hit.t	   = secondHit.t;
						hit.normal = secondHit.normal;
						hit.primID = secondHit.primID;
						hit.uv	   = secondHit.uv;
						hasHit	   = true;
					}
					triangleMask |= Triangle1Processed;
				}

				if ( rangeEnd )
				{
					triangleMask = InvalidValue; // indicate range end by 'invalid value'
					break;
				}

				triPairIndex++;
				triangleMask = 0;
				if ( nodeEnd )
				{
					triPairIndex = 0;
					leafAddr++;
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
			hit.t = ray.maxT;
			hiprtHit firstHit;
			while ( true )
			{
				bool	 nodeEnd  = false;
				bool	 rangeEnd = false;
				uint32_t hitMask =
					this->testTrianglePair( ray, nodes, leafAddr, triPairIndex, firstHit, secondHit, nodeEnd, rangeEnd );

				bool firstHasHit  = hitMask & 1;
				bool secondHasHit = hitMask & 2;
				if ( firstHasHit && ( !hasHit || hit.t > firstHit.t ) )
				{
					hit.t	   = firstHit.t;
					hit.normal = firstHit.normal;
					hit.primID = firstHit.primID;
					hit.uv	   = firstHit.uv;
					hasHit	   = true;
				}

				if ( secondHasHit && ( !hasHit || hit.t > secondHit.t ) )
				{
					hit.t	   = secondHit.t;
					hit.normal = secondHit.normal;
					hit.primID = secondHit.primID;
					hit.uv	   = secondHit.uv;
					hasHit	   = true;
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
	}
	else
	{
		if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
		{
			if ( ( triangleMask & Triangle0Processed ) == 0 )
			{
				hasHit = this->testTriangle( ray, invD, nodes, leafAddr, TriangleType, hit );
				triangleMask |= Triangle0Processed;
			}

			if ( !hasHit )
			{
				hasHit = this->testTriangle( ray, invD, nodes, leafAddr, TriangleType + 1, hit );
				triangleMask |= Triangle1Processed;
			}

			if ( triangleMask & Triangle1Processed ) triangleMask = InvalidValue;
		}
		else
		{
			hasHit = this->testTriangle( ray, invD, nodes, leafAddr, TriangleType, hit );

			hiprtHit secondHit;
			bool	 secondHasHit = this->testTriangle( ray, invD, nodes, leafAddr, TriangleType + 1, secondHit );

			if ( secondHasHit && ( !hasHit || hit.t > secondHit.t ) )
			{
				hit.t	   = secondHit.t;
				hit.normal = secondHit.normal;
				hit.primID = secondHit.primID;
				hit.uv	   = secondHit.uv;
				hasHit	   = true;
			}
		}
	}
	return hasHit;
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE bool TraversalBase<Stack, TraversalType>::testTriangle(
	const hiprtRay&				   ray,
	[[maybe_unused]] const float3& invD,
	TriangleNode*				   nodes,
	uint32_t					   leafAddr,
	uint32_t					   leafType,
	hiprtHit&					   hit )
{
	const TriangleNode& node = nodes[leafAddr];
	if ( node.getPrimIndex( 0 ) == node.getPrimIndex( 1 ) && leafType == TriangleType + 1 ) return false;
	bool hasHit = false;
#if HIPRT_RTIP < 20
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
	float t		   = __int_as_float( result[0] ) * invDenom;
	hasHit		   = ray.minT <= t && t <= ray.maxT;
	if ( hasHit )
	{
		hit.t	   = t;
		hit.uv.x   = __int_as_float( result[2] ) * invDenom;
		hit.uv.y   = __int_as_float( result[3] ) * invDenom;
		hit.primID = leafType & 1 ? node.getPrimIndex( 1 ) : node.getPrimIndex( 0 );
		hit.normal = node.getNormal( leafType & 1 );
	}
#endif
	return hasHit;
}

template <typename Stack, hiprtTraversalType TraversalType>
HIPRT_DEVICE uint32_t TraversalBase<Stack, TraversalType>::testTrianglePair(
	const hiprtRay& ray,
	TriangleNode*	nodes,
	uint32_t		leafAddr,
	uint32_t		triPairIndex,
	hiprtHit&		hit0,
	hiprtHit&		hit1,
	bool&			nodeEnd,
	bool&			rangeEnd )
{
#if HIPRT_RTIP >= 31
	const TriangleNode& node = nodes[leafAddr];

	hip_float3 dummy0, dummy1;
	auto	   result = __builtin_amdgcn_image_bvh8_intersect_ray(
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
		float t		 = __int_as_float( result[0] );
		bool  hasHit = ray.minT <= t && t <= ray.maxT;
		hitMask |= hasHit ? 1 : 0;
		if ( hasHit )
		{
			hit0.t		= t;
			hit0.uv.x	= __int_as_float( result[1] );
			hit0.uv.y	= __int_as_float( result[2] );
			hit0.primID = result[3] >> 1;
			hit0.normal = node.getNormal( triPairIndex, 0 );
		}
	}

	{
		float t		 = __int_as_float( result[4] );
		bool  hasHit = ray.minT <= t && t <= ray.maxT;
		hitMask |= hasHit ? 2 : 0;
		if ( hasHit )
		{
			hit1.t		= t;
			hit1.uv.x	= __int_as_float( result[5] );
			hit1.uv.y	= __int_as_float( result[6] );
			hit1.primID = result[7] >> 1;
			hit1.normal = node.getNormal( triPairIndex, 1 );
		}
	}

	nodeEnd	 = ( result[8] & 3 ) == 1;
	rangeEnd = ( result[8] & 3 ) == 3;

	return hitMask;
#endif
}

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
class GeomTraversal : public TraversalBase<Stack, TraversalType>
{
  public:
	HIPRT_DEVICE
	GeomTraversal(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		Stack&			   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0u );

	HIPRT_DEVICE bool
	testLeafNode( const hiprtRay& ray, const float3& invD, uint32_t& leafIndex, uint32_t& triangleMask, hiprtHit& hit );

	HIPRT_DEVICE hiprtHit getNextHit();

  protected:
	using TraversalBase<Stack, TraversalType>::m_ray;
	using TraversalBase<Stack, TraversalType>::m_tableHeader;
	using TraversalBase<Stack, TraversalType>::m_state;
	using TraversalBase<Stack, TraversalType>::m_stack;
	using TraversalBase<Stack, TraversalType>::m_boxNodes;
	using TraversalBase<Stack, TraversalType>::m_payload;
	using TraversalBase<Stack, TraversalType>::m_nodeIndex;
	using TraversalBase<Stack, TraversalType>::m_triangleMask;
	using TraversalBase<Stack, TraversalType>::m_rayType;
#if HIPRT_RTIP >= 20
	using TraversalBase<Stack, TraversalType>::m_descriptor;
#endif

	PrimitiveNode* m_primNodes;
	uint32_t	   m_geomType;
	uint32_t	   m_leafIndex;
};

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
HIPRT_DEVICE GeomTraversal<Stack, PrimitiveNode, TraversalType>::GeomTraversal(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	Stack&			   stack,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: TraversalBase<Stack, TraversalType>( ray, stack, hint, payload, funcTable, rayType ), m_leafIndex( InvalidValue )
{
	GeomHeader* geomHeader = reinterpret_cast<GeomHeader*>( geom );
	m_boxNodes			   = geomHeader->m_boxNodes;
	m_primNodes			   = reinterpret_cast<PrimitiveNode*>( geomHeader->m_primNodes );
	m_geomType			   = geomHeader->m_geomType;
	m_stack.reset();
}

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
HIPRT_DEVICE bool GeomTraversal<Stack, PrimitiveNode, TraversalType>::testLeafNode(
	const hiprtRay&				   ray,
	[[maybe_unused]] const float3& invD,
	uint32_t&					   leafIndex,
	[[maybe_unused]] uint32_t&	   triangleMask,
	hiprtHit&					   hit )
{
	bool hasHit = false;
	if constexpr ( is_same<PrimitiveNode, TriangleNode>::value )
	{
		hasHit = this->testTriangleNode( ray, invD, m_primNodes, leafIndex, triangleMask, hit );
	}
	else
	{
		hit.primID = m_primNodes[getNodeAddr( leafIndex )].m_primIndex;
		hasHit	   = intersectFunc( m_geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit );
		if ( !hasHit ) hit.primID = InvalidValue;
	}
	return hasHit;
}

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
HIPRT_DEVICE hiprtHit GeomTraversal<Stack, PrimitiveNode, TraversalType>::getNextHit()
{
	hiprtRay ray = m_ray;
	float3	 invD;
	if constexpr ( Rtip < 31 ) invD = rcp( m_ray.direction );

	if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
	{
		if ( m_leafIndex == InvalidValue && isLeafNode( m_nodeIndex ) )
		{
			m_leafIndex = m_nodeIndex;
			m_nodeIndex = m_stack.pop();
		}
	}
	hiprtHit result;

	if ( m_stack.empty() ) m_stack.push( InvalidValue );

	while ( m_nodeIndex != InvalidValue || m_leafIndex != InvalidValue )
	{
		while ( isInternalNode( m_nodeIndex ) )
		{
			if ( !this->testInternalNode( ray, invD, m_boxNodes, m_nodeIndex ) ) m_nodeIndex = m_stack.pop();

			if ( m_state == hiprtTraversalStateStackOverflow ) return hiprtHit();

			if ( isLeafNode( m_nodeIndex ) && m_leafIndex == InvalidValue )
			{
				m_leafIndex = m_nodeIndex;
				m_nodeIndex = m_stack.pop();
			}

			if ( !hiprt::any( m_leafIndex == InvalidValue ) ) break;
		}

		while ( m_leafIndex != InvalidValue )
		{
			hiprtHit hit;
			if ( testLeafNode( ray, invD, m_leafIndex, m_triangleMask, hit ) )
			{
				if ( m_geomType == InvalidValue || m_tableHeader.funcDataSets == nullptr ||
					 !filterFunc( m_geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit ) )
				{
					if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
					{
						if ( getNodeType( m_leafIndex ) == CustomType || m_triangleMask == InvalidValue )
						{
							m_triangleMask = 0;
							m_leafIndex	   = InvalidValue;
						}
						m_state = hiprtTraversalStateHit;
						return hit;
					}
					else
					{
						result	 = hit;
						ray.maxT = hit.t;
					}
				}
			}

			m_triangleMask = 0;
			m_leafIndex	   = InvalidValue;
			if ( isLeafNode( m_nodeIndex ) )
			{
				m_leafIndex = m_nodeIndex;
				m_nodeIndex = m_stack.pop();
			}
		}
	}

	if ( m_state != hiprtTraversalStateStackOverflow ) m_state = hiprtTraversalStateFinished;

	return result;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
class SceneTraversal : public TraversalBase<Stack, TraversalType>
{
  public:
	HIPRT_DEVICE SceneTraversal(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		Stack&			   stack,
		InstanceStack&	   instanceStack,
		hiprtRayMask	   mask		 = InvalidValue,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0,
		float			   time		 = 0.0f );

	HIPRT_DEVICE const uint32_t& instanceId() const
	{
		if constexpr ( !is_same<InstanceStack, hiprtEmptyInstanceStack>::value )
			return m_instanceIds[m_level];
		else
			return m_instanceId;
	}

	HIPRT_DEVICE uint32_t& instanceId()
	{
		if constexpr ( !is_same<InstanceStack, hiprtEmptyInstanceStack>::value )
			return m_instanceIds[m_level];
		else
			return m_instanceId;
	}

	HIPRT_DEVICE bool transformRay( uint32_t nodeIndex, hiprtRay& ray, float3& invD );

	HIPRT_DEVICE void restoreRay( hiprtRay& ray, float3& invD ) const;

	HIPRT_DEVICE bool testLeafNode(
		void*			primNodes,
		const hiprtRay& ray,
		const float3&	invD,
		uint32_t&		leafIndex,
		uint32_t&		triangleMask,
		uint32_t		geomType,
		hiprtHit&		hit );

	HIPRT_DEVICE hiprtHit getNextHit();

  protected:
	using TraversalBase<Stack, TraversalType>::m_tableHeader;
	using TraversalBase<Stack, TraversalType>::m_ray;
	using TraversalBase<Stack, TraversalType>::m_state;
	using TraversalBase<Stack, TraversalType>::m_stack;
	using TraversalBase<Stack, TraversalType>::m_boxNodes;
	using TraversalBase<Stack, TraversalType>::m_payload;
	using TraversalBase<Stack, TraversalType>::m_nodeIndex;
	using TraversalBase<Stack, TraversalType>::m_triangleMask;
	using TraversalBase<Stack, TraversalType>::m_rayType;
	using TraversalBase<Stack, TraversalType>::m_hint;
#if HIPRT_RTIP >= 20
	using TraversalBase<Stack, TraversalType>::m_descriptor;
#endif

	union
	{
		uint32_t m_instanceId;
		uint32_t m_instanceIds[MaxInstanceLevels];
	};

	InstanceStack& m_instanceStack;
	SceneHeader*   m_scene;
	InstanceNode*  m_instanceNodes;
	Frame*		   m_frames;
	hiprtRayMask   m_mask;
	uint32_t	   m_level;
	uint32_t	   m_instanceIndex;
	float		   m_time;
};

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE SceneTraversal<Stack, InstanceStack, TraversalType>::SceneTraversal(
	hiprtScene		   scene,
	const hiprtRay&	   ray,
	Stack&			   stack,
	InstanceStack&	   instanceStack,
	hiprtRayMask	   mask,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType,
	float			   time )
	: TraversalBase<Stack, TraversalType>( ray, stack, hint, payload, funcTable, rayType ), m_time( time ), m_mask( mask ),
	  m_instanceStack( instanceStack ), m_level( 0u )
{
	SceneHeader* sceneHeader = reinterpret_cast<SceneHeader*>( scene );
	m_boxNodes				 = sceneHeader->m_boxNodes;
	m_instanceNodes			 = sceneHeader->m_primNodes;
	m_frames				 = sceneHeader->m_frames;
	m_stack.reset();
	m_instanceIndex = InvalidValue;
	instanceId()	= InvalidValue;
	if constexpr ( !is_same<InstanceStack, hiprtEmptyInstanceStack>::value )
	{
		m_instanceStack.reset();
		m_scene = sceneHeader;
	}
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE bool SceneTraversal<Stack, InstanceStack, TraversalType>::transformRay(
	uint32_t nodeIndex, hiprtRay& ray, [[maybe_unused]] float3& invD )
{
	const InstanceNode& instanceNode = m_instanceNodes[getNodeAddr( nodeIndex )];
	if ( instanceNode.m_identity == 0 )
	{
		if ( instanceNode.m_static != 0 )
		{
#if HIPRT_RTIP >= 31
			hip_float3 origin, direction;
			auto	   result = __builtin_amdgcn_image_bvh8_intersect_ray(
				  encodeBaseAddr( m_instanceNodes ),
				  ray.maxT,
				  0xff,
				  { ray.origin.x, ray.origin.y, ray.origin.z },
				  { ray.direction.x, ray.direction.y, ray.direction.z },
				  nodeIndex,
				  { m_descriptor.x, m_descriptor.y, m_descriptor.z, m_descriptor.w },
				  &origin,
				  &direction );

			if ( result[7] == InvalidValue ) return false;

			ray.origin	  = { origin.x, origin.y, origin.z };
			ray.direction = { direction.x, direction.y, direction.z };
#else
			ray = instanceNode.transformRay( ray );
#endif
		}
		else
		{
			Transform tr( m_frames, instanceNode.m_transform.frameIndex, instanceNode.m_transform.frameCount );
			ray = tr.transformRay( ray, m_time );
		}
		if constexpr ( Rtip < 31 ) invD = rcp( ray.direction );
	}

	return true;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE void SceneTraversal<Stack, InstanceStack, TraversalType>::restoreRay( hiprtRay& ray, float3& invD ) const
{
	ray.origin	  = m_ray.origin;
	ray.direction = m_ray.direction;
	if constexpr ( Rtip < 31 ) invD = rcp( m_ray.direction );
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE bool SceneTraversal<Stack, InstanceStack, TraversalType>::testLeafNode(
	void*						   primNodes,
	const hiprtRay&				   ray,
	[[maybe_unused]] const float3& invD,
	uint32_t&					   leafIndex,
	[[maybe_unused]] uint32_t&	   triangleMask,
	uint32_t					   geomType,
	hiprtHit&					   hit )
{
	bool hasHit = false;
	if constexpr ( !is_same<InstanceStack, hiprtEmptyInstanceStack>::value )
	{
#pragma unroll
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
		TriangleNode* nodes = reinterpret_cast<TriangleNode*>( primNodes );
		hasHit				= this->testTriangleNode( ray, invD, nodes, leafIndex, triangleMask, hit );
	}
	else
	{
		CustomNode* nodes = reinterpret_cast<CustomNode*>( primNodes );
		hit.primID		  = nodes[getNodeAddr( leafIndex )].m_primIndex;
		hasHit			  = intersectFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit );
		if ( !hasHit ) hit.primID = InvalidValue;
	}

	return hasHit;
}

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
HIPRT_DEVICE hiprtHit SceneTraversal<Stack, InstanceStack, TraversalType>::getNextHit()
{
	BoxNode* nodes	   = m_boxNodes;
	void*	 primNodes = nullptr;
	uint32_t geomType  = InvalidValue;

	hiprtRay ray = m_ray;
	float3	 invD;
	if constexpr ( Rtip < 31 )
	{
		if ( instanceId() == InvalidValue ) invD = rcp( m_ray.direction );
	}

	if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
	{
		if ( instanceId() != InvalidValue )
		{
			transformRay( m_instanceIndex, ray, invD );
			nodes	  = m_instanceNodes[getNodeAddr( m_instanceIndex )].m_geometry->m_boxNodes;
			primNodes = m_instanceNodes[getNodeAddr( m_instanceIndex )].m_geometry->m_primNodes;
			geomType  = m_instanceNodes[getNodeAddr( m_instanceIndex )].m_geometry->m_geomType;
		}
	}

	hiprtHit result;

	if ( m_stack.empty() ) m_stack.push( InvalidValue );

	while ( m_nodeIndex != InvalidValue && m_state != hiprtTraversalStateStackOverflow )
	{
		if ( isInternalNode( m_nodeIndex ) )
		{
			if ( this->testInternalNode( ray, invD, nodes, m_nodeIndex ) ) continue;
		}
		else
		{
			if ( instanceId() != InvalidValue )
			{
				hiprtHit hit;
				if ( testLeafNode( primNodes, ray, invD, m_nodeIndex, m_triangleMask, geomType, hit ) )
				{
					if ( geomType == InvalidValue || m_tableHeader.funcDataSets == nullptr ||
						 !filterFunc( geomType >> 1, m_rayType, m_tableHeader, ray, m_payload, hit ) )
					{
						if constexpr ( TraversalType == hiprtTraversalTerminateAtAnyHit )
						{
							m_state = hiprtTraversalStateHit;
							if ( getNodeType( m_nodeIndex ) == CustomType || m_triangleMask == InvalidValue )
							{
								m_triangleMask = 0;
								m_nodeIndex	   = m_stack.pop();

								while ( m_nodeIndex == InvalidValue && !m_stack.empty() )
								{
									if constexpr ( !is_same<InstanceStack, hiprtEmptyInstanceStack>::value )
									{
										if ( instanceId() == InvalidValue )
										{
											hiprtInstanceStackEntry instanceEntry = m_instanceStack.pop();
											m_ray								  = instanceEntry.ray;
											m_scene = reinterpret_cast<SceneHeader*>( instanceEntry.scene );
											m_level--;

											m_boxNodes		= m_scene->m_boxNodes;
											m_instanceNodes = m_scene->m_primNodes;
											m_frames		= m_scene->m_frames;
										}
									}

									instanceId()   = InvalidValue;
									m_triangleMask = 0;
									m_nodeIndex	   = m_stack.pop();
								}
							}
							return hit;
						}
						else
						{
							ray.maxT = hit.t;
							result	 = hit;
						}
					}
				}
			}
			else
			{
				const uint32_t instanceAddr = getNodeAddr( m_nodeIndex );
				if ( ( m_instanceNodes[instanceAddr].m_mask & m_mask ) && transformRay( m_nodeIndex, ray, invD ) )
				{
					if ( m_stack.vacancy() < 1 )
					{
						m_state = hiprtTraversalStateStackOverflow;
						continue;
					}

					m_instanceIndex = m_nodeIndex;
					instanceId()	= m_instanceNodes[instanceAddr].m_primIndex;

					m_nodeIndex = RootIndex;
					m_stack.push( InvalidValue );

					if constexpr ( !is_same<InstanceStack, hiprtEmptyInstanceStack>::value )
					{
						if ( m_instanceNodes[instanceAddr].m_type == hiprtInstanceTypeScene )
						{
							m_instanceStack.push( { m_ray, reinterpret_cast<hiprtScene>( m_scene ) } );
							m_ray	= ray;
							m_scene = m_instanceNodes[instanceAddr].m_scene;
							m_level++;
							instanceId() = InvalidValue;

							m_boxNodes		= m_scene->m_boxNodes;
							m_instanceNodes = m_scene->m_primNodes;
							m_frames		= m_scene->m_frames;

							nodes = m_boxNodes;
							continue;
						}
					}
					nodes	  = m_instanceNodes[instanceAddr].m_geometry->m_boxNodes;
					primNodes = m_instanceNodes[instanceAddr].m_geometry->m_primNodes;
					geomType  = m_instanceNodes[instanceAddr].m_geometry->m_geomType;
					continue;
				}
			}
		}

		m_triangleMask = 0;
		m_nodeIndex	   = m_stack.pop();
		while ( m_nodeIndex == InvalidValue && !m_stack.empty() )
		{
			if constexpr ( !is_same<InstanceStack, hiprtEmptyInstanceStack>::value )
			{
				if ( instanceId() == InvalidValue )
				{
					hiprtInstanceStackEntry instanceEntry = m_instanceStack.pop();
					m_ray								  = instanceEntry.ray;
					m_scene								  = reinterpret_cast<SceneHeader*>( instanceEntry.scene );
					m_level--;

					m_boxNodes		= m_scene->m_boxNodes;
					m_instanceNodes = m_scene->m_primNodes;
					m_frames		= m_scene->m_frames;
				}
			}
			instanceId() = InvalidValue;
			m_nodeIndex	 = m_stack.pop();
			nodes		 = m_boxNodes;
			restoreRay( ray, invD );
		}
	}

	if ( m_state != hiprtTraversalStateStackOverflow ) m_state = hiprtTraversalStateFinished;

	return result;
}

template <typename PrimitiveNode, hiprtTraversalType TraversalType>
class GeomTraversalPrivateStack
{
  public:
	typedef hiprtPrivateStack Stack;

	HIPRT_DEVICE GeomTraversalPrivateStack(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 )
		: m_traversal( geom, ray, m_stack, hint, payload, funcTable, rayType )
	{
	}

	HIPRT_DEVICE hiprtHit getNextHit() { return m_traversal.getNextHit(); }

	HIPRT_DEVICE hiprtTraversalState getCurrentState() { return m_traversal.getCurrentState(); }

  private:
	Stack											   m_stack;
	GeomTraversal<Stack, PrimitiveNode, TraversalType> m_traversal;
};

template <hiprtTraversalType TraversalType>
class SceneTraversalPrivateStack
{
  public:
	typedef hiprtPrivateStack		  Stack;
	typedef hiprtPrivateInstanceStack InstanceStack;

	HIPRT_DEVICE SceneTraversalPrivateStack(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtRayMask	   mask		 = hiprtFullRayMask,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0,
		float			   time		 = 0.0f )
		: m_traversal( scene, ray, m_stack, m_instanceStack, mask, hint, payload, funcTable, rayType, time )
	{
	}

	HIPRT_DEVICE hiprtHit getNextHit() { return m_traversal.getNextHit(); }

	HIPRT_DEVICE hiprtTraversalState getCurrentState() { return m_traversal.getCurrentState(); }

  private:
	Stack												m_stack;
	InstanceStack										m_instanceStack;
	SceneTraversal<Stack, InstanceStack, TraversalType> m_traversal;
};
} // namespace hiprt

// Impl classes
template <typename StackEntry, uint32_t StackSize>
class hiprtPrivateStack_impl
{
  public:
	HIPRT_DEVICE hiprtPrivateStack_impl() : m_stack() {}
	~hiprtPrivateStack_impl() = default;
	HIPRT_DEVICE StackEntry pop() { return m_stack.pop(); }
	HIPRT_DEVICE void		push( StackEntry val ) { m_stack.push( val ); }
	HIPRT_DEVICE bool		empty() const { return m_stack.empty(); }
	HIPRT_DEVICE uint32_t	vacancy() const { return m_stack.vacancy(); }
	HIPRT_DEVICE void		reset() { m_stack.reset(); }

  private:
	hiprt::PrivateStack<StackEntry, StackSize> m_stack;
};

template <typename StackEntry, bool DynamicAssignment>
class hiprtGlobalStack_impl
{
  public:
	HIPRT_DEVICE hiprtGlobalStack_impl( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer )
		: m_stack( globalStackBuffer, sharedStackBuffer )
	{
	}
	~hiprtGlobalStack_impl() = default;
	HIPRT_DEVICE StackEntry pop() { return m_stack.pop(); }
	HIPRT_DEVICE void		push( StackEntry val ) { m_stack.push( val ); }
	HIPRT_DEVICE bool		empty() const { return m_stack.empty(); }
	HIPRT_DEVICE uint32_t	vacancy() const { return m_stack.vacancy(); }
	HIPRT_DEVICE void		reset() { m_stack.reset(); }

  private:
	hiprt::GlobalStack<StackEntry, DynamicAssignment> m_stack;
};

template <hiprtPrimitiveNodeType PrimitiveNodeType, hiprtTraversalType TraversalType>
class hiprtGeomTraversal_impl
{
  public:
	HIPRT_DEVICE hiprtGeomTraversal_impl(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 )
		: m_traversal( geom, ray, hint, payload, funcTable, rayType )
	{
	}

	HIPRT_DEVICE hiprtHit getNextHit() { return m_traversal.getNextHit(); }

	HIPRT_DEVICE hiprtTraversalState getCurrentState() { return m_traversal.getCurrentState(); }

  private:
	typedef typename hiprt::conditional<PrimitiveNodeType == hiprtTriangleNode, hiprt::TriangleNode, hiprt::CustomNode>::type
															  NodeType;
	hiprt::GeomTraversalPrivateStack<NodeType, TraversalType> m_traversal;
};

template <hiprtTraversalType TraversalType>
class hiprtSceneTraversal_impl
{
  public:
	HIPRT_DEVICE hiprtSceneTraversal_impl(
		hiprtScene		   scene,
		const hiprtRay&	   ray,
		hiprtRayMask	   mask		 = hiprtFullRayMask,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0,
		float			   time		 = 0.0f )
		: m_traversal( scene, ray, mask, hint, payload, funcTable, rayType, time )
	{
	}

	HIPRT_DEVICE hiprtHit getNextHit() { return m_traversal.getNextHit(); }

	HIPRT_DEVICE hiprtTraversalState getCurrentState() { return m_traversal.getCurrentState(); }

  private:
	hiprt::SceneTraversalPrivateStack<TraversalType> m_traversal;
};

template <typename hiprtStack, hiprtPrimitiveNodeType PrimitiveNodeType, hiprtTraversalType TraversalType>
class hiprtGeomTraversalCustomStack_impl
{
  public:
	HIPRT_DEVICE hiprtGeomTraversalCustomStack_impl(
		hiprtGeometry	   geom,
		const hiprtRay&	   ray,
		hiprtStack&		   stack,
		hiprtTraversalHint hint		 = hiprtTraversalHintDefault,
		void*			   payload	 = nullptr,
		hiprtFuncTable	   funcTable = nullptr,
		uint32_t		   rayType	 = 0 )
		: m_traversal( geom, ray, stack, hint, payload, funcTable, rayType )
	{
	}

	HIPRT_DEVICE hiprtHit getNextHit() { return m_traversal.getNextHit(); }

	HIPRT_DEVICE hiprtTraversalState getCurrentState() { return m_traversal.getCurrentState(); }

  private:
	typedef typename hiprt::conditional<PrimitiveNodeType == hiprtTriangleNode, hiprt::TriangleNode, hiprt::CustomNode>::type
															  NodeType;
	hiprt::GeomTraversal<hiprtStack, NodeType, TraversalType> m_traversal;
};

template <typename hiprtStack, typename hiprtInstanceStack, hiprtTraversalType TraversalType>
class hiprtSceneTraversalCustomStack_impl
{
  public:
	HIPRT_DEVICE hiprtSceneTraversalCustomStack_impl(
		hiprtScene			scene,
		const hiprtRay&		ray,
		hiprtStack&			stack,
		hiprtInstanceStack& instanceStack,
		hiprtRayMask		mask	  = hiprtFullRayMask,
		hiprtTraversalHint	hint	  = hiprtTraversalHintDefault,
		void*				payload	  = nullptr,
		hiprtFuncTable		funcTable = nullptr,
		uint32_t			rayType	  = 0,
		float				time	  = 0.0f )
		: m_traversal( scene, ray, stack, instanceStack, mask, hint, payload, funcTable, rayType, time )
	{
	}

	HIPRT_DEVICE hiprtHit getNextHit() { return m_traversal.getNextHit(); }

	HIPRT_DEVICE hiprtTraversalState getCurrentState() { return m_traversal.getCurrentState(); }

  private:
	hiprt::SceneTraversal<hiprtStack, hiprtInstanceStack, TraversalType> m_traversal;
};

// hiprt_device classes

// hiprtPrivateStack
HIPRT_DEVICE hiprtPrivateStack::hiprtPrivateStack() : m_impl() {}

HIPRT_DEVICE hiprtPrivateStack::~hiprtPrivateStack() { m_impl->~hiprtPrivateStack_impl(); }

HIPRT_DEVICE uint32_t hiprtPrivateStack::pop() { return m_impl->pop(); }

HIPRT_DEVICE void hiprtPrivateStack::push( uint32_t val ) { m_impl->push( val ); }

HIPRT_DEVICE bool hiprtPrivateStack::empty() const { return m_impl->empty(); }

HIPRT_DEVICE uint32_t hiprtPrivateStack::vacancy() const { return m_impl->vacancy(); }

HIPRT_DEVICE void hiprtPrivateStack::reset() { m_impl->reset(); }

// hiprtPrivateInstanceStack
HIPRT_DEVICE hiprtPrivateInstanceStack::hiprtPrivateInstanceStack() : m_impl() {}

HIPRT_DEVICE hiprtPrivateInstanceStack::~hiprtPrivateInstanceStack() { m_impl->~hiprtPrivateStack_impl(); }

HIPRT_DEVICE hiprtInstanceStackEntry hiprtPrivateInstanceStack::pop() { return m_impl->pop(); }

HIPRT_DEVICE void hiprtPrivateInstanceStack::push( hiprtInstanceStackEntry val ) { m_impl->push( val ); }

HIPRT_DEVICE bool hiprtPrivateInstanceStack::empty() const { return m_impl->empty(); }

HIPRT_DEVICE uint32_t hiprtPrivateInstanceStack::vacancy() const { return m_impl->vacancy(); }

HIPRT_DEVICE void hiprtPrivateInstanceStack::reset() { m_impl->reset(); }

// hiprtGlobalStack
HIPRT_DEVICE
hiprtGlobalStack::hiprtGlobalStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer )
	: m_impl( globalStackBuffer, sharedStackBuffer )
{
}

HIPRT_DEVICE hiprtGlobalStack::~hiprtGlobalStack() { m_impl->~hiprtGlobalStack_impl(); }

HIPRT_DEVICE uint32_t hiprtGlobalStack::pop() { return m_impl->pop(); }

HIPRT_DEVICE void hiprtGlobalStack::push( uint32_t val ) { m_impl->push( val ); }

HIPRT_DEVICE bool hiprtGlobalStack::empty() const { return m_impl->empty(); }

HIPRT_DEVICE uint32_t hiprtGlobalStack::vacancy() const { return m_impl->vacancy(); }

HIPRT_DEVICE void hiprtGlobalStack::reset() { m_impl->reset(); }

// hiprtGlobalInstanceStack
HIPRT_DEVICE
hiprtGlobalInstanceStack::hiprtGlobalInstanceStack(
	hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer )
	: m_impl( globalStackBuffer, sharedStackBuffer )
{
}

HIPRT_DEVICE hiprtGlobalInstanceStack::~hiprtGlobalInstanceStack() { m_impl->~hiprtGlobalStack_impl(); }

HIPRT_DEVICE hiprtInstanceStackEntry hiprtGlobalInstanceStack::pop() { return m_impl->pop(); }

HIPRT_DEVICE void hiprtGlobalInstanceStack::push( hiprtInstanceStackEntry val ) { m_impl->push( val ); }

HIPRT_DEVICE bool hiprtGlobalInstanceStack::empty() const { return m_impl->empty(); }

HIPRT_DEVICE uint32_t hiprtGlobalInstanceStack::vacancy() const { return m_impl->vacancy(); }

HIPRT_DEVICE void hiprtGlobalInstanceStack::reset() { m_impl->reset(); }

// hiprtDynamicStack
HIPRT_DEVICE
hiprtDynamicStack::hiprtDynamicStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer )
	: m_impl( globalStackBuffer, sharedStackBuffer )
{
}

HIPRT_DEVICE hiprtDynamicStack::~hiprtDynamicStack() { m_impl->~hiprtGlobalStack_impl(); }

HIPRT_DEVICE uint32_t hiprtDynamicStack::pop() { return m_impl->pop(); }

HIPRT_DEVICE void hiprtDynamicStack::push( uint32_t val ) { m_impl->push( val ); }

HIPRT_DEVICE bool hiprtDynamicStack::empty() const { return m_impl->empty(); }

HIPRT_DEVICE uint32_t hiprtDynamicStack::vacancy() const { return m_impl->vacancy(); }

HIPRT_DEVICE void hiprtDynamicStack::reset() { m_impl->reset(); }

// hiprtDynamicInstanceStack
HIPRT_DEVICE
hiprtDynamicInstanceStack::hiprtDynamicInstanceStack(
	hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer )
	: m_impl( globalStackBuffer, sharedStackBuffer )
{
}

HIPRT_DEVICE hiprtDynamicInstanceStack::~hiprtDynamicInstanceStack() { m_impl->~hiprtGlobalStack_impl(); }

HIPRT_DEVICE hiprtInstanceStackEntry hiprtDynamicInstanceStack::pop() { return m_impl->pop(); }

HIPRT_DEVICE void hiprtDynamicInstanceStack::push( hiprtInstanceStackEntry val ) { m_impl->push( val ); }

HIPRT_DEVICE bool hiprtDynamicInstanceStack::empty() const { return m_impl->empty(); }

HIPRT_DEVICE uint32_t hiprtDynamicInstanceStack::vacancy() const { return m_impl->vacancy(); }

HIPRT_DEVICE void hiprtDynamicInstanceStack::reset() { m_impl->reset(); }

// hiprtGeomTraversalClosest
HIPRT_DEVICE hiprtGeomTraversalClosest::hiprtGeomTraversalClosest(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, hint, payload, funcTable, rayType )
{
}

HIPRT_DEVICE hiprtHit hiprtGeomTraversalClosest::getNextHit() { return m_impl->getNextHit(); }

HIPRT_DEVICE hiprtTraversalState hiprtGeomTraversalClosest::getCurrentState() { return m_impl->getCurrentState(); }

// hiprtGeomTraversalAnyHit
HIPRT_DEVICE hiprtGeomTraversalAnyHit::hiprtGeomTraversalAnyHit(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, hint, payload, funcTable, rayType )
{
}

HIPRT_DEVICE hiprtHit hiprtGeomTraversalAnyHit::getNextHit() { return m_impl->getNextHit(); }

HIPRT_DEVICE hiprtTraversalState hiprtGeomTraversalAnyHit::getCurrentState() { return m_impl->getCurrentState(); }

// hiprtGeomCustomTraversalClosest
HIPRT_DEVICE hiprtGeomCustomTraversalClosest::hiprtGeomCustomTraversalClosest(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, hint, payload, funcTable, rayType )
{
}

HIPRT_DEVICE hiprtHit hiprtGeomCustomTraversalClosest::getNextHit() { return m_impl->getNextHit(); }

HIPRT_DEVICE hiprtTraversalState hiprtGeomCustomTraversalClosest::getCurrentState() { return m_impl->getCurrentState(); }

// hiprtGeomCustomTraversalAnyHit
HIPRT_DEVICE hiprtGeomCustomTraversalAnyHit::hiprtGeomCustomTraversalAnyHit(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, hint, payload, funcTable, rayType )
{
}

HIPRT_DEVICE hiprtHit hiprtGeomCustomTraversalAnyHit::getNextHit() { return m_impl->getNextHit(); }

HIPRT_DEVICE hiprtTraversalState hiprtGeomCustomTraversalAnyHit::getCurrentState() { return m_impl->getCurrentState(); }

// hiprtSceneTraversalClosest
HIPRT_DEVICE hiprtSceneTraversalClosest::hiprtSceneTraversalClosest(
	hiprtScene		   scene,
	const hiprtRay&	   ray,
	hiprtRayMask	   mask,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType,
	float			   time )
	: m_impl( scene, ray, mask, hint, payload, funcTable, rayType, time )
{
}

HIPRT_DEVICE hiprtHit hiprtSceneTraversalClosest::getNextHit() { return m_impl->getNextHit(); }

HIPRT_DEVICE hiprtTraversalState hiprtSceneTraversalClosest::getCurrentState() { return m_impl->getCurrentState(); }

// hiprtSceneTraversalAnyHit
HIPRT_DEVICE hiprtSceneTraversalAnyHit::hiprtSceneTraversalAnyHit(
	hiprtScene		   scene,
	const hiprtRay&	   ray,
	hiprtRayMask	   mask,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType,
	float			   time )
	: m_impl( scene, ray, mask, hint, payload, funcTable, rayType, time )
{
}

HIPRT_DEVICE hiprtHit hiprtSceneTraversalAnyHit::getNextHit() { return m_impl->getNextHit(); }

HIPRT_DEVICE hiprtTraversalState hiprtSceneTraversalAnyHit::getCurrentState() { return m_impl->getCurrentState(); }

// hiprtGeomTraversalClosestCustomStack
template <typename hiprtStack>
HIPRT_DEVICE hiprtGeomTraversalClosestCustomStack<hiprtStack>::hiprtGeomTraversalClosestCustomStack(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtStack&		   stack,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, stack, hint, payload, funcTable, rayType )
{
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtHit hiprtGeomTraversalClosestCustomStack<hiprtStack>::getNextHit()
{
	return m_impl->getNextHit();
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtTraversalState hiprtGeomTraversalClosestCustomStack<hiprtStack>::getCurrentState()
{
	return m_impl->getCurrentState();
}

// hiprtGeomTraversalAnyHitCustomStack
template <typename hiprtStack>
HIPRT_DEVICE hiprtGeomTraversalAnyHitCustomStack<hiprtStack>::hiprtGeomTraversalAnyHitCustomStack(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtStack&		   stack,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, stack, hint, payload, funcTable, rayType )
{
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtHit hiprtGeomTraversalAnyHitCustomStack<hiprtStack>::getNextHit()
{
	return m_impl->getNextHit();
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtTraversalState hiprtGeomTraversalAnyHitCustomStack<hiprtStack>::getCurrentState()
{
	return m_impl->getCurrentState();
}

// hiprtGeomCustomTraversalClosestCustomStack
template <typename hiprtStack>
HIPRT_DEVICE hiprtGeomCustomTraversalClosestCustomStack<hiprtStack>::hiprtGeomCustomTraversalClosestCustomStack(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtStack&		   stack,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, stack, hint, payload, funcTable, rayType )
{
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtHit hiprtGeomCustomTraversalClosestCustomStack<hiprtStack>::getNextHit()
{
	return m_impl->getNextHit();
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtTraversalState hiprtGeomCustomTraversalClosestCustomStack<hiprtStack>::getCurrentState()
{
	return m_impl->getCurrentState();
}

// hiprtGeomCustomTraversalAnyHitCustomStack
template <typename hiprtStack>
HIPRT_DEVICE hiprtGeomCustomTraversalAnyHitCustomStack<hiprtStack>::hiprtGeomCustomTraversalAnyHitCustomStack(
	hiprtGeometry	   geom,
	const hiprtRay&	   ray,
	hiprtStack&		   stack,
	hiprtTraversalHint hint,
	void*			   payload,
	hiprtFuncTable	   funcTable,
	uint32_t		   rayType )
	: m_impl( geom, ray, stack, hint, payload, funcTable, rayType )
{
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtHit hiprtGeomCustomTraversalAnyHitCustomStack<hiprtStack>::getNextHit()
{
	return m_impl->getNextHit();
}

template <typename hiprtStack>
HIPRT_DEVICE hiprtTraversalState hiprtGeomCustomTraversalAnyHitCustomStack<hiprtStack>::getCurrentState()
{
	return m_impl->getCurrentState();
}

// hiprtSceneTraversalClosestCustomStack
template <typename hiprtStack, typename hiprtInstanceStack>
HIPRT_DEVICE hiprtSceneTraversalClosestCustomStack<hiprtStack, hiprtInstanceStack>::hiprtSceneTraversalClosestCustomStack(
	hiprtScene			scene,
	const hiprtRay&		ray,
	hiprtStack&			stack,
	hiprtInstanceStack& instanceStack,
	hiprtRayMask		mask,
	hiprtTraversalHint	hint,
	void*				payload,
	hiprtFuncTable		funcTable,
	uint32_t			rayType,
	float				time )
	: m_impl( scene, ray, stack, instanceStack, mask, hint, payload, funcTable, rayType, time )
{
}

// hiprtSceneTraversalClosestCustomStack
template <typename hiprtStack, typename hiprtInstanceStack>
HIPRT_DEVICE hiprtHit hiprtSceneTraversalClosestCustomStack<hiprtStack, hiprtInstanceStack>::getNextHit()
{
	return m_impl->getNextHit();
}

template <typename hiprtStack, typename hiprtInstanceStack>
HIPRT_DEVICE hiprtTraversalState hiprtSceneTraversalClosestCustomStack<hiprtStack, hiprtInstanceStack>::getCurrentState()
{
	return m_impl->getCurrentState();
}

// hiprtSceneTraversalAnyHitCustomStack
template <typename hiprtStack, typename hiprtInstanceStack>
HIPRT_DEVICE hiprtSceneTraversalAnyHitCustomStack<hiprtStack, hiprtInstanceStack>::hiprtSceneTraversalAnyHitCustomStack(
	hiprtScene			scene,
	const hiprtRay&		ray,
	hiprtStack&			stack,
	hiprtInstanceStack& instanceStack,
	hiprtRayMask		mask,
	hiprtTraversalHint	hint,
	void*				payload,
	hiprtFuncTable		funcTable,
	uint32_t			rayType,
	float				time )
	: m_impl( scene, ray, stack, instanceStack, mask, hint, payload, funcTable, rayType, time )
{
}

template <typename hiprtStack, typename hiprtInstanceStack>
HIPRT_DEVICE hiprtHit hiprtSceneTraversalAnyHitCustomStack<hiprtStack, hiprtInstanceStack>::getNextHit()
{
	return m_impl->getNextHit();
}

template <typename hiprtStack, typename hiprtInstanceStack>
HIPRT_DEVICE hiprtTraversalState hiprtSceneTraversalAnyHitCustomStack<hiprtStack, hiprtInstanceStack>::getCurrentState()
{
	return m_impl->getCurrentState();
}

// transformation getters
HIPRT_DEVICE hiprtFrameSRT hiprtGetObjectToWorldFrameSRT( hiprtScene scene, uint32_t instanceID, float time )
{
	hiprt::SceneHeader* sceneHeader = reinterpret_cast<hiprt::SceneHeader*>( scene );
	hiprt::Transform	tr(
		   sceneHeader->m_frames,
		   sceneHeader->m_instances[instanceID].m_frameIndex,
		   sceneHeader->m_instances[instanceID].m_frameCount );
	hiprt::Frame	frame	 = tr.interpolateFrames( time );
	hiprt::SRTFrame srtFrame = hiprt::SRTFrame::getSRTFrame( frame );
	return *reinterpret_cast<hiprtFrameSRT*>( &srtFrame );
}

HIPRT_DEVICE hiprtFrameSRT hiprtGetWorldToObjectFrameSRT( hiprtScene scene, uint32_t instanceID, float time )
{
	hiprt::SceneHeader* sceneHeader = reinterpret_cast<hiprt::SceneHeader*>( scene );
	hiprt::Transform	tr(
		   sceneHeader->m_frames,
		   sceneHeader->m_instances[instanceID].m_frameIndex,
		   sceneHeader->m_instances[instanceID].m_frameCount );
	hiprt::Frame	frame	 = tr.interpolateFrames( time );
	hiprt::SRTFrame srtFrame = hiprt::SRTFrame::getSRTFrameInv( frame );
	return *reinterpret_cast<hiprtFrameSRT*>( &srtFrame );
}

HIPRT_DEVICE hiprtFrameMatrix hiprtGetObjectToWorldFrameMatrix( hiprtScene scene, uint32_t instanceID, float time )
{
	hiprt::SceneHeader* sceneHeader = reinterpret_cast<hiprt::SceneHeader*>( scene );
	hiprt::Transform	tr(
		   sceneHeader->m_frames,
		   sceneHeader->m_instances[instanceID].m_frameIndex,
		   sceneHeader->m_instances[instanceID].m_frameCount );
	hiprt::Frame	   frame	   = tr.interpolateFrames( time );
	hiprt::MatrixFrame matrixFrame = hiprt::MatrixFrame::getMatrixFrame( frame );
	return *reinterpret_cast<hiprtFrameMatrix*>( &matrixFrame );
}

HIPRT_DEVICE hiprtFrameMatrix hiprtGetWorldToObjectFrameMatrix( hiprtScene scene, uint32_t instanceID, float time )
{
	hiprt::SceneHeader* sceneHeader = reinterpret_cast<hiprt::SceneHeader*>( scene );
	hiprt::Transform	tr(
		   sceneHeader->m_frames,
		   sceneHeader->m_instances[instanceID].m_frameIndex,
		   sceneHeader->m_instances[instanceID].m_frameCount );
	hiprt::Frame	   frame	   = tr.interpolateFrames( time );
	hiprt::MatrixFrame matrixFrame = hiprt::MatrixFrame::getMatrixFrameInv( frame );
	return *reinterpret_cast<hiprtFrameMatrix*>( &matrixFrame );
}

// very unoptimized (hopefully correct) transform functions
// it is implemented via the matrix frame (instead of srt) to account for shear
HIPRT_DEVICE float3 hiprtPointObjectToWorld( float3 point, hiprtScene scene, uint32_t instanceID, float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetObjectToWorldFrameMatrix( scene, instanceID, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transform( point );
}

HIPRT_DEVICE float3 hiprtPointWorldToObject( float3 point, hiprtScene scene, uint32_t instanceID, float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetWorldToObjectFrameMatrix( scene, instanceID, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transform( point );
}

HIPRT_DEVICE float3 hiprtVectorObjectToWorld( float3 vector, hiprtScene scene, uint32_t instanceID, float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetObjectToWorldFrameMatrix( scene, instanceID, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transformVector( vector );
}

HIPRT_DEVICE float3 hiprtVectorWorldToObject( float3 vector, hiprtScene scene, uint32_t instanceID, float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetWorldToObjectFrameMatrix( scene, instanceID, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transformVector( vector );
}

// tranformation getters for mlas
HIPRT_DEVICE hiprtFrameSRT
hiprtGetObjectToWorldFrameSRT( hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetObjectToWorldFrameMatrix( scene, instanceIDs, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	hiprt::SRTFrame	   srtFrame	   = hiprt::SRTFrame::getSRTFrame( frame );
	return *reinterpret_cast<hiprtFrameSRT*>( &srtFrame );
}

HIPRT_DEVICE hiprtFrameSRT
hiprtGetWorldToObjectFrameSRT( hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetWorldToObjectFrameMatrix( scene, instanceIDs, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	hiprt::SRTFrame	   srtFrame	   = hiprt::SRTFrame::getSRTFrame( frame );
	return *reinterpret_cast<hiprtFrameSRT*>( &srtFrame );
}

HIPRT_DEVICE hiprtFrameMatrix
hiprtGetObjectToWorldFrameMatrix( hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprt::SceneHeader* sceneHeader = reinterpret_cast<hiprt::SceneHeader*>( scene );
	hiprt::Transform	tr(
		   sceneHeader->m_frames,
		   sceneHeader->m_instances[instanceIDs[0]].m_frameIndex,
		   sceneHeader->m_instances[instanceIDs[0]].m_frameCount );
	hiprt::Frame	   frame	   = tr.interpolateFrames( time );
	hiprt::MatrixFrame matrixFrame = hiprt::MatrixFrame::getMatrixFrame( frame );
	for ( uint32_t i = 1; i < hiprtMaxInstanceLevels; ++i )
	{
		if ( sceneHeader->m_instances[instanceIDs[i - 1]].m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = sceneHeader->m_instances[instanceIDs[i - 1]].m_scene;
		tr			= hiprt::Transform(
			 sceneHeader->m_frames,
			 sceneHeader->m_instances[instanceIDs[i]].m_frameIndex,
			 sceneHeader->m_instances[instanceIDs[i]].m_frameCount );
		frame		= tr.interpolateFrames( time );
		matrixFrame = hiprt::MatrixFrame::multiply( matrixFrame, hiprt::MatrixFrame::getMatrixFrame( frame ) );
	}
	return *reinterpret_cast<hiprtFrameMatrix*>( &matrixFrame );
}

HIPRT_DEVICE hiprtFrameMatrix
hiprtGetWorldToObjectFrameMatrix( hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprt::SceneHeader* sceneHeader = reinterpret_cast<hiprt::SceneHeader*>( scene );
	hiprt::Transform	tr(
		   sceneHeader->m_frames,
		   sceneHeader->m_instances[instanceIDs[0]].m_frameIndex,
		   sceneHeader->m_instances[instanceIDs[0]].m_frameCount );
	hiprt::Frame	   frame	   = tr.interpolateFrames( time );
	hiprt::MatrixFrame matrixFrame = hiprt::MatrixFrame::getMatrixFrameInv( frame );
	for ( uint32_t i = 1; i < hiprtMaxInstanceLevels; ++i )
	{
		if ( sceneHeader->m_instances[instanceIDs[i - 1]].m_type != hiprtInstanceTypeScene ) break;
		sceneHeader = sceneHeader->m_instances[instanceIDs[i - 1]].m_scene;
		tr			= hiprt::Transform(
			 sceneHeader->m_frames,
			 sceneHeader->m_instances[instanceIDs[i]].m_frameIndex,
			 sceneHeader->m_instances[instanceIDs[i]].m_frameCount );
		frame		= tr.interpolateFrames( time );
		matrixFrame = hiprt::MatrixFrame::multiply( hiprt::MatrixFrame::getMatrixFrameInv( frame ), matrixFrame );
	}
	return *reinterpret_cast<hiprtFrameMatrix*>( &matrixFrame );
}

HIPRT_DEVICE float3
hiprtPointObjectToWorld( float3 point, hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetObjectToWorldFrameMatrix( scene, instanceIDs, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transform( point );
}

HIPRT_DEVICE float3
hiprtPointWorldToObject( float3 point, hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetWorldToObjectFrameMatrix( scene, instanceIDs, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transform( point );
}

HIPRT_DEVICE float3
hiprtVectorObjectToWorld( float3 vector, hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetObjectToWorldFrameMatrix( scene, instanceIDs, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transformVector( vector );
}

HIPRT_DEVICE float3
hiprtVectorWorldToObject( float3 vector, hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], float time )
{
	hiprtFrameMatrix   hiprtMatrix = hiprtGetWorldToObjectFrameMatrix( scene, instanceIDs, time );
	hiprt::MatrixFrame matrixFrame = *reinterpret_cast<hiprt::MatrixFrame*>( &hiprtMatrix );
	hiprt::Frame	   frame	   = matrixFrame.convert();
	return frame.transformVector( vector );
}

// explicit template instatiation
template class hiprtPrivateStack_impl<uint32_t, hiprtPrivateStack::StackSize>;
template class hiprtGlobalStack_impl<uint32_t, false>;
template class hiprtGlobalStack_impl<uint32_t, true>;

template class hiprtPrivateStack_impl<hiprtInstanceStackEntry, hiprtPrivateInstanceStack::StackSize>;
template class hiprtGlobalStack_impl<hiprtInstanceStackEntry, false>;
template class hiprtGlobalStack_impl<hiprtInstanceStackEntry, true>;

template class hiprtGeomTraversal_impl<hiprtTriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversal_impl<hiprtTriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class hiprtGeomTraversal_impl<hiprtCustomNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversal_impl<hiprtCustomNode, hiprtTraversalTerminateAtAnyHit>;

template class hiprtSceneTraversal_impl<hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversal_impl<hiprtTraversalTerminateAtAnyHit>;

template class hiprtGeomTraversalCustomStack_impl<hiprtPrivateStack, hiprtTriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtPrivateStack, hiprtCustomNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtPrivateStack, hiprtTriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtPrivateStack, hiprtCustomNode, hiprtTraversalTerminateAtAnyHit>;

template class hiprtGeomTraversalCustomStack_impl<hiprtGlobalStack, hiprtTriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtGlobalStack, hiprtCustomNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtGlobalStack, hiprtTriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtGlobalStack, hiprtCustomNode, hiprtTraversalTerminateAtAnyHit>;

template class hiprtGeomTraversalCustomStack_impl<hiprtDynamicStack, hiprtTriangleNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtDynamicStack, hiprtCustomNode, hiprtTraversalTerminateAtClosestHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtDynamicStack, hiprtTriangleNode, hiprtTraversalTerminateAtAnyHit>;
template class hiprtGeomTraversalCustomStack_impl<hiprtDynamicStack, hiprtCustomNode, hiprtTraversalTerminateAtAnyHit>;

template class hiprtSceneTraversalCustomStack_impl<
	hiprtPrivateStack,
	hiprtEmptyInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<hiprtPrivateStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtGlobalStack,
	hiprtEmptyInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<hiprtGlobalStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtDynamicStack,
	hiprtEmptyInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<hiprtDynamicStack, hiprtEmptyInstanceStack, hiprtTraversalTerminateAtAnyHit>;

template class hiprtSceneTraversalCustomStack_impl<
	hiprtPrivateStack,
	hiprtPrivateInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtPrivateStack,
	hiprtPrivateInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtGlobalStack,
	hiprtPrivateInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtGlobalStack,
	hiprtPrivateInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtDynamicStack,
	hiprtPrivateInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtDynamicStack,
	hiprtPrivateInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;

template class hiprtSceneTraversalCustomStack_impl<
	hiprtPrivateStack,
	hiprtGlobalInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtPrivateStack,
	hiprtGlobalInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtGlobalStack,
	hiprtGlobalInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<hiprtGlobalStack, hiprtGlobalInstanceStack, hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtDynamicStack,
	hiprtGlobalInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtDynamicStack,
	hiprtGlobalInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;

template class hiprtSceneTraversalCustomStack_impl<
	hiprtPrivateStack,
	hiprtDynamicInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtPrivateStack,
	hiprtDynamicInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtGlobalStack,
	hiprtDynamicInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtGlobalStack,
	hiprtDynamicInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtDynamicStack,
	hiprtDynamicInstanceStack,
	hiprtTraversalTerminateAtClosestHit>;
template class hiprtSceneTraversalCustomStack_impl<
	hiprtDynamicStack,
	hiprtDynamicInstanceStack,
	hiprtTraversalTerminateAtAnyHit>;

template class hiprtGeomTraversalClosestCustomStack<hiprtPrivateStack>;
template class hiprtGeomCustomTraversalClosestCustomStack<hiprtPrivateStack>;
template class hiprtGeomTraversalAnyHitCustomStack<hiprtPrivateStack>;
template class hiprtGeomCustomTraversalAnyHitCustomStack<hiprtPrivateStack>;

template class hiprtGeomTraversalClosestCustomStack<hiprtGlobalStack>;
template class hiprtGeomCustomTraversalClosestCustomStack<hiprtGlobalStack>;
template class hiprtGeomTraversalAnyHitCustomStack<hiprtGlobalStack>;
template class hiprtGeomCustomTraversalAnyHitCustomStack<hiprtGlobalStack>;

template class hiprtGeomTraversalClosestCustomStack<hiprtDynamicStack>;
template class hiprtGeomCustomTraversalClosestCustomStack<hiprtDynamicStack>;
template class hiprtGeomTraversalAnyHitCustomStack<hiprtDynamicStack>;
template class hiprtGeomCustomTraversalAnyHitCustomStack<hiprtDynamicStack>;

template class hiprtSceneTraversalClosestCustomStack<hiprtPrivateStack, hiprtEmptyInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtPrivateStack, hiprtEmptyInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtGlobalStack, hiprtEmptyInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtGlobalStack, hiprtEmptyInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtDynamicStack, hiprtEmptyInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtDynamicStack, hiprtEmptyInstanceStack>;

template class hiprtSceneTraversalClosestCustomStack<hiprtPrivateStack, hiprtPrivateInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtPrivateStack, hiprtPrivateInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtGlobalStack, hiprtPrivateInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtGlobalStack, hiprtPrivateInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtDynamicStack, hiprtPrivateInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtDynamicStack, hiprtPrivateInstanceStack>;

template class hiprtSceneTraversalClosestCustomStack<hiprtPrivateStack, hiprtGlobalInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtPrivateStack, hiprtGlobalInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtGlobalStack, hiprtGlobalInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtGlobalStack, hiprtGlobalInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtDynamicStack, hiprtGlobalInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtDynamicStack, hiprtGlobalInstanceStack>;

template class hiprtSceneTraversalClosestCustomStack<hiprtPrivateStack, hiprtDynamicInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtPrivateStack, hiprtDynamicInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtGlobalStack, hiprtDynamicInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtGlobalStack, hiprtDynamicInstanceStack>;
template class hiprtSceneTraversalClosestCustomStack<hiprtDynamicStack, hiprtDynamicInstanceStack>;
template class hiprtSceneTraversalAnyHitCustomStack<hiprtDynamicStack, hiprtDynamicInstanceStack>;
