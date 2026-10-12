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
#include <hiprt/hiprt_types.h>
#include <hiprt/hiprt_vec.h>

namespace hiprt
{
template <typename StackEntry, uint32_t StackSize>
class PrivateStack
{
  public:
	HIPRT_DEVICE PrivateStack();

	HIPRT_DEVICE StackEntry             pop();
	HIPRT_DEVICE void                   push( const StackEntry val );
	[[nodiscard]] HIPRT_DEVICE bool     empty() const;
	[[nodiscard]] HIPRT_DEVICE uint32_t vacancy() const;
	HIPRT_DEVICE void                   reset();

  private:
	StackEntry m_stackBuffer[StackSize];
	uint32_t   m_top;
};

template <typename StackEntry, bool DynamicAssignment>
class GlobalStack
{
  public:
	static constexpr uint32_t Stride    = WarpSize;
	static constexpr uint32_t LogStride = Log2( Stride );

	HIPRT_DEVICE
	GlobalStack( hiprtGlobalStackBuffer globalStackBuffer, hiprtSharedStackBuffer sharedStackBuffer );

	HIPRT_DEVICE ~GlobalStack();

	HIPRT_DEVICE StackEntry             pop();
	HIPRT_DEVICE void                   push( const StackEntry val );
	[[nodiscard]] HIPRT_DEVICE uint32_t vacancy() const;
	[[nodiscard]] HIPRT_DEVICE bool     empty() const;
	HIPRT_DEVICE void                   reset();

  private:
	uint32_t*   m_globalStackLock;
	StackEntry* m_globalStackBuffer;
	StackEntry* m_sharedStackBuffer;
	uint32_t    m_globalStackSize;
	uint32_t    m_sharedStackSize;
	int32_t     m_globalIndex;
	int32_t     m_sharedIndex;
	uint32_t    m_sharedCount;
};
} // namespace hiprt

/** \brief An empty dummy instance stack.
 *
 * Use this stack if you use one level of instancing.
 */
class hiprtEmptyInstanceStack
{
};

/** \brief A stack using (slow) local memory internally.
 *
 */
using hiprtPrivateStack         = hiprt::PrivateStack<uint32_t, 64u>;
using hiprtPrivateInstanceStack = hiprt::PrivateStack<hiprtInstanceStackEntry, hiprtMaxInstanceLevels - 1>;

/** \brief A stack using both (fast) shared memory and (slow) global memory.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
using hiprtGlobalStack         = hiprt::GlobalStack<uint32_t, false>;
using hiprtGlobalInstanceStack = hiprt::GlobalStack<hiprtInstanceStackEntry, false>;

/** \brief A stack using both (fast) shared memory and (slow) global memory with dynamic assignment.
 *
 * The stack uses shared memory if there is enough space.
 * Otherwise, it uses global memory as a backup.
 */
using hiprtDynamicStack         = hiprt::GlobalStack<uint32_t, true>;
using hiprtDynamicInstanceStack = hiprt::GlobalStack<hiprtInstanceStackEntry, true>;

namespace hiprt
{
struct SRTFrame;
struct MatrixFrame;
struct SceneHeader;

#if defined( HIPRT_MATRIX_FRAME )
using Frame = MatrixFrame;
#else
using Frame        = SRTFrame;
#endif

struct CustomNode;

namespace rtip11
{
struct TrianglePairNode;
struct BoxNode;
struct InstanceNode;
}; // namespace rtip11

namespace rtip31
{
struct TrianglePacketNode;
struct BoxNode;
struct InstanceNode;
}; // namespace rtip31

#if HIPRT_RTIP >= 31
using TriangleNode = rtip31::TrianglePacketNode;
using BoxNode      = rtip31::BoxNode;
using InstanceNode = rtip31::InstanceNode;
#else
using TriangleNode = rtip11::TrianglePairNode;
using BoxNode      = rtip11::BoxNode;
using InstanceNode = rtip11::InstanceNode;
#endif

template <typename Stack, hiprtTraversalType TraversalType>
class TraversalBase
{
  public:
	HIPRT_DEVICE TraversalBase(
		const hiprtRay&          ray,
		Stack&                   stack,
		const hiprtTraversalHint hint,
		void*                    payload,
		const hiprtFuncTable     funcTable,
		const uint32_t           rayType );

	[[nodiscard]] HIPRT_DEVICE hiprtTraversalState getCurrentState() const;

  protected:
#if HIPRT_RTIP >= 11
	HIPRT_DEVICE void packDescriptor( const uint32_t boxSortHeuristic = 0u, const bool compressed = false );
#endif

	[[nodiscard]] HIPRT_DEVICE bool
	testInternalNode( const hiprtRay& ray, const float3& invD, const BoxNode* nodes, uint32_t& nodeIndex );

	[[nodiscard]] HIPRT_DEVICE bool testTriangleNode(
		const hiprtRay&     ray,
		const float3&       invD,
		const TriangleNode* nodes,
		const uint32_t      geomType,
		uint32_t&           leafIndex,
		uint32_t&           triangleMask,
		hiprtHit&           hit );

	[[nodiscard]] HIPRT_DEVICE bool testTriangle(
		const hiprtRay&                 ray,
		const float3&                   invD,
		const rtip11::TrianglePairNode* nodes,
		const uint32_t                  leafAddr,
		const uint32_t                  leafType,
		hiprtHit&                       hit );

	[[nodiscard]] HIPRT_DEVICE uint32_t testTrianglePair(
		const hiprtRay&                   ray,
		const rtip31::TrianglePacketNode* nodes,
		const uint32_t                    leafAddr,
		const uint32_t                    triPairIndex,
		hiprtHit&                         hit0,
		hiprtHit&                         hit1,
		bool&                             nodeEnd,
		bool&                             rangeEnd );

  protected:
	hiprtRay             m_ray;
	hiprtFuncTableHeader m_tableHeader = { 0, 1, nullptr };
	uint4                m_descriptor{};
	Stack&               m_stack;
	const BoxNode*       m_boxNodes = nullptr;
	void*                m_payload  = nullptr;
	uint32_t             m_nodeIndex{};
	uint32_t             m_rayType{};
	uint32_t             m_triangleMask{};
	hiprtTraversalState  m_state = hiprtTraversalStateInit;
	hiprtTraversalHint   m_hint  = hiprtTraversalHintDefault;
};

template <typename Stack, typename PrimitiveNode, hiprtTraversalType TraversalType>
class GeomTraversal : public TraversalBase<Stack, TraversalType>
{
  public:
	HIPRT_DEVICE
	GeomTraversal(
		const hiprtGeometry      geometry,
		const hiprtRay&          ray,
		Stack&                   stack,
		const hiprtTraversalHint hint      = hiprtTraversalHintDefault,
		void*                    payload   = nullptr,
		const hiprtFuncTable     funcTable = nullptr,
		const uint32_t           rayType   = 0u );

	[[nodiscard]] HIPRT_DEVICE hiprtHit getNextHit();

  protected:
	[[nodiscard]] HIPRT_DEVICE bool
	testLeafNode( const hiprtRay& ray, const float3& invD, uint32_t& leafIndex, uint32_t& triangleMask, hiprtHit& hit );

  protected:
	const PrimitiveNode* m_primNodes = nullptr;
	uint32_t             m_geomType{};
	uint32_t             m_leafIndex{};
};

template <typename Stack, typename InstanceStack, hiprtTraversalType TraversalType>
class SceneTraversal : public TraversalBase<Stack, TraversalType>
{
  public:
	HIPRT_DEVICE SceneTraversal(
		const hiprtScene         scene,
		const hiprtRay&          ray,
		Stack&                   stack,
		InstanceStack&           instanceStack,
		const hiprtRayMask       mask      = InvalidValue,
		const hiprtTraversalHint hint      = hiprtTraversalHintDefault,
		void*                    payload   = nullptr,
		const hiprtFuncTable     funcTable = nullptr,
		const uint32_t           rayType   = 0,
		const float              time      = 0.0f );

	[[nodiscard]] HIPRT_DEVICE hiprtHit getNextHit();

  protected:
	[[nodiscard]] HIPRT_DEVICE const uint32_t& instanceId() const;

	[[nodiscard]] HIPRT_DEVICE uint32_t& instanceId();

	HIPRT_DEVICE bool transformRay( const uint32_t nodeIndex, hiprtRay& ray, float3& invD );

	HIPRT_DEVICE void restoreRay( hiprtRay& ray, float3& invD ) const;

	[[nodiscard]] HIPRT_DEVICE bool testLeafNode(
		const void*     primNodes,
		const hiprtRay& ray,
		const float3&   invD,
		const uint32_t  geomType,
		uint32_t&       leafIndex,
		uint32_t&       triangleMask,
		hiprtHit&       hit );

  protected:
	union
	{
		uint32_t m_instanceId{};
		uint32_t m_instanceIds[MaxInstanceLevels];
	};

	InstanceStack&      m_instanceStack;
	const SceneHeader*  m_scene         = nullptr;
	const InstanceNode* m_instanceNodes = nullptr;
	const Frame*        m_frames        = nullptr;
	const hiprtRayMask  m_mask          = FullRayMask;
	uint32_t            m_level{};
	uint32_t            m_instanceIndex{};
	const float         m_time{};
};

template <typename PrimitiveNode, hiprtTraversalType TraversalType>
class GeomTraversalPrivateStack
{
  public:
	using Stack = hiprtPrivateStack;

	HIPRT_DEVICE GeomTraversalPrivateStack(
		hiprtGeometry            geom,
		const hiprtRay&          ray,
		const hiprtTraversalHint hint      = hiprtTraversalHintDefault,
		void*                    payload   = nullptr,
		const hiprtFuncTable     funcTable = nullptr,
		const uint32_t           rayType   = 0 );

	[[nodiscard]] HIPRT_DEVICE hiprtHit getNextHit();

	[[nodiscard]] HIPRT_DEVICE hiprtTraversalState getCurrentState() const;

  private:
	Stack                                              m_stack;
	GeomTraversal<Stack, PrimitiveNode, TraversalType> m_traversal;
};

template <hiprtTraversalType TraversalType>
class SceneTraversalPrivateStack
{
  public:
	using Stack         = hiprtPrivateStack;
	using InstanceStack = hiprtPrivateInstanceStack;

	HIPRT_DEVICE SceneTraversalPrivateStack(
		hiprtScene               scene,
		const hiprtRay&          ray,
		const hiprtRayMask       mask      = hiprtFullRayMask,
		const hiprtTraversalHint hint      = hiprtTraversalHintDefault,
		void*                    payload   = nullptr,
		const hiprtFuncTable     funcTable = nullptr,
		const uint32_t           rayType   = 0,
		const float              time      = 0.0f );

	[[nodiscard]] HIPRT_DEVICE hiprtHit getNextHit();

	[[nodiscard]] HIPRT_DEVICE hiprtTraversalState getCurrentState() const;

  private:
	Stack                                               m_stack;
	InstanceStack                                       m_instanceStack;
	SceneTraversal<Stack, InstanceStack, TraversalType> m_traversal;
};
} // namespace hiprt

/** \brief A traversal object for finding the closest / any hit with hiprtGeometry containing triangles.
 *
 * It uses a private stack with size 64 internally.
 */
using hiprtGeomTraversalClosest = hiprt::GeomTraversalPrivateStack<hiprt::TriangleNode, hiprtTraversalTerminateAtClosestHit>;
using hiprtGeomTraversalAnyHit  = hiprt::GeomTraversalPrivateStack<hiprt::TriangleNode, hiprtTraversalTerminateAtAnyHit>;

/** \brief A traversal object for finding the closest / any hit with hiprtGeometry containing custom primitives.
 *
 * It uses a private stack with size 64 internally.
 */
using hiprtGeomCustomTraversalClosest =
	hiprt::GeomTraversalPrivateStack<hiprt::CustomNode, hiprtTraversalTerminateAtClosestHit>;
using hiprtGeomCustomTraversalAnyHit = hiprt::GeomTraversalPrivateStack<hiprt::CustomNode, hiprtTraversalTerminateAtAnyHit>;

/** \brief A traversal object for finding the closest / any hit with hiprtScene.
 *
 * It uses a private stack with size 64 internally.
 */
using hiprtSceneTraversalClosest = hiprt::SceneTraversalPrivateStack<hiprtTraversalTerminateAtClosestHit>;
using hiprtSceneTraversalAnyHit  = hiprt::SceneTraversalPrivateStack<hiprtTraversalTerminateAtAnyHit>;

/** \brief A traversal object for finding the closest / any hit with hiprtGeometry containing triangles.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
using hiprtGeomTraversalClosestCustomStack =
	hiprt::GeomTraversal<hiprtStack, hiprt::TriangleNode, hiprtTraversalTerminateAtClosestHit>;

template <typename hiprtStack>
using hiprtGeomTraversalAnyHitCustomStack =
	hiprt::GeomTraversal<hiprtStack, hiprt::TriangleNode, hiprtTraversalTerminateAtAnyHit>;

/** \brief A traversal object for finding the closest / any hit with hiprtGeometry containing custom primitives.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack>
using hiprtGeomCustomTraversalClosestCustomStack =
	hiprt::GeomTraversal<hiprtStack, hiprt::CustomNode, hiprtTraversalTerminateAtClosestHit>;

template <typename hiprtStack>
using hiprtGeomCustomTraversalAnyHitCustomStack =
	hiprt::GeomTraversal<hiprtStack, hiprt::CustomNode, hiprtTraversalTerminateAtAnyHit>;

/** \brief A traversal object for finding the closest / any hit with hiprtScene.
 *
 * \tparam hiprtStack A custom stack.
 */
template <typename hiprtStack, typename hiprtInstanceStack>
using hiprtSceneTraversalClosestCustomStack =
	hiprt::SceneTraversal<hiprtStack, hiprtInstanceStack, hiprtTraversalTerminateAtClosestHit>;

template <typename hiprtStack, typename hiprtInstanceStack>
using hiprtSceneTraversalAnyHitCustomStack =
	hiprt::SceneTraversal<hiprtStack, hiprtInstanceStack, hiprtTraversalTerminateAtAnyHit>;

/** \brief Returns the object to world transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT
hiprtGetObjectToWorldFrameSRT( const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT
hiprtGetWorldToObjectFrameSRT( const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Returns the object to world transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix
hiprtGetObjectToWorldFrameMatrix( const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix
hiprtGetWorldToObjectFrameMatrix( const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Returns the object to world transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT hiprtGetObjectToWorldFrameSRT(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the SRT frame.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameSRT hiprtGetWorldToObjectFrameSRT(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time = 0.0f );

/** \brief Returns the object to world transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix hiprtGetObjectToWorldFrameMatrix(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time = 0.0f );

/** \brief Returns the world to object transformation for a given instance and time in the form of the matrix.
 *
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
HIPRT_DEVICE hiprtFrameMatrix hiprtGetWorldToObjectFrameMatrix(
	const hiprtScene scene, const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels], const float time = 0.0f );

/** \brief Transforms a point from the object space to the world space.
 *
 * \param point A point in the object space.
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3
hiprtPointObjectToWorld( const float3& point, const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Transforms a point from the world space to the object space.
 *
 * \param point A point in the world space.
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3
hiprtPointWorldToObject( const float3& point, const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Transforms a vector from the object space to the world space.
 *
 * \param vector A vector in object space.
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3
hiprtVectorObjectToWorld( const float3& vector, const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Transforms a vector from the world space to the object space.
 *
 * \param vector A vector in the world space.
 * \param scene A scene.
 * \param instanceID Instance ID.
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3
hiprtVectorWorldToObject( const float3& vector, const hiprtScene scene, const uint32_t instanceID, const float time = 0.0f );

/** \brief Transforms a point from the object space to the world space.
 *
 * \param point A point in the object space.
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3 hiprtPointObjectToWorld(
	const float3&    point,
	const hiprtScene scene,
	const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels],
	const float time = 0.0f );

/** \brief Transforms a point from the world space to the object space.
 *
 * \param point A point in the world space.
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3 hiprtPointWorldToObject(
	const float3&    point,
	const hiprtScene scene,
	const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels],
	const float time = 0.0f );

/** \brief Transforms a vector from the object space to the world space.
 *
 * \param vector A vector in object space.
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3 hiprtVectorObjectToWorld(
	const float3&    vector,
	const hiprtScene scene,
	const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels],
	const float time = 0.0f );

/** \brief Transforms a vector from the world space to the object space.
 *
 * \param vector A vector in the world space.
 * \param scene A scene.
 * \param instanceIDs Instance IDs (multi-level instancing).
 * \param time The time.
 */
[[nodiscard]] HIPRT_DEVICE float3 hiprtVectorWorldToObject(
	const float3&    vector,
	const hiprtScene scene,
	const uint32_t ( &instanceIDs )[hiprtMaxInstanceLevels],
	const float time = 0.0f );
