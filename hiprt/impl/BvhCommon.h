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
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Instance.h>
#include <hiprt/impl/Scene.h>

namespace hiprt
{
HIPRT_INLINE HIPRT_HOST_DEVICE size_t getPrimCount( const hiprtGeometryBuildInput& buildInput )
{
	size_t primCount;
	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		primCount = buildInput.primitive.triangleMesh.triangleCount;
		if ( buildInput.primitive.triangleMesh.trianglePairCount > 0 )
			primCount = buildInput.primitive.triangleMesh.trianglePairCount;
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		primCount = buildInput.primitive.aabbList.aabbCount;
		break;
	}
#if !defined( __KERNELCC__ )
	default:
		throw std::runtime_error( "Not supported" );
#endif
	}
	return primCount;
}

HIPRT_INLINE HIPRT_HOST_DEVICE size_t getPrimNodeSize( const hiprtGeometryBuildInput& buildInput )
{
	size_t nodeSize;
	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		nodeSize = sizeof( TriangleNode );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		nodeSize = sizeof( CustomNode );
		break;
	}
#if !defined( __KERNELCC__ )
	default:
		throw std::runtime_error( "Not supported" );
#endif
	}
	return nodeSize;
}

HIPRT_INLINE HIPRT_HOST_DEVICE size_t
getGeometryStorageBufferSize( const size_t primNodeCount, const size_t boxNodeCount, const size_t primNodeSize )
{
	return RoundUp( sizeof( GeomHeader ), DefaultAlignment ) + RoundUp( primNodeCount * primNodeSize, DefaultAlignment ) +
		   RoundUp( boxNodeCount * sizeof( BoxNode ), DefaultAlignment );
}

HIPRT_INLINE HIPRT_HOST_DEVICE size_t getSceneStorageBufferSize(
	const size_t primCount, const size_t primNodeCount, const size_t boxNodeCount, const size_t frameCount )
{
	return RoundUp( sizeof( SceneHeader ), DefaultAlignment ) + RoundUp( boxNodeCount * sizeof( BoxNode ), DefaultAlignment ) +
		   RoundUp( primNodeCount * sizeof( InstanceNode ), DefaultAlignment ) +
		   RoundUp( primCount * sizeof( Instance ), DefaultAlignment ) +
		   RoundUp( frameCount * sizeof( Frame ), DefaultAlignment );
}

HIPRT_INLINE HIPRT_HOST_DEVICE bool
batchBuild( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return getPrimCount( buildInput ) <= buildOptions.batchBuildMaxPrimCount &&
		   ( buildOptions.buildFlags & 7 ) != hiprtBuildFlagBitCustomBvhImport;
}

HIPRT_INLINE HIPRT_HOST_DEVICE bool batchBuild( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return buildInput.instanceCount <= buildOptions.batchBuildMaxPrimCount &&
		   ( buildOptions.buildFlags & 7 ) != hiprtBuildFlagBitCustomBvhImport;
}
} // namespace hiprt
