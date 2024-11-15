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

#include <hiprt/impl/AabbList.h>
#include <hiprt/impl/BvhCommon.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/LbvhBuilder.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/TriangleMesh.h>

namespace hiprt
{
/// @brief Calculate the temporary buffer size for kernel execution
/// @param count The input size (The number of primitives or instances)
/// @return The size in byte
size_t LbvhBuilder::getTemporaryBufferSize( const size_t count )
{
	return RoundUp( sizeof( Aabb ), DefaultAlignment ) + 3 * RoundUp( sizeof( uint32_t ) * count, DefaultAlignment ) +
		   RoundUp( count * sizeof( ScratchNode ), DefaultAlignment ) +
		   RoundUp( count * sizeof( ReferenceNode ), DefaultAlignment ) + RoundUp( sizeof( uint32_t ), DefaultAlignment );
}

size_t LbvhBuilder::getTemporaryBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t primCount	   = getPrimCount( buildInput );
	size_t		 size		   = getTemporaryBufferSize( primCount );
	bool		 pairTriangles = false;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh )
	{
		const hiprtTriangleMeshPrimitive& mesh	   = buildInput.primitive.triangleMesh;
		const bool						  pairable = mesh.triangleCount > 2 && mesh.trianglePairCount == 0;
		pairTriangles = pairable && !( buildOptions.buildFlags & hiprtBuildFlagBitDisableTrianglePairing );
		if ( pairTriangles ) size += RoundUp( sizeof( uint2 ) * primCount, DefaultAlignment );
	}
	return size;
}

size_t LbvhBuilder::getTemporaryBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return getTemporaryBufferSize( buildInput.instanceCount );
}

size_t LbvhBuilder::getStorageBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t primCount	  = getPrimCount( buildInput );
	const size_t primNodeSize = getPrimNodeSize( buildInput );
	const size_t boxNodeCount = DivideRoundUp( 2 * primCount, 3 );
	return getGeometryStorageBufferSize( primCount, boxNodeCount, primNodeSize );
}

size_t LbvhBuilder::getStorageBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t frameCount	  = buildInput.frameCount;
	const size_t primCount	  = buildInput.instanceCount;
	const size_t boxNodeCount = DivideRoundUp( 2 * primCount, 3 );
	return getSceneStorageBufferSize( primCount, primCount, boxNodeCount, frameCount );
}

void LbvhBuilder::build(
	Context&					   context,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	oroStream					   stream,
	hiprtDevicePtr				   buffer )
{
	const size_t storageSize = getStorageBufferSize( buildInput, buildOptions );
	const size_t tempSize	 = getTemporaryBufferSize( buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );
	MemoryArena	 temporaryMemoryArena( temporaryBuffer, tempSize, DefaultAlignment );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		build<TriangleNode>(
			context, mesh, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		build<CustomNode>( context, list, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void LbvhBuilder::build(
	Context&					context,
	const hiprtSceneBuildInput& buildInput,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	oroStream					stream,
	hiprtDevicePtr				buffer )
{
	const size_t storageSize = getStorageBufferSize( buildInput, buildOptions );
	const auto	 tempSize	 = getTemporaryBufferSize( buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );
	MemoryArena	 temporaryMemoryArena( temporaryBuffer, tempSize, DefaultAlignment );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<SRTFrame> list( buildInput );
		build<InstanceNode>( context, list, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<MatrixFrame> list( buildInput );
		build<InstanceNode>( context, list, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void LbvhBuilder::update(
	Context&					   context,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	oroStream					   stream,
	hiprtDevicePtr				   buffer )
{
	const size_t storageSize = getStorageBufferSize( buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		update<TriangleNode>( context, mesh, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		update<CustomNode>( context, list, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void LbvhBuilder::update(
	Context&					context,
	const hiprtSceneBuildInput& buildInput,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	oroStream					stream,
	hiprtDevicePtr				buffer )
{
	const size_t storageSize = getStorageBufferSize( buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<SRTFrame> list( buildInput );
		update<InstanceNode>( context, list, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<MatrixFrame> list( buildInput );
		update<InstanceNode>( context, list, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}
} // namespace hiprt
