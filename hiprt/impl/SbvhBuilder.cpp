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
#include <hiprt/impl/SbvhBuilder.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/TriangleMesh.h>

namespace hiprt
{
size_t SbvhBuilder::getTemporaryBufferSize( const size_t count, const hiprtBuildOptions buildOptions )
{
	const bool	 spatialSplits	   = !( buildOptions.buildFlags & hiprtBuildFlagBitDisableSpatialSplits );
	const float	 alpha			   = !spatialSplits ? 1.0f : Alpha;
	const size_t maxReferenceCount = alpha * count;
	return roundUp( sizeof( Aabb ), DefaultAlignment ) + roundUp( maxReferenceCount * sizeof( Task ), DefaultAlignment ) +
		   roundUp( maxReferenceCount * sizeof( ScratchNode ), DefaultAlignment ) +
		   roundUp( maxReferenceCount * sizeof( ReferenceNode ), DefaultAlignment ) +
		   roundUp( maxReferenceCount * sizeof( uint32_t ), DefaultAlignment ) +
		   ( !spatialSplits ? 1 : 2 ) *
			   roundUp( ( maxReferenceCount / 2 ) * sizeof( Bin ) * 3 * MinBinCount, DefaultAlignment ) +
		   3 * roundUp( sizeof( uint32_t ), DefaultAlignment );
}

size_t SbvhBuilder::getTemporaryBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t primCount = getPrimCount( buildInput );
	return getTemporaryBufferSize( primCount, buildOptions );
}

size_t SbvhBuilder::getTemporaryBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return getTemporaryBufferSize( buildInput.instanceCount, buildOptions );
}

size_t SbvhBuilder::getStorageBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const float	 alpha			   = buildOptions.buildFlags & hiprtBuildFlagBitDisableSpatialSplits ? 1.0f : Alpha;
	const size_t primCount		   = getPrimCount( buildInput );
	const size_t nodeSize		   = getNodeSize( buildInput );
	const size_t maxReferenceCount = alpha * primCount;
	const size_t nodeCount		   = divideRoundUp( 2 * maxReferenceCount, 3 );
	return getGeometryStorageBufferSize( primCount, nodeCount, nodeSize );
}

size_t SbvhBuilder::getStorageBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const float	 alpha			   = buildOptions.buildFlags & hiprtBuildFlagBitDisableSpatialSplits ? 1.0f : Alpha;
	const size_t frameCount		   = buildInput.frameCount;
	const size_t primCount		   = buildInput.instanceCount;
	const size_t maxReferenceCount = alpha * primCount;
	const size_t nodeCount		   = divideRoundUp( 2 * maxReferenceCount, 3 );
	return getSceneStorageBufferSize( primCount, nodeCount, frameCount );
}

void SbvhBuilder::build(
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
	}
}

void SbvhBuilder::build(
	Context&					context,
	const hiprtSceneBuildInput& buildInput,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	oroStream					stream,
	hiprtDevicePtr				buffer )
{
	const size_t storageSize = getStorageBufferSize( buildInput, buildOptions );
	const size_t tempSize	 = getTemporaryBufferSize( buildInput, buildOptions );
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

void SbvhBuilder::update(
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
	}
}

void SbvhBuilder::update(
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
