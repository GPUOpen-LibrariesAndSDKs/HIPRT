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
#include <hiprt/impl/BvhImporter.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/Scene.h>
#include <hiprt/impl/TriangleMesh.h>

namespace hiprt
{
size_t BvhImporter::getTemporaryBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return 0;
}

size_t BvhImporter::getTemporaryBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return 0;
}

size_t BvhImporter::getStorageBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t primCount	  = getPrimCount( buildInput );
	const size_t primNodeSize = getPrimNodeSize( buildInput );
	const size_t boxNodeCount = buildInput.nodeList.nodeCount;
	return getGeometryStorageBufferSize( primCount, boxNodeCount, primNodeSize );
}

size_t BvhImporter::getStorageBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t frameCount	  = buildInput.frameCount;
	const size_t primCount	  = buildInput.instanceCount;
	const size_t boxNodeCount = buildInput.nodeList.nodeCount;
	return getSceneStorageBufferSize( primCount, primCount, boxNodeCount, frameCount );
}

void BvhImporter::build(
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

	ApiNodeList nodes( buildInput.nodeList );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		build<TriangleNode>(
			context, mesh, nodes, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		build<CustomNode>(
			context, list, nodes, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void BvhImporter::build(
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

	ApiNodeList nodes( buildInput.nodeList );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<SRTFrame> list( buildInput );
		build<InstanceNode>(
			context, list, nodes, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<MatrixFrame> list( buildInput );
		build<InstanceNode>(
			context, list, nodes, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void BvhImporter::update(
	Context&					   context,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	oroStream					   stream,
	hiprtDevicePtr				   buffer )
{
	const size_t storageSize = getStorageBufferSize( buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	ApiNodeList nodes( buildInput.nodeList );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		update<TriangleNode>( context, mesh, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		update<CustomNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}
void BvhImporter::update(
	Context&					context,
	const hiprtSceneBuildInput& buildInput,
	const hiprtBuildOptions		buildOptions,
	hiprtDevicePtr				temporaryBuffer,
	oroStream					stream,
	hiprtDevicePtr				buffer )
{
	const size_t storageSize = getStorageBufferSize( buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	ApiNodeList nodes( buildInput.nodeList );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<SRTFrame> list( buildInput );
		update<InstanceNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<MatrixFrame> list( buildInput );
		update<InstanceNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}
} // namespace hiprt
