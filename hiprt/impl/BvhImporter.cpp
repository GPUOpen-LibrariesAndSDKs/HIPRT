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
#include <hiprt/impl/Header.h>
#include <hiprt/impl/InstanceList.h>
#include <hiprt/impl/TriangleMesh.h>

namespace hiprt
{
size_t BvhImporter::getTemporaryBufferSize(
	Context& context, const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return getTemporaryBufferSize<hiprtGeometryBuildInput>( context, buildInput, buildOptions );
}

size_t BvhImporter::getTemporaryBufferSize(
	Context& context, const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	return getTemporaryBufferSize<hiprtSceneBuildInput>( context, buildInput, buildOptions );
}

size_t BvhImporter::getStorageBufferSize(
	Context& context, const hiprtGeometryBuildInput& buildInput, [[maybe_unused]] const hiprtBuildOptions buildOptions )
{
	const size_t primCount		   = getPrimCount( buildInput );
	const size_t maxReferenceCount = buildInput.nodeList.nodeCount;
	const size_t primNodeCount	   = getMaxPrimNodeCount( buildInput, context.getRtip(), maxReferenceCount );
	const size_t primNodeSize	   = getPrimNodeSize( buildInput, context.getTriangleNodeSize() );
	const size_t boxNodeCount	   = getMaxBoxNodeCount( buildInput, context.getRtip(), maxReferenceCount );
	return getGeometryStorageBufferSize( primNodeCount, boxNodeCount, primNodeSize, context.getBoxNodeSize() );
}

size_t BvhImporter::getStorageBufferSize(
	Context& context, const hiprtSceneBuildInput& buildInput, [[maybe_unused]] const hiprtBuildOptions buildOptions )
{
	const size_t primCount		   = buildInput.instanceCount;
	const size_t maxReferenceCount = buildInput.nodeList.nodeCount;
	const size_t boxNodeCount	   = getMaxBoxNodeCount( buildInput, context.getRtip(), maxReferenceCount );
	return getSceneStorageBufferSize(
		primCount,
		maxReferenceCount,
		boxNodeCount,
		context.getInstanceNodeSize(),
		context.getBoxNodeSize(),
		buildInput.frameCount );
}

void BvhImporter::build(
	Context&					   context,
	const hiprtGeometryBuildInput& buildInput,
	const hiprtBuildOptions		   buildOptions,
	hiprtDevicePtr				   temporaryBuffer,
	oroStream					   stream,
	hiprtDevicePtr				   buffer )
{
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	const size_t tempSize	 = getTemporaryBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );
	MemoryArena	 temporaryMemoryArena( temporaryBuffer, tempSize, DefaultAlignment );

	NodeList nodes( buildInput.nodeList );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		if ( context.getRtip() >= 31 )
			build<Box8Node, TrianglePacketNode>(
				context, mesh, nodes, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, TrianglePairNode>(
				context, mesh, nodes, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		if ( context.getRtip() >= 31 )
			build<Box8Node, CustomNode>(
				context, list, nodes, buildOptions, buildInput.geomType, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, CustomNode>(
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
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	const size_t tempSize	 = getTemporaryBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );
	MemoryArena	 temporaryMemoryArena( temporaryBuffer, tempSize, DefaultAlignment );

	NodeList nodes( buildInput.nodeList );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<hiprtFrameSRT> list( buildInput );
		if ( context.getRtip() >= 31 )
			build<Box8Node, HwInstanceNode>(
				context, list, nodes, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, UserInstanceNode>(
				context, list, nodes, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<hiprtFrameMatrix> list( buildInput );
		if ( context.getRtip() >= 31 )
			build<Box8Node, HwInstanceNode>(
				context, list, nodes, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		else
			build<Box4Node, UserInstanceNode>(
				context, list, nodes, buildOptions, hiprtInvalidValue, temporaryMemoryArena, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}

void BvhImporter::update(
	Context&						context,
	const hiprtGeometryBuildInput&	buildInput,
	const hiprtBuildOptions			buildOptions,
	[[maybe_unused]] hiprtDevicePtr temporaryBuffer,
	oroStream						stream,
	hiprtDevicePtr					buffer )
{
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	NodeList nodes( buildInput.nodeList );

	switch ( buildInput.type )
	{
	case hiprtPrimitiveTypeTriangleMesh: {
		TriangleMesh mesh( buildInput.primitive.triangleMesh );
		if ( context.getRtip() >= 31 )
			update<Box8Node, TrianglePacketNode>( context, mesh, nodes, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, TrianglePairNode>( context, mesh, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtPrimitiveTypeAABBList: {
		AabbList list( buildInput.primitive.aabbList );
		if ( context.getRtip() >= 31 )
			update<Box8Node, CustomNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, CustomNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}
void BvhImporter::update(
	Context&						context,
	const hiprtSceneBuildInput&		buildInput,
	const hiprtBuildOptions			buildOptions,
	[[maybe_unused]] hiprtDevicePtr temporaryBuffer,
	oroStream						stream,
	hiprtDevicePtr					buffer )
{
	const size_t storageSize = getStorageBufferSize( context, buildInput, buildOptions );
	MemoryArena	 storageMemoryArena( buffer, storageSize, DefaultAlignment );

	NodeList nodes( buildInput.nodeList );

	switch ( buildInput.frameType )
	{
	case hiprtFrameTypeSRT: {
		InstanceList<hiprtFrameSRT> list( buildInput );
		if ( context.getRtip() >= 31 )
			update<Box8Node, HwInstanceNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, UserInstanceNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	case hiprtFrameTypeMatrix: {
		InstanceList<hiprtFrameMatrix> list( buildInput );
		if ( context.getRtip() >= 31 )
			update<Box8Node, HwInstanceNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		else
			update<Box4Node, UserInstanceNode>( context, list, nodes, buildOptions, stream, storageMemoryArena );
		break;
	}
	default:
		throw std::runtime_error( "Not supported" );
	}
}
} // namespace hiprt
