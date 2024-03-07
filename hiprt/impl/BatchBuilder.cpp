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

#include <hiprt/impl/BvhCommon.h>
#include <hiprt/impl/BatchBuilder.h>

namespace hiprt
{
#if !defined( __KERNELCC__ )
DECLARE_TYPE_TRAITS( hiprtGeometryBuildInput );
DECLARE_TYPE_TRAITS( hiprtSceneBuildInput );
#endif

size_t BatchBuilder::getStorageBufferSize( const hiprtGeometryBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t primCount = getPrimCount( buildInput );
	const size_t nodeSize  = getNodeSize( buildInput );
	const size_t nodeCount = divideRoundUp( 2 * primCount, 3 );
	return getGeometryStorageBufferSize( primCount, nodeCount, nodeSize );
}

size_t BatchBuilder::getStorageBufferSize( const hiprtSceneBuildInput& buildInput, const hiprtBuildOptions buildOptions )
{
	const size_t frameCount = buildInput.frameCount;
	const size_t primCount	= buildInput.instanceCount;
	const size_t nodeCount	= divideRoundUp( 2 * primCount, 3 );
	return getSceneStorageBufferSize( primCount, nodeCount, frameCount );
}
} // namespace hiprt
