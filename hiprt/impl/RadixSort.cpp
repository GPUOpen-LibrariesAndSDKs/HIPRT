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

#include "RadixSort.h"
#include "Utility.h"

#include <Orochi/OrochiUtils.h>

namespace
{
// NOTE: change the path to match where the kernels are stored.
constexpr auto hiprtKernelPath{ "contrib/Orochi/ParallelPrimitives/RadixSortKernels.h" };
constexpr auto hiprtIncludeDir{ "contrib/Orochi/" };
} // namespace

namespace hiprt
{

RadixSort::RadixSort( oroDevice device, oroStream stream, OrochiUtils& oroutils )
	: m_sort(
		  device,
		  oroutils,
		  stream,
		  ( Utility::getRootDir() / hiprtKernelPath ).string(),
		  ( Utility::getRootDir() / hiprtIncludeDir ).string() )
{
}

void RadixSort::sort(
	uint32_t* inputKeys,
	uint32_t* inputValues,
	uint32_t* outputKeys,
	uint32_t* outputValues,
	size_t	  size,
	oroStream stream ) noexcept
{
	Oro::RadixSort::KeyValueSoA srcGpu{};
	Oro::RadixSort::KeyValueSoA dstGpu{};

	srcGpu.key	 = inputKeys;
	srcGpu.value = inputValues;

	dstGpu.key	 = outputKeys;
	dstGpu.value = outputValues;

	static constexpr auto startBit{ 0 };
	static constexpr auto endBit{ 32 };

	m_sort.sort( srcGpu, dstGpu, static_cast<int>( size ), startBit, endBit, stream );
}

} // namespace hiprt
