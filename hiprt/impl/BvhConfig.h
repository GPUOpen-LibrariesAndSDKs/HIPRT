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

namespace hiprt
{
// Cost
static constexpr bool LogBvhCost = false;
// BVH
static constexpr uint32_t BvhBuilderReductionBlockSize = 256u;
static constexpr uint32_t BatchBuilderMaxBlockSize	   = MaxBatchBuildMaxPrimCount;
static constexpr uint32_t CollapseBlockSize			   = 1024u;
// LBVH
static constexpr uint32_t LbvhEmitBlockSize = 512u;
// PLOC
static constexpr uint32_t PlocMainBlockSize = 1024u;
static constexpr uint32_t PlocRadius		= 8u;
// SBVH
static constexpr uint32_t SbvhMinBinCount = 8u;
static constexpr uint32_t SbvhMaxBinCount = 32u;
static constexpr float	  SbvhAlpha		  = 1.5f;
static constexpr float	  SbvhBeta		  = 1.0e-4f;
static constexpr float	  SbvhGamma		  = 1.0e-3f;
static constexpr float	  SbvhEpsilon	  = 1.0e-2f;
}; // namespace hiprt
