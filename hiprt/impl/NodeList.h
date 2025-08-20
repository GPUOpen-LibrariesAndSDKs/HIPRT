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
#include <hiprt/hiprt_types.h>
#include <hiprt/impl/BvhNode.h>

namespace hiprt
{
class NodeList
{
  public:
	HIPRT_HOST_DEVICE NodeList( const hiprtBvhNodeList& list ) : m_referenceCount( list.nodeCount )
	{
		m_apiNodes		 = reinterpret_cast<ApiNode*>( list.internalNodes );
		m_referenceNodes = reinterpret_cast<ReferenceNode*>( list.leafNodes );
	}

	HIPRT_HOST_DEVICE const ApiNode* getApiNodes() const { return m_apiNodes; }

	HIPRT_HOST_DEVICE const ReferenceNode* getReferenceNodes() const { return m_referenceNodes; }

	HIPRT_HOST_DEVICE uint32_t getReferenceCount() const { return m_referenceCount; }

  private:
	const ApiNode*		 m_apiNodes;
	const ReferenceNode* m_referenceNodes;
	uint32_t			 m_referenceCount;
};
} // namespace hiprt
