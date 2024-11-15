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

namespace hiprt
{
class MemoryArena
{
  public:
	HIPRT_HOST_DEVICE MemoryArena( hiprtDevicePtr data, size_t storageSize, uint32_t alignment )
		: m_data( data ), m_storageSize( storageSize ), m_alignment( alignment ), m_offset( 0 )
	{
	}

	template <typename T>
	HIPRT_HOST_DEVICE T* allocate( size_t size = 1 )
	{
		if ( size == 0 ) return nullptr;
		T* p = reinterpret_cast<T*>( reinterpret_cast<uint8_t*>( m_data ) + m_offset );
		m_offset += RoundUp( sizeof( T ) * size, m_alignment );
		HIPRT_ASSERT( m_offset <= m_storageSize );
		return p;
	}

	HIPRT_HOST_DEVICE size_t getStorageSize() { return m_storageSize; }

  private:
	hiprtDevicePtr m_data;
	uint32_t	   m_alignment;
	size_t		   m_offset;
	size_t		   m_storageSize;
};
} // namespace hiprt
