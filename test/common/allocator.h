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

#include <atomic>

template <class T, int ChunkBits>
class PoolAllocator
{
  public:
	enum
	{
		ChunkItems = 1 << ChunkBits,
		ChunkMask  = ChunkItems - 1,
		ItemMask   = ~ChunkMask,
	};

	struct Chunk
	{
		T                     m_items[ChunkItems];
		uint32_t              m_index = 0;
		std::atomic<uint32_t> m_head  = 0;
	};

	PoolAllocator()
	{
		Chunk* chunk = new Chunk();
		m_ptr        = chunk;
		m_chunks.push_back( chunk );
	}

	~PoolAllocator()
	{
		for ( auto chunk : m_chunks )
			delete chunk;
	}

	PoolAllocator( const PoolAllocator& )  = delete;
	void operator=( const PoolAllocator& ) = delete;

	void allocate( uint32_t* handle, T** ptr )
	{
		while ( true )
		{
			Chunk* chunk = m_ptr.load();
			if ( ChunkItems <= chunk->m_head.load() )
			{
				// the chunk is full but other thread should be allocating a new chunk.
				continue;
			}

			const uint32_t itemIndex = chunk->m_head++;
			if ( itemIndex < ChunkItems )
			{
				// allocation succeeded
				*handle = ( chunk->m_index << ChunkBits ) | itemIndex;
				*ptr    = chunk->m_items + itemIndex;

				if ( itemIndex + 1 ==
				     ChunkItems ) // run out of chunk. let's allocate a new chunk. only one thread can allocate it.
				{
					Chunk* newChunk   = new Chunk();
					newChunk->m_index = chunk->m_index + 1;
					newChunk->m_head  = 0;
					m_chunks.push_back( newChunk );
					m_ptr = newChunk;
				}
				return;
			}
		}
	}

	// these methods conflicts with allocate
	[[nodiscard]] T* item( const uint32_t handle )
	{
		const uint32_t chunkIndex = ( handle & ItemMask ) >> ChunkBits;
		const uint32_t itemIndex  = handle & ChunkMask;
		return m_chunks[chunkIndex]->m_items + itemIndex;
	}

	[[nodiscard]] const T* item( uint32_t handle ) const
	{
		using ME = PoolAllocator<T, ChunkBits>;
		return const_cast<ME*>( this )->item( handle );
	}

	[[nodiscard]] int      chunkCount() const { return m_chunks.size(); }
	[[nodiscard]] T*       chunk( const uint32_t i ) const { return m_chunks[i]->m_items; }
	[[nodiscard]] uint32_t chunkBytes() const { return ChunkItems * sizeof( T ); }
	[[nodiscard]] uint32_t count() const
	{
		return static_cast<uint32_t>( ( m_chunks.size() - 1 ) * ChunkItems + m_ptr.load()->m_head.load() );
	}

  private:
	std::atomic<Chunk*> m_ptr;
	std::vector<Chunk*> m_chunks;
};
