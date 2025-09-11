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

// from FireRender
template <class T, int CHUNK_BITS>
class PoolAllocator
{
  public:
	enum
	{
		CHUNK_ITEMS = 1 << CHUNK_BITS,
		CHUNK_MASK	= CHUNK_ITEMS - 1,
		ITEM_MASK	= ~CHUNK_MASK,
	};

	struct Chunk
	{
		T					  items[CHUNK_ITEMS];
		uint32_t			  index = 0;
		std::atomic<uint32_t> head	= 0;
	};

	PoolAllocator()
	{
		Chunk* chunk = new Chunk();
		m_ptr		 = chunk;
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
		for ( ;; )
		{
			Chunk* chunk = m_ptr.load();
			if ( CHUNK_ITEMS <= chunk->head.load() )
			{
				// the chunk is full but other thread should be allocating a new chunk.
				continue;
			}

			uint32_t itemIndex = chunk->head++;
			if ( itemIndex < CHUNK_ITEMS )
			{
				// allocation succeeded
				*handle = ( chunk->index << CHUNK_BITS ) | itemIndex;
				*ptr	= chunk->items + itemIndex;

				if ( itemIndex + 1 ==
					 CHUNK_ITEMS ) // run out of chunk. let's allocate a new chunk. only one thread can allocate it.
				{
					Chunk* newChunk = new Chunk();
					newChunk->index = chunk->index + 1;
					newChunk->head	= 0;
					m_chunks.push_back( newChunk );
					m_ptr = newChunk;
				}
				return;
			}
		}
	}

	// these methods conflicts with allocate
	T* item( uint32_t handle )
	{
		uint32_t m			= CHUNK_MASK;
		uint32_t chunkIndex = ( handle & ITEM_MASK ) >> CHUNK_BITS;
		uint32_t itemIndex	= handle & CHUNK_MASK;
		return m_chunks[chunkIndex]->items + itemIndex;
	}

	const T* item( uint32_t handle ) const
	{
		using ME = PoolAllocator<T, CHUNK_BITS>;
		return const_cast<ME*>( this )->item( handle );
	}

	int		 chunkCount() const { return m_chunks.size(); }
	T*		 chunk( int i ) const { return m_chunks[i]->items; }
	uint32_t chunkBytes() const { return CHUNK_ITEMS * sizeof( T ); }
	uint32_t count() const
	{
		return static_cast<uint32_t>( ( m_chunks.size() - 1 ) * CHUNK_ITEMS + m_ptr.load()->head.load() );
	}

  private:
	std::atomic<Chunk*> m_ptr;
	std::vector<Chunk*> m_chunks;
};
