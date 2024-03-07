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
#include <Orochi/Orochi.h>

namespace hiprt
{
class Kernel
{
  public:
	struct Argument
	{
		size_t		m_size;
		size_t		m_align;
		const void* m_value;

		template <class T>
		Argument( const T& value )
		{
			m_size	= sizeof( T );
			m_align = __alignof( T );
			m_value = &value;
		}
	};

	Kernel( oroFunction function = 0 ) : m_function( function ) {}

	void setArgs( std::vector<Argument> args );

	void launch(
		uint32_t  gx,
		uint32_t  gy,
		uint32_t  gz,
		uint32_t  bx,
		uint32_t  by,
		uint32_t  bz,
		uint32_t  sharedMemBytes,
		oroStream stream );

	void launch( uint32_t nx, oroStream stream = 0, uint32_t sharedMemBytes = 0 );
	void launch( uint32_t nx, uint32_t tx, oroStream stream = 0, uint32_t sharedMemBytes = 0 );

	uint32_t getNumSmem();
	uint32_t getNumRegs();

  private:
	oroFunction			 m_function;
	std::vector<uint8_t> m_args;
	std::vector<void*>	 m_argPtrs;
};
} // namespace hiprt
