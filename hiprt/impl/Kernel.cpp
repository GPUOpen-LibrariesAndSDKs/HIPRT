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

#include <hiprt/hiprt_math.h>
#include <hiprt/impl/Kernel.h>
#include <hiprt/impl/Error.h>
#include <string.h>

namespace hiprt
{
void Kernel::launch(
	const uint32_t gx,
	const uint32_t gy,
	const uint32_t gz,
	const uint32_t bx,
	const uint32_t by,
	const uint32_t bz,
	const uint32_t sharedMemBytes,
	oroStream	   stream )
{
	checkOro( oroModuleLaunchKernel( m_function, gx, gy, gz, bx, by, bz, sharedMemBytes, stream, m_argPtrs.data(), 0 ) );
}

void Kernel::setArgs( const std::vector<Argument> args )
{
	size_t size = 0;
	for ( size_t i = 0; i < args.size(); i++ )
	{
		size = ( size + args[i].m_align - 1 ) & ~( args[i].m_align - 1 );
		size += args[i].m_size;
	}

	m_args.clear();
	m_args.resize( size );
	m_argPtrs.clear();
	m_argPtrs.resize( size );

	size_t ofs = 0;
	for ( size_t i = 0; i < args.size(); i++ )
	{
		ofs = ( ofs + args[i].m_align - 1 ) & ~( args[i].m_align - 1 );
		std::memcpy( m_args.data() + ofs, args[i].m_value, args[i].m_size );
		m_argPtrs[i] = m_args.data() + ofs;
		ofs += args[i].m_size;
	}
}

void Kernel::launch( const uint32_t nx, oroStream stream, const uint32_t sharedMemBytes )
{
	int tb, minNb;
	checkOro( oroModuleOccupancyMaxPotentialBlockSize( &minNb, &tb, m_function, 0, 0 ) );
	const uint32_t nb = DivideRoundUp( nx, static_cast<uint32_t>( tb ) );
	launch( nb, 1, 1, tb, 1, 1, sharedMemBytes, stream );
}

void Kernel::launch( const uint32_t nx, const uint32_t tx, oroStream stream, const uint32_t sharedMemBytes )
{
	const uint32_t tb = tx;
	const uint32_t nb = DivideRoundUp( nx, tb );
	launch( nb, 1, 1, tb, 1, 1, sharedMemBytes, stream );
}

uint32_t Kernel::getNumSmem() const
{
	int numSmem;
	checkOro( oroFuncGetAttribute( &numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, m_function ) );
	return numSmem;
}

uint32_t Kernel::getNumRegs() const
{
	int numRegs;
	checkOro( oroFuncGetAttribute( &numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, m_function ) );
	return numRegs;
}
} // namespace hiprt
