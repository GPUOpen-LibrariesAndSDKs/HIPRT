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

namespace hiprt
{
class Logger
{
  public:
	static Logger& getInstance();

	void setLevel( uint32_t level ) { m_level = level; }
	int	 getLevel() const { return m_level; }

	void print( uint32_t filter, const char* fmt, ... );

  protected:
	uint32_t m_level = 0;
};

template <typename... Args>
void log( uint32_t filter, Args... args )
{
	Logger::getInstance().print( filter, args... );
}

template <typename... Args>
void logInfo( Args... args )
{
	Logger::getInstance().print( hiprtLogLevelInfo, args... );
}

template <typename... Args>
void logWarn( Args... args )
{
	Logger::getInstance().print( hiprtLogLevelWarn, args... );
}

template <typename... Args>
void logError( Args... args )
{
	Logger::getInstance().print( hiprtLogLevelError, args... );
}
}; // namespace hiprt
