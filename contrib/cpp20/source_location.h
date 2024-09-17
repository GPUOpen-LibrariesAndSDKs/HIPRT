//
// Copyright (c) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#pragma once


#if __cplusplus >= 202002L  // if C++20, we can use the default source_location class

#include <source_location>
using source_location = std::source_location;

#else // .. otherwise, we use a custom implementation


class source_location 
{
public:
	constexpr source_location(const char* file = "unknown", const char* function = "unknown", int line = 0, int column = 0) noexcept
		: file_(file), function_(function), line_(line), column_(column) {}

	constexpr const char* file_name() const noexcept { return file_; }
	constexpr const char* function_name() const noexcept { return function_; }
	constexpr int line() const noexcept { return line_; }
	constexpr int column() const noexcept { return column_; }

	 static constexpr source_location current(const char* file = __builtin_FILE(), const char* function = __builtin_FUNCTION(), int line = __builtin_LINE(), int column = 0) noexcept {
		return source_location(file, function, line, column);
	 }

private:
	const char* file_;
	const char* function_;
	int line_;
	int column_;
};


#endif




