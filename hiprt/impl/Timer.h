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

#include <Orochi/Orochi.h>
#include <hiprt/impl/Error.h>
#include <functional>
#include <unordered_map>

namespace hiprt
{

class Timer final
{
  public:
	static constexpr bool EnableTimer = false;

	using TokenType = int;
	using TimeUnit	= float;

	Timer() = default;

	Timer( const Timer& ) = default;
	Timer( Timer&& )	  = default;

	Timer& operator=( const Timer& )  = default;
	Timer& operator=( Timer&& other ) = default;

	~Timer() = default;

	class Profiler;

	/// Call the callable and measure the elapsed time using Orochi events.
	/// @param[in] token The token of the time record.
	/// @param[in] callable The callable object to be called.
	/// @param[in] args The parameters of the callable.
	/// @return The forwarded returned result of the callable.
	template <typename CallableType, typename... Args>
	decltype( auto ) measure( const TokenType token, CallableType&& callable, Args&&... args ) noexcept
	{
		TimeUnit time{};
		oroEvent start{};
		oroEvent stop{};
		if constexpr ( EnableTimer )
		{
			checkOro( oroEventCreateWithFlags( &start, 0 ) );
			checkOro( oroEventCreateWithFlags( &stop, 0 ) );
			checkOro( oroEventRecord( start, 0 ) );
		}

		using return_type = std::invoke_result_t<CallableType, Args...>;
		if constexpr ( std::is_void_v<return_type> )
		{
			std::invoke( std::forward<CallableType>( callable ), std::forward<Args>( args )... );
			if constexpr ( EnableTimer )
			{
				checkOro( oroEventRecord( stop, 0 ) );
				checkOro( oroEventSynchronize( stop ) );
				checkOro( oroEventElapsedTime( &time, start, stop ) );
				checkOro( oroEventDestroy( start ) );
				checkOro( oroEventDestroy( stop ) );
				timeRecord[token] += time;
			}
			return;
		}
		else
		{
			decltype( auto ) result{ std::invoke( std::forward<CallableType>( callable ), std::forward<Args>( args )... ) };
			if constexpr ( EnableTimer )
			{
				checkOro( oroEventRecord( stop, 0 ) );
				checkOro( oroEventSynchronize( stop ) );
				checkOro( oroEventElapsedTime( &time, start, stop ) );
				checkOro( oroEventDestroy( start ) );
				checkOro( oroEventDestroy( stop ) );
				timeRecord[token] += time;
			}
			return result;
		}
	}

	[[nodiscard]] TimeUnit getTimeRecord( const TokenType token ) const noexcept
	{
		if ( timeRecord.find( token ) != timeRecord.end() ) return timeRecord.at( token );
		return TimeUnit{};
	}

	void reset( const TokenType token ) noexcept
	{
		if ( timeRecord.count( token ) > 0UL )
		{
			timeRecord[token] = TimeUnit{};
		}
	}

	void clear() noexcept { timeRecord.clear(); }

  private:
	using TimeRecord = std::unordered_map<TokenType, TimeUnit>;
	TimeRecord timeRecord;
};

} // namespace hiprt