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
#include <hiprt/impl/Quaternion.h>

namespace hiprt
{
HIPRT_HOST_DEVICE static float hypot3f( const float x, const float y, const float z )
{
	float a = fabsf( x );
	float b = fabsf( y );
	float c = fabsf( z );
	float d = a < b ? b < c ? c : b : a < c ? c : a;
	return ( d != 0.0f ? ( d * sqrtf( ( a / d ) * ( a / d ) + ( b / d ) * ( b / d ) + ( c / d ) * ( c / d ) ) ) : 0.0f );
}

HIPRT_HOST_DEVICE static void qr( const float* a, float* q, float* r )
{
	// A and Q may be the same. QR using Modified Gram-Schmidt
	// method, A must be fullrank.
	float w;

	if ( a != q )
	{
		q[0] = a[0];
		q[1] = a[1];
		q[2] = a[2];
		q[3] = a[3];
		q[4] = a[4];
		q[5] = a[5];
		q[6] = a[6];
		q[7] = a[7];
		q[8] = a[8];
	}

	r[0] = 0.0f;
	r[1] = 0.0f;
	r[2] = 0.0f;
	r[3] = 0.0f;
	r[4] = 0.0f;
	r[5] = 0.0f;
	r[6] = 0.0f;
	r[7] = 0.0f;
	r[8] = 0.0f;

	w	 = hypot3f( q[0], q[3], q[6] );
	r[0] = w;
	if ( w != 0.0f )
	{
		w	 = 1.0f / w;
		q[0] = q[0] * w;
		q[3] = q[3] * w;
		q[6] = q[6] * w;
	}
	else
	{
		q[0] = 1.0f;
	}

	w	 = q[0] * q[1] + q[3] * q[4] + q[6] * q[7];
	r[1] = w;

	q[1] = q[1] - w * q[0];
	q[4] = q[4] - w * q[3];
	q[7] = q[7] - w * q[6];

	w	 = hypot3f( q[1], q[4], q[7] );
	r[4] = w;
	if ( w != 0.0f )
	{
		w	 = 1.0f / w;
		q[1] = q[1] * w;
		q[4] = q[4] * w;
		q[7] = q[7] * w;
	}
	else
	{
		q[1] = 1.0f;
	}

	w	 = q[0] * q[2] + q[3] * q[5] + q[6] * q[8];
	r[2] = w;

	q[2] = q[2] - w * q[0];
	q[5] = q[5] - w * q[3];
	q[8] = q[8] - w * q[6];

	w	 = q[1] * q[2] + q[4] * q[5] + q[7] * q[8];
	r[5] = w;

	q[2] = q[2] - w * q[1];
	q[5] = q[5] - w * q[4];
	q[8] = q[8] - w * q[7];

	w	 = hypot3f( q[2], q[5], q[8] );
	r[8] = w;
	if ( w != 0.0f )
	{
		w	 = 1.0f / w;
		q[2] = q[2] * w;
		q[5] = q[5] * w;
		q[8] = q[8] * w;
	}
	else
	{
		q[2] = 1.0f;
	}

	float d = q[0] * q[4] * q[8] + q[1] * q[5] * q[6] + q[3] * q[7] * q[2] - q[2] * q[4] * q[6] - q[1] * q[3] * q[8] -
			  q[5] * q[7] * q[0];

	if ( d < 0.0f )
	{
		q[0] = -q[0];
		q[3] = -q[3];
		q[6] = -q[6];
		r[0] = -r[0];
		r[1] = -r[1];
		r[2] = -r[2];
	}
}
} // namespace hiprt
