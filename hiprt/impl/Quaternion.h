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
#include <hiprt/hiprt_math.h>

namespace hiprt
{
HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtFromAxisAngle( const float4& axisAngle )
{
	float3 axis	 = normalize( make_float3( axisAngle ) );
	float  angle = axisAngle.w;

	float4 q;
	q.x = axis.x * sinf( angle / 2.0f );
	q.y = axis.y * sinf( angle / 2.0f );
	q.z = axis.z * sinf( angle / 2.0f );
	q.w = cosf( angle / 2.0f );
	return q;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtToAxisAngle( const float4& q )
{
	float3 axis = make_float3( q );
	float  norm = sqrtf( dot( axis, axis ) );
	if ( norm == 0.0f ) return float4{ 0.0f, 0.0f, 1.0f, 0.0f };
	float angle = 2.0f * atan2f( norm, q.w );
	return make_float4( axis / norm, angle );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtFromRotationMatrix( const float R[3][3] )
{
	const float tr = R[0][0] + R[1][1] + R[2][2];
	float4		q;

	if ( tr > 0.0f )
	{
		const float S = sqrtf( tr + 1.0f ) * 2.0f;
		q.w			  = 0.25f * S;
		q.x			  = ( R[2][1] - R[1][2] ) / S;
		q.y			  = ( R[0][2] - R[2][0] ) / S;
		q.z			  = ( R[1][0] - R[0][1] ) / S;
	}
	else if ( ( R[0][0] > R[1][1] ) && ( R[0][0] > R[2][2] ) )
	{
		const float S = sqrtf( 1.0f + R[0][0] - R[1][1] - R[2][2] ) * 2.0f;
		q.w			  = ( R[2][1] - R[1][2] ) / S;
		q.x			  = 0.25f * S;
		q.y			  = ( R[0][1] + R[1][0] ) / S;
		q.z			  = ( R[0][2] + R[2][0] ) / S;
	}
	else if ( R[1][1] > R[2][2] )
	{
		const float S = sqrtf( 1.0f + R[1][1] - R[0][0] - R[2][2] ) * 2.0f;
		q.w			  = ( R[0][2] - R[2][0] ) / S;
		q.x			  = ( R[0][1] + R[1][0] ) / S;
		q.y			  = 0.25f * S;
		q.z			  = ( R[1][2] + R[2][1] ) / S;
	}
	else
	{
		const float S = sqrtf( 1.0f + R[2][2] - R[0][0] - R[1][1] ) * 2.0f;
		q.w			  = ( R[1][0] - R[0][1] ) / S;
		q.x			  = ( R[0][2] + R[2][0] ) / S;
		q.y			  = ( R[1][2] + R[2][1] ) / S;
		q.z			  = 0.25f * S;
	}
	return q;
}

HIPRT_HOST_DEVICE HIPRT_INLINE void qtToRotationMatrix( const float4& q, float R[3][3] )
{
	const float4 q2{ q.x * q.x, q.y * q.y, q.z * q.z, 0.0f };

	R[0][0] = 1 - 2 * q2.y - 2 * q2.z;
	R[0][1] = 2 * q.x * q.y - 2 * q.w * q.z;
	R[0][2] = 2 * q.x * q.z + 2 * q.w * q.y;

	R[1][0] = 2 * q.x * q.y + 2 * q.w * q.z;
	R[1][1] = 1 - 2 * q2.x - 2 * q2.z;
	R[1][2] = 2 * q.y * q.z - 2 * q.w * q.x;

	R[2][0] = 2 * q.x * q.z - 2 * q.w * q.y;
	R[2][1] = 2 * q.y * q.z + 2 * q.w * q.x;
	R[2][2] = 1 - 2 * q2.x - 2 * q2.y;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtGetIdentity() { return float4{ 0.0f, 0.0f, 0.0f, 1.0f }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float qtDot( const float4& q0, const float4& q1 )
{
	return q0.x * q1.x + q0.y * q1.y + q0.z * q1.z + q0.w * q1.w;
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtNormalize( const float4& q ) { return q / sqrtf( qtDot( q, q ) ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtMul( const float4& a, const float4& b )
{
	const float3 c = cross( make_float3( a ), make_float3( b ) ) + a.w * make_float3( b ) + b.w * make_float3( a );
	return { c.x, c.y, c.z, a.w * b.w - dot( make_float3( a ), make_float3( b ) ) };
}

HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtInvert( const float4& q ) { return float4{ -q.x, -q.y, -q.z, q.w }; }

HIPRT_HOST_DEVICE HIPRT_INLINE float3 qtRotate( const float4& q, const float3& p )
{
	const float4 qp	  = make_float4( p, 0.0f );
	const float4 qInv = qtInvert( q );
	const float4 out  = qtMul( qtMul( q, qp ), qInv );
	return make_float3( out );
}

HIPRT_HOST_DEVICE HIPRT_INLINE float3 qtInvRotate( const float4& q, const float3& p ) { return qtRotate( qtInvert( q ), p ); }

HIPRT_HOST_DEVICE HIPRT_INLINE float4 qtMix( float4 v0, float4 v1, float t )
{
	// Only unit quaternions are valid rotations.
	// Normalize to avoid undefined behavior.
	v0 = qtNormalize( v0 );
	v1 = qtNormalize( v1 );

	// Compute the cosine of the angle between the two vectors.
	float dot = qtDot( v0, v1 );

	// If the dot product is negative, slerp won't take
	// the shorter path. Note that v1 and -v1 are equivalent when
	// the negation is applied to all four components. Fix by
	// reversing one quaternion.
	if ( dot < 0.0f )
	{
		v1	= -v1;
		dot = -dot;
	}

	const float DOT_THRESHOLD = 0.9995;
	if ( dot > DOT_THRESHOLD )
	{
		// If the inputs are too close for comfort, linearly interpolate
		// and normalize the result.

		float4 result = v0 + ( v1 - v0 ) * t;
		result		  = qtNormalize( result );
		return result;
	}

	// Since dot is in range [0, DOT_THRESHOLD], acos is safe
	const float theta_0		= acosf( dot );	   // theta_0 = angle between input vectors
	const float theta		= theta_0 * t;	   // theta = angle between v0 and result
	const float sin_theta	= sinf( theta );   // compute this value only once
	const float sin_theta_0 = sinf( theta_0 ); // compute this value only once

	const float s0 = cosf( theta ) - dot * sin_theta / sin_theta_0; // == sin(theta_0 - theta) / sin(theta_0)
	const float s1 = sin_theta / sin_theta_0;

	return ( v0 * s0 ) + ( v1 * s1 );
}
} // namespace hiprt
