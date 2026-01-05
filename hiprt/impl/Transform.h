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
#include <hiprt/impl/Aabb.h>
#include <hiprt/impl/QrDecomposition.h>
#include <hiprt/impl/Quaternion.h>

namespace hiprt
{
struct SRTFrame;
struct MatrixFrame;

#if defined( HIPRT_MATRIX_FRAME )
using Frame = MatrixFrame;
#else
using Frame = SRTFrame;
#endif

HIPRT_HOST_DEVICE HIPRT_INLINE static bool
identitySRT( const float3& scale, const float3& shear, const float4& rotation, const float3& translation )
{
	if ( scale.x != 1.0f || scale.y != 1.0f || scale.z != 1.0f ) return false;
	if ( shear.x != 0.0f || shear.y != 0.0f || shear.z != 0.0f ) return false;
	if ( translation.x != 0.0f || translation.y != 0.0f || translation.z != 0.0f ) return false;
	if ( rotation.w != 1.0f ) return false;
	return true;
}

HIPRT_HOST_DEVICE HIPRT_INLINE static bool identityMatrix( const float ( &matrix )[3][4] )
{
	if ( matrix[0][0] != 1.0f || matrix[1][1] != 1.0f || matrix[2][2] != 1.0f ) return false;
	if ( matrix[0][1] != 0.0f || matrix[0][2] != 0.0f || matrix[0][3] != 0.0f ) return false;
	if ( matrix[1][0] != 0.0f || matrix[1][2] != 0.0f || matrix[1][3] != 0.0f ) return false;
	if ( matrix[2][0] != 0.0f || matrix[2][1] != 0.0f || matrix[2][3] != 0.0f ) return false;
	return true;
}

HIPRT_HOST_DEVICE static void SRTToInvMatrix(
	const float3& scale, const float3& shear, const float4& rotation, const float3& translation, float ( &matrixInv )[3][4] )
{
	float Q[3][3];
	qtToRotationMatrix( rotation, Q );

	float Ri[3][3];
	Ri[0][0] = 1.0f / scale.x;
	Ri[1][1] = 1.0f / scale.y;
	Ri[2][2] = 1.0f / scale.z;
	Ri[0][1] = -shear.x / ( scale.x * scale.y );
	Ri[0][2] = ( shear.x * shear.z - shear.y * scale.y ) / ( scale.x * scale.y * scale.z );
	Ri[1][2] = -shear.z / ( scale.y * scale.z );
	Ri[1][0] = 0.0f;
	Ri[2][0] = 0.0f;
	Ri[2][1] = 0.0f;

#ifdef __KERNECC__
#pragma unroll
#endif
	for ( uint32_t i = 0; i < 3; ++i )
	{
#ifdef __KERNECC__
#pragma unroll
#endif
		for ( uint32_t j = 0; j < 3; ++j )
		{
			matrixInv[i][j] = dot( { Ri[i][0], Ri[i][1], Ri[i][2] }, { Q[j][0], Q[j][1], Q[j][2] } );
		}
	}

	matrixInv[0][3] = -dot( { matrixInv[0][0], matrixInv[0][1], matrixInv[0][2] }, translation );
	matrixInv[1][3] = -dot( { matrixInv[1][0], matrixInv[1][1], matrixInv[1][2] }, translation );
	matrixInv[2][3] = -dot( { matrixInv[2][0], matrixInv[2][1], matrixInv[2][2] }, translation );
}

HIPRT_HOST_DEVICE static void matrixToInvMatrix( const float ( &matrix )[3][4], float ( &matrixInv )[3][4] )
{
	const auto& m = matrix;

	const float det = m[0][0] * ( m[1][1] * m[2][2] - m[1][2] * m[2][1] ) -
					  m[0][1] * ( m[1][0] * m[2][2] - m[1][2] * m[2][0] ) + m[0][2] * ( m[1][0] * m[2][1] - m[1][1] * m[2][0] );

	constexpr float Epsilon = 1e-10f;
	if ( fabs( det ) < Epsilon )
	{
		memset( &matrixInv[0][0], 0, 12 * sizeof( float ) );
		return;
	}

	const float invDet = 1.0f / det;

	matrixInv[0][0] = ( m[1][1] * m[2][2] - m[1][2] * m[2][1] ) * invDet;
	matrixInv[0][1] = ( m[0][2] * m[2][1] - m[0][1] * m[2][2] ) * invDet;
	matrixInv[0][2] = ( m[0][1] * m[1][2] - m[0][2] * m[1][1] ) * invDet;
	matrixInv[1][0] = ( m[1][2] * m[2][0] - m[1][0] * m[2][2] ) * invDet;
	matrixInv[1][1] = ( m[0][0] * m[2][2] - m[0][2] * m[2][0] ) * invDet;
	matrixInv[1][2] = ( m[0][2] * m[1][0] - m[0][0] * m[1][2] ) * invDet;
	matrixInv[2][0] = ( m[1][0] * m[2][1] - m[1][1] * m[2][0] ) * invDet;
	matrixInv[2][1] = ( m[0][1] * m[2][0] - m[0][0] * m[2][1] ) * invDet;
	matrixInv[2][2] = ( m[0][0] * m[1][1] - m[0][1] * m[1][0] ) * invDet;

	const float3 translation{ matrix[0][3], matrix[1][3], matrix[2][3] };
	matrixInv[0][3] = -dot( { matrixInv[0][0], matrixInv[0][1], matrixInv[0][2] }, translation );
	matrixInv[1][3] = -dot( { matrixInv[1][0], matrixInv[1][1], matrixInv[1][2] }, translation );
	matrixInv[2][3] = -dot( { matrixInv[2][0], matrixInv[2][1], matrixInv[2][2] }, translation );
}

struct alignas( 64 ) SRTFrame
{
	SRTFrame() = default;

	HIPRT_HOST_DEVICE SRTFrame( const hiprtFrameSRT& frame )
	{
		m_rotation	  = qtFromAxisAngle( frame.rotation );
		m_scale		  = frame.scale;
		m_shear		  = make_float3( 0.0f );
		m_translation = frame.translation;
		m_time		  = frame.time;
	}

	HIPRT_HOST_DEVICE SRTFrame( const hiprtFrameMatrix& frame )
	{
		const bool identity = identityMatrix( frame.matrix );
		if ( identity )
		{
			m_scale		  = make_float3( 1.0f );
			m_shear		  = make_float3( 0.0f );
			m_translation = make_float3( 0.0f );
			m_rotation	  = { 0.0f, 0.0f, 0.0f, 1.0f };
		}
		else
		{
			float QR[3][3], Q[3][3], R[3][3];
#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t i = 0; i < 3; ++i )
			{
#ifdef __KERNECC__
#pragma unroll
#endif
				for ( uint32_t j = 0; j < 3; ++j )
				{
					QR[i][j] = frame.matrix[i][j];
				}
			}

			qr( &QR[0][0], &Q[0][0], &R[0][0] );
			m_translation = { frame.matrix[0][3], frame.matrix[1][3], frame.matrix[2][3] };
			m_rotation	  = qtFromRotationMatrix( Q );
			m_scale		  = { R[0][0], R[1][1], R[2][2] };
			m_shear		  = { R[0][1], R[0][2], R[1][2] };
		}
		m_time = frame.time;
	}

	static HIPRT_HOST_DEVICE SRTFrame interpolate( const SRTFrame& f0, const SRTFrame& f1, const float t )
	{
		SRTFrame f{};
		f.m_scale		= mix( f0.m_scale, f1.m_scale, t );
		f.m_shear		= mix( f0.m_shear, f1.m_shear, t );
		f.m_translation = mix( f0.m_translation, f1.m_translation, t );
		f.m_rotation	= qtMix( f0.m_rotation, f1.m_rotation, t );
		return f;
	}

	HIPRT_HOST_DEVICE float3 transform( const float3& p ) const
	{
		if ( identity() ) return p;
		float3 result = p;
		result *= m_scale;
		result += float3{ p.y * m_shear.x + p.z * m_shear.y, p.z * m_shear.z, 0.0f };
		result = qtRotate( m_rotation, result );
		result += m_translation;
		return result;
	}

	HIPRT_HOST_DEVICE float3 transformVector( const float3& v ) const
	{
		if ( identity() ) return v;
		float3 result = v;
		result /= m_scale;
		result.y -= v.x * m_shear.x / m_scale.y;
		result.z -= ( m_shear.y * result.x + m_shear.z * result.y ) / m_scale.z;
		result = qtRotate( m_rotation, result );
		return result;
	}

	HIPRT_HOST_DEVICE float3 invTransform( const float3& p ) const
	{
		if ( identity() ) return p;
		float3 result = p;
		result -= m_translation;
		result = qtInvRotate( m_rotation, result );
		result /= m_scale;
		result.y -= p.z * m_shear.z / m_scale.y;
		result.x -= ( m_shear.x * result.y + m_shear.y * result.z ) / m_scale.x;
		return result;
	}

	HIPRT_HOST_DEVICE float3 invTransformVector( const float3& v ) const
	{
		if ( identity() ) return v;
		float3 result = v;
		result		  = qtInvRotate( m_rotation, result );
		result *= m_scale;
		result += float3{ 0.0f, v.x * m_shear.x, v.x * m_shear.y + v.y * m_shear.z };
		return result;
	}

	HIPRT_HOST_DEVICE bool identity() const { return identitySRT( m_scale, m_shear, m_rotation, m_translation ); }

	float4 m_rotation{ 0.0f, 0.0f, 0.0f, 1.0f };
	float3 m_scale{ 1.0f, 1.0f, 1.0f };
	float3 m_shear{};
	float3 m_translation{};
	float  m_time{};
};
HIPRT_STATIC_ASSERT( sizeof( SRTFrame ) == 64 );

struct alignas( 64 ) MatrixFrame
{
	MatrixFrame() = default;

	HIPRT_HOST_DEVICE MatrixFrame( const hiprtFrameSRT& frame )
	{
		const float4 rotation = qtFromAxisAngle( frame.rotation );
		const bool	 identity = identitySRT( frame.scale, make_float3( 0.0f ), rotation, frame.translation );
		if ( identity )
		{
#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t i = 0; i < 3; ++i )
			{
#ifdef __KERNECC__
#pragma unroll
#endif
				for ( uint32_t j = 0; j < 4; ++j )
				{
					if ( i == j )
						m_matrix[i][j] = 1.0f;
					else
						m_matrix[i][j] = 0.0f;
				}
			}
		}
		else
		{
			float Q[3][3];
			qtToRotationMatrix( rotation, Q );

#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t i = 0; i < 3; ++i )
			{
				m_matrix[i][0] = Q[i][0] * frame.scale.x;
				m_matrix[i][1] = Q[i][1] * frame.scale.y;
				m_matrix[i][2] = Q[i][2] * frame.scale.z;
			}

			m_matrix[0][3] = frame.translation.x;
			m_matrix[1][3] = frame.translation.y;
			m_matrix[2][3] = frame.translation.z;
		}
		m_time = frame.time;
	}

	HIPRT_HOST_DEVICE MatrixFrame( const hiprtFrameMatrix& frame )
	{
		m_time = frame.time;
		memcpy( &m_matrix[0][0], &frame.matrix[0][0], 12 * sizeof( float ) );
	}

	HIPRT_HOST_DEVICE static MatrixFrame interpolate( const MatrixFrame& f0, const MatrixFrame& f1, const float t )
	{
		MatrixFrame f{};
#ifdef __KERNECC__
#pragma unroll
#endif
		for ( uint32_t i = 0; i < 3; ++i )
		{
#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t j = 0; j < 4; ++j )
			{
				f.m_matrix[i][j] = mix( f0.m_matrix[i][j], f1.m_matrix[i][j], t );
			}
		}
		return f;
	}

	HIPRT_HOST_DEVICE float3 transform( const float3& p ) const
	{
		if ( identity() ) return p;
		float3 result{};
		result.x = dot( { m_matrix[0][0], m_matrix[0][1], m_matrix[0][2] }, p );
		result.y = dot( { m_matrix[1][0], m_matrix[1][1], m_matrix[1][2] }, p );
		result.z = dot( { m_matrix[2][0], m_matrix[2][1], m_matrix[2][2] }, p );
		result += { m_matrix[0][3], m_matrix[1][3], m_matrix[2][3] };
		return result;
	}

	HIPRT_HOST_DEVICE float3 transformVector( const float3& v ) const
	{
		if ( identity() ) return v;
		float matrixInv[3][4];
		matrixToInvMatrix( m_matrix, matrixInv );
		float3 result{};
		result.x = dot( { matrixInv[0][0], matrixInv[1][0], matrixInv[2][0] }, v );
		result.y = dot( { matrixInv[0][1], matrixInv[1][1], matrixInv[2][1] }, v );
		result.z = dot( { matrixInv[0][2], matrixInv[1][2], matrixInv[2][2] }, v );
		return result;
	}

	HIPRT_HOST_DEVICE float3 invTransform( const float3& p ) const
	{
		if ( identity() ) return p;
		float matrixInv[3][4];
		matrixToInvMatrix( m_matrix, matrixInv );
		float3 result{};
		result.x = dot( { matrixInv[0][0], matrixInv[0][1], matrixInv[0][2] }, p );
		result.y = dot( { matrixInv[1][0], matrixInv[1][1], matrixInv[1][2] }, p );
		result.z = dot( { matrixInv[2][0], matrixInv[2][1], matrixInv[2][2] }, p );
		result += { matrixInv[0][3], matrixInv[1][3], matrixInv[2][3] };
		return result;
	}

	HIPRT_HOST_DEVICE float3 invTransformVector( const float3& v ) const
	{
		if ( identity() ) return v;
		float3 result{};
		result.x = dot( { m_matrix[0][0], m_matrix[1][0], m_matrix[2][0] }, v );
		result.y = dot( { m_matrix[0][1], m_matrix[1][1], m_matrix[2][1] }, v );
		result.z = dot( { m_matrix[0][2], m_matrix[1][2], m_matrix[2][2] }, v );
		return result;
	}

	HIPRT_HOST_DEVICE bool identity() const { return identityMatrix( m_matrix ); }

	float m_matrix[3][4] = { { 1.0f, 0.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f, 0.0f }, { 0.0f, 0.0f, 1.0f, 0.0f } };
	float m_time{};
};
HIPRT_STATIC_ASSERT( sizeof( MatrixFrame ) == 64 );

HIPRT_HOST_DEVICE static bool computeInvTransformMatrix( const SRTFrame& frame, float ( &matrixInv )[3][4] )
{
	if ( identitySRT( frame.m_scale, frame.m_shear, frame.m_rotation, frame.m_translation ) )
	{
#ifdef __KERNECC__
#pragma unroll
#endif
		for ( uint32_t i = 0; i < 3; ++i )
		{
#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t j = 0; j < 4; ++j )
			{
				if ( i == j )
					matrixInv[i][j] = 1.0f;
				else
					matrixInv[i][j] = 0.0f;
			}
		}
		return true;
	}

	SRTToInvMatrix( frame.m_scale, frame.m_shear, frame.m_rotation, frame.m_translation, matrixInv );

	return false;
}

HIPRT_HOST_DEVICE static bool computeInvTransformMatrix( const MatrixFrame& frame, float ( &matrixInv )[3][4] )
{
	if ( identityMatrix( frame.m_matrix ) )
	{
		memcpy( &matrixInv[0][0], &frame.m_matrix[0][0], 12 * sizeof( float ) );
		return true;
	}

	matrixToInvMatrix( frame.m_matrix, matrixInv );

	return false;
}

class Transform
{
  public:
	HIPRT_HOST_DEVICE Transform( const Frame* frameData, uint32_t frameIndex, uint32_t frameCount )
		: m_frameCount( frameCount ), m_frames( nullptr )
	{
		if ( frameData != nullptr ) m_frames = frameData + frameIndex;
	}

	HIPRT_HOST_DEVICE Frame interpolateFrames( float time ) const
	{
		if ( m_frameCount == 0 || m_frames == nullptr ) return Frame();

		Frame f0 = m_frames[0];
		if ( m_frameCount == 1 || time == 0.0f || time <= f0.m_time ) return f0;

		Frame f1 = m_frames[m_frameCount - 1];
		if ( time >= f1.m_time ) return f1;

		for ( uint32_t i = 1; i < m_frameCount; ++i )
		{
			f1 = m_frames[i];
			if ( time >= f0.m_time && time <= f1.m_time ) break;
			f0 = f1;
		}

		const float t = ( time - f0.m_time ) / ( f1.m_time - f0.m_time );

		return Frame::interpolate( f0, f1, t );
	}

	HIPRT_HOST_DEVICE hiprtRay transformRay( const hiprtRay& ray, float time ) const
	{
		hiprtRay	outRay;
		const Frame frame = interpolateFrames( time );
		if ( frame.identity() ) return ray;
		outRay.origin	 = frame.invTransform( ray.origin );
		outRay.direction = frame.invTransform( ray.origin + ray.direction );
		outRay.direction = outRay.direction - outRay.origin;
		outRay.minT		 = ray.minT;
		outRay.maxT		 = ray.maxT;
		return outRay;
	}

	HIPRT_HOST_DEVICE float3 transformNormal( const float3& normal, float time ) const
	{
		const Frame frame = interpolateFrames( time );
		return frame.transformVector( normal );
	}

	HIPRT_HOST_DEVICE Aabb boundPointMotion( const float3& p ) const
	{
		Aabb outAabb;

		if ( m_frameCount == 0 || m_frames == nullptr )
		{
			outAabb.grow( p );
			return outAabb;
		}

		Frame f0 = m_frames[0];
		outAabb.grow( f0.transform( p ) );

		if ( m_frameCount == 1 ) return outAabb;

		constexpr uint32_t Steps = 3;
		constexpr float	   Delta = 1.0f / float( Steps + 1 );

		Frame f1;
		for ( uint32_t i = 1; i < m_frameCount; ++i )
		{
			f1		= m_frames[i];
			float t = Delta;
			for ( uint32_t j = 1; j <= Steps; ++j )
			{
				Frame f = Frame::interpolate( f0, f1, t );
				outAabb.grow( f.transform( p ) );
				t += Delta;
			}
			f0 = f1;
			outAabb.grow( f0.transform( p ) );
		}

		return outAabb;
	}

	HIPRT_HOST_DEVICE Aabb motionBounds( const Aabb& aabb ) const
	{
		const float3 p0 = aabb.m_min;
		const float3 p1 = { aabb.m_min.x, aabb.m_min.y, aabb.m_max.z };
		const float3 p2 = { aabb.m_min.x, aabb.m_max.y, aabb.m_min.z };
		const float3 p3 = { aabb.m_min.x, aabb.m_max.y, aabb.m_max.z };
		const float3 p4 = { aabb.m_max.x, aabb.m_min.y, aabb.m_min.z };
		const float3 p5 = { aabb.m_max.x, aabb.m_min.y, aabb.m_max.z };
		const float3 p6 = { aabb.m_max.x, aabb.m_max.y, aabb.m_min.z };
		const float3 p7 = aabb.m_max;

		Aabb outAabb;
		outAabb.grow( boundPointMotion( p0 ) );
		outAabb.grow( boundPointMotion( p1 ) );
		outAabb.grow( boundPointMotion( p2 ) );
		outAabb.grow( boundPointMotion( p3 ) );
		outAabb.grow( boundPointMotion( p4 ) );
		outAabb.grow( boundPointMotion( p5 ) );
		outAabb.grow( boundPointMotion( p6 ) );
		outAabb.grow( boundPointMotion( p7 ) );
		return outAabb;
	}

  private:
	uint32_t	 m_frameCount;
	const Frame* m_frames;
};
} // namespace hiprt
