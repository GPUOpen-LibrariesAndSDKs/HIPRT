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

namespace hiprt
{
struct SRTFrame;
struct MatrixFrame;

struct alignas( 64 ) Frame
{
	HIPRT_HOST_DEVICE Frame() : m_time( 0.0f )
	{
		m_scale		  = make_float3( 1.0f );
		m_shear		  = make_float3( 0.0f );
		m_translation = make_float3( 0.0f );
		m_rotation	  = { 0.0f, 0.0f, 0.0f, 1.0f };
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

	HIPRT_HOST_DEVICE bool identity() const
	{
		if ( m_scale.x != 1.0f || m_scale.y != 1.0f || m_scale.z != 1.0f ) return false;
		if ( m_shear.x != 0.0f || m_shear.y != 0.0f || m_shear.z != 0.0f ) return false;
		if ( m_translation.x != 0.0f || m_translation.y != 0.0f || m_translation.z != 0.0f ) return false;
		if ( m_rotation.w != 1.0f ) return false;
		return true;
	}

	float4 m_rotation;
	float3 m_scale;
	float3 m_shear;
	float3 m_translation;
	float  m_time;
};
HIPRT_STATIC_ASSERT( sizeof( Frame ) == 64 );

struct alignas( 16 ) SRTFrame
{
	float4 m_rotation;
	float3 m_scale;
	float3 m_translation;
	float  m_time;

	HIPRT_HOST_DEVICE Frame convert() const
	{
		Frame frame;
		frame.m_time		= m_time;
		frame.m_rotation	= qtFromAxisAngle( m_rotation );
		frame.m_scale		= m_scale;
		frame.m_shear		= make_float3( 0.0f );
		frame.m_translation = m_translation;
		return frame;
	}

	static HIPRT_HOST_DEVICE SRTFrame getSRTFrame( const Frame& frame )
	{
		SRTFrame srtFrame;
		srtFrame.m_time		   = frame.m_time;
		srtFrame.m_translation = frame.m_translation;
		srtFrame.m_scale	   = frame.m_scale;
		srtFrame.m_rotation	   = qtToAxisAngle( frame.m_rotation );
		return srtFrame;
	}

	static HIPRT_HOST_DEVICE SRTFrame getSRTFrameInv( const Frame& frame )
	{
		SRTFrame srtFrame;
		srtFrame.m_time		   = frame.m_time;
		srtFrame.m_translation = -frame.m_translation;
		srtFrame.m_scale	   = 1.0f / frame.m_scale;
		srtFrame.m_rotation	   = qtToAxisAngle( frame.m_rotation );
		srtFrame.m_rotation.w *= -1.0f;
		return srtFrame;
	}
};
HIPRT_STATIC_ASSERT( sizeof( SRTFrame ) == 48 );

struct alignas( 64 ) MatrixFrame
{
	float m_matrix[3][4];
	float m_time;

	HIPRT_HOST_DEVICE Frame convert() const
	{
		float QR[3][3], Q[3][3], R[3][3];
#ifdef __KERNECC__
#pragma unroll
#endif
		for ( uint32_t i = 0; i < 3; ++i )
#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t j = 0; j < 3; ++j )
				QR[i][j] = m_matrix[i][j];
		qr( &QR[0][0], &Q[0][0], &R[0][0] );

		Frame frame;
		frame.m_time		= m_time;
		frame.m_translation = { m_matrix[0][3], m_matrix[1][3], m_matrix[2][3] };
		frame.m_rotation	= qtFromRotationMatrix( Q );
		frame.m_scale		= { R[0][0], R[1][1], R[2][2] };
		frame.m_shear		= { R[0][1], R[0][2], R[1][2] };
		return frame;
	}

	static HIPRT_HOST_DEVICE MatrixFrame getMatrixFrame( const Frame& frame )
	{
		MatrixFrame matrixFrame{};
		matrixFrame.m_time = frame.m_time;

		if ( frame.identity() )
		{
			matrixFrame.m_matrix[0][0] = 1.0f;
			matrixFrame.m_matrix[1][1] = 1.0f;
			matrixFrame.m_matrix[2][2] = 1.0f;
			return matrixFrame;
		}

		float Q[3][3];
		qtToRotationMatrix( frame.m_rotation, Q );

		float R[3][3];
		R[0][0] = frame.m_scale.x;
		R[1][1] = frame.m_scale.y;
		R[2][2] = frame.m_scale.z;
		R[0][1] = frame.m_shear.x;
		R[0][2] = frame.m_shear.y;
		R[1][2] = frame.m_shear.z;
		R[1][0] = 0.0f;
		R[2][0] = 0.0f;
		R[2][1] = 0.0f;

#ifdef __KERNECC__
#pragma unroll
#endif
		for ( uint32_t i = 0; i < 3; ++i )
#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t j = 0; j < 3; ++j )
#ifdef __KERNECC__
#pragma unroll
#endif
				for ( uint32_t k = 0; k < 3; ++k )
					matrixFrame.m_matrix[i][j] += Q[i][k] * R[k][j];

		matrixFrame.m_matrix[0][3] = frame.m_translation.x;
		matrixFrame.m_matrix[1][3] = frame.m_translation.y;
		matrixFrame.m_matrix[2][3] = frame.m_translation.z;

		return matrixFrame;
	}

	static HIPRT_HOST_DEVICE MatrixFrame getMatrixFrameInv( const Frame& frame )
	{
		MatrixFrame matrixFrame{};
		matrixFrame.m_time = frame.m_time;

		if ( frame.identity() )
		{
			matrixFrame.m_matrix[0][0] = 1.0f;
			matrixFrame.m_matrix[1][1] = 1.0f;
			matrixFrame.m_matrix[2][2] = 1.0f;
			return matrixFrame;
		}

		float Q[3][3];
		qtToRotationMatrix( frame.m_rotation, Q );

		float Ri[3][3];
		Ri[0][0] = 1.0f / frame.m_scale.x;
		Ri[1][1] = 1.0f / frame.m_scale.y;
		Ri[2][2] = 1.0f / frame.m_scale.z;
		Ri[0][1] = -frame.m_shear.x / ( frame.m_scale.x * frame.m_scale.y );
		Ri[0][2] = ( frame.m_shear.x * frame.m_shear.z - frame.m_shear.y * frame.m_scale.y ) /
				   ( frame.m_scale.x * frame.m_scale.y * frame.m_scale.z );
		Ri[1][2] = -frame.m_shear.z / ( frame.m_scale.y * frame.m_scale.z );
		Ri[1][0] = 0.0f;
		Ri[2][0] = 0.0f;
		Ri[2][1] = 0.0f;

#ifdef __KERNECC__
#pragma unroll
#endif
		for ( uint32_t i = 0; i < 3; ++i )
#ifdef __KERNECC__
#pragma unroll
#endif
			for ( uint32_t j = 0; j < 3; ++j )
#ifdef __KERNECC__
#pragma unroll
#endif
				for ( uint32_t k = 0; k < 3; ++k )
					matrixFrame.m_matrix[i][j] += Ri[i][k] * Q[j][k];

		matrixFrame.m_matrix[0][3] =
			-( matrixFrame.m_matrix[0][0] * frame.m_translation.x + matrixFrame.m_matrix[0][1] * frame.m_translation.y +
			   matrixFrame.m_matrix[0][2] * frame.m_translation.z );
		matrixFrame.m_matrix[1][3] =
			-( matrixFrame.m_matrix[1][0] * frame.m_translation.x + matrixFrame.m_matrix[1][1] * frame.m_translation.y +
			   matrixFrame.m_matrix[1][2] * frame.m_translation.z );
		matrixFrame.m_matrix[2][3] =
			-( matrixFrame.m_matrix[2][0] * frame.m_translation.x + matrixFrame.m_matrix[2][1] * frame.m_translation.y +
			   matrixFrame.m_matrix[2][2] * frame.m_translation.z );

		return matrixFrame;
	}

	static HIPRT_HOST_DEVICE MatrixFrame multiply( const MatrixFrame& matrix0, const MatrixFrame& matrix1 )
	{
		MatrixFrame matrix{};
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
#ifdef __KERNECC__
#pragma unroll
#endif
				for ( uint32_t k = 0; k < 4; ++k )
				{
					float m0 = matrix0.m_matrix[i][k];
					float m1 = j < 3 ? 0.0f : 1.0f;
					if ( k < 3 ) m1 = matrix1.m_matrix[k][j];
					matrix.m_matrix[i][j] += m0 * m1;
				}
			}
		}
		return matrix;
	}
};
HIPRT_STATIC_ASSERT( sizeof( MatrixFrame ) == 64 );

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

		float t = ( time - f0.m_time ) / ( f1.m_time - f0.m_time );

		Frame f;
		f.m_scale		= mix( f0.m_scale, f1.m_scale, t );
		f.m_shear		= mix( f0.m_shear, f1.m_shear, t );
		f.m_translation = mix( f0.m_translation, f1.m_translation, t );
		f.m_rotation	= qtMix( f0.m_rotation, f1.m_rotation, t );

		return f;
	}

	HIPRT_HOST_DEVICE hiprtRay transformRay( const hiprtRay& ray, float time ) const
	{
		hiprtRay outRay;
		Frame	 frame = interpolateFrames( time );
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
		Frame frame = interpolateFrames( time );
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
				Frame f;
				f.m_scale		= mix( f0.m_scale, f1.m_scale, t );
				f.m_shear		= mix( f0.m_shear, f1.m_shear, t );
				f.m_translation = mix( f0.m_translation, f1.m_translation, t );
				f.m_rotation	= qtMix( f0.m_rotation, f1.m_rotation, t );
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
		float3 p0 = aabb.m_min;
		float3 p1 = { aabb.m_min.x, aabb.m_min.y, aabb.m_max.z };
		float3 p2 = { aabb.m_min.x, aabb.m_max.y, aabb.m_min.z };
		float3 p3 = { aabb.m_min.x, aabb.m_max.y, aabb.m_max.z };
		float3 p4 = { aabb.m_max.x, aabb.m_min.y, aabb.m_max.z };
		float3 p5 = { aabb.m_max.x, aabb.m_max.y, aabb.m_min.z };
		float3 p6 = { aabb.m_max.x, aabb.m_max.y, aabb.m_max.z };
		float3 p7 = aabb.m_max;

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
