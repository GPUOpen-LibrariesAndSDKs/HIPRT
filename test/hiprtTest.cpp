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

#include <test/hiprtTest.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <contrib/stbi/stbi_image_write.h>
#define STB_IMAGE_IMPLEMENTATION
#include <contrib/stbi/stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include "common/tiny_obj_loader.h"
#include "common/allocator.h"
#include "common/bvhbuilder.h"
#include <chrono>
#include <thread>
#include <algorithm>

CmdArguments g_parsedArgs;

void checkOro( oroError res, const std::source_location& location )
{
	if ( res != oroSuccess )
	{
		const char* msg;
		oroGetErrorString( res, &msg );
		std::cerr << "Orochi error: '" << msg << "' on line " << location.line() << " "
				  << " in '" << location.file_name() << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkOrortc( orortcResult res, const std::source_location& location )
{
	if ( res != ORORTC_SUCCESS )
	{
		std::cerr << "Orortc error: '" << orortcGetErrorString( res ) << "' [ " << res << " ] on line " << location.line()
				  << " "
				  << " in '" << location.file_name() << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

void checkHiprt( hiprtError res, const std::source_location& location )
{
	if ( res != hiprtSuccess )
	{
		std::cerr << "Hiprt error: '" << res << "' on line " << location.line() << " "
				  << " in '" << location.file_name() << "'." << std::endl;
		exit( EXIT_FAILURE );
	}
}

inline float3 operator-( float3& a, float3& b ) { return make_float3( a.x - b.x, a.y - b.y, a.z - b.z ); }

struct BuilderContext
{
	PoolAllocator<hiprtBvhNode, 16> m_nodeAllocator;
	PoolAllocator<uint32_t, 16>		m_leafAllocator;
	void*							m_geomData;
};

struct GeometryData
{
	const float3*	m_vertices;
	const uint32_t* m_indices;
	const int2*		m_pairIndices = nullptr;
};

void hiprtTest::buildBvh( hiprtGeometryBuildInput& buildInput )
{
	std::vector<hiprtBvhNode> nodes;
	if ( buildInput.type == hiprtPrimitiveTypeTriangleMesh )
	{
		std::vector<Aabb>	 primBoxes( buildInput.primitive.triangleMesh.triangleCount );
		std::vector<uint8_t> verticesRaw(
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride );
		std::vector<uint8_t> trianglesRaw(
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride );
		copyDtoH(
			verticesRaw.data(),
			reinterpret_cast<uint8_t*>( buildInput.primitive.triangleMesh.vertices ),
			buildInput.primitive.triangleMesh.vertexCount * buildInput.primitive.triangleMesh.vertexStride );
		copyDtoH(
			trianglesRaw.data(),
			reinterpret_cast<uint8_t*>( buildInput.primitive.triangleMesh.triangleIndices ),
			buildInput.primitive.triangleMesh.triangleCount * buildInput.primitive.triangleMesh.triangleStride );
		for ( uint32_t i = 0; i < buildInput.primitive.triangleMesh.triangleCount; ++i )
		{
			int3 triangle =
				*reinterpret_cast<int3*>( trianglesRaw.data() + i * buildInput.primitive.triangleMesh.triangleStride );
			float3 v0 = *reinterpret_cast<const float3*>(
				verticesRaw.data() + triangle.x * buildInput.primitive.triangleMesh.vertexStride );
			float3 v1 = *reinterpret_cast<const float3*>(
				verticesRaw.data() + triangle.y * buildInput.primitive.triangleMesh.vertexStride );
			float3 v2 = *reinterpret_cast<const float3*>(
				verticesRaw.data() + triangle.z * buildInput.primitive.triangleMesh.vertexStride );
			primBoxes[i].reset();
			primBoxes[i].grow( v0 );
			primBoxes[i].grow( v1 );
			primBoxes[i].grow( v2 );
		}
		BvhBuilder::build( buildInput.primitive.triangleMesh.triangleCount, primBoxes, nodes );
	}
	else if ( buildInput.type == hiprtPrimitiveTypeAABBList )
	{
		std::vector<Aabb>	 primBoxes( buildInput.primitive.aabbList.aabbCount );
		std::vector<uint8_t> primBoxesRaw( buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride );
		copyDtoH(
			primBoxesRaw.data(),
			reinterpret_cast<uint8_t*>( buildInput.primitive.aabbList.aabbs ),
			buildInput.primitive.aabbList.aabbCount * buildInput.primitive.aabbList.aabbStride );
		for ( uint32_t i = 0; i < buildInput.primitive.aabbList.aabbCount; ++i )
		{
			float4* ptr = reinterpret_cast<float4*>( primBoxesRaw.data() + i * buildInput.primitive.aabbList.aabbStride );
			primBoxes[i].m_min = make_float3( ptr[0] );
			primBoxes[i].m_max = make_float3( ptr[1] );
		}
		BvhBuilder::build( buildInput.primitive.aabbList.aabbCount, primBoxes, nodes );
	}
	malloc( reinterpret_cast<hiprtBvhNode*&>( buildInput.nodeList.nodes ), nodes.size() );
	copyHtoD( reinterpret_cast<hiprtBvhNode*>( buildInput.nodeList.nodes ), nodes.data(), nodes.size() );
	buildInput.nodeList.nodeCount = static_cast<uint32_t>( nodes.size() );
}

void hiprtTest::buildEmbreeBvh(
	RTCDevice embreeDevice, std::vector<RTCBuildPrimitive>& embreePrims, std::vector<hiprtBvhNode>& nodes, void* geomData )
{
	const float Alpha = 1.5f;
	enum
	{
		LeafFlag = 1 << 30
	};

	size_t primCount	= embreePrims.size();
	size_t primCapacity = Alpha * embreePrims.size();
	embreePrims.resize( primCapacity );

	BuilderContext context;
	context.m_geomData = geomData;

	RTCBVH			  embreeBvh		  = rtcNewBVH( embreeDevice );
	RTCBuildArguments embreeArgs	  = rtcDefaultBuildArguments();
	embreeArgs.byteSize				  = sizeof( embreeArgs );
	embreeArgs.buildQuality			  = RTC_BUILD_QUALITY_HIGH;
	embreeArgs.maxBranchingFactor	  = 4;
	embreeArgs.bvh					  = embreeBvh;
	embreeArgs.primitives			  = embreePrims.data();
	embreeArgs.primitiveCount		  = primCount;
	embreeArgs.primitiveArrayCapacity = primCapacity;
	embreeArgs.minLeafSize			  = 1;
	embreeArgs.maxLeafSize			  = 1;
	embreeArgs.splitPrimitive		  = nullptr;
	embreeArgs.userPtr				  = &context;

	embreeArgs.createNode = []( RTCThreadLocalAllocator allocator, uint32_t childCount, void* userPtr ) -> void* {
		BuilderContext* ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		uint32_t		handle;
		hiprtBvhNode*	ptr;
		ctxt->m_nodeAllocator.allocate( &handle, &ptr );
		return reinterpret_cast<void*>( static_cast<uintptr_t>( handle ) );
	};

	embreeArgs.setNodeChildren = []( void* nodePtr, void** children, uint32_t childCount, void* userPtr ) {
		BuilderContext* ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		hiprtBvhNode*	node = ctxt->m_nodeAllocator.item( static_cast<uint32_t>( reinterpret_cast<uintptr_t>( nodePtr ) ) );
		for ( uint32_t i = 0; i < childCount; i++ )
		{
			node->childIndices[i]	= static_cast<uint32_t>( reinterpret_cast<uintptr_t>( children[i] ) );
			node->childNodeTypes[i] = node->childIndices[i] & LeafFlag ? hiprtBvhNodeTypeLeaf : hiprtBvhNodeTypeInternal;
		}
		for ( uint32_t i = childCount; i < 4; i++ )
			node->childIndices[i] = hiprtInvalidValue;
	};

	embreeArgs.setNodeBounds = []( void* nodePtr, const struct RTCBounds** bounds, uint32_t childCount, void* userPtr ) {
		BuilderContext* ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		hiprtBvhNode*	node = ctxt->m_nodeAllocator.item( static_cast<uint32_t>( reinterpret_cast<uintptr_t>( nodePtr ) ) );
		for ( uint32_t i = 0; i < childCount; i++ )
		{
			node->childAabbsMin[i] = make_float3( bounds[i]->lower_x, bounds[i]->lower_y, bounds[i]->lower_z );
			node->childAabbsMax[i] = make_float3( bounds[i]->upper_x, bounds[i]->upper_y, bounds[i]->upper_z );
		}
	};

	embreeArgs.createLeaf = []( RTCThreadLocalAllocator			allocator,
								const struct RTCBuildPrimitive* primitives,
								size_t							primitiveCount,
								void*							userPtr ) -> void* {
		BuilderContext* ctxt = reinterpret_cast<BuilderContext*>( userPtr );
		uint32_t		handle;
		uint32_t*		ptr;
		ctxt->m_leafAllocator.allocate( &handle, &ptr );
		*ptr = primitives->primID;
		return reinterpret_cast<void*>( static_cast<uintptr_t>( handle | LeafFlag ) );
	};

	if ( geomData == nullptr )
	{
		embreeArgs.splitPrimitive = []( const struct RTCBuildPrimitive* primitive,
										uint32_t						dimension,
										float							position,
										struct RTCBounds*				leftBounds,
										struct RTCBounds*				rightBounds,
										void*							userPtr ) {
			leftBounds->lower_x = rightBounds->lower_x = primitive->lower_x;
			leftBounds->lower_y = rightBounds->lower_y = primitive->lower_y;
			leftBounds->lower_z = rightBounds->lower_z = primitive->lower_z;
			leftBounds->upper_x = rightBounds->upper_x = primitive->upper_x;
			leftBounds->upper_y = rightBounds->upper_y = primitive->upper_y;
			leftBounds->upper_z = rightBounds->upper_z = primitive->upper_z;
			( &leftBounds->upper_x )[dimension]		   = position;
			( &rightBounds->lower_x )[dimension]	   = position;
		};
	}
	else
	{
		embreeArgs.splitPrimitive = []( const struct RTCBuildPrimitive* primitive,
										uint32_t						dimension,
										float							position,
										struct RTCBounds*				leftBounds,
										struct RTCBounds*				rightBounds,
										void*							userPtr ) {
			BuilderContext* ctxt	 = reinterpret_cast<BuilderContext*>( userPtr );
			GeometryData*	geomData = reinterpret_cast<GeometryData*>( ctxt->m_geomData );

			auto splitTriangle =
				[]( float3( &vertices )[3], uint32_t axis, float position, const Aabb& box, Aabb& leftBox, Aabb& rightBox ) {
					const float3* v1 = &vertices[2];
					for ( uint32_t i = 0; i < 3; i++ )
					{
						const float3* v0 = v1;
						v1				 = &vertices[i];
						float v0p		 = ( &v0->x )[axis];
						float v1p		 = ( &v1->x )[axis];

						if ( v0p <= position ) leftBox.grow( *v0 );
						if ( v0p >= position ) rightBox.grow( *v0 );

						if ( ( v0p < position && v1p > position ) || ( v0p > position && v1p < position ) )
						{
							float3 t = hiprt::mix( *v0, *v1, fmaxf( fminf( ( position - v0p ) / ( v1p - v0p ), 1.0f ), 0.0f ) );
							leftBox.grow( t );
							rightBox.grow( t );
						}
					}

					( &leftBox.m_max.x )[axis]	= position;
					( &rightBox.m_min.x )[axis] = position;
					leftBox.intersect( box );
					rightBox.intersect( box );
				};

			int2 primID = make_int2( primitive->primID );
			if ( geomData->m_pairIndices != nullptr ) primID = geomData->m_pairIndices[primitive->primID];

			Aabb box, leftBox, rightBox;
			box.grow( make_float3( primitive->lower_x, primitive->lower_y, primitive->lower_z ) );
			box.grow( make_float3( primitive->upper_x, primitive->upper_y, primitive->upper_z ) );

			float3 vertices[3];
			vertices[0] = geomData->m_vertices[geomData->m_indices[3 * primID.x + 0]];
			vertices[1] = geomData->m_vertices[geomData->m_indices[3 * primID.x + 1]];
			vertices[2] = geomData->m_vertices[geomData->m_indices[3 * primID.x + 2]];
			splitTriangle( vertices, dimension, position, box, leftBox, rightBox );

			if ( primID.x != primID.y )
			{
				Aabb secLeftBox, secRightBox;
				vertices[0] = geomData->m_vertices[geomData->m_indices[3 * primID.y + 0]];
				vertices[1] = geomData->m_vertices[geomData->m_indices[3 * primID.y + 1]];
				vertices[2] = geomData->m_vertices[geomData->m_indices[3 * primID.y + 2]];
				splitTriangle( vertices, dimension, position, box, secLeftBox, secRightBox );
				leftBox.grow( secLeftBox );
				rightBox.grow( secRightBox );
			}

			leftBounds->lower_x = leftBox.m_min.x;
			leftBounds->lower_y = leftBox.m_min.y;
			leftBounds->lower_z = leftBox.m_min.z;
			leftBounds->upper_x = leftBox.m_max.x;
			leftBounds->upper_y = leftBox.m_max.y;
			leftBounds->upper_z = leftBox.m_max.z;

			rightBounds->lower_x = rightBox.m_min.x;
			rightBounds->lower_y = rightBox.m_min.y;
			rightBounds->lower_z = rightBox.m_min.z;
			rightBounds->upper_x = rightBox.m_max.x;
			rightBounds->upper_y = rightBox.m_max.y;
			rightBounds->upper_z = rightBox.m_max.z;
		};
	}

	rtcBuildBVH( &embreeArgs );
	uint32_t nodeCount = context.m_nodeAllocator.count();
	nodes.resize( nodeCount );

	for ( uint32_t i = 0; i < nodeCount; ++i )
	{
		hiprtBvhNode* node = context.m_nodeAllocator.item( i );
		for ( uint32_t j = 0; j < 4; j++ )
		{
			if ( node->childIndices[j] == hiprtInvalidValue ) continue;
			uint32_t childIndex = node->childIndices[j];
			if ( childIndex & LeafFlag )
			{
				uint32_t* primID	  = context.m_leafAllocator.item( childIndex & ( ~LeafFlag ) );
				node->childIndices[j] = *primID;
			}
		}
		nodes[i] = *node;
	}

	rtcReleaseBVH( embreeBvh );
}

void hiprtTest::buildEmbreeGeometryBvh(
	RTCDevice embreeDevice, const float3* vertices, const uint32_t* indices, hiprtGeometryBuildInput& buildInput )
{
	uint32_t triangleCount = buildInput.primitive.triangleMesh.triangleCount;

	GeometryData geomData;
	geomData.m_vertices = vertices;
	geomData.m_indices	= indices;

	std::vector<hiprtBvhNode> nodes;

	if ( triangleCount > 2 )
	{
		auto tryPairTriangles = [&]( const int3& a, const int3& b ) {
			int3 lb = make_int3( 3 );

			lb.x = ( b.x == a.x ) ? 0 : lb.x;
			lb.y = ( b.y == a.x ) ? 0 : lb.y;
			lb.z = ( b.z == a.x ) ? 0 : lb.z;

			lb.x = ( b.x == a.y ) ? 1 : lb.x;
			lb.y = ( b.y == a.y ) ? 1 : lb.y;
			lb.z = ( b.z == a.y ) ? 1 : lb.z;

			lb.x = ( b.x == a.z ) ? 2 : lb.x;
			lb.y = ( b.y == a.z ) ? 2 : lb.y;
			lb.z = ( b.z == a.z ) ? 2 : lb.z;

			if ( ( lb.x == 3 ) + ( lb.y == 3 ) + ( lb.z == 3 ) <= 1 ) return lb;
			return make_int3( hiprt::InvalidValue );
		};

		std::vector<int2> pairIndices;
		uint32_t		  groups = hiprt::DivideRoundUp( triangleCount, 32 );
		for ( uint32_t i = 0; i < groups; ++i )
		{
			const uint32_t	  offset = i * 32;
			std::vector<bool> active( 32 );
			for ( uint32_t j = 0; j < 32; ++j )
				active[j] = offset + j < triangleCount;

			for ( uint32_t j = 0; j < 32; ++j )
			{
				if ( !active[j] ) continue;
				int2 pair		= make_int2( offset + j );
				int3 triIndices = *reinterpret_cast<const int3*>( &indices[3 * ( offset + j )] );
				for ( uint32_t k = j + 1; k < 32; ++k )
				{
					if ( !active[k] ) continue;
					int3 secondTriIndices = *reinterpret_cast<const int3*>( &indices[3 * ( offset + k )] );
					bool pairable		  = tryPairTriangles( secondTriIndices, triIndices ).x != hiprt::InvalidValue;
					if ( pairable )
					{
						pair.y	  = offset + k;
						active[k] = false;
						break;
					}
				}
				pairIndices.push_back( pair );
				active[j] = false;
			}
		}

		buildInput.primitive.triangleMesh.trianglePairCount = static_cast<uint32_t>( pairIndices.size() );
		malloc( reinterpret_cast<int2*&>( buildInput.primitive.triangleMesh.trianglePairIndices ), pairIndices.size() );
		copyHtoD(
			reinterpret_cast<int2*>( buildInput.primitive.triangleMesh.trianglePairIndices ),
			pairIndices.data(),
			pairIndices.size() );

		std::vector<RTCBuildPrimitive> embreePrims( pairIndices.size() );
		for ( size_t i = 0; i < pairIndices.size(); ++i )
		{
			Aabb box;
			for ( uint32_t j = 0; j < 2; ++j )
			{
				int3 triIndices = *reinterpret_cast<const int3*>( &indices[3 * ( &pairIndices[i].x )[j]] );
				box.grow( vertices[triIndices.x] );
				box.grow( vertices[triIndices.y] );
				box.grow( vertices[triIndices.z] );

				embreePrims[i].primID  = static_cast<uint32_t>( i );
				embreePrims[i].lower_x = box.m_min.x;
				embreePrims[i].lower_y = box.m_min.y;
				embreePrims[i].lower_z = box.m_min.z;
				embreePrims[i].upper_x = box.m_max.x;
				embreePrims[i].upper_y = box.m_max.y;
				embreePrims[i].upper_z = box.m_max.z;
			}
		}

		geomData.m_pairIndices = pairIndices.data();
		buildEmbreeBvh( embreeDevice, embreePrims, nodes, &geomData );
	}
	else
	{
		std::vector<RTCBuildPrimitive> embreePrims( triangleCount );
		for ( uint32_t i = 0; i < triangleCount; ++i )
		{
			Aabb box;
			box.grow( vertices[indices[3 * i + 0]] );
			box.grow( vertices[indices[3 * i + 1]] );
			box.grow( vertices[indices[3 * i + 2]] );

			embreePrims[i].primID  = i;
			embreePrims[i].lower_x = box.m_min.x;
			embreePrims[i].lower_y = box.m_min.y;
			embreePrims[i].lower_z = box.m_min.z;
			embreePrims[i].upper_x = box.m_max.x;
			embreePrims[i].upper_y = box.m_max.y;
			embreePrims[i].upper_z = box.m_max.z;
		}

		if ( embreePrims.size() == 1 )
		{
			hiprtBvhNode node;
			node.childAabbsMin[0]  = { embreePrims.back().lower_x, embreePrims.back().lower_y, embreePrims.back().lower_z };
			node.childAabbsMax[0]  = { embreePrims.back().upper_x, embreePrims.back().upper_y, embreePrims.back().upper_z };
			node.childIndices[0]   = 0;
			node.childIndices[1]   = hiprt::InvalidValue;
			node.childIndices[2]   = hiprt::InvalidValue;
			node.childIndices[3]   = hiprt::InvalidValue;
			node.childNodeTypes[0] = hiprtBvhNodeTypeLeaf;
			nodes.push_back( node );
		}
		else
		{
			buildEmbreeBvh( embreeDevice, embreePrims, nodes, &geomData );
		}
	}

	malloc( reinterpret_cast<hiprtBvhNode*&>( buildInput.nodeList.nodes ), nodes.size() );
	copyHtoD( reinterpret_cast<hiprtBvhNode*>( buildInput.nodeList.nodes ), nodes.data(), nodes.size() );
	buildInput.nodeList.nodeCount = static_cast<uint32_t>( nodes.size() );
}

void hiprtTest::buildEmbreeSceneBvh(
	RTCDevice						  embreeDevice,
	const std::vector<Aabb>&		  geomBoxes,
	const std::vector<hiprtFrameSRT>& frames,
	hiprtSceneBuildInput&			  buildInput )
{
	uint32_t instanceCount = buildInput.instanceCount;

	struct BuilderContext
	{
		PoolAllocator<hiprtBvhNode, 16> nodeAllocator;
		PoolAllocator<uint32_t, 16>		leafAllocator;
	} context;

	std::vector<RTCBuildPrimitive> embreePrims( instanceCount );
	for ( uint32_t i = 0; i < instanceCount; ++i )
	{
		const Aabb&			 geomBox = geomBoxes[i];
		const hiprtFrameSRT& f		 = frames[i];

		float3 p[8];
		p[0] = geomBox.m_min;
		p[1] = make_float3( geomBox.m_min.x, geomBox.m_min.y, geomBox.m_max.z );
		p[2] = make_float3( geomBox.m_min.x, geomBox.m_max.y, geomBox.m_min.z );
		p[3] = make_float3( geomBox.m_min.x, geomBox.m_max.y, geomBox.m_max.z );
		p[4] = make_float3( geomBox.m_max.x, geomBox.m_min.y, geomBox.m_max.z );
		p[5] = make_float3( geomBox.m_max.x, geomBox.m_max.y, geomBox.m_min.z );
		p[6] = make_float3( geomBox.m_max.x, geomBox.m_max.y, geomBox.m_max.z );
		p[7] = geomBox.m_max;

		Aabb box;
		for ( uint32_t i = 0; i < 8; ++i )
		{
			p[i] *= f.scale;
			p[i] = rotate( f.rotation, p[i] );
			p[i] += f.translation;
			box.grow( p[i] );
		}

		embreePrims[i].primID  = i;
		embreePrims[i].lower_x = box.m_min.x;
		embreePrims[i].lower_y = box.m_min.y;
		embreePrims[i].lower_z = box.m_min.z;
		embreePrims[i].upper_x = box.m_max.x;
		embreePrims[i].upper_y = box.m_max.y;
		embreePrims[i].upper_z = box.m_max.z;
	}

	std::vector<hiprtBvhNode> nodes;
	if ( embreePrims.size() == 1 )
	{
		hiprtBvhNode node;
		node.childAabbsMin[0]  = { embreePrims.back().lower_x, embreePrims.back().lower_y, embreePrims.back().lower_z };
		node.childAabbsMax[0]  = { embreePrims.back().upper_x, embreePrims.back().upper_y, embreePrims.back().upper_z };
		node.childIndices[0]   = 0;
		node.childIndices[1]   = hiprt::InvalidValue;
		node.childIndices[2]   = hiprt::InvalidValue;
		node.childIndices[3]   = hiprt::InvalidValue;
		node.childNodeTypes[0] = hiprtBvhNodeTypeLeaf;
		nodes.push_back( node );
	}
	else
	{
		buildEmbreeBvh( embreeDevice, embreePrims, nodes, nullptr );
	}

	malloc( reinterpret_cast<hiprtBvhNode*&>( buildInput.nodeList.nodes ), nodes.size() );
	copyHtoD( reinterpret_cast<hiprtBvhNode*>( buildInput.nodeList.nodes ), nodes.data(), nodes.size() );
	buildInput.nodeList.nodeCount = static_cast<uint32_t>( nodes.size() );
}

bool hiprtTest::readSourceCode(
	const std::filesystem::path& srcPath, std::string& sourceCode, std::optional<std::vector<std::filesystem::path>> includes )
{
	std::fstream f( srcPath );
	if ( f.is_open() )
	{
		size_t sizeFile;
		f.seekg( 0, std::fstream::end );
		size_t size = sizeFile = static_cast<size_t>( f.tellg() );
		f.seekg( 0, std::fstream::beg );
		if ( includes )
		{
			sourceCode.clear();
			std::string line;
			while ( std::getline( f, line ) )
			{
				if ( line.find( "#include" ) != std::string::npos )
				{
					size_t		pa	= line.find( "<" );
					size_t		pb	= line.find( ">" );
					std::string buf = line.substr( pa + 1, pb - pa - 1 );
					includes.value().push_back( buf );
					sourceCode += line + '\n';
				}
				sourceCode += line + '\n';
			}
		}
		else
		{
			sourceCode.resize( size, ' ' );
			f.read( &sourceCode[0], size );
		}
		f.close();
	}
	else
		return false;
	return true;
}

hiprtError hiprtTest::buildTraceKernelsFromBitcode(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	std::vector<const char*>					 functionNames,
	std::vector<hiprtApiFunction>&				 functionsOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::vector<const char*> options;

	size_t							   binarySize = 0;
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;

	if ( !readSourceCode( srcPath, sourceCode, includeNamesData ) )
	{
		std::cerr << "Unable to open '" << srcPath << "'" << std::endl;
		return hiprtErrorInternal;
	}

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( size_t i = 0; i < includeNamesData.size(); i++ )
	{
		if ( !readSourceCode( std::filesystem::path( "../" ) / includeNamesData[i], headersData[i] ) )
		{
			std::cerr << "Unable to find header '" << includeNamesData[i] << "' at ../" << std::endl;
			return hiprtErrorInternal;
		}
		includeNames.push_back( includeNamesData[i].string().c_str() );
		headers.push_back( headersData[i].c_str() );
	}

	if ( opts )
	{
		for ( const auto o : *opts )
		{
			options.push_back( o );
		}
	}

	const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
	if ( isAmd )
	{
		options.push_back( "-fgpu-rdc" );
		options.push_back( "-Xclang" );
		options.push_back( "-disable-llvm-passes" );
		options.push_back( "-Xclang" );
		options.push_back( "-mno-constructor-aliases" );
	}
	else
	{
		options.push_back( "--device-c" );
		options.push_back( "-arch=compute_60" );
	}
	options.push_back( "-std=c++17" );
	options.push_back( "-I../" );

	orortcProgram prog;
	checkOrortc( orortcCreateProgram(
		&prog,
		sourceCode.data(),
		srcPath.string().c_str(),
		static_cast<int>( headers.size() ),
		headers.data(),
		includeNames.data() ) );

	for ( auto functionName : functionNames )
	{
		checkOrortc( orortcAddNameExpression( prog, functionName ) );
	}
	orortcResult e = orortcCompileProgram( prog, static_cast<int>( options.size() ), options.data() );

	if ( e != ORORTC_SUCCESS )
	{
		size_t logSize;
		checkOrortc( orortcGetProgramLogSize( prog, &logSize ) );

		if ( logSize )
		{
			std::string log( logSize, '\0' );
			checkOrortc( orortcGetProgramLog( prog, &log[0] ) );
			std::cerr << log << std::endl;
			return hiprtErrorInternal;
		}
	}

	std::string bitcodeBinary;
	size_t		size = 0;
	if ( isAmd )
		checkOrortc( orortcGetBitcodeSize( prog, &size ) );
	else
		checkOrortc( orortcGetCodeSize( prog, &size ) );
	bitcodeBinary.resize( size );
	if ( isAmd )
		checkOrortc( orortcGetBitcode( prog, bitcodeBinary.data() ) );
	else
		checkOrortc( orortcGetCode( prog, bitcodeBinary.data() ) );

	functionsOut.resize( functionNames.size() );
	hiprtError error = hiprtBuildTraceKernelsFromBitcode(
		ctxt,
		static_cast<uint32_t>( functionNames.size() ),
		functionNames.data(),
		srcPath.string().c_str(),
		bitcodeBinary.data(),
		size,
		numGeomTypes,
		numRayTypes,
		funcNameSets ? funcNameSets.value().data() : nullptr,
		functionsOut.data(),
		true );

	return error;
}

hiprtError hiprtTest::buildTraceKernelFromBitcode(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	const std::string&							 functionName,
	oroFunction&								 functionOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	bool						  usePrecompiledBitcode = g_parsedArgs.m_usePrecompiledBitcodes;
	std::vector<hiprtApiFunction> functions;
	hiprtError					  e = hiprtSuccess;

	std::vector<const char*> options;
	if ( opts ) options = *opts;

	std::string bitcodeLinkingDef = "-DHIPRT_BITCODE_LINKING";
	options.push_back( bitcodeLinkingDef.c_str() );

	if ( !usePrecompiledBitcode )
	{
		e = buildTraceKernelsFromBitcode(
			ctxt, srcPath, { functionName.c_str() }, functions, options, funcNameSets, numGeomTypes, numRayTypes );
		ASSERT( functions.size() == 1 );
		functionOut = *reinterpret_cast<oroFunction*>( &functions.back() );
	}
	else
	{
		const bool isAmd = oroGetCurAPI( 0 ) == ORO_API_HIP;
		HIPRT_ASSERT( isAmd == true ); // precompiled path supported only on AMD

		auto loadFile = []( const std::filesystem::path& path, std::vector<uint8_t>& dst ) {
			std::fstream f( path, std::ios::binary | std::ios::in );
			if ( f.is_open() )
			{
				size_t sizeFile;
				f.seekg( 0, std::fstream::end );
				size_t size = sizeFile = static_cast<size_t>( f.tellg() );
				dst.resize( size );
				f.seekg( 0, std::fstream::beg );
				f.read( reinterpret_cast<char*>( dst.data() ), size );
				f.close();
			}
		};

		std::filesystem::path path =
			std::string( "../dist/bin/Release/hiprt" ) + HIPRT_VERSION_STR + "_" + HIP_VERSION_STR + "_";
#if !defined( __GNUC__ )
		path += "precompiled_bitcode_win.hipfb";
#else
		path += "precompiled_bitcode_linux.hipfb";
#endif
		std::vector<uint8_t> binary;
		loadFile( path, binary );

		oroModule module;
		oroError  res = oroModuleLoadData( &module, binary.data() );
		if ( res != oroSuccess )
		{
			// add some verbose to help debugging missing file.
			std::cout << "oroModuleLoadData FAILED (error=" << res << ") loading file: " << path.string().c_str() << std::endl;
		}
		checkOro( res );
		checkOro( oroModuleGetFunction( &functionOut, module, functionName.c_str() ) );
	}
	return e;
}

hiprtError hiprtTest::buildTraceKernels(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	std::vector<const char*>					 functionNames,
	std::vector<hiprtApiFunction>&				 functionsOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::vector<std::filesystem::path> includeNamesData;
	std::string						   sourceCode;
	readSourceCode( srcPath, sourceCode, includeNamesData );

	std::vector<std::string> headersData( includeNamesData.size() );
	std::vector<const char*> headers;
	std::vector<const char*> includeNames;
	for ( size_t i = 0; i < includeNamesData.size(); i++ )
	{
		readSourceCode( std::filesystem::path( "../" ) / includeNamesData[i], headersData[i] );
		includeNames.push_back( includeNamesData[i].string().c_str() );
		headers.push_back( headersData[i].c_str() );
	}

	functionsOut.resize( functionNames.size() );
	return hiprtBuildTraceKernels(
		ctxt,
		static_cast<uint32_t>( functionNames.size() ),
		functionNames.data(),
		sourceCode.c_str(),
		srcPath.string().c_str(),
		static_cast<uint32_t>( headers.size() ),
		headers.data(),
		includeNames.data(),
		opts ? static_cast<uint32_t>( opts.value().size() ) : 0,
		opts ? opts.value().data() : nullptr,
		numGeomTypes,
		numRayTypes,
		funcNameSets ? funcNameSets.value().data() : nullptr,
		functionsOut.data(),
		nullptr,
		true );
}

hiprtError hiprtTest::buildTraceKernel(
	hiprtContext								 ctxt,
	const std::filesystem::path&				 srcPath,
	const std::string&							 functionName,
	oroFunction&								 functionOut,
	std::optional<std::vector<const char*>>		 opts,
	std::optional<std::vector<hiprtFuncNameSet>> funcNameSets,
	uint32_t									 numGeomTypes,
	uint32_t									 numRayTypes )
{
	std::vector<hiprtApiFunction> functions;
	hiprtError					  e =
		buildTraceKernels( ctxt, srcPath, { functionName.c_str() }, functions, opts, funcNameSets, numGeomTypes, numRayTypes );
	ASSERT( functions.size() == 1 );
	functionOut = *reinterpret_cast<oroFunction*>( &functions.back() );
	return e;
}

void hiprtTest::validateAndWriteImage(
	const std::filesystem::path&		 imgPath,
	uint32_t							 width,
	uint32_t							 height,
	uint8_t*							 data,
	std::optional<std::filesystem::path> refPath,
	std::optional<std::filesystem::path> refFilename )
{
	std::vector<uint8_t> image( width * height * 4 );
	copyDtoH( image.data(), data, width * height * 4 );
	writeImage( imgPath, width, height, image.data() );

	if ( refFilename && refPath )
	{
		int refW;
		int refH;
		int refB;

		std::filesystem::path fullRefFilename = refPath.value() / refFilename.value();
		uint8_t*			  ref			  = stbi_load( fullRefFilename.string().c_str(), &refW, &refH, &refB, 0 );
		if ( ref == 0 )
		{
			std::cerr << "Unable to open reference image '" << fullRefFilename << "'!" << std::endl;
			EXPECT_FALSE( 1 );
			return;
		}

		if ( refW != width || refH != height )
		{
			std::cerr << "Framebuffer resolution does not match!" << std::endl;
			EXPECT_FALSE( 1 );
			return;
		}

		uint32_t pixelThreshold = 10;
		uint32_t maxDiff		= 0;
		uint32_t nDiffPixels	= 0;

		for ( uint32_t i = 0; i < width * height; i++ )
		{
			uint32_t r = abs( image[i * 4 + 0] - ref[i * 4 + 0] );
			uint32_t g = abs( image[i * 4 + 1] - ref[i * 4 + 1] );
			uint32_t b = abs( image[i * 4 + 2] - ref[i * 4 + 2] );
			uint32_t a = abs( image[i * 4 + 3] - ref[i * 4 + 3] );

			if ( r > pixelThreshold || g > pixelThreshold || b > pixelThreshold || a > pixelThreshold )
			{
				maxDiff = std::max( maxDiff, std::max( r, std::max( g, b ) ) );
				nDiffPixels++;
			}
		}

		const float fail = 100.0f * nDiffPixels / ( static_cast<float>( width * height ) );
		if ( nDiffPixels != 0 )
			std::cerr << "Pixel difference: " << nDiffPixels << " (" << std::setprecision( 1 ) << fail
					  << "%)	(max diff: " << maxDiff << "/255)" << std::endl;

		if ( !( fail < 0.3f || maxDiff < 10 ) )
		{
			EXPECT_FALSE( 1 );
		}

		stbi_image_free( ref );
	}
}

void hiprtTest::writeImage( const std::filesystem::path& imgPath, uint32_t width, uint32_t height, uint8_t* data )
{
	stbi_write_png( imgPath.string().c_str(), width, height, 4, data, width * 4 );
}

void hiprtTest::launchKernel( oroFunction func, uint32_t nx, uint32_t ny, void** args, uint32_t sharedMemoryBytes )
{
	constexpr uint32_t tx  = 16u;
	constexpr uint32_t ty  = 16u;
	uint32_t		   nbx = hiprt::divideRoundUp( nx, tx );
	uint32_t		   nby = hiprt::divideRoundUp( ny, ty );
	checkOro( oroModuleLaunchKernel( func, nbx, nby, 1, tx, ty, 1, sharedMemoryBytes, 0, args, 0 ) );
}

void hiprtTest::launchKernel(
	oroFunction func, uint32_t nx, uint32_t ny, uint32_t tx, uint32_t ty, void** args, uint32_t sharedMemoryBytes )
{
	uint32_t nbx = hiprt::divideRoundUp( nx, tx );
	uint32_t nby = hiprt::divideRoundUp( ny, ty );
	checkOro( oroModuleLaunchKernel( func, nbx, nby, 1, tx, ty, 1, sharedMemoryBytes, 0, args, 0 ) );
}

void ObjTestCases::createScene(
	SceneData&					 scene,
	const std::string&			 filename,
	const std::string&			 mtlBaseDir,
	bool						 enableRayMask,
	std::optional<hiprtFrameSRT> frame,
	hiprtBuildFlags				 bvhBuildFlag,
	bool						 time )
{
	hiprtCreateContext( HIPRT_API_VERSION, m_ctxtInput, scene.m_ctx );

	tinyobj::attrib_t				 attrib;
	std::vector<tinyobj::shape_t>	 shapes;
	std::vector<tinyobj::material_t> materials;
	std::string						 err;
	std::string						 warning;

	bool ret = tinyobj::LoadObj( &attrib, &shapes, &materials, &warning, &err, filename.c_str(), mtlBaseDir.c_str() );

	if ( !warning.empty() )
	{
		std::cerr << "OBJ Loader WARN : " << warning << std::endl;
	}

	if ( !err.empty() )
	{
		std::cerr << "OBJ Loader ERROR : " << err << std::endl;
		std::exit( EXIT_FAILURE );
	}

	if ( !ret )
	{
		std::cerr << "Failed to load obj file" << std::endl;
		std::exit( EXIT_FAILURE );
	}

	if ( shapes.empty() )
	{
		std::cerr << "No shapes in obj file (run 'git lfs fetch' and 'git lfs pull' in 'test/common/meshes/lfs')" << std::endl;
		std::exit( EXIT_FAILURE );
	}

	std::vector<Material> shapeMaterials; // materials for all instances
	std::vector<Light>	  lights;
	std::vector<uint32_t> materialIndices; // material ids for all instances
	std::vector<uint32_t> instanceMask;
	std::vector<float3>	  allVertices;
	std::vector<float3>	  allNormals;
	std::vector<uint32_t> allIndices;
	std::vector<Aabb>	  geomBoxes;

	uint32_t numOfLights = 0;

	// Prefix sum to calculate the offsets in to global vert,index and material buffer
	uint32_t				 vertexPrefixSum = 0u;
	uint32_t				 normalPrefixSum = 0u;
	uint32_t				 indexPrefixSum	 = 0u;
	uint32_t				 matIdxPrefixSum = 0u;
	std::vector<uint32_t>	 indicesOffsets;
	std::vector<uint32_t>	 verticesOffsets;
	std::vector<uint32_t>	 normalsOffsets;
	std::vector<uint32_t>	 matIdxOffset;
	std::chrono::nanoseconds bvhBuildTime{};

	indicesOffsets.resize( shapes.size() );
	verticesOffsets.resize( shapes.size() );
	normalsOffsets.resize( shapes.size() );
	matIdxOffset.resize( shapes.size() );

	auto convert = []( const tinyobj::real_t c[3] ) -> float3 { return float3{ c[0], c[1], c[2] }; };

	for ( const auto& mat : materials )
	{
		Material m;
		m.m_diffuse	 = convert( mat.diffuse );
		m.m_emission = convert( mat.emission );
		shapeMaterials.push_back( m );
	}

	RTCDevice embreeDevice;
	if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
	{
		embreeDevice = rtcNewDevice( "" );
		rtcSetDeviceErrorFunction(
			embreeDevice,
			[]( void* userPtr, enum RTCError code, const char* str ) { std::cerr << str << std::endl; },
			nullptr );
	}

	auto compare = []( const tinyobj::index_t& a, const tinyobj::index_t& b ) {
		if ( a.vertex_index < b.vertex_index ) return true;
		if ( a.vertex_index > b.vertex_index ) return false;

		if ( a.normal_index < b.normal_index ) return true;
		if ( a.normal_index > b.normal_index ) return false;

		if ( a.texcoord_index < b.texcoord_index ) return true;
		if ( a.texcoord_index > b.texcoord_index ) return false;

		return false;
	};

	for ( size_t i = 0; i < shapes.size(); ++i )
	{
		std::vector<float3>										  vertices;
		std::vector<float3>										  normals;
		std::vector<uint32_t>									  indices;
		float3*													  v = reinterpret_cast<float3*>( attrib.vertices.data() );
		std::map<tinyobj::index_t, uint32_t, decltype( compare )> knownIndex( compare );
		Aabb													  geomBox;

		for ( size_t face = 0; face < shapes[i].mesh.num_face_vertices.size(); face++ )
		{
			tinyobj::index_t idx0 = shapes[i].mesh.indices[3 * face + 0];
			tinyobj::index_t idx1 = shapes[i].mesh.indices[3 * face + 1];
			tinyobj::index_t idx2 = shapes[i].mesh.indices[3 * face + 2];

			if ( knownIndex.find( idx0 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx0] );
			}
			else
			{
				knownIndex[idx0] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx0] );
				vertices.push_back( v[idx0.vertex_index] );
				normals.push_back( v[idx0.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( knownIndex.find( idx1 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx1] );
			}
			else
			{
				knownIndex[idx1] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx1] );
				vertices.push_back( v[idx1.vertex_index] );
				normals.push_back( v[idx1.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( knownIndex.find( idx2 ) != knownIndex.end() )
			{
				indices.push_back( knownIndex[idx2] );
			}
			else
			{
				knownIndex[idx2] = static_cast<uint32_t>( vertices.size() );
				indices.push_back( knownIndex[idx2] );
				vertices.push_back( v[idx2.vertex_index] );
				normals.push_back( v[idx2.normal_index] );
				geomBox.grow( vertices.back() );
			}

			if ( !shapeMaterials.empty() && shapeMaterials[shapes[i].mesh.material_ids[face]].light() )
			{
				Light l;
				l.m_le = make_float3(
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.x + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.y + 40.f,
					shapeMaterials[shapes[i].mesh.material_ids[face]].m_emission.z + 40.f );

				size_t idx = indices.size() - 1;
				l.m_lv0	   = vertices[indices[idx - 2]];
				l.m_lv1	   = vertices[indices[idx - 1]];
				l.m_lv2	   = vertices[indices[idx - 0]];

				lights.push_back( l );
				numOfLights++;
			}

			materialIndices.push_back(
				shapes[i].mesh.material_ids[face] >= 0 ? shapes[i].mesh.material_ids[face] : hiprtInvalidValue );
		}

		verticesOffsets[i] = vertexPrefixSum;
		vertexPrefixSum += static_cast<uint32_t>( vertices.size() );
		indicesOffsets[i] = indexPrefixSum;
		indexPrefixSum += static_cast<uint32_t>( indices.size() );
		matIdxOffset[i] = matIdxPrefixSum;
		matIdxPrefixSum += static_cast<uint32_t>( shapes[i].mesh.material_ids.size() );
		normalsOffsets[i] = normalPrefixSum;
		normalPrefixSum += static_cast<uint32_t>( normals.size() );

		uint32_t mask = ~0u;
		if ( enableRayMask && ( i % 2 == 0 ) ) mask = 0u;

		instanceMask.push_back( mask );
		geomBoxes.push_back( geomBox );

		allVertices.insert( allVertices.end(), vertices.begin(), vertices.end() );
		allNormals.insert( allNormals.end(), normals.begin(), normals.end() );
		allIndices.insert( allIndices.end(), indices.begin(), indices.end() );
	}

	uint32_t threadCount = std::min( std::thread::hardware_concurrency(), 16u );
	if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport ) threadCount = 1;
	std::vector<std::thread>			  threads( threadCount );
	std::vector<std::chrono::nanoseconds> bvhBuildTimes( threadCount );
	std::vector<oroStream>				  streams( threadCount );
	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		checkOro( oroStreamCreate( &streams[threadIndex] ) );
	}

	oroCtx ctx;
	checkOro( oroCtxGetCurrent( &ctx ) );

	m_scene.m_geometries.resize( shapes.size() );
	m_scene.m_instances.resize( shapes.size() );
	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		threads[threadIndex] = std::thread(
			[&]( uint32_t threadIndex ) {
				checkOro( oroCtxSetCurrent( ctx ) );

				std::vector<hiprtGeometry*>			 geomAddrs;
				std::vector<hiprtGeometryBuildInput> geomInputs;
				for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
				{
					hiprtTriangleMeshPrimitive mesh;

					uint32_t* indices	= &allIndices[indicesOffsets[i]];
					mesh.triangleCount	= static_cast<uint32_t>( shapes[i].mesh.num_face_vertices.size() );
					mesh.triangleStride = sizeof( uint32_t ) * 3;
					malloc( reinterpret_cast<uint8_t*&>( mesh.triangleIndices ), 3 * mesh.triangleCount * sizeof( uint32_t ) );
					copyHtoDAsync(
						reinterpret_cast<uint32_t*>( mesh.triangleIndices ),
						indices,
						3 * mesh.triangleCount,
						streams[threadIndex] );

					float3* vertices  = &allVertices[verticesOffsets[i]];
					mesh.vertexCount  = ( i + 1 == shapes.size() ) ? vertexPrefixSum - verticesOffsets[i]
																   : verticesOffsets[i + 1] - verticesOffsets[i];
					mesh.vertexStride = sizeof( float3 );
					malloc( reinterpret_cast<uint8_t*&>( mesh.vertices ), mesh.vertexCount * sizeof( float3 ) );
					copyHtoDAsync(
						reinterpret_cast<float3*>( mesh.vertices ), vertices, mesh.vertexCount, streams[threadIndex] );

					hiprtGeometryBuildInput geomInput;
					geomInput.type					 = hiprtPrimitiveTypeTriangleMesh;
					geomInput.primitive.triangleMesh = mesh;
					geomInput.geomType				 = 0;

					if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
						buildEmbreeGeometryBvh( embreeDevice, vertices, indices, geomInput );

					geomInputs.push_back( geomInput );
					geomAddrs.push_back( &m_scene.m_geometries[i] );
				}

				if ( !geomInputs.empty() )
				{
					hiprtBuildOptions options;
					options.buildFlags = bvhBuildFlag;

					size_t geomTempSize;
					checkHiprt( hiprtGetGeometriesBuildTemporaryBufferSize(
						scene.m_ctx, static_cast<uint32_t>( geomInputs.size() ), geomInputs.data(), options, geomTempSize ) );

					hiprtDevicePtr tempGeomBuffer = nullptr;
					if ( geomTempSize > 0 ) malloc( reinterpret_cast<uint8_t*&>( tempGeomBuffer ), geomTempSize );

					checkHiprt( hiprtCreateGeometries(
						scene.m_ctx,
						static_cast<uint32_t>( geomInputs.size() ),
						geomInputs.data(),
						options,
						geomAddrs.data() ) );

					std::vector<hiprtGeometry> geoms;
					for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
						geoms.push_back( m_scene.m_geometries[i] );

					std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
					checkHiprt( hiprtBuildGeometries(
						scene.m_ctx,
						hiprtBuildOperationBuild,
						static_cast<uint32_t>( geomInputs.size() ),
						geomInputs.data(),
						options,
						tempGeomBuffer,
						streams[threadIndex],
						geoms.data() ) );
					std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
					bvhBuildTimes[threadIndex] += end - begin;

					size_t j = 0;
					for ( size_t i = threadIndex; i < shapes.size(); i += threadCount )
					{
						m_scene.m_geometries[i]			= geoms[j++];
						m_scene.m_instances[i].type		= hiprtInstanceTypeGeometry;
						m_scene.m_instances[i].geometry = m_scene.m_geometries[i];
					}

					for ( auto& geomInput : geomInputs )
					{
						free( geomInput.primitive.triangleMesh.triangleIndices );
						free( geomInput.primitive.triangleMesh.vertices );
						if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
						{
							free( geomInput.nodeList.nodes );
							free( geomInput.primitive.triangleMesh.trianglePairIndices );
						}
					}

					if ( geomTempSize > 0 ) free( tempGeomBuffer );

					waitForCompletion( streams[threadIndex] );
				}
			},
			threadIndex );
	}

	for ( size_t threadIndex = 0; threadIndex < threadCount; ++threadIndex )
	{
		threads[threadIndex].join();
		checkOro( oroStreamDestroy( streams[threadIndex] ) );
		bvhBuildTime = std::max( bvhBuildTime, bvhBuildTimes[threadIndex] );
	}

	// copy vertex offset
	malloc( scene.m_vertexOffsets, verticesOffsets.size() );
	copyHtoD( scene.m_vertexOffsets, verticesOffsets.data(), verticesOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_vertexOffsets );

	// copy normals
	malloc( scene.m_normals, allNormals.size() );
	copyHtoD( scene.m_normals, allNormals.data(), allNormals.size() );
	scene.m_garbageCollector.push_back( scene.m_normals );

	// copy normal offsets
	malloc( scene.m_normalOffsets, normalsOffsets.size() );
	copyHtoD( scene.m_normalOffsets, normalsOffsets.data(), normalsOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_normalOffsets );

	// copy indices
	malloc( scene.m_indices, allIndices.size() );
	copyHtoD( scene.m_indices, allIndices.data(), allIndices.size() );
	scene.m_garbageCollector.push_back( scene.m_indices );

	// copy index offsets
	malloc( scene.m_indexOffsets, indicesOffsets.size() );
	copyHtoD( scene.m_indexOffsets, indicesOffsets.data(), indicesOffsets.size() );
	scene.m_garbageCollector.push_back( scene.m_indexOffsets );

	// copy material indices
	malloc( scene.m_bufMaterialIndices, materialIndices.size() );
	copyHtoD( scene.m_bufMaterialIndices, materialIndices.data(), materialIndices.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMaterialIndices );

	// copy material offset
	malloc( scene.m_bufMatIdsPerInstance, matIdxOffset.size() );
	copyHtoD( scene.m_bufMatIdsPerInstance, matIdxOffset.data(), matIdxOffset.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMatIdsPerInstance );

	// copy materials
	if ( shapeMaterials.empty() )
	{ // default material to prevent crash
		Material mat;
		mat.m_diffuse  = make_float3( 1.0f );
		mat.m_emission = make_float3( 0.0f );
		shapeMaterials.push_back( mat );
	}
	malloc( scene.m_bufMaterials, shapeMaterials.size() );
	copyHtoD( scene.m_bufMaterials, shapeMaterials.data(), shapeMaterials.size() );
	scene.m_garbageCollector.push_back( scene.m_bufMaterials );

	// copy light
	if ( !lights.empty() )
	{
		malloc( scene.m_lights, lights.size() );
		copyHtoD( scene.m_lights, lights.data(), lights.size() );
		scene.m_garbageCollector.push_back( scene.m_lights );
	}

	// copy light num
	malloc( scene.m_numOfLights, 1 );
	copyHtoD( scene.m_numOfLights, &numOfLights, 1 );
	scene.m_garbageCollector.push_back( scene.m_numOfLights );

	// prepare scene
	hiprtScene			 sceneLocal;
	hiprtDevicePtr		 sceneTemp = nullptr;
	hiprtSceneBuildInput sceneInput;
	{
		sceneInput.instanceCount = static_cast<uint32_t>( shapes.size() );
		malloc( reinterpret_cast<uint32_t*&>( sceneInput.instanceMasks ), sceneInput.instanceCount );
		copyHtoD( reinterpret_cast<uint32_t*>( sceneInput.instanceMasks ), instanceMask.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( sceneInput.instanceMasks );

		malloc( reinterpret_cast<hiprtInstance*&>( sceneInput.instances ), sceneInput.instanceCount );
		copyHtoD(
			reinterpret_cast<hiprtInstance*>( sceneInput.instances ), m_scene.m_instances.data(), sceneInput.instanceCount );
		scene.m_garbageCollector.push_back( sceneInput.instances );

		std::vector<hiprtFrameSRT> frames;
		hiprtFrameSRT			   transform;
		if ( !frame )
		{
			transform.translation = make_float3( 0.0f, 0.0f, 0.0f );
			transform.scale		  = make_float3( 1.0f, 1.0f, 1.0f );
			transform.rotation	  = make_float4( 0.0f, 0.0f, 1.0f, 0.0f );
		}

		sceneInput.frameCount				= sceneInput.instanceCount;
		sceneInput.instanceTransformHeaders = nullptr;

		for ( uint32_t i = 0; i < sceneInput.instanceCount; i++ )
		{
			frames.push_back( frame ? frame.value() : transform );
		}

		malloc( reinterpret_cast<hiprtFrameSRT*&>( sceneInput.instanceFrames ), frames.size() );
		copyHtoD( reinterpret_cast<hiprtFrameSRT*>( sceneInput.instanceFrames ), frames.data(), frames.size() );
		scene.m_garbageCollector.push_back( sceneInput.instanceFrames );

		size_t			  sceneTempSize;
		hiprtBuildOptions options;
		options.buildFlags = bvhBuildFlag;
		checkHiprt( hiprtGetSceneBuildTemporaryBufferSize( scene.m_ctx, sceneInput, options, sceneTempSize ) );
		if ( sceneTempSize > 0 )
		{
			malloc( reinterpret_cast<uint8_t*&>( sceneTemp ), sceneTempSize );
			scene.m_garbageCollector.push_back( sceneTemp );
		}

		if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport )
		{
			buildEmbreeSceneBvh( embreeDevice, geomBoxes, frames, sceneInput );
			scene.m_garbageCollector.push_back( sceneInput.nodeList.nodes );
		}

		checkHiprt( hiprtCreateScene( scene.m_ctx, sceneInput, options, sceneLocal ) );

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		checkHiprt( hiprtBuildScene( scene.m_ctx, hiprtBuildOperationBuild, sceneInput, options, sceneTemp, 0, sceneLocal ) );
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		bvhBuildTime += ( end - begin );

		if ( time )
			std::cout << "Bvh build time : " << std::chrono::duration_cast<std::chrono::milliseconds>( bvhBuildTime ).count()
					  << " ms" << std::endl;
		scene.m_scene = sceneLocal;
	}

	if ( bvhBuildFlag == hiprtBuildFlagBitCustomBvhImport ) rtcReleaseDevice( embreeDevice );
}

void ObjTestCases::setupScene(
	Camera&						 camera,
	const std::string&			 filePath,
	const std::string&			 dirPath,
	bool						 enableRayMask,
	std::optional<hiprtFrameSRT> frame,
	hiprtBuildFlags				 bvhBuildFlag,
	bool						 time )
{
	m_camera = camera;
	createScene( m_scene, filePath, dirPath, enableRayMask, frame, bvhBuildFlag, time );
}

void ObjTestCases::deleteScene( SceneData& scene )
{

	checkHiprt( hiprtDestroyScene( scene.m_ctx, scene.m_scene ) );
	checkHiprt(
		hiprtDestroyGeometries( scene.m_ctx, static_cast<uint32_t>( scene.m_geometries.size() ), scene.m_geometries.data() ) );
	checkHiprt( hiprtDestroyContext( scene.m_ctx ) );
}

void ObjTestCases::render(
	std::optional<std::filesystem::path> imgPath,
	const std::filesystem::path&		 kernelPath,
	const std::string&					 funcName,
	std::optional<std::filesystem::path> refFilename,
	bool								 time,
	float								 aoRadius )
{
	uint8_t* dst;
	malloc( dst, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	memset( dst, 0, g_parsedArgs.m_ww * g_parsedArgs.m_wh * 4 );
	m_scene.m_garbageCollector.push_back( dst );

	uint32_t	   stackSize		  = 64u;
	const uint32_t sharedStackSize	  = 16u;
	const uint32_t blockWidth		  = 8u;
	const uint32_t blockHeight		  = 8u;
	const uint32_t blockSize		  = blockWidth * blockHeight;
	std::string	   blockSizeDef		  = "-DBLOCK_SIZE=" + std::to_string( blockSize );
	std::string	   sharedStackSizeDef = "-DSHARED_STACK_SIZE=" + std::to_string( sharedStackSize );

	std::vector<const char*> opts;
	opts.push_back( blockSizeDef.c_str() );
	opts.push_back( sharedStackSizeDef.c_str() );
	// opts.push_back( "-G" );

	hiprtGlobalStackBufferInput stackBufferInput{
		hiprtStackTypeGlobal,
		hiprtStackEntryTypeInteger,
		stackSize,
		static_cast<uint32_t>( g_parsedArgs.m_ww * g_parsedArgs.m_wh ) };
	if constexpr ( UseDynamicStack ) stackBufferInput.type = hiprtStackTypeDynamic;
	hiprtGlobalStackBuffer stackBuffer;
	checkHiprt( hiprtCreateGlobalStackBuffer( m_scene.m_ctx, stackBufferInput, stackBuffer ) );

	oroFunction	   func;
	hiprtFuncTable funcTable = nullptr;

	if constexpr ( UseFilter )
	{
		hiprtFuncNameSet funcNameSet;
		funcNameSet.filterFuncName				   = "filter";
		std::vector<hiprtFuncNameSet> funcNameSets = { funcNameSet };

		hiprtFuncDataSet funcDataSet;
		checkHiprt( hiprtCreateFuncTable( m_scene.m_ctx, 1, 1, funcTable ) );
		checkHiprt( hiprtSetFuncTable( m_scene.m_ctx, funcTable, 0, 0, funcDataSet ) );
		if constexpr ( UseBitcode )
			buildTraceKernelFromBitcode( m_scene.m_ctx, kernelPath, funcName, func, opts, funcNameSets, 1, 1 );
		else
			buildTraceKernel( m_scene.m_ctx, kernelPath, funcName, func, opts, funcNameSets, 1, 1 );
	}
	else
	{
		if constexpr ( UseBitcode )
			buildTraceKernelFromBitcode( m_scene.m_ctx, kernelPath, funcName, func, opts );

		else
			buildTraceKernel( m_scene.m_ctx, kernelPath, funcName, func, opts );
	}

	int2  res	 = make_int2( g_parsedArgs.m_ww, g_parsedArgs.m_wh );
	void* args[] = {
		&m_scene.m_scene,
		&dst,
		&res,
		&stackBuffer,
		&m_camera,
		&m_scene.m_bufMaterialIndices,
		&m_scene.m_bufMaterials,
		&m_scene.m_bufMatIdsPerInstance,
		&m_scene.m_indices,
		&m_scene.m_indexOffsets,
		&m_scene.m_normals,
		&m_scene.m_normalOffsets,
		&m_scene.m_numOfLights,
		&m_scene.m_lights,
		&aoRadius,
		&funcTable };

	int numRegs;
	checkOro( oroFuncGetAttribute( &numRegs, ORO_FUNC_ATTRIBUTE_NUM_REGS, func ) );

	int numSmem;
	checkOro( oroFuncGetAttribute( &numSmem, ORO_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func ) );

	std::cout << "Trace kernel: registers " << numRegs << ", shared memory " << numSmem << std::endl;
	waitForCompletion();
	std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
	launchKernel( func, g_parsedArgs.m_ww, g_parsedArgs.m_wh, blockWidth, blockHeight, args );

	waitForCompletion();
	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	checkHiprt( hiprtDestroyGlobalStackBuffer( m_scene.m_ctx, stackBuffer ) );
	if constexpr ( UseFilter ) checkHiprt( hiprtDestroyFuncTable( m_scene.m_ctx, funcTable ) );

	if ( time )
		std::cout << "Ray cast time: " << std::chrono::duration_cast<std::chrono::milliseconds>( end - begin ).count() << " ms"
				  << std::endl;

	if ( imgPath )
		validateAndWriteImage(
			imgPath.value(), g_parsedArgs.m_ww, g_parsedArgs.m_wh, dst, g_parsedArgs.m_referencePath, refFilename );
}
