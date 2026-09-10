// SPDX-FileCopyrightText: 2025 Erin Catto
// SPDX-License-Identifier: MIT

#include "broad_phase.h"

#include "aabb.h"
#include "arena_allocator.h"
#include "body.h"
#include "contact.h"
#include "core.h"
#include "parallel_for.h"
#include "physics_world.h"
#include "platform.h"
#include "shape.h"

#include <string.h>
#include <stdlib.h>

void b3CreateBroadPhase( b3BroadPhase* bp, const b3Capacity* capacity )
{
	_Static_assert( b3_bodyTypeCount == 3, "must be three body types" );

	bp->movedProxies[b3_staticBody] = b3CreateBitSet( b3MaxInt( 16, capacity->staticShapeCount ) );
	bp->movedProxies[b3_kinematicBody] = b3CreateBitSet( 16 );
	bp->movedProxies[b3_dynamicBody] = b3CreateBitSet( b3MaxInt( 16, capacity->dynamicShapeCount ) );
	b3Array_Reserve( bp->moveArray, capacity->dynamicShapeCount );
	bp->moveResults = NULL;
	bp->movePairs = NULL;
	bp->movePairCapacity = 0;
	b3AtomicStoreInt( &bp->movePairIndex, 0 );
	bp->pairSet = b3CreateSet( 2 * capacity->contactCount );

	int staticCapacity = b3MaxInt( 16, capacity->staticShapeCount );
	bp->trees[b3_staticBody] = b3DynamicTree_Create( staticCapacity );

	int kinematicCapacity = 16;
	bp->trees[b3_kinematicBody] = b3DynamicTree_Create( kinematicCapacity );

	int dynamicCapacity = b3MaxInt( 16, capacity->dynamicShapeCount );
	bp->trees[b3_dynamicBody] = b3DynamicTree_Create( dynamicCapacity );
}

void b3DestroyBroadPhase( b3BroadPhase* bp )
{
	for ( int i = 0; i < b3_bodyTypeCount; ++i )
	{
		b3DynamicTree_Destroy( bp->trees + i );
	}

	for ( int i = 0; i < b3_bodyTypeCount; ++i )
	{
		b3DestroyBitSet( &bp->movedProxies[i] );
	}
	b3Array_Destroy( bp->moveArray );
	b3DestroySet( &bp->pairSet );

	*bp = (b3BroadPhase){ 0 };

	memset( bp, 0, sizeof( b3BroadPhase ) );
}

static void b3UnBufferMove( b3BroadPhase* bp, int proxyKey )
{
	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );
	b3BitSet* set = &bp->movedProxies[proxyType];

	if ( b3GetBit( set, proxyId ) )
	{
		b3ClearBit( set, proxyId );

		// Purge from move buffer. Linear search.
		// todo if I can iterate the move set then I don't need the moveArray
		int count = bp->moveArray.count;
		for ( int i = 0; i < count; ++i )
		{
			if ( bp->moveArray.data[i] == proxyKey )
			{
				b3Array_RemoveSwap( bp->moveArray, i );
				break;
			}
		}
	}
}

int b3BroadPhase_CreateProxy( b3BroadPhase* bp, b3BodyType proxyType, b3AABB aabb, uint64_t categoryBits, int shapeIndex,
							  bool forcePairCreation )
{
	B3_ASSERT( 0 <= proxyType && proxyType < b3_bodyTypeCount );
	int proxyId = b3DynamicTree_CreateProxy( bp->trees + proxyType, aabb, categoryBits, shapeIndex );
	int proxyKey = B3_PROXY_KEY( proxyId, proxyType );
	if ( proxyType != b3_staticBody || forcePairCreation )
	{
		b3BufferMove( bp, proxyKey );
	}
	return proxyKey;
}

void b3BroadPhase_DestroyProxy( b3BroadPhase* bp, int proxyKey )
{
	b3UnBufferMove( bp, proxyKey );

	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	B3_ASSERT( 0 <= proxyType && proxyType <= b3_bodyTypeCount );
	b3DynamicTree_DestroyProxy( bp->trees + proxyType, proxyId );
}

void b3BroadPhase_MoveProxy( b3BroadPhase* bp, int proxyKey, b3AABB aabb )
{
	b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	b3DynamicTree_MoveProxy( bp->trees + proxyType, proxyId, aabb );
	b3BufferMove( bp, proxyKey );
}

void b3BroadPhase_EnlargeProxy( b3BroadPhase* bp, int proxyKey, b3AABB aabb )
{
	B3_ASSERT( proxyKey != B3_NULL_INDEX );
	int typeIndex = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	B3_ASSERT( typeIndex != b3_staticBody );

	b3DynamicTree_EnlargeProxy( bp->trees + typeIndex, proxyId, aabb );
	b3BufferMove( bp, proxyKey );
}

typedef struct b3MovePair
{
	int shapeIndexA;
	int shapeIndexB;
	int childIndex;
	b3MovePair* next;
	bool heap;
} b3MovePair;

typedef struct b3MoveResult
{
	b3MovePair* pairList;
} b3MoveResult;

typedef struct b3QueryPairContext
{
	b3World* world;
	b3MoveResult* moveResult;
	b3AABB aabb;
	b3BodyType queryTreeType;
	int queryProxyKey;
	int queryShapeIndex;

	int compoundProxyId;
	int compoundShapeIndex;
} b3QueryPairContext;

// This is called from b3DynamicTree::Query when we are gathering pairs.
static bool b3PairQueryCallback( int proxyId, uint64_t userData, void* context )
{
	b3QueryPairContext* queryContext = (b3QueryPairContext*)context;
	b3World* world = queryContext->world;
	int shapeIndex;
	int childIndex = 0;

	if ( queryContext->compoundShapeIndex == B3_NULL_INDEX )
	{
		// Outer query: userData is a shape index.
		shapeIndex = (int)userData;

		// A proxy cannot form a pair with itself.
		if ( shapeIndex == queryContext->queryShapeIndex )
		{
			return true;
		}

		b3Shape* shape = b3Array_Get( world->shapes, shapeIndex );
		if ( shape->type == b3_compoundShape )
		{
			// Query bounds are float world space, so the demoted transform is the matching float frame
			b3Transform compoundTransform = b3ToRelativeTransform( b3GetBodyTransform( world, shape->bodyId ), b3Pos_zero );
			b3AABB localAABB = b3AABB_Transform( b3InvertTransform( compoundTransform ), queryContext->aabb );

			// recurse
			queryContext->compoundShapeIndex = shapeIndex;
			queryContext->compoundProxyId = proxyId;

			b3DynamicTree_Query( &shape->compound->tree, localAABB, B3_DEFAULT_MASK_BITS, false, b3PairQueryCallback, context );
			queryContext->compoundShapeIndex = B3_NULL_INDEX;
			queryContext->compoundProxyId = B3_NULL_INDEX;
			return true;
		}
	}
	else
	{
		// Inner query into a compound shape: userData is the compound child index, not a shape
		// index, so do not compare it against queryShapeIndex.
		shapeIndex = queryContext->compoundShapeIndex;
		proxyId = queryContext->compoundProxyId;
		childIndex = (int)userData;
	}

	b3BroadPhase* broadPhase = &queryContext->world->broadPhase;

	int proxyKey = B3_PROXY_KEY( proxyId, queryContext->queryTreeType );
	int queryProxyKey = queryContext->queryProxyKey;

	// A proxy cannot form a pair with itself.
	B3_ASSERT( proxyKey != queryContext->queryProxyKey );

	b3BodyType treeType = queryContext->queryTreeType;
	b3BodyType queryProxyType = B3_PROXY_TYPE( queryProxyKey );

	// De-duplication
	// It is important to prevent duplicate contacts from being created. Ideally I can prevent duplicates
	// early and in the worker. Most of the time the movedProxies bit sets contain dynamic and kinematic
	// proxies, but sometimes static proxies are in there too (b3ShapeDef::invokeContactCreation or a
	// modified static shape), so we always have to check.

	// Is this proxy also moving?
	if ( queryProxyType == b3_dynamicBody )
	{
		if ( treeType == b3_dynamicBody && proxyKey < queryProxyKey )
		{
			bool moved = b3GetBit( &broadPhase->movedProxies[treeType], proxyId );
			if ( moved )
			{
				// Both proxies are moving. Avoid duplicate pairs.
				return true;
			}
		}
	}
	else
	{
		B3_ASSERT( treeType == b3_dynamicBody );
		bool moved = b3GetBit( &broadPhase->movedProxies[treeType], proxyId );
		if ( moved )
		{
			// Both proxies are moving. Avoid duplicate pairs.
			return true;
		}
	}

	uint64_t pairKey = b3ShapePairKey( shapeIndex, queryContext->queryShapeIndex, childIndex );
	if ( b3ContainsKey( &broadPhase->pairSet, pairKey ) )
	{
		// contact exists
		return true;
	}

	// Order shapes so that B3_SHAPE_PAIR_KEY works correctly
	int shapeIdA = shapeIndex;
	int shapeIdB = queryContext->queryShapeIndex;
	b3Shape* shapeA = b3Array_Get( world->shapes, shapeIdA );
	b3Shape* shapeB = b3Array_Get( world->shapes, shapeIdB );
	int bodyIdA = shapeA->bodyId;
	int bodyIdB = shapeB->bodyId;

	// Are the shapes on the same body?
	if ( bodyIdA == bodyIdB )
	{
		return true;
	}

	// Sensors are handled elsewhere
	if ( shapeA->sensorIndex != B3_NULL_INDEX || shapeB->sensorIndex != B3_NULL_INDEX )
	{
		return true;
	}

	if ( b3ShouldShapesCollide( shapeA->filter, shapeB->filter ) == false )
	{
		return true;
	}

	// Does a joint override collision?
	b3Body* bodyA = b3Array_Get( world->bodies, bodyIdA );
	b3Body* bodyB = b3Array_Get( world->bodies, bodyIdB );
	if ( b3ShouldBodiesCollide( world, bodyA, bodyB ) == false )
	{
		return true;
	}

	// Custom user filter
	if ( ( shapeA->flags & b3_enableCustomFiltering ) || ( shapeB->flags & b3_enableCustomFiltering ) )
	{
		b3CustomFilterFcn* customFilterFcn = queryContext->world->customFilterFcn;
		if ( customFilterFcn != NULL )
		{
			b3ShapeId idA = { shapeIdA + 1, world->worldId, shapeA->generation };
			b3ShapeId idB = { shapeIdB + 1, world->worldId, shapeB->generation };
			bool shouldCollide = customFilterFcn( idA, idB, queryContext->world->customFilterContext );
			if ( shouldCollide == false )
			{
				return true;
			}
		}
	}

	// todo per thread to eliminate atomic?
	int pairIndex = b3AtomicFetchAddInt( &broadPhase->movePairIndex, 1 );

	b3MovePair* pair;
	if ( pairIndex < broadPhase->movePairCapacity )
	{
		pair = broadPhase->movePairs + pairIndex;
		pair->heap = false;
	}
	else
	{
		// todo experimenting with ignoring this pair if we ran out of space
		return true;
		// pair = (b3MovePair*)b3Alloc( sizeof( b3MovePair ) );
		// pair->heap = true;
	}

	pair->shapeIndexA = shapeIdA;
	pair->shapeIndexB = shapeIdB;
	pair->childIndex = childIndex;
	pair->next = queryContext->moveResult->pairList;
	queryContext->moveResult->pairList = pair;

	// continue the query
	return true;
}

typedef struct b3FindPairsContext
{
	b3World* world;
	const int* bodyIds;
} b3FindPairsContext;

static void b3BuildTreeBodyIds( const b3World* world, const b3DynamicTree* tree, int* bodyIds )
{
	int previous = B3_NULL_INDEX;
	int index = tree->root;
	while ( index != B3_NULL_INDEX )
	{
		const b3TreeNode* node = tree->nodes + index;
		int next;
		if ( previous == node->parent )
		{
			if ( node->flags & b3_leafNode )
			{
				bodyIds[index] = world->shapes.data[(int)node->userData].bodyId;
				next = node->parent;
			}
			else
			{
				next = node->children.child1;
			}
		}
		else if ( previous == node->children.child1 )
		{
			next = node->children.child2;
		}
		else
		{
			int a = bodyIds[node->children.child1];
			int b = bodyIds[node->children.child2];
			bodyIds[index] = a == b ? a : B3_NULL_INDEX;
			next = node->parent;
		}
		previous = index;
		index = next;
	}
}

static void b3QueryOtherBodies( const b3DynamicTree* tree, b3AABB aabb, const int* bodyIds, int bodyId,
								 b3QueryPairContext* context )
{
	if ( tree->nodeCount == 0 )
	{
		return;
	}
	int stack[1024];
	int count = 0;
	stack[count++] = tree->root;
	while ( count > 0 )
	{
		int index = stack[--count];
		const b3TreeNode* node = tree->nodes + index;
		if ( bodyIds[index] == bodyId || ( node->categoryBits & B3_DEFAULT_MASK_BITS ) == 0 ||
			 b3AABB_Overlaps( node->aabb, aabb ) == false )
		{
			continue;
		}
		if ( node->flags & b3_leafNode )
		{
			if ( b3PairQueryCallback( index, node->userData, context ) == false )
			{
				return;
			}
		}
		else
		{
			B3_ASSERT( count < 1023 );
			if ( count < 1023 )
			{
				stack[count++] = node->children.child1;
				stack[count++] = node->children.child2;
			}
		}
	}
}

static void b3FindPairsTask( int startIndex, int endIndex, int workerIndex, void* context )
{
	b3TracyCZoneNC( pair_task, "Pair Task", b3_colorAquamarine, true );

	B3_UNUSED( workerIndex );

	b3FindPairsContext* task = (b3FindPairsContext*)context;
	b3World* world = task->world;
	b3BroadPhase* bp = &world->broadPhase;

	b3QueryPairContext queryContext = { 0 };
	queryContext.world = world;
	queryContext.compoundShapeIndex = B3_NULL_INDEX;

	for ( int i = startIndex; i < endIndex; ++i )
	{
		// Initialize move result for this moved proxy
		queryContext.moveResult = bp->moveResults + i;
		queryContext.moveResult->pairList = NULL;

		int proxyKey = bp->moveArray.data[i];
		b3BodyType proxyType = B3_PROXY_TYPE( proxyKey );

		int proxyId = B3_PROXY_ID( proxyKey );
		queryContext.queryProxyKey = proxyKey;

		const b3DynamicTree* baseTree = bp->trees + proxyType;

		// We have to query the tree with the fat AABB so that
		// we don't fail to create a contact that may touch later.
		b3AABB fatAABB = b3DynamicTree_GetAABB( baseTree, proxyId );
		queryContext.queryShapeIndex = (int)b3DynamicTree_GetUserData( baseTree, proxyId );
		queryContext.aabb = fatAABB;

		// Compound shape collision invocation is not supported
		B3_VALIDATE( world->shapes.data[queryContext.queryShapeIndex].type != b3_compoundShape );

		// Query trees. Only dynamic proxies collide with kinematic and static proxies.
		// Using B3_DEFAULT_MASK_BITS so that b3Filter::groupIndex works.
		// consider using bits = groupIndex > 0 ? B3_DEFAULT_MASK_BITS : maskBits
		bool requireAllBits = false;
		if ( proxyType == b3_dynamicBody )
		{
			queryContext.queryTreeType = b3_kinematicBody;
			b3DynamicTree_Query( bp->trees + b3_kinematicBody, fatAABB, B3_DEFAULT_MASK_BITS, requireAllBits, b3PairQueryCallback,
								 &queryContext );

			queryContext.queryTreeType = b3_staticBody;
			b3DynamicTree_Query( bp->trees + b3_staticBody, fatAABB, B3_DEFAULT_MASK_BITS, requireAllBits, b3PairQueryCallback,
								 &queryContext );
		}

		// All proxies collide with dynamic proxies
		// Using B3_DEFAULT_MASK_BITS so that b3Filter::groupIndex works.
		queryContext.queryTreeType = b3_dynamicBody;
		if ( task->bodyIds != NULL && proxyType == b3_dynamicBody )
		{
			int bodyId = world->shapes.data[queryContext.queryShapeIndex].bodyId;
			b3QueryOtherBodies( bp->trees + b3_dynamicBody, fatAABB, task->bodyIds, bodyId, &queryContext );
		}
		else
		{
			b3DynamicTree_Query( bp->trees + b3_dynamicBody, fatAABB, B3_DEFAULT_MASK_BITS, requireAllBits,
								 b3PairQueryCallback, &queryContext );
		}
	}

	b3TracyCZoneEnd( pair_task );
}

static void b3UpdateTreesTask( void* context )
{
	b3TracyCZoneNC( tree_task, "Rebuild Trees", b3_colorFireBrick, true );

	b3World* world = (b3World*)context;
	b3DynamicTree_Rebuild( world->broadPhase.trees + b3_dynamicBody, false );
	b3DynamicTree_Rebuild( world->broadPhase.trees + b3_kinematicBody, false );

	b3TracyCZoneEnd( tree_task );
}

void b3UpdateBroadPhasePairs( b3World* world )
{
	b3BroadPhase* bp = &world->broadPhase;

	int moveCount = bp->moveArray.count;

	if ( moveCount == 0 )
	{
		return;
	}

	b3TracyCZoneNC( update_pairs, "Pairs", b3_colorMediumSlateBlue, true );

	b3Stack* alloc = &world->stack;

	// todo these could be in the step context
	bp->moveResults = (b3MoveResult*)b3StackAlloc( alloc, moveCount * sizeof( b3MoveResult ), "move results" );
	bp->movePairCapacity = 16 * moveCount;
	bp->movePairs = (b3MovePair*)b3StackAlloc( alloc, bp->movePairCapacity * sizeof( b3MovePair ), "move pairs" );

	b3AtomicStoreInt( &bp->movePairIndex, 0 );

#ifndef NDEBUG
	extern b3AtomicInt b3_probeCount;
	b3AtomicStoreInt( &b3_probeCount, 0 );
#endif

	const char* option = getenv( "B3_PRUNE_SELF_SUBTREES" );
	bool prune = option == NULL || strcmp( option, "0" ) != 0;
	int* bodyIds = NULL;
	if ( prune && bp->trees[b3_dynamicBody].nodeCount > 0 )
	{
		bodyIds = b3StackAlloc( alloc, bp->trees[b3_dynamicBody].nodeCapacity * sizeof( int ), "tree body ids" );
		b3BuildTreeBodyIds( world, bp->trees + b3_dynamicBody, bodyIds );
#if B3_ENABLE_VALIDATION
		const b3DynamicTree* tree = bp->trees + b3_dynamicBody;
		for ( int i = 0; i < tree->nodeCapacity; ++i )
		{
			const b3TreeNode* node = tree->nodes + i;
			if ( ( node->flags & b3_allocatedNode ) == 0 )
			{
				continue;
			}
			if ( node->flags & b3_leafNode )
			{
				B3_ASSERT( bodyIds[i] == world->shapes.data[(int)node->userData].bodyId );
			}
			else
			{
				int a = bodyIds[node->children.child1];
				int b = bodyIds[node->children.child2];
				B3_ASSERT( bodyIds[i] == ( a == b ? a : B3_NULL_INDEX ) );
			}
		}
#endif
	}
	b3FindPairsContext task = { world, bodyIds };
	int minRange = 64;
	b3ParallelFor( world, b3FindPairsTask, moveCount, minRange, &task, "pairs" );
	if ( bodyIds != NULL )
	{
		b3StackFree( alloc, bodyIds );
	}

	b3TracyCZoneNC( create_contacts, "Create Contacts", b3_colorCoral, true );

	// Task that can be done in parallel with the narrow-phase
	// - rebuild the collision tree for dynamic and kinematic bodies to keep their query performance good
	if ( world->taskCount < B3_MAX_TASKS )
	{
		world->userTreeTask = world->enqueueTaskFcn( &b3UpdateTreesTask, world, world->userTaskContext, "rebuild tree" );
		world->taskCount += 1;
		world->activeTaskCount += world->userTreeTask == NULL ? 0 : 1;
	}
	else
	{
		world->userTreeTask = NULL;
		b3UpdateTreesTask( world );
	}

	// Single-threaded work
	// - Clear move flags
	// - Create contacts in deterministic order
	for ( int i = 0; i < moveCount; ++i )
	{
		b3MoveResult* result = bp->moveResults + i;
		b3MovePair* pair = result->pairList;
		while ( pair != NULL )
		{
			int shapeIdA = pair->shapeIndexA;
			int shapeIdB = pair->shapeIndexB;
			int childIndex = pair->childIndex;

			b3Shape* shapeA = b3Array_Get( world->shapes, shapeIdA );
			b3Shape* shapeB = b3Array_Get( world->shapes, shapeIdB );

			b3CreateContact( world, shapeA, shapeB, childIndex );

			if ( pair->heap )
			{
				b3MovePair* temp = pair;
				pair = pair->next;
				b3Free( temp, sizeof( b3MovePair ) );
			}
			else
			{
				pair = pair->next;
			}
		}
	}

	// Reset move buffer: clear only the bits that were set this step.
	// Invariant: bit set in movedProxies[type] iff proxyKey is present in moveArray.
	for ( int i = 0; i < bp->moveArray.count; ++i )
	{
		int proxyKey = bp->moveArray.data[i];
		b3ClearBit( &bp->movedProxies[B3_PROXY_TYPE( proxyKey )], B3_PROXY_ID( proxyKey ) );
	}
	b3Array_Clear( bp->moveArray );

	b3StackFree( alloc, bp->movePairs );
	bp->movePairs = NULL;
	b3StackFree( alloc, bp->moveResults );
	bp->moveResults = NULL;

	b3ValidateSolverSets( world );

	b3TracyCZoneEnd( create_contacts );

	b3TracyCZoneEnd( update_pairs );
}

bool b3BroadPhase_TestOverlap( const b3BroadPhase* bp, int proxyKeyA, int proxyKeyB )
{
	int typeIndexA = B3_PROXY_TYPE( proxyKeyA );
	int proxyIdA = B3_PROXY_ID( proxyKeyA );
	int typeIndexB = B3_PROXY_TYPE( proxyKeyB );
	int proxyIdB = B3_PROXY_ID( proxyKeyB );

	b3AABB aabbA = b3DynamicTree_GetAABB( bp->trees + typeIndexA, proxyIdA );
	b3AABB aabbB = b3DynamicTree_GetAABB( bp->trees + typeIndexB, proxyIdB );
	return b3AABB_Overlaps( aabbA, aabbB );
}

int b3BroadPhase_GetShapeIndex( b3BroadPhase* bp, int proxyKey )
{
	int typeIndex = B3_PROXY_TYPE( proxyKey );
	int proxyId = B3_PROXY_ID( proxyKey );

	return (int)b3DynamicTree_GetUserData( bp->trees + typeIndex, proxyId );
}

void b3ValidateBroadPhase( const b3BroadPhase* bp )
{
	b3DynamicTree_Validate( bp->trees + b3_dynamicBody );
	b3DynamicTree_Validate( bp->trees + b3_kinematicBody );

	// todo validate every shape AABB is contained in tree AABB
}

void b3ValidateNoEnlarged( const b3BroadPhase* bp )
{
#if B3_ENABLE_VALIDATION == 1
	for ( int j = 0; j < b3_bodyTypeCount; ++j )
	{
		const b3DynamicTree* tree = bp->trees + j;
		b3DynamicTree_ValidateNoEnlarged( tree );
	}
#else
	B3_UNUSED( bp );
#endif
}
