// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#include "arena_allocator.h" // b3Arena
#include "contact.h"
#include "core.h"
#include "hull.h"
#include "manifold.h" // b3MakeFeatureId
#include "physics_world.h"
#include "shape.h"
#include "voxel_collide.h"
#include "voxel_shape.h"

#include "box3d/collision.h" // b3CollideHullAnd*, b3MakeOffsetBoxHull
#include "box3d/constants.h"
#include "box3d/types.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#define B3_VOXEL_NORMAL_MATCH 0.995f

typedef struct b3VoxelOldManifold
{
	b3Vec3 normal;
	b3Vec3 frictionImpulse;
	b3Vec3 rollingImpulse;
	float twistImpulse;
	int pointCount;
	uint32_t featureId[B3_MAX_MANIFOLD_POINTS];
	b3Vec3 anchorA[B3_MAX_MANIFOLD_POINTS];
	float normalImpulse[B3_MAX_MANIFOLD_POINTS];
	bool claimed[B3_MAX_MANIFOLD_POINTS];
	bool consumed;
} b3VoxelOldManifold;

static int b3Voxel_reduceCluster( const b3VoxelContact* contacts, const int* members, int n, int maxPts, int* outIdx )
{
	if ( n <= maxPts )
	{
		for ( int i = 0; i < n; ++i )
			outIdx[i] = members[i];
		return n;
	}

	bool used[B3_VOXEL_MAX_CONTACTS] = { false };

	int deepest = 0;
	for ( int i = 1; i < n; ++i )
	{
		if ( contacts[members[i]].initialPenetration > contacts[members[deepest]].initialPenetration )
			deepest = i;
	}

	int nsel = 0;
	outIdx[nsel++] = members[deepest];
	used[deepest] = true;

	while ( nsel < maxPts )
	{
		int bestIdx = -1;
		float bestMin = -1.0f;
		for ( int i = 0; i < n; ++i )
		{
			if ( used[i] )
				continue;
			float minD = FLT_MAX;
			for ( int j = 0; j < nsel; ++j )
			{
				b3Vec3 diff = b3Sub( contacts[members[i]].body0Point, contacts[outIdx[j]].body0Point );
				float d = b3Dot( diff, diff );
				if ( d < minD )
					minD = d;
			}
			if ( minD > bestMin )
			{
				bestMin = minD;
				bestIdx = i;
			}
		}
		if ( bestIdx < 0 )
			break;
		outIdx[nsel++] = members[bestIdx];
		used[bestIdx] = true;
	}
	return nsel;
}

static bool b3Voxel_emitManifolds( b3World* world, b3Contact* contact, const b3VoxelContact* contacts, int count,
								   b3WorldTransform xfA, b3WorldTransform xfB, const b3Shape* shapeA, const b3Shape* shapeB,
								   b3VoxelCounters* counters )
{
	if ( count == 0 )
	{
		if ( contact->manifoldCount > 0 )
		{
			b3FreeManifolds( world, contact->manifolds, contact->manifoldCount );
			contact->manifolds = NULL;
			contact->manifoldCount = 0;
		}
		return false;
	}

	b3VoxelOldManifold oldManifolds[B3_VOXEL_MAX_CONTACTS];
	int oldCount = contact->manifoldCount;
	if ( oldCount > B3_VOXEL_MAX_CONTACTS )
		oldCount = B3_VOXEL_MAX_CONTACTS;
	for ( int i = 0; i < oldCount; ++i )
	{
		const b3Manifold* m = contact->manifolds + i;
		b3VoxelOldManifold* o = oldManifolds + i;
		o->normal = m->normal;
		o->frictionImpulse = m->frictionImpulse;
		o->rollingImpulse = m->rollingImpulse;
		o->twistImpulse = m->twistImpulse;
		o->pointCount = m->pointCount;
		o->consumed = false;
		for ( int j = 0; j < m->pointCount && j < B3_MAX_MANIFOLD_POINTS; ++j )
		{
			o->featureId[j] = m->points[j].featureId;
			o->anchorA[j] = m->points[j].anchorA;
			o->normalImpulse[j] = m->points[j].normalImpulse;
			o->claimed[j] = false;
		}
	}

	int clusterOf[B3_VOXEL_MAX_CONTACTS];
	b3Vec3 clusterNormal[B3_VOXEL_MAX_CONTACTS];
	int clusterCount = 0;
	for ( int i = 0; i < count; ++i )
	{
		int cl = -1;
		for ( int k = 0; k < clusterCount; ++k )
		{
			if ( b3Dot( clusterNormal[k], contacts[i].normal ) > B3_VOXEL_CLUSTER_DOT )
			{
				cl = k;
				break;
			}
		}
		if ( cl < 0 )
		{
			cl = clusterCount;
			clusterNormal[clusterCount] = contacts[i].normal;
			clusterCount += 1;
		}
		clusterOf[i] = cl;
	}

	if ( contact->manifoldCount != clusterCount )
	{
		if ( counters != NULL )
		{
			counters->manifoldReallocations += 1;
		}
		if ( contact->manifoldCount > 0 )
		{
			b3FreeManifolds( world, contact->manifolds, contact->manifoldCount );
		}
		contact->manifolds = b3AllocateManifolds( world, clusterCount );
		contact->manifoldCount = (uint16_t)clusterCount;
	}
	else
	{
		memset( contact->manifolds, 0, contact->manifoldCount * sizeof( b3Manifold ) );
	}
	if ( counters != NULL )
	{
		counters->normalClusters += clusterCount;
		counters->emittedManifolds += clusterCount;
	}

	b3Matrix3 matrixA = b3MakeMatrixFromQuat( xfA.q );
	b3Vec3 offsetAB = b3SubPos( xfA.p, xfB.p );

	for ( int cl = 0; cl < clusterCount; ++cl )
	{
		int members[B3_VOXEL_MAX_CONTACTS];
		int n = 0;
		for ( int i = 0; i < count; ++i )
		{
			if ( clusterOf[i] == cl )
				members[n++] = i;
		}

		int sel[B3_MAX_MANIFOLD_POINTS];
		int nsel = b3Voxel_reduceCluster( contacts, members, n, B3_MAX_MANIFOLD_POINTS, sel );
		if ( counters != NULL )
		{
			counters->emittedPoints += nsel;
		}

		b3Manifold* manifold = contact->manifolds + cl;
		manifold->pointCount = nsel;

		manifold->normal = b3MulMV( matrixA, b3Neg( clusterNormal[cl] ) );

		for ( int j = 0; j < nsel; ++j )
		{
			const b3VoxelContact* c = contacts + sel[j];
			b3ManifoldPoint* mp = manifold->points + j;
			b3Vec3 point = b3MulMV( matrixA, b3MulSV( 0.5f, b3Add( c->body0Point, c->body1Point ) ) );
			mp->anchorA = point;
			mp->anchorB = b3Add( point, offsetAB );
			mp->separation = -c->initialPenetration;
			mp->featureId = c->featureId;
			mp->triangleIndex = B3_NULL_INDEX;
			mp->normalVelocity = 0.0f;
			mp->totalNormalImpulse = 0.0f;
			mp->normalImpulse = 0.0f;
			mp->persisted = false;
		}

		int best = -1;
		float bestDot = B3_VOXEL_NORMAL_MATCH;
		for ( int k = 0; k < oldCount; ++k )
		{
			if ( oldManifolds[k].consumed )
				continue;
			float d = b3Dot( oldManifolds[k].normal, manifold->normal );
			if ( d > bestDot )
			{
				bestDot = d;
				best = k;
			}
		}
		if ( best >= 0 )
		{
			b3VoxelOldManifold* om = oldManifolds + best;
			manifold->frictionImpulse = om->frictionImpulse;
			manifold->rollingImpulse = om->rollingImpulse;
			manifold->twistImpulse = om->twistImpulse;
			om->consumed = true;

			for ( int j = 0; j < nsel; ++j )
			{
				b3ManifoldPoint* mp = manifold->points + j;
				for ( int k = 0; k < om->pointCount; ++k )
				{
					b3Vec3 anchorDelta = b3Sub( om->anchorA[k], mp->anchorA );
					float recycleDistance = B3_CONTACT_RECYCLE_DISTANCE;
					if ( !om->claimed[k] && om->featureId[k] == mp->featureId &&
						 b3Dot( anchorDelta, anchorDelta ) <= recycleDistance * recycleDistance )
					{
						mp->normalImpulse = om->normalImpulse[k];
						mp->persisted = true;
						if ( counters != NULL )
						{
							counters->persistedPoints += 1;
						}
						// The cell-pair id is a compact hash. The local anchor guard makes a
						// collision harmless: an impulse is only reused at the same geometric
						// feature, within Box3D's normal contact-recycling tolerance.
						om->claimed[k] = true;
						break;
					}
				}
			}
		}
	}

	const b3SurfaceMaterial* materialA = b3GetShapeMaterials( shapeA );
	const b3SurfaceMaterial* materialB = b3GetShapeMaterials( shapeB );
	contact->friction =
		world->frictionCallback( materialA->friction, materialA->userMaterialId, materialB->friction, materialB->userMaterialId );
	contact->restitution = world->restitutionCallback( materialA->restitution, materialA->userMaterialId, materialB->restitution,
													   materialB->userMaterialId );
	contact->rollingResistance = 0.0f;
	contact->tangentVelocity =
		b3Sub( b3RotateVector( xfA.q, materialA->tangentVelocity ), b3RotateVector( xfB.q, materialB->tangentVelocity ) );

	return true;
}

static bool b3Voxel_emitPatchManifolds( b3World* world, b3Contact* contact, const b3VoxelPatchCollision* collision,
									 b3WorldTransform xfA, b3WorldTransform xfB, const b3Shape* shapeA,
									 const b3Shape* shapeB, b3VoxelCounters* counters )
{
	B3_ASSERT( contact->flags & b3_simVoxelContact );
	b3VoxelContactWorkspace* workspace = &contact->voxelContact;

	// Preserve the established bounded normal-cluster solver output. Each
	// cluster chooses one retained canonical patch as its exact persistence
	// owner, so no impulse may migrate to an unrelated same-normal patch.
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	uint8_t patchOfContact[B3_VOXEL_MAX_CONTACTS];
	int contactCount = 0;
	for ( int patchIndex = 0; patchIndex < collision->manifoldCount; ++patchIndex )
	{
		const b3VoxelPatchManifold* patch = collision->manifolds + patchIndex;
		for ( int pointIndex = 0; pointIndex < patch->pointCount; ++pointIndex )
		{
			if ( contactCount >= B3_VOXEL_MAX_CONTACTS )
			{
				B3_ASSERT( false );
				break;
			}
			contacts[contactCount] = patch->points[pointIndex];
			patchOfContact[contactCount] = (uint8_t)patchIndex;
			contactCount += 1;
		}
	}
	if ( contactCount == 0 )
	{
		if ( contact->manifoldCount > 0 )
		{
			b3FreeManifolds( world, contact->manifolds, contact->manifoldCount );
			contact->manifolds = NULL;
			contact->manifoldCount = 0;
		}
		workspace->states.count = 0;
		return false;
	}

	b3Manifold oldManifolds[B3_VOXEL_MAX_CONTACTS];
	b3VoxelManifoldState oldStates[B3_VOXEL_MAX_CONTACTS];
	int oldCount = 0;
	if ( workspace->states.count == contact->manifoldCount )
	{
		oldCount = contact->manifoldCount;
		B3_ASSERT( oldCount <= B3_VOXEL_MAX_CONTACTS );
		if ( oldCount > 0 )
		{
			memcpy( oldManifolds, contact->manifolds, (size_t)oldCount * sizeof( b3Manifold ) );
			memcpy( oldStates, workspace->states.data, (size_t)oldCount * sizeof( b3VoxelManifoldState ) );
		}
	}

	int clusterOf[B3_VOXEL_MAX_CONTACTS];
	b3Vec3 clusterNormal[B3_VOXEL_MAX_CONTACTS];
	int clusterCount = 0;
	for ( int i = 0; i < contactCount; ++i )
	{
		int clusterIndex = B3_NULL_INDEX;
		for ( int j = 0; j < clusterCount; ++j )
		{
			if ( b3Dot( clusterNormal[j], contacts[i].normal ) > B3_VOXEL_CLUSTER_DOT )
			{
				clusterIndex = j;
				break;
			}
		}
		if ( clusterIndex == B3_NULL_INDEX )
		{
			clusterIndex = clusterCount;
			clusterNormal[clusterCount++] = contacts[i].normal;
		}
		clusterOf[i] = clusterIndex;
	}

	B3_ASSERT( clusterCount <= B3_VOXEL_MAX_CONTACTS );
	if ( contact->manifoldCount != clusterCount )
	{
		if ( counters != NULL )
			counters->manifoldReallocations += 1;
		if ( contact->manifoldCount > 0 )
			b3FreeManifolds( world, contact->manifolds, contact->manifoldCount );
		contact->manifolds = b3AllocateManifolds( world, clusterCount );
		contact->manifoldCount = clusterCount;
		memset( contact->manifolds, 0, (size_t)clusterCount * sizeof( b3Manifold ) );
	}
	else
	{
		memset( contact->manifolds, 0, (size_t)clusterCount * sizeof( b3Manifold ) );
	}
	b3Array_Resize( workspace->states, clusterCount );

	if ( counters != NULL )
	{
		counters->normalClusters += clusterCount;
		counters->emittedManifolds += clusterCount;
	}

	b3Matrix3 matrixA = b3MakeMatrixFromQuat( xfA.q );
	b3Vec3 offsetAB = b3SubPos( xfA.p, xfB.p );
	bool oldConsumed[B3_VOXEL_MAX_CONTACTS] = { false };
	bool oldPointClaimed[B3_VOXEL_MAX_CONTACTS][B3_MAX_MANIFOLD_POINTS] = { { false } };

	for ( int clusterIndex = 0; clusterIndex < clusterCount; ++clusterIndex )
	{
		int members[B3_VOXEL_MAX_CONTACTS];
		int memberCount = 0;
		for ( int i = 0; i < contactCount; ++i )
		{
			if ( clusterOf[i] == clusterIndex )
				members[memberCount++] = i;
		}
		int selected[B3_MAX_MANIFOLD_POINTS];
		int selectedCount = b3Voxel_reduceCluster( contacts, members, memberCount, B3_MAX_MANIFOLD_POINTS, selected );
		B3_ASSERT( selectedCount > 0 );

		b3Manifold* manifold = contact->manifolds + clusterIndex;
		manifold->normal = b3MulMV( matrixA, b3Neg( clusterNormal[clusterIndex] ) );

		// Keep an exact owner sticky while that patch remains in the cluster.
		// This preserves the cluster-level impulses without ever falling back to
		// normal-only identity. If every old owner disappeared, canonical key
		// order supplies the deterministic replacement.
		const b3VoxelPatchManifold* owner = collision->manifolds + patchOfContact[members[0]];
		int stickyOldIndex = B3_NULL_INDEX;
		for ( int oldIndex = 0; oldIndex < oldCount && stickyOldIndex == B3_NULL_INDEX; ++oldIndex )
		{
			if ( oldConsumed[oldIndex] || b3Dot( oldManifolds[oldIndex].normal, manifold->normal ) <= B3_VOXEL_NORMAL_MATCH )
				continue;
			for ( int memberIndex = 0; memberIndex < memberCount; ++memberIndex )
			{
				const b3VoxelPatchManifold* candidate =
					collision->manifolds + patchOfContact[members[memberIndex]];
				if ( oldStates[oldIndex].kind == candidate->kind &&
					 b3VoxelPatchKeyEqual( &oldStates[oldIndex].key, &candidate->key ) )
				{
					owner = candidate;
					stickyOldIndex = oldIndex;
					break;
				}
			}
		}
	b3VoxelManifoldState* state = workspace->states.data + clusterIndex;
	memset( state, 0, sizeof( *state ) );
	state->key = owner->key;
	state->kind = owner->kind;

		manifold->pointCount = selectedCount;
		for ( int j = 0; j < selectedCount; ++j )
		{
			const b3VoxelContact* point = contacts + selected[j];
			const b3VoxelPatchManifold* pointPatch = collision->manifolds + patchOfContact[selected[j]];
			state->pointKeys[j] = pointPatch->key;
			state->pointKinds[j] = pointPatch->kind;
			b3ManifoldPoint* mp = manifold->points + j;
			b3Vec3 anchor = b3MulMV( matrixA, b3MulSV( 0.5f, b3Add( point->body0Point, point->body1Point ) ) );
			mp->anchorA = anchor;
			mp->anchorB = b3Add( anchor, offsetAB );
			mp->separation = -point->initialPenetration;
			mp->featureId = point->featureId;
			mp->triangleIndex = B3_NULL_INDEX;
		}
		if ( counters != NULL )
			counters->emittedPoints += selectedCount;

		int oldIndex = stickyOldIndex;
		for ( int j = 0; oldIndex == B3_NULL_INDEX && j < oldCount; ++j )
		{
			if ( !oldConsumed[j] && oldStates[j].kind == state->kind &&
				 b3VoxelPatchKeyEqual( &oldStates[j].key, &state->key ) &&
				 b3Dot( oldManifolds[j].normal, manifold->normal ) > B3_VOXEL_NORMAL_MATCH )
			{
				oldIndex = j;
				break;
			}
		}
		if ( oldIndex == B3_NULL_INDEX )
		{
			if ( counters != NULL )
				counters->workspaceKeyMisses += 1;
		}
		else
		{
			if ( counters != NULL )
				counters->workspaceKeyHits += 1;
			oldConsumed[oldIndex] = true;
			const b3Manifold* old = oldManifolds + oldIndex;
			manifold->frictionImpulse = old->frictionImpulse;
			manifold->rollingImpulse = old->rollingImpulse;
			manifold->twistImpulse = old->twistImpulse;
		}

		// Point ownership is finer than the normal cluster owner. Match every
		// point by its own exact patch key before consulting its compact feature
		// id and anchor guard, even if the cluster owner itself changed.
		for ( int j = 0; j < manifold->pointCount; ++j )
		{
			b3ManifoldPoint* mp = manifold->points + j;
			bool matched = false;
			for ( int oldManifoldIndex = 0; oldManifoldIndex < oldCount && !matched; ++oldManifoldIndex )
			{
				const b3Manifold* old = oldManifolds + oldManifoldIndex;
				if ( b3Dot( old->normal, manifold->normal ) <= B3_VOXEL_NORMAL_MATCH )
					continue;
				for ( int k = 0; k < old->pointCount; ++k )
				{
					if ( oldPointClaimed[oldManifoldIndex][k] || old->points[k].featureId != mp->featureId ||
						 oldStates[oldManifoldIndex].pointKinds[k] != state->pointKinds[j] ||
						 !b3VoxelPatchKeyEqual( oldStates[oldManifoldIndex].pointKeys + k, state->pointKeys + j ) )
						continue;
					b3Vec3 anchorDelta = b3Sub( old->points[k].anchorA, mp->anchorA );
					float recycleDistance = B3_CONTACT_RECYCLE_DISTANCE;
					if ( b3Dot( anchorDelta, anchorDelta ) > recycleDistance * recycleDistance )
					{
						if ( counters != NULL )
							counters->featureRemapRejects += 1;
						continue;
					}
					mp->normalImpulse = old->points[k].normalImpulse;
					mp->persisted = true;
					oldPointClaimed[oldManifoldIndex][k] = true;
					matched = true;
					if ( counters != NULL )
					{
						counters->persistedPoints += 1;
						counters->exactPatchPersistedPoints += 1;
					}
					break;
				}
			}
		}
	}

	const b3SurfaceMaterial* materialA = b3GetShapeMaterials( shapeA );
	const b3SurfaceMaterial* materialB = b3GetShapeMaterials( shapeB );
	contact->friction =
		world->frictionCallback( materialA->friction, materialA->userMaterialId, materialB->friction, materialB->userMaterialId );
	contact->restitution = world->restitutionCallback( materialA->restitution, materialA->userMaterialId, materialB->restitution,
											   materialB->userMaterialId );
	contact->rollingResistance = 0.0f;
	contact->tangentVelocity =
		b3Sub( b3RotateVector( xfA.q, materialA->tangentVelocity ), b3RotateVector( xfB.q, materialB->tangentVelocity ) );
	return true;
}

static uint32_t b3Voxel_cellId( b3Vec3i c )
{
	uint32_t h = (uint32_t)c.x * 73856093u ^ (uint32_t)c.y * 19349663u ^ (uint32_t)c.z * 83492791u;
	h ^= h >> 16;
	h *= 0x7feb352du;
	h ^= h >> 15;
	h *= 0x846ca68bu;
	h ^= h >> 16;
	return h ? h : 1u;
}

typedef struct b3VoxelConvexPatchAlias
{
	struct b3VoxelConvexPatchAlias* next;
	b3Vec3i cell;
	uint8_t exposed;
} b3VoxelConvexPatchAlias;

// A voxel/convex canonical key has only one patch. Within one query its domain
// sentinels are fixed, so the exact identity is the exposure mask plus the cell
// coordinate on axes that expose either side. Keeping the general two-patch
// key here used 48 bytes and compared a permanently empty second half.
typedef struct b3VoxelConvexPatchKey
{
	b3Vec3i coordinate;
	uint8_t exposed;
} b3VoxelConvexPatchKey;

typedef struct b3VoxelConvexPatchEntry
{
	b3VoxelConvexPatchKey key;
	b3VoxelContact points[4];
	b3Vec3 gridPoints[4];
	b3VoxelConvexPatchAlias* firstAlias;
	b3VoxelConvexPatchAlias* lastAlias;
	int pointCount;
	int representedAliases;
	uint8_t selectedMask;
	bool separated;
} b3VoxelConvexPatchEntry;

typedef struct b3VoxelConvexPatchTable
{
	b3Arena* arena;
	b3VoxelConvexPatchEntry* entries;
	int count;
	int capacity;
	int growthCount;
	int scratchBytes;
} b3VoxelConvexPatchTable;

static b3VoxelConvexPatchKey b3Voxel_makeConvexPatchKey( b3Vec3i cell, uint8_t exposed )
{
	exposed &= 0x3Fu;
	b3VoxelConvexPatchKey key = { .exposed = exposed };
	if ( ( exposed & 0x03u ) != 0 )
		key.coordinate.x = cell.x;
	if ( ( exposed & 0x0Cu ) != 0 )
		key.coordinate.y = cell.y;
	if ( ( exposed & 0x30u ) != 0 )
		key.coordinate.z = cell.z;
	return key;
}

static bool b3Voxel_convexPatchKeyEqual( b3VoxelConvexPatchKey a, b3VoxelConvexPatchKey b )
{
	return a.exposed == b.exposed && a.coordinate.x == b.coordinate.x && a.coordinate.y == b.coordinate.y &&
		   a.coordinate.z == b.coordinate.z;
}

static void b3Voxel_initConvexPatchTable( b3VoxelConvexPatchTable* table, b3Arena* arena )
{
	memset( table, 0, sizeof( *table ) );
	table->arena = arena;
}

static void b3Voxel_growConvexPatchTable( b3VoxelConvexPatchTable* table )
{
	B3_ASSERT( table->capacity <= INT_MAX / 2 );
	int newCapacity = table->capacity == 0 ? 16 : 2 * table->capacity;
	b3VoxelConvexPatchEntry* entries = b3Bump( table->arena, newCapacity * (int)sizeof( b3VoxelConvexPatchEntry ) );
	table->growthCount += 1;
	table->scratchBytes += newCapacity * (int)sizeof( b3VoxelConvexPatchEntry );
	if ( table->count > 0 )
		memcpy( entries, table->entries, (size_t)table->count * sizeof( b3VoxelConvexPatchEntry ) );
	table->entries = entries;
	table->capacity = newCapacity;
}

static int b3Voxel_findOrInsertConvexPatch( b3VoxelConvexPatchTable* table, b3Vec3i cell, uint8_t exposed,
										bool* inserted )
{
	b3VoxelConvexPatchKey key = b3Voxel_makeConvexPatchKey( cell, exposed );
	for ( int keyIndex = 0; keyIndex < table->count; ++keyIndex )
	{
		if ( b3Voxel_convexPatchKeyEqual( table->entries[keyIndex].key, key ) )
		{
			*inserted = false;
			return keyIndex;
		}
	}

	if ( table->count == table->capacity )
		b3Voxel_growConvexPatchTable( table );

	int entryIndex = table->count++;
	b3VoxelConvexPatchEntry* entry = table->entries + entryIndex;
	memset( entry, 0, sizeof( *entry ) );
	entry->key = key;
	*inserted = true;
	return entryIndex;
}

typedef struct b3VoxelConvexQueryContext
{
	const b3VoxelData* voxels;
	const b3Shape* convex;
	b3Transform transformBtoA;
	float contactDistance;
	float halfVoxel;
	b3BoxHull cellHull;
	bool useBoxObb;
	bool convexObbPrepared;
	bool obbPairPrepared;
	b3VoxelOBB cellObb;
	b3VoxelOBB convexObb;
	b3ObbPairContext obbPairContext;
	b3Vec3 origin;
	b3Vec3i domainMin;
	b3Vec3i domainMax;
	float voxelSize;
	b3VoxelConvexPatchTable patchTable;
	bool patchTablePrepared;
	int patchScratchBytes;
	b3SATCache satCache;
	b3VoxelContactReducer reducer;
	b3VoxelContact fallback;
	bool hasFallback;
	b3Vec3i fallbackSeedCell;
	float fallbackSeedPenetration;
	uint32_t fallbackSeedFeatureId;
	bool hasFallbackSeed;
	bool useBoxPatches;
	bool replayExact;
	b3VoxelCounters* counters;
} b3VoxelConvexQueryContext;

static bool b3Voxel_makeBoxObb( const b3HullData* hull, b3Transform transform, bool certifiedBox, b3VoxelOBB* obb )
{
	// The compact b3BoxHull layout can also be produced by the generic hull
	// builder. Require the canonical box topology before interpreting its point
	// and plane indices as the fixed +/-X, +/-Y, +/-Z box ordering.
	if ( !certifiedBox && !b3IsBoxHull( hull ) )
	{
		return false;
	}

	const b3Plane* planes = b3GetHullPlanes( hull );
	const b3Vec3* points = b3GetHullPoints( hull );
	b3Vec3 axes[3] = { planes[1].normal, planes[3].normal, planes[5].normal };
	b3Vec3 fromCenter = b3Sub( points[0], hull->center );
	b3Vec3 half = { b3Dot( axes[0], fromCenter ), b3Dot( axes[1], fromCenter ), b3Dot( axes[2], fromCenter ) };

	obb->center = b3TransformPoint( transform, hull->center );
	obb->half = half;
	for ( int i = 0; i < 3; ++i )
	{
		obb->axes[i] = b3RotateVector( transform.q, axes[i] );
	}
	return true;
}

static void b3Voxel_placeCellHull( b3BoxHull* hull, b3Vec3 center, float h )
{
	hull->base.center = center;
	hull->base.aabb = (b3AABB){ { center.x - h, center.y - h, center.z - h }, { center.x + h, center.y + h, center.z + h } };

	hull->boxPlanes[0].offset = h - center.x;
	hull->boxPlanes[1].offset = h + center.x;
	hull->boxPlanes[2].offset = h - center.y;
	hull->boxPlanes[3].offset = h + center.y;
	hull->boxPlanes[4].offset = h - center.z;
	hull->boxPlanes[5].offset = h + center.z;

	hull->boxPoints[0] = (b3Vec3){ h + center.x, h + center.y, h + center.z };
	hull->boxPoints[1] = (b3Vec3){ -h + center.x, h + center.y, h + center.z };
	hull->boxPoints[2] = (b3Vec3){ -h + center.x, -h + center.y, h + center.z };
	hull->boxPoints[3] = (b3Vec3){ h + center.x, -h + center.y, h + center.z };
	hull->boxPoints[4] = (b3Vec3){ h + center.x, h + center.y, -h + center.z };
	hull->boxPoints[5] = (b3Vec3){ -h + center.x, h + center.y, -h + center.z };
	hull->boxPoints[6] = (b3Vec3){ -h + center.x, -h + center.y, -h + center.z };
	hull->boxPoints[7] = (b3Vec3){ h + center.x, -h + center.y, -h + center.z };
	for ( int i = 0; i < 8; ++i )
	{
		hull->vx[i] = hull->boxPoints[i].x;
		hull->vy[i] = hull->boxPoints[i].y;
		hull->vz[i] = hull->boxPoints[i].z;
	}
}

static b3Vec3 b3Voxel_getConvexSupport( const b3VoxelConvexQueryContext* context, b3Vec3 direction )
{
	const b3Shape* convex = context->convex;
	switch ( convex->type )
	{
		case b3_sphereShape:
		{
			b3Vec3 center = b3TransformPoint( context->transformBtoA, convex->sphere.center );
			return b3MulAdd( center, convex->sphere.radius, direction );
		}
		case b3_capsuleShape:
		{
			b3Vec3 center1 = b3TransformPoint( context->transformBtoA, convex->capsule.center1 );
			b3Vec3 center2 = b3TransformPoint( context->transformBtoA, convex->capsule.center2 );
			b3Vec3 center = b3Dot( center1, direction ) > b3Dot( center2, direction ) ? center1 : center2;
			return b3MulAdd( center, convex->capsule.radius, direction );
		}
		case b3_hullShape:
		{
			b3Vec3 localDirection = b3InvRotateVector( context->transformBtoA.q, direction );
			int index = b3FindHullSupportVertex( convex->hull, localDirection );
			return b3TransformPoint( context->transformBtoA, b3GetHullPoints( convex->hull )[index] );
		}
		default:
			return b3Vec3_zero;
	}
}

static void b3Voxel_considerConvexFallback( b3VoxelConvexQueryContext* context, uint8_t exposed, b3Vec3 cellCenter,
											uint32_t featureId )
{
	// A cellwise separating axis can point through an internal voxel seam even
	// though the convex overlaps the represented union. Build an escape contact
	// from each exposed cell face and retain the shallowest valid one. The voxel
	// witness is on that face and the convex witness is an exact support point,
	// so the signed witness relation remains geometric rather than synthetic.
	for ( int bit = 0; bit < 6; ++bit )
	{
		if ( ( exposed & ( 1u << bit ) ) == 0 )
			continue;
		int axis = bit / 2;
		float sign = ( bit & 1 ) != 0 ? 1.0f : -1.0f;
		b3Vec3 outward = b3Vec3_zero;
		( &outward.x )[axis] = sign;
		b3Vec3 normal = b3Neg( outward );
		b3Vec3 convexPoint = b3Voxel_getConvexSupport( context, normal );
		b3Vec3 voxelPoint = convexPoint;
		for ( int tangentAxis = 0; tangentAxis < 3; ++tangentAxis )
		{
			float center = ( &cellCenter.x )[tangentAxis];
			if ( tangentAxis == axis )
				( &voxelPoint.x )[tangentAxis] = center + sign * context->halfVoxel;
			else
				( &voxelPoint.x )[tangentAxis] =
					b3ClampFloat( ( &voxelPoint.x )[tangentAxis], center - context->halfVoxel, center + context->halfVoxel );
		}
		float penetration = b3Dot( b3Sub( convexPoint, voxelPoint ), normal );
		uint32_t faceFeatureId = featureId ^ ( (uint32_t)( bit + 1 ) * 0x9E3779B9u );
		if ( penetration > 0.0f &&
			 ( !context->hasFallback || penetration < context->fallback.initialPenetration ||
			   ( penetration == context->fallback.initialPenetration && faceFeatureId < context->fallback.featureId ) ) )
		{
			context->fallback = (b3VoxelContact){
				.normal = normal,
				.initialPenetration = penetration,
				.penetrationDepth = penetration,
				.body0Point = voxelPoint,
				.body1Point = convexPoint,
				.featureId = faceFeatureId,
			};
			context->hasFallback = true;
		}
	}
}

static void b3Voxel_considerConvexEscape( b3VoxelConvexQueryContext* context )
{
	int cellCount = b3Voxel_GetCellCount( context->voxels );
	for ( int bit = 0; bit < 6; ++bit )
	{
		int axis = bit / 2;
		int direction = ( bit & 1 ) != 0 ? 1 : -1;
		b3Vec3i boundaryCell = context->fallbackSeedCell;
		for ( int step = 0; step < cellCount; ++step )
		{
			b3Vec3i next = boundaryCell;
			int* coordinate = &( &next.x )[axis];
			if ( ( direction < 0 && *coordinate == INT_MIN ) || ( direction > 0 && *coordinate == INT_MAX ) )
			{
				break;
			}
			*coordinate += direction;
			if ( !b3VoxelData_IsSolid( context->voxels, next ) )
			{
				b3Vec3 center = b3Voxel_GetCellCenter( context->voxels, boundaryCell );
				uint32_t featureId = context->fallbackSeedFeatureId ^ b3Voxel_cellId( boundaryCell );
				b3Voxel_considerConvexFallback( context, (uint8_t)( 1u << bit ), center, featureId );
				break;
			}
			boundaryCell = next;
		}
	}
}

static bool b3Voxel_isOnExposedCellFace( const b3VoxelConvexQueryContext* context, uint8_t exposed, b3Vec3 cellCenter,
										 b3Vec3 point )
{
	float tolerance = 2.0e-4f * ( 1.0f + context->halfVoxel );
	b3Vec3 offset = b3Sub( point, cellCenter );
	return b3VoxelPointOnExposedCubeFace( offset, context->halfVoxel, exposed, tolerance );
}

static bool b3Voxel_isGridPointOnExposedCellFace( const b3VoxelConvexQueryContext* context, uint8_t exposed,
												b3Vec3i cell, b3Vec3 gridPoint )
{
	float tolerance = 2.0e-4f * ( 1.0f + context->halfVoxel );
	b3Vec3 offset = {
		gridPoint.x - (float)cell.x * context->voxelSize,
		gridPoint.y - (float)cell.y * context->voxelSize,
		gridPoint.z - (float)cell.z * context->voxelSize,
	};
	return b3VoxelPointOnExposedCubeFace( offset, context->halfVoxel, exposed, tolerance );
}

static uint32_t b3Voxel_convexPatchFeatureId( uint64_t hash, int pointIndex )
{
	uint32_t feature = (uint32_t)hash ^ (uint32_t)( hash >> 32 ) ^ (uint32_t)( pointIndex + 1 ) * 0x9e3779b1u;
	return feature != 0 ? feature : 1u;
}

static void b3Voxel_buildConvexPatchEntry( b3VoxelConvexQueryContext* context, int entryIndex,
										 const b3VoxelPatchKey* key )
{
	b3VoxelConvexPatchEntry* entry = context->patchTable.entries + entryIndex;

	b3Vec3 lower;
	b3Vec3 upper;
	for ( int axis = 0; axis < 3; ++axis )
	{
		( &lower.x )[axis] = ( &context->origin.x )[axis] +
							 context->voxelSize * (float)( &key->patch0.lower.x )[axis];
		( &upper.x )[axis] = ( &context->origin.x )[axis] +
							 context->voxelSize * (float)( &key->patch0.upper.x )[axis];
	}
	b3VoxelOBB pseudo = context->cellObb;
	pseudo.center = b3MulSV( 0.5f, b3Add( lower, upper ) );
	b3Vec3 cellHalf = { context->halfVoxel, context->halfVoxel, context->halfVoxel };
	pseudo.half = b3Add( b3MulSV( 0.5f, b3Sub( upper, lower ) ), cellHalf );
	if ( pseudo.half.x <= 0.0f || pseudo.half.y <= 0.0f || pseudo.half.z <= 0.0f || !b3IsValidVec3( pseudo.half ) )
	{
		entry->separated = true;
		return;
	}

	b3ObbPairContext pairContext;
	b3VoxelPrepareAxisAlignedOBBPair( &pairContext, pseudo.half, &context->convexObb );
	if ( context->counters != NULL )
	{
		context->counters->pseudoSatCalls += 1;
	}
	b3ObbSat sat;
	if ( !b3VoxelComputeOBBSat( b3Sub( pseudo.center, context->convexObb.center ), &pairContext,
									context->contactDistance, &sat ) )
	{
		entry->separated = true;
		if ( context->counters != NULL )
		{
			context->counters->pseudoSeparatedKeys += 1;
		}
		return;
	}

	entry->pointCount =
		b3VoxelManifoldFromSat( &pseudo, &context->convexObb, &sat, context->contactDistance, 4, entry->points );
	uint64_t featureHash = b3VoxelPatchKeyHash( key );
	for ( int pointIndex = 0; pointIndex < entry->pointCount; ++pointIndex )
	{
		entry->gridPoints[pointIndex] = b3Sub( entry->points[pointIndex].body0Point, context->origin );
		entry->points[pointIndex].featureId = b3Voxel_convexPatchFeatureId( featureHash, pointIndex );
	}
}

static void b3Voxel_selectConvexPatchAlias( b3VoxelConvexQueryContext* context, b3VoxelConvexPatchEntry* entry,
										b3Vec3i cell, uint8_t exposed )
{
	entry->representedAliases += 1;
	if ( context->counters != NULL )
	{
		context->counters->representedLeafPairs += 1;
	}
	if ( entry->separated )
		return;

	b3VoxelOBB realCell;
	bool hasRealCell = false;
	for ( int pointIndex = 0; pointIndex < entry->pointCount; ++pointIndex )
	{
		uint8_t bit = (uint8_t)( 1u << pointIndex );
		if ( ( entry->selectedMask & bit ) != 0 )
			continue;
		const b3VoxelContact* point = entry->points + pointIndex;
		if ( !b3Voxel_ExposureMaskContains( exposed, b3Neg( point->normal ) ) ||
			 !b3Voxel_isGridPointOnExposedCellFace( context, exposed, cell, entry->gridPoints[pointIndex] ) )
		{
			if ( context->counters != NULL )
				context->counters->pseudoWitnessRejects += 1;
			continue;
		}
		if ( point->initialPenetration > 0.0f )
		{
			if ( !hasRealCell )
			{
				realCell = context->cellObb;
				realCell.center = b3Voxel_GetCellCenter( context->voxels, cell );
				hasRealCell = true;
			}
			b3Vec3 support0 = b3VoxelObbSupport( &realCell, b3Neg( point->normal ) );
			b3Vec3 support1 = b3VoxelObbSupport( &context->convexObb, point->normal );
			float realPenetration = b3Dot( b3Sub( support1, support0 ), point->normal );
			float tolerance = 16.0f * FLT_EPSILON * ( 1.0f + fabsf( point->initialPenetration ) );
			if ( realPenetration + tolerance < point->initialPenetration )
			{
				if ( context->counters != NULL )
					context->counters->pseudoDepthRejects += 1;
				continue;
			}
		}
		entry->selectedMask |= bit;
	}
}

static void b3Voxel_addConvexPatchAlias( b3VoxelConvexQueryContext* context, b3Vec3i cell, uint8_t exposed )
{
	if ( !context->patchTablePrepared )
	{
		context->origin = b3Voxel_GetOrigin( context->voxels );
		bool hasDomain = b3Voxel_GetDomain( context->voxels, &context->domainMin, &context->domainMax );
		B3_ASSERT( hasDomain );
		B3_UNUSED( hasDomain );
		b3Voxel_initConvexPatchTable( &context->patchTable, context->patchTable.arena );
		context->patchTablePrepared = true;
	}
	bool inserted;
	int entryIndex = b3Voxel_findOrInsertConvexPatch( &context->patchTable, cell, exposed, &inserted );
	if ( context->counters != NULL )
	{
		context->counters->patchVisits += 1;
		context->counters->patchEligibleVisits += 1;
		context->counters->patchDuplicateVisits += !inserted;
	}
	if ( inserted )
	{
		b3VoxelPatchKey key = {
			.patch0 = b3VoxelCanonicalPatchRange( cell, exposed, context->domainMin, context->domainMax ),
		};
		b3Voxel_buildConvexPatchEntry( context, entryIndex, &key );
	}

	b3VoxelConvexPatchEntry* entry = context->patchTable.entries + entryIndex;
	b3Voxel_selectConvexPatchAlias( context, entry, cell, exposed );
	if ( entry->selectedMask != 0 )
	{
		// Aliases are retained solely for exact replay when no pseudo-cuboid
		// witness belongs to a real cell. Once any point is selected, this key
		// can never take that fallback, so avoid per-cell arena traffic.
		return;
	}
	b3VoxelConvexPatchAlias* alias = b3Bump( context->patchTable.arena, sizeof( b3VoxelConvexPatchAlias ) );
	*alias = (b3VoxelConvexPatchAlias){ .cell = cell, .exposed = exposed };
	if ( entry->lastAlias != NULL )
		entry->lastAlias->next = alias;
	else
		entry->firstAlias = alias;
	entry->lastAlias = alias;
	context->patchScratchBytes += sizeof( b3VoxelConvexPatchAlias );
}

static void b3Voxel_addConvexCellContact( b3VoxelConvexQueryContext* context, b3Vec3i cell, uint8_t exposed,
											  b3Vec3 cellCenter, b3VoxelContact contact )
{
	if ( context->counters != NULL )
	{
		context->counters->rawContactPoints += 1;
	}

	// Contact normals point from the convex toward the voxel cell. Exposure
	// masks use the opposite, outward direction from the cell toward the convex.
	if ( !b3Voxel_ExposureMaskContains( exposed, b3Neg( contact.normal ) ) ||
		 !b3Voxel_isOnExposedCellFace( context, exposed, cellCenter, contact.body0Point ) )
	{
		if ( context->counters != NULL )
		{
			context->counters->surfaceRejects += 1;
		}
		// A convex embedded in solid occupancy, or a deep overlap whose cellwise
		// minimum axis crosses an internal seam, can have no accepted raw point.
		if ( contact.initialPenetration > 0.0f )
		{
			b3Voxel_considerConvexFallback( context, exposed, cellCenter, contact.featureId );
			if ( !context->hasFallbackSeed || contact.initialPenetration > context->fallbackSeedPenetration ||
				 ( contact.initialPenetration == context->fallbackSeedPenetration &&
				   contact.featureId < context->fallbackSeedFeatureId ) )
			{
				context->fallbackSeedCell = cell;
				context->fallbackSeedPenetration = contact.initialPenetration;
				context->fallbackSeedFeatureId = contact.featureId;
				context->hasFallbackSeed = true;
			}
		}
		return;
	}
	b3VoxelReducer_Add( &context->reducer, &contact );
}

static bool b3Voxel_collideConvexCell( b3Vec3i cell, uint8_t exposed, void* rawContext )
{
	b3VoxelConvexQueryContext* context = rawContext;
	if ( context->counters != NULL )
	{
		context->counters->convexLeafTests += 1;
	}
	if ( context->useBoxObb && !context->convexObbPrepared )
	{
		bool valid = b3Voxel_makeBoxObb( context->convex->hull, context->transformBtoA, true, &context->convexObb );
		B3_ASSERT( valid );
		B3_UNUSED( valid );
		context->convexObbPrepared = true;
	}
	if ( context->useBoxPatches && !context->replayExact && exposed != 0 )
	{
		b3Voxel_addConvexPatchAlias( context, cell, exposed );
		return true;
	}
	b3Vec3 center = b3Voxel_GetCellCenter( context->voxels, cell );
	if ( context->useBoxObb )
	{
		if ( !context->obbPairPrepared )
		{
			b3VoxelPrepareAxisAlignedOBBPair( &context->obbPairContext, context->cellObb.half, &context->convexObb );
			context->obbPairPrepared = true;
		}
		b3VoxelOBB cellObb = context->cellObb;
		cellObb.center = center;
		b3VoxelContact contacts[4];
		if ( context->counters != NULL )
		{
			context->counters->obbTests += 1;
		}
		b3ObbSat sat;
		if ( !b3VoxelComputeOBBSat( b3Sub( cellObb.center, context->convexObb.center ), &context->obbPairContext,
									context->contactDistance, &sat ) )
		{
			return true;
		}
		int count = b3VoxelManifoldFromSat( &cellObb, &context->convexObb, &sat, context->contactDistance, 1, contacts );
		uint32_t cellId = b3Voxel_cellId( cell );
		if ( !b3Voxel_ExposureMaskContains( exposed, b3Neg( contacts[0].normal ) ) ||
			 !b3Voxel_isOnExposedCellFace( context, exposed, center, contacts[0].body0Point ) )
		{
			// Preserve the exact deepest-point fallback for a convex embedded in
			// solid occupancy. Reuse the SAT result, but fully clip rejected faces.
			count = b3VoxelManifoldFromSat( &cellObb, &context->convexObb, &sat, context->contactDistance, 4, contacts );
			for ( int k = 0; k < count; ++k )
			{
				contacts[k].featureId = cellId ^ ( (uint32_t)( k + 1 ) * 2654435761u );
				b3Voxel_addConvexCellContact( context, cell, exposed, center, contacts[k] );
			}
			return true;
		}

		int clusterIndex;
		if ( !b3VoxelReducer_ShouldClip( &context->reducer, contacts, &clusterIndex ) )
		{
			// The one-point manifold was already checked against the represented
			// union, and ShouldClip already found its normal cluster. Avoid
			// repeating both tests in the generic per-point admission path.
			contacts[0].featureId = cellId ^ 2654435761u;
			if ( context->counters != NULL )
			{
				context->counters->rawContactPoints += 1;
			}
			b3VoxelReducer_AddToCluster( &context->reducer, contacts, clusterIndex );
			return true;
		}

		count = b3VoxelManifoldFromSat( &cellObb, &context->convexObb, &sat, context->contactDistance, 4, contacts );
		for ( int k = 0; k < count; ++k )
		{
			contacts[k].featureId = cellId ^ ( (uint32_t)( k + 1 ) * 2654435761u );
			b3Voxel_addConvexCellContact( context, cell, exposed, center, contacts[k] );
		}
		return true;
	}
	b3Voxel_placeCellHull( &context->cellHull, center, context->halfVoxel );

	b3LocalManifoldPoint pts[8];
	b3LocalManifold m = { 0 };
	m.points = pts;

	switch ( context->convex->type )
	{
		case b3_sphereShape:
		{
			b3SimplexCache sc = { 0 };
			b3CollideHullAndSphere( &m, 8, &context->cellHull.base, &context->convex->sphere, context->transformBtoA, &sc );
			break;
		}
		case b3_capsuleShape:
		{
			b3SimplexCache sc = { 0 };
			b3CollideHullAndCapsule( &m, 8, &context->cellHull.base, &context->convex->capsule, context->transformBtoA, &sc );
			break;
		}
		case b3_hullShape:
		{
			b3SATCache* satCache = &context->satCache;
			b3CollideHulls( &m, 8, &context->cellHull.base, context->convex->hull, context->transformBtoA, satCache );
			if ( context->counters != NULL )
			{
				context->counters->hullSatCalls += 1;
				context->counters->hullSatCacheHits += satCache->hit;
			}
			if ( m.pointCount > 0 )
			{
				// Reusing a separating feature is exact, but Box3D's temporal
				// contact reconstruction may choose a different valid manifold for
				// an adjacent cell. Regenerate cached contacts from a cold SAT query
				// so this fast path cannot change the collider's deterministic output.
				if ( satCache->hit )
				{
					b3SATCache exact = { 0 };
					m.pointCount = 0;
					b3CollideHulls( &m, 8, &context->cellHull.base, context->convex->hull, context->transformBtoA, &exact );
					if ( context->counters != NULL )
					{
						context->counters->hullSatCalls += 1;
					}
				}
				*satCache = (b3SATCache){ 0 };
			}
			break;
		}
		default:
			return true;
	}

	uint32_t cellId = 0;
	bool hasCellId = false;
	for ( int k = 0; k < m.pointCount; ++k )
	{
		if ( m.points[k].separation >= context->contactDistance )
			continue;

		b3VoxelContact c;
		c.normal = b3Neg( m.normal );
		c.initialPenetration = -m.points[k].separation;
		c.penetrationDepth = c.initialPenetration > 0.0f ? c.initialPenetration : 0.0f;
		b3Vec3 halfSeparation = b3MulSV( 0.5f * m.points[k].separation, m.normal );
		c.body0Point = b3Sub( m.points[k].point, halfSeparation );
		c.body1Point = b3Add( m.points[k].point, halfSeparation );
		if ( !hasCellId )
		{
			cellId = b3Voxel_cellId( cell );
			hasCellId = true;
		}
		c.featureId = cellId ^ ( b3MakeFeatureId( m.points[k].pair ) * 2654435761u );
		b3Voxel_addConvexCellContact( context, cell, exposed, center, c );
	}
	return true;
}

static void b3Voxel_finishConvexPatches( b3VoxelConvexQueryContext* context )
{
	if ( !context->useBoxPatches || !context->patchTablePrepared )
		return;

	if ( context->counters != NULL )
	{
		context->counters->patchUniqueKeys += context->patchTable.count;
		context->counters->patchEligibleUniqueKeys += context->patchTable.count;
		context->counters->patchMaxUniqueKeys =
			b3MaxInt( context->counters->patchMaxUniqueKeys, context->patchTable.count );
	}

	for ( int entryIndex = 0; entryIndex < context->patchTable.count; ++entryIndex )
	{
		b3VoxelConvexPatchEntry* entry = context->patchTable.entries + entryIndex;
		if ( context->counters != NULL )
		{
			context->counters->patchMaxMultiplicity =
				b3MaxInt( context->counters->patchMaxMultiplicity, entry->representedAliases );
			int bucket = entry->representedAliases == 1 ? 0 : entry->representedAliases <= 3 ? 1 :
					 entry->representedAliases <= 7   ? 2 : entry->representedAliases <= 15 ? 3 :
					 entry->representedAliases <= 31  ? 4 : 5;
			context->counters->patchMultiplicityCounts[bucket] += 1;
		}
		if ( entry->separated )
			continue;

		if ( entry->selectedMask != 0 )
		{
			if ( context->counters != NULL )
			{
				context->counters->selectedPatchKeys += 1;
			}
			for ( int pointIndex = 0; pointIndex < entry->pointCount; ++pointIndex )
			{
				if ( ( entry->selectedMask & (uint8_t)( 1u << pointIndex ) ) == 0 )
					continue;
				if ( context->counters != NULL )
				{
					context->counters->rawContactPoints += 1;
				}
				b3VoxelReducer_Add( &context->reducer, entry->points + pointIndex );
			}
			continue;
		}

		if ( context->counters != NULL )
		{
			context->counters->emptyPatchFallbackKeys += 1;
		}
		context->replayExact = true;
		for ( b3VoxelConvexPatchAlias* alias = entry->firstAlias; alias != NULL; alias = alias->next )
		{
			if ( context->counters != NULL )
			{
				context->counters->patchLeafFallbackTests += 1;
			}
			b3Voxel_collideConvexCell( alias->cell, alias->exposed, context );
		}
		context->replayExact = false;
	}

	if ( context->counters != NULL )
	{
		context->counters->patchTableGrowths += context->patchTable.growthCount;
		int scratchBytes = context->patchTable.scratchBytes + context->patchScratchBytes;
		context->counters->patchScratchPeakBytes = b3MaxInt( context->counters->patchScratchPeakBytes, scratchBytes );
	}
}

static int b3VoxelCollideConvexImpl( const b3VoxelData* v, const b3Shape* convex, b3Transform btoa, float contactDistance,
									int maxContacts, b3VoxelContact* out, b3Arena* arena, b3VoxelCounters* counters,
									bool forceBoxPatches )
{
	if ( counters != NULL )
	{
		counters->voxelConvexCalls += 1;
	}
	float vs = b3Voxel_GetVoxelSize( v );
	float h = 0.5f * vs;

	b3Vec3 lo, hi;
	if ( convex->type == b3_sphereShape )
	{
		b3Vec3 c = b3TransformPoint( btoa, convex->sphere.center );
		float r = convex->sphere.radius;
		lo = (b3Vec3){ c.x - r, c.y - r, c.z - r };
		hi = (b3Vec3){ c.x + r, c.y + r, c.z + r };
	}
	else if ( convex->type == b3_capsuleShape )
	{
		b3Vec3 c1 = b3TransformPoint( btoa, convex->capsule.center1 );
		b3Vec3 c2 = b3TransformPoint( btoa, convex->capsule.center2 );
		float r = convex->capsule.radius;
		lo = (b3Vec3){ b3MinFloat( c1.x, c2.x ) - r, b3MinFloat( c1.y, c2.y ) - r, b3MinFloat( c1.z, c2.z ) - r };
		hi = (b3Vec3){ b3MaxFloat( c1.x, c2.x ) + r, b3MaxFloat( c1.y, c2.y ) + r, b3MaxFloat( c1.z, c2.z ) + r };
	}
	else
	{
		b3AABB hb = b3AABB_Transform( btoa, convex->hull->aabb );
		lo = hb.lowerBound;
		hi = hb.upperBound;
	}
	float cd = contactDistance;
	b3AABB query = { { lo.x - cd, lo.y - cd, lo.z - cd }, { hi.x + cd, hi.y + cd, hi.z + cd } };

	// Do not aggregate-initialize this context. Its reducer owns a large array
	// of contact clusters whose inactive slots never need initialization; a
	// zeroed aggregate made every voxel-convex pair clear the entire array.
	b3VoxelConvexQueryContext context;
	context.voxels = v;
	context.convex = convex;
	context.transformBtoA = btoa;
	context.contactDistance = contactDistance;
	context.halfVoxel = h;
	context.voxelSize = vs;
	context.useBoxObb = false;
	context.convexObbPrepared = false;
	context.obbPairPrepared = false;
	context.useBoxPatches = false;
	context.replayExact = false;
	context.patchTablePrepared = false;
	context.patchScratchBytes = 0;
	context.satCache = (b3SATCache){ 0 };
	context.hasFallback = false;
	context.hasFallbackSeed = false;
	context.counters = counters;
	if ( convex->type == b3_hullShape )
	{
		context.cellObb = (b3VoxelOBB){
			.axes = { b3Vec3_axisX, b3Vec3_axisY, b3Vec3_axisZ },
			.half = { h, h, h },
		};
		bool certifiedBox = ( convex->flags & b3_boxHull ) != 0;
		if ( certifiedBox )
		{
			context.useBoxObb = true;
		}
		else
		{
			context.useBoxObb = b3Voxel_makeBoxObb( convex->hull, btoa, false, &context.convexObb );
			context.convexObbPrepared = context.useBoxObb;
		}
		if ( context.useBoxObb )
		{
			context.useBoxPatches = arena != NULL && certifiedBox &&
									( forceBoxPatches || b3Voxel_GetCellCount( v ) >= 512 );
			if ( context.useBoxPatches )
			{
				// Preserve the arena pointer without initializing the table until
				// the first exposed candidate actually enters the patch path.
				context.patchTable.arena = arena;
			}
		}
	}
	if ( !context.useBoxObb )
	{
		context.cellHull = b3MakeBoxHullForCollision( h, h, h );
	}
	b3VoxelReducer_Init( &context.reducer, maxContacts, counters );
	b3Voxel_ForEachCellTracked( v, query, b3Voxel_collideConvexCell, &context, counters );
	b3Voxel_finishConvexPatches( &context );
	if ( context.reducer.totalPointCount == 0 && !context.hasFallback && context.hasFallbackSeed )
	{
		b3Voxel_considerConvexEscape( &context );
	}
	if ( context.reducer.totalPointCount == 0 && context.hasFallback )
	{
		b3VoxelReducer_Add( &context.reducer, &context.fallback );
		if ( counters != NULL )
		{
			counters->deepOverlapFallbacks += 1;
		}
	}
	return b3VoxelReducer_Finish( &context.reducer, out );
}

int b3VoxelCollideConvex( const b3VoxelData* v, const b3Shape* convex, b3Transform btoa, float contactDistance, int maxContacts,
						  b3VoxelContact* out, b3Arena* arena, b3VoxelCounters* counters )
{
	return b3VoxelCollideConvexImpl( v, convex, btoa, contactDistance, maxContacts, out, arena, counters, false );
}

int b3VoxelCollideConvexCanonicalWithArena( const b3VoxelData* v, const b3Shape* convex, b3Transform btoa,
											float contactDistance, int maxContacts, b3VoxelContact* out, b3Arena* arena,
											b3VoxelCounters* counters )
{
	B3_ASSERT( arena != NULL );
	return b3VoxelCollideConvexImpl( v, convex, btoa, contactDistance, maxContacts, out, arena, counters, true );
}

void b3CollideVoxelAndHull( b3LocalManifold* manifold, int capacity, const b3VoxelData* voxelA, const b3HullData* hullB,
							b3Transform transformBtoA )
{
	B3_ASSERT( manifold != NULL && manifold->points != NULL );
	manifold->pointCount = 0;
	if ( manifold == NULL || manifold->points == NULL || capacity <= 0 || voxelA == NULL || hullB == NULL )
		return;

	b3Shape convex = { 0 };
	convex.type = b3_hullShape;
	convex.hull = hullB;
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	int count =
		b3VoxelCollideConvex( voxelA, &convex, transformBtoA, B3_LINEAR_SLOP, B3_VOXEL_MAX_CONTACTS, contacts, NULL, NULL );
	if ( count == 0 )
		return;

	int deepest = 0;
	for ( int i = 1; i < count; ++i )
	{
		if ( contacts[i].initialPenetration > contacts[deepest].initialPenetration )
			deepest = i;
	}
	manifold->normal = b3Neg( contacts[deepest].normal );
	int pointCount = 0;
	for ( int i = 0; i < count && pointCount < capacity; ++i )
	{
		if ( b3Dot( contacts[i].normal, contacts[deepest].normal ) < B3_VOXEL_CLUSTER_DOT )
			continue;
		b3LocalManifoldPoint* point = manifold->points + pointCount;
		memset( point, 0, sizeof( *point ) );
		point->point = b3MulSV( 0.5f, b3Add( contacts[i].body0Point, contacts[i].body1Point ) );
		point->separation = -contacts[i].initialPenetration;
		pointCount += 1;
	}
	manifold->pointCount = pointCount;
}

bool b3ComputeVoxelManifolds( b3World* world, int workerIndex, b3Contact* contact, const b3Shape* shapeA, b3WorldTransform xfA,
							  const b3Shape* shapeB, b3WorldTransform xfB, b3Arena arena )
{
	b3VoxelCounters* counters = &world->taskContexts.data[workerIndex].voxelCounters;

	b3Transform transformBtoA = b3InvMulWorldTransforms( xfA, xfB );

	float contactDistance = ( contact->flags & b3_enableSpeculativePoints ) ? B3_SPECULATIVE_DISTANCE : B3_LINEAR_SLOP;

	bool touching;
	if ( shapeB->type == b3_voxelShape )
	{
		if ( b3VoxelUseCanonicalPatches( shapeA->voxel, b3Quat_identity, shapeB->voxel, transformBtoA.q ) )
		{
			b3VoxelPatchCollision collision;
			b3VoxelBuildCanonicalPatches( shapeA->voxel, b3Transform_identity, shapeB->voxel, transformBtoA,
										  contactDistance, &collision, &arena, counters );
			touching = b3Voxel_emitPatchManifolds( world, contact, &collision, xfA, xfB, shapeA, shapeB, counters );
		}
		else
		{
			b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
			int count = b3VoxelCollideLeafTracked( shapeA->voxel, b3Transform_identity, shapeB->voxel, transformBtoA,
											 contactDistance, B3_VOXEL_MAX_CONTACTS, contacts, counters );
			contact->voxelContact.states.count = 0;
			touching = b3Voxel_emitManifolds( world, contact, contacts, count, xfA, xfB, shapeA, shapeB, counters );
		}
	}
	else
	{
		b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
		int count = b3VoxelCollideConvex( shapeA->voxel, shapeB, transformBtoA, contactDistance, B3_VOXEL_MAX_CONTACTS, contacts,
										  &arena, counters );
		B3_ASSERT( contact->voxelContact.states.count == 0 );
		touching = b3Voxel_emitManifolds( world, contact, contacts, count, xfA, xfB, shapeA, shapeB, counters );
	}
	if ( touching )
	{
		counters->scalarContacts += contact->manifoldCount != 1;
		counters->singleManifoldContacts += contact->manifoldCount == 1;
	}
	return touching;
}
