// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#pragma once

#include "box3d/math_functions.h"
#include "box3d/types.h"
#include "box3d/voxel.h"

typedef struct b3Arena b3Arena;
typedef struct b3Shape b3Shape;
typedef struct b3VoxelCounters b3VoxelCounters;

typedef struct b3VoxelOBB
{
	b3Vec3 center;
	b3Vec3 axes[3]; // unit
	b3Vec3 half;
} b3VoxelOBB;

// Canonical topology and pseudo-cuboid identity for a real occupied cell.
// A range endpoint outside the occupied integer domain is a sentinel for an
// unbounded direction. The range is identity only; collision code clips the
// corresponding pseudo-cuboid against the opposing shape before use.
typedef enum b3VoxelTopology
{
	b3_voxelInterior = 0,
	b3_voxelFace = 1,
	b3_voxelEdge = 2,
	b3_voxelVertex = 3,
} b3VoxelTopology;

typedef struct b3VoxelPatchRange
{
	b3Vec3i lower;
	b3Vec3i upper;
} b3VoxelPatchRange;

typedef struct b3VoxelPatchKey
{
	b3VoxelPatchRange patch0;
	b3VoxelPatchRange patch1;
} b3VoxelPatchKey;

typedef struct b3VoxelPatchSlot
{
	uint64_t hash;
	int entryIndex;
} b3VoxelPatchSlot;

// Arena-backed exact-key table. Slots are acceleration only; keys retain
// discovery order and full equality always decides identity.
typedef struct b3VoxelPatchTable
{
	b3Arena* arena;
	b3VoxelPatchKey* keys;
	b3VoxelPatchSlot* slots;
	int count;
	int keyCapacity;
	int slotCapacity;
	int growthCount;
	int scratchBytes;
} b3VoxelPatchTable;

b3VoxelTopology b3VoxelClassifyTopology( uint8_t exposed );
b3VoxelPatchRange b3VoxelCanonicalPatchRange( b3Vec3i cell, uint8_t exposed, b3Vec3i domainMin, b3Vec3i domainMax );
b3VoxelPatchKey b3VoxelSwapPatchKey( b3VoxelPatchKey key );
uint64_t b3VoxelPatchKeyHash( const b3VoxelPatchKey* key );
uint64_t b3VoxelPatchKeyTableHash( const b3VoxelPatchKey* key );
bool b3VoxelPatchKeyEqual( const b3VoxelPatchKey* a, const b3VoxelPatchKey* b );
void b3VoxelPatchTableInit( b3VoxelPatchTable* table, b3Arena* arena );
int b3VoxelPatchTableFindOrInsert( b3VoxelPatchTable* table, const b3VoxelPatchKey* key, uint64_t hash, bool* inserted );

// Axis data shared by every cell in an OBB shape pair. Centers are supplied
// separately, so the narrow phase can prepare this once and reuse it.
typedef struct b3ObbPairAxis
{
	b3Vec3 axis;
	float radius;
	int srcBox;
	bool isEdge;
	// Face axes: reference axis then incident axis. Edge axes: the generating
	// edge on box0 then box1.
	uint8_t axis0;
	uint8_t axis1;
	uint8_t surfaceMask0[2];
	uint8_t surfaceMask1[2];
} b3ObbPairAxis;

typedef struct b3ObbPairContext
{
	b3ObbPairAxis axes[15];
	int count;
} b3ObbPairContext;

typedef struct b3ObbSat
{
	float minOverlap;
	b3Vec3 minAxis; // box1 -> box0
	uint8_t surfaceMask0;
	uint8_t surfaceMask1;
	int srcBox;
	bool isEdge;
	int sepCount;
	b3Vec3 sepAxis;
	const b3ObbPairAxis* minPrepared;
	bool minPositive;
} b3ObbSat;

typedef struct b3VoxelContact
{
	b3Vec3 normal;			  // unit, points from box1 (B) toward box0 (A)
	float initialPenetration; // signed: > 0 overlapping, <= 0 speculative gap
	float penetrationDepth;	  // ranking only (>= 0)
	b3Vec3 body0Point;		  // world witness on box0 (A) surface
	b3Vec3 body1Point;		  // world witness on box1 (B) surface
	uint32_t featureId;		  // stable id (the colliding cell pair) for warm-starting; 0 from the raw box primitives
} b3VoxelContact;

#define B3_VOXEL_MAX_CONTACTS 64
#define B3_VOXEL_MAX_CLUSTERS 16
#define B3_VOXEL_MAX_PATCH_MANIFOLDS 16
#define B3_VOXEL_POINTS_PER_CLUSTER 4
#define B3_VOXEL_SUPPORT_DIRECTIONS 8
#define B3_VOXEL_CANDIDATES_PER_CLUSTER ( 1 + B3_VOXEL_SUPPORT_DIRECTIONS )
#define B3_VOXEL_CLUSTER_DOT 0.98f

typedef struct b3VoxelContactCluster
{
	b3Vec3 normal;
	b3Vec3 tangent1;
	b3Vec3 tangent2;
	// Slot zero is the deepest contact. The remaining slots retain planar
	// support extrema in eight fixed directions; Finish reduces this bounded,
	// order-independent candidate set to the four solver points.
	b3VoxelContact candidates[B3_VOXEL_CANDIDATES_PER_CLUSTER];
	float support[B3_VOXEL_SUPPORT_DIRECTIONS];
	float deepestPenetration;
	int contactCount;
} b3VoxelContactCluster;

/// Streaming, deterministic manifold reducer. Candidate enumeration remains
/// complete; each normal patch retains its deepest point plus a spread set.
typedef struct b3VoxelContactReducer
{
	b3VoxelContactCluster clusters[B3_VOXEL_MAX_CLUSTERS];
	int clusterCount;
	int totalPointCount;
	int maxContacts;
	b3VoxelCounters* counters;
} b3VoxelContactReducer;

typedef struct b3VoxelPatchManifold
{
	b3VoxelPatchKey key;
	b3VoxelContact points[B3_VOXEL_POINTS_PER_CLUSTER];
	int pointCount;
	int discoveryOrdinal;
	uint8_t kind; // zero canonical patch, one whole-pair deep escape
} b3VoxelPatchManifold;

typedef struct b3VoxelPatchCollision
{
	b3VoxelPatchManifold manifolds[B3_VOXEL_MAX_PATCH_MANIFOLDS];
	int manifoldCount;
	int uniqueKeyCount;
} b3VoxelPatchCollision;

B3_FORCE_INLINE b3Vec3 b3VoxelObbSupport( const b3VoxelOBB* box, b3Vec3 direction )
{
	b3Vec3 point = box->center;
	for ( int axis = 0; axis < 3; ++axis )
	{
		float projection = b3Dot( direction, box->axes[axis] );
		float half = ( &box->half.x )[axis];
		if ( projection > 1.0e-8f )
			point = b3MulAdd( point, half, box->axes[axis] );
		else if ( projection < -1.0e-8f )
			point = b3MulAdd( point, -half, box->axes[axis] );
	}
	return point;
}

B3_FORCE_INLINE bool b3VoxelPointOnExposedCubeFace( b3Vec3 offset, float half, uint8_t exposed, float tolerance )
{
	float expandedHalf = half + tolerance;
	if ( b3AbsFloat( offset.x ) > expandedHalf || b3AbsFloat( offset.y ) > expandedHalf ||
		 b3AbsFloat( offset.z ) > expandedHalf )
	{
		return false;
	}

	for ( int axis = 0; axis < 3; ++axis )
	{
		float coordinate = ( &offset.x )[axis];
		uint8_t negativeBit = (uint8_t)( 1u << ( 2 * axis ) );
		uint8_t positiveBit = (uint8_t)( 1u << ( 2 * axis + 1 ) );
		if ( ( ( exposed & negativeBit ) != 0 && b3AbsFloat( coordinate + half ) <= tolerance ) ||
			 ( ( exposed & positiveBit ) != 0 && b3AbsFloat( coordinate - half ) <= tolerance ) )
		{
			return true;
		}
	}
	return false;
}

b3AABB b3VoxelOBB_Bounds( const b3VoxelOBB* box );

int b3VoxelCollideOBB( const b3VoxelOBB* box0, const b3VoxelOBB* box1, float contactDistance, int maxContacts,
					   b3VoxelContact* out );

void b3VoxelPrepareOBBPair( b3ObbPairContext* context, const b3VoxelOBB* box0, const b3VoxelOBB* box1 );
void b3VoxelPrepareAxisAlignedOBBPair( b3ObbPairContext* context, b3Vec3 half0, const b3VoxelOBB* box1 );
bool b3VoxelComputeOBBSat( b3Vec3 centerDelta, const b3ObbPairContext* pairContext, float contactDistance, b3ObbSat* sat );
int b3VoxelManifoldFromSat( const b3VoxelOBB* box0, const b3VoxelOBB* box1, const b3ObbSat* sat, float contactDistance,
							int maxContacts, b3VoxelContact* out );
int b3VoxelCollideOBBPrepared( const b3VoxelOBB* box0, const b3VoxelOBB* box1, const b3ObbPairContext* pairContext,
							   float contactDistance, int maxContacts, b3VoxelContact* out );

int b3VoxelCollideAABB( const b3AABB* box0, const b3AABB* box1, float contactDistance, int maxContacts, b3VoxelContact* out );

int b3VoxelCollide( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1, float contactDistance,
					int maxContacts, b3VoxelContact* out );

// Internal voxel/convex driver. Kept visible to the native differential tests;
// this is not part of the public Box3D ABI.
int b3VoxelCollideConvex( const b3VoxelData* voxel, const b3Shape* convex, b3Transform transformBtoA, float contactDistance,
						  int maxContacts, b3VoxelContact* out, b3Arena* arena, b3VoxelCounters* counters );
int b3VoxelCollideConvexCanonicalWithArena( const b3VoxelData* voxel, const b3Shape* convex, b3Transform transformBtoA,
											float contactDistance, int maxContacts, b3VoxelContact* out, b3Arena* arena,
											b3VoxelCounters* counters );

int b3VoxelCollideWithArena( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
							 float contactDistance, int maxContacts, b3VoxelContact* out, b3Arena* arena,
							 b3VoxelCounters* counters );

bool b3VoxelUseCanonicalPatches( const b3VoxelData* v0, b3Quat q0, const b3VoxelData* v1, b3Quat q1 );
int b3VoxelCollideLeafTracked( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
							  float contactDistance, int maxContacts, b3VoxelContact* out, b3VoxelCounters* counters );

// Standalone canonical pseudo-cuboid path used by native differential tests
// until production contact ownership and persistence are wired.
int b3VoxelBuildCanonicalPatches( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
								  float contactDistance, b3VoxelPatchCollision* result, b3Arena* arena,
								  b3VoxelCounters* counters );
int b3VoxelCollideCanonicalWithArena( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
								 float contactDistance, int maxContacts, b3VoxelContact* out, b3Arena* arena,
								 b3VoxelCounters* counters );

// Continuous collision for a moving voxel shape. Convex targets use exact
// per-cell conservative advancement. Aggregate targets use bounded-motion
// overlap sampling followed by bisection, so arbitrary voxel rotation cannot
// fall back to a centroid ray.
b3TOIOutput b3VoxelShapeTimeOfImpact( const b3Shape* target, const b3Sweep* targetSweep, const b3VoxelData* movingVoxel,
									  const b3Sweep* movingSweep, float maxFraction, b3VoxelCounters* counters );

void b3VoxelReducer_Init( b3VoxelContactReducer* reducer, int maxContacts, b3VoxelCounters* counters );
void b3VoxelReducer_Add( b3VoxelContactReducer* reducer, const b3VoxelContact* contact );
int b3VoxelReducer_AddToCluster( b3VoxelContactReducer* reducer, const b3VoxelContact* contact, int clusterIndex );
bool b3VoxelReducer_ShouldClip( const b3VoxelContactReducer* reducer, const b3VoxelContact* contact, int* clusterIndexOut );
int b3VoxelReducer_Finish( const b3VoxelContactReducer* reducer, b3VoxelContact* out );
