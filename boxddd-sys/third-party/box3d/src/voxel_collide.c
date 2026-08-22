// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#include "voxel_collide.h"

#include "arena_allocator.h"
#include "core.h"
#include "hull.h"
#include "math_internal.h"
#include "shape.h"
#include "voxel_shape.h"

#include "box3d/collision.h"
#include "box3d/constants.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

b3VoxelTopology b3VoxelClassifyTopology( uint8_t exposed )
{
	exposed &= 0x3Fu;
	int axisCount = 0;
	axisCount += ( exposed & 0x03u ) != 0;
	axisCount += ( exposed & 0x0Cu ) != 0;
	axisCount += ( exposed & 0x30u ) != 0;
	return (b3VoxelTopology)axisCount;
}

b3VoxelPatchRange b3VoxelCanonicalPatchRange( b3Vec3i cell, uint8_t exposed, b3Vec3i domainMin, b3Vec3i domainMax )
{
	b3VoxelPatchRange range;
	for ( int axis = 0; axis < 3; ++axis )
	{
		uint8_t negativeBit = (uint8_t)( 1u << ( 2 * axis ) );
		uint8_t positiveBit = (uint8_t)( 1u << ( 2 * axis + 1 ) );
		( &range.lower.x )[axis] = ( exposed & negativeBit ) != 0 ? ( &cell.x )[axis] : ( &domainMin.x )[axis] - 1;
		( &range.upper.x )[axis] = ( exposed & positiveBit ) != 0 ? ( &cell.x )[axis] : ( &domainMax.x )[axis] + 1;
	}
	return range;
}

b3VoxelPatchKey b3VoxelSwapPatchKey( b3VoxelPatchKey key )
{
	b3VoxelPatchRange swap = key.patch0;
	key.patch0 = key.patch1;
	key.patch1 = swap;
	return key;
}

static uint64_t b3VoxelPatchMix( uint64_t hash, uint32_t value )
{
	// FNV-1a over fixed-width integer values. The hash is only a table index;
	// exact key equality remains authoritative for collision safety.
	for ( int byte = 0; byte < 4; ++byte )
	{
		hash ^= (uint8_t)( value >> ( 8 * byte ) );
		hash *= UINT64_C( 1099511628211 );
	}
	return hash;
}

uint64_t b3VoxelPatchKeyHash( const b3VoxelPatchKey* key )
{
	uint64_t hash = UINT64_C( 14695981039346656037 );
	const b3VoxelPatchRange* ranges[2] = { &key->patch0, &key->patch1 };
	for ( int rangeIndex = 0; rangeIndex < 2; ++rangeIndex )
	{
		for ( int axis = 0; axis < 3; ++axis )
		{
			hash = b3VoxelPatchMix( hash, (uint32_t)( &ranges[rangeIndex]->lower.x )[axis] );
			hash = b3VoxelPatchMix( hash, (uint32_t)( &ranges[rangeIndex]->upper.x )[axis] );
		}
	}
	return hash != 0 ? hash : UINT64_MAX;
}

uint64_t b3VoxelPatchKeyTableHash( const b3VoxelPatchKey* key )
{
	// The stable FNV hash above is also used as a persistent contact feature.
	// Hash-table slots need no such compatibility, so mix the six pairs of
	// integer bounds directly. Exact key equality still decides every match.
	uint64_t hash = UINT64_C( 0x9e3779b97f4a7c15 );
	const b3VoxelPatchRange* ranges[2] = { &key->patch0, &key->patch1 };
	for ( int rangeIndex = 0; rangeIndex < 2; ++rangeIndex )
	{
		for ( int axis = 0; axis < 3; ++axis )
		{
			uint64_t bounds = (uint64_t)(uint32_t)( &ranges[rangeIndex]->lower.x )[axis] << 32 |
							  (uint32_t)( &ranges[rangeIndex]->upper.x )[axis];
			hash ^= bounds + UINT64_C( 0x9e3779b97f4a7c15 ) + ( hash << 6 ) + ( hash >> 2 );
		}
	}
	return hash != 0 ? hash : UINT64_MAX;
}

bool b3VoxelPatchKeyEqual( const b3VoxelPatchKey* a, const b3VoxelPatchKey* b )
{
	return a->patch0.lower.x == b->patch0.lower.x && a->patch0.lower.y == b->patch0.lower.y &&
		   a->patch0.lower.z == b->patch0.lower.z && a->patch0.upper.x == b->patch0.upper.x &&
		   a->patch0.upper.y == b->patch0.upper.y && a->patch0.upper.z == b->patch0.upper.z &&
		   a->patch1.lower.x == b->patch1.lower.x && a->patch1.lower.y == b->patch1.lower.y &&
		   a->patch1.lower.z == b->patch1.lower.z && a->patch1.upper.x == b->patch1.upper.x &&
		   a->patch1.upper.y == b->patch1.upper.y && a->patch1.upper.z == b->patch1.upper.z;
}

void b3VoxelPatchTableInit( b3VoxelPatchTable* table, b3Arena* arena )
{
	memset( table, 0, sizeof( *table ) );
	table->arena = arena;
}

static int b3VoxelPatchTableFind( const b3VoxelPatchTable* table, const b3VoxelPatchKey* key, uint64_t hash, int* emptySlot )
{
	if ( table->slotCapacity == 0 )
	{
		*emptySlot = -1;
		return -1;
	}

	int mask = table->slotCapacity - 1;
	int slotIndex = (int)( hash & (uint64_t)mask );
	for ( ;; )
	{
		const b3VoxelPatchSlot* slot = table->slots + slotIndex;
		if ( slot->hash == 0 )
		{
			*emptySlot = slotIndex;
			return -1;
		}
		if ( slot->hash == hash && b3VoxelPatchKeyEqual( table->keys + slot->entryIndex, key ) )
		{
			*emptySlot = -1;
			return slot->entryIndex;
		}
		slotIndex = ( slotIndex + 1 ) & mask;
	}
}

static void b3VoxelPatchTableGrowSlots( b3VoxelPatchTable* table )
{
	B3_ASSERT( table->slotCapacity <= INT_MAX / 2 );
	int newCapacity = table->slotCapacity == 0 ? 16 : 2 * table->slotCapacity;
	b3VoxelPatchSlot* newSlots = b3Bump( table->arena, newCapacity * (int)sizeof( b3VoxelPatchSlot ) );
	memset( newSlots, 0, (size_t)newCapacity * sizeof( b3VoxelPatchSlot ) );
	table->growthCount += 1;
	table->scratchBytes += newCapacity * (int)sizeof( b3VoxelPatchSlot );

	for ( int oldIndex = 0; oldIndex < table->slotCapacity; ++oldIndex )
	{
		b3VoxelPatchSlot slot = table->slots[oldIndex];
		if ( slot.hash == 0 )
			continue;
		int mask = newCapacity - 1;
		int newIndex = (int)( slot.hash & (uint64_t)mask );
		while ( newSlots[newIndex].hash != 0 )
		{
			newIndex = ( newIndex + 1 ) & mask;
		}
		newSlots[newIndex] = slot;
	}

	table->slots = newSlots;
	table->slotCapacity = newCapacity;
}

static void b3VoxelPatchTableGrowKeys( b3VoxelPatchTable* table )
{
	B3_ASSERT( table->keyCapacity <= INT_MAX / 2 );
	int newCapacity = table->keyCapacity == 0 ? 16 : 2 * table->keyCapacity;
	b3VoxelPatchKey* newKeys = b3Bump( table->arena, newCapacity * (int)sizeof( b3VoxelPatchKey ) );
	table->growthCount += 1;
	table->scratchBytes += newCapacity * (int)sizeof( b3VoxelPatchKey );
	if ( table->count > 0 )
	{
		memcpy( newKeys, table->keys, (size_t)table->count * sizeof( b3VoxelPatchKey ) );
	}
	table->keys = newKeys;
	table->keyCapacity = newCapacity;
}

int b3VoxelPatchTableFindOrInsert( b3VoxelPatchTable* table, const b3VoxelPatchKey* key, uint64_t hash, bool* inserted )
{
	hash = hash != 0 ? hash : UINT64_MAX;
	int emptySlot = -1;
	int existing = b3VoxelPatchTableFind( table, key, hash, &emptySlot );
	if ( existing >= 0 )
	{
		*inserted = false;
		return existing;
	}

	if ( table->slotCapacity == 0 || 10 * ( table->count + 1 ) >= 7 * table->slotCapacity )
	{
		b3VoxelPatchTableGrowSlots( table );
		existing = b3VoxelPatchTableFind( table, key, hash, &emptySlot );
		B3_ASSERT( existing == -1 && emptySlot >= 0 );
	}
	if ( table->count == table->keyCapacity )
	{
		b3VoxelPatchTableGrowKeys( table );
	}

	int entryIndex = table->count++;
	table->keys[entryIndex] = *key;
	table->slots[emptySlot] = (b3VoxelPatchSlot){ hash, entryIndex };
	*inserted = true;
	return entryIndex;
}

static inline float b3Voxel_comp( b3Vec3 v, int i )
{
	return ( &v.x )[i];
}

static void b3Obb_supportingEdge( const b3VoxelOBB* box, int edgeAxis, b3Vec3 direction, b3Vec3* point0, b3Vec3* point1 )
{
	b3Vec3 center = box->center;
	for ( int axis = 0; axis < 3; ++axis )
	{
		if ( axis == edgeAxis )
			continue;

		float sign = b3Dot( direction, box->axes[axis] ) >= 0.0f ? 1.0f : -1.0f;
		center = b3MulAdd( center, sign * b3Voxel_comp( box->half, axis ), box->axes[axis] );
	}

	float half = b3Voxel_comp( box->half, edgeAxis );
	*point0 = b3MulAdd( center, -half, box->axes[edgeAxis] );
	*point1 = b3MulAdd( center, half, box->axes[edgeAxis] );
}

static void b3Obb_makeSupportContact( const b3VoxelOBB* box0, const b3VoxelOBB* box1, b3Vec3 normal, float contactDistance,
									  b3VoxelContact* contact )
{
	contact->normal = normal;
	contact->body0Point = b3VoxelObbSupport( box0, b3Neg( normal ) );
	contact->body1Point = b3VoxelObbSupport( box1, normal );
	contact->initialPenetration = b3Dot( b3Sub( contact->body1Point, contact->body0Point ), normal );
	contact->penetrationDepth =
		contact->initialPenetration > 0.0f ? contact->initialPenetration : contact->initialPenetration + contactDistance;
}

b3AABB b3VoxelOBB_Bounds( const b3VoxelOBB* b )
{
	b3Vec3 r;
	for ( int c = 0; c < 3; ++c )
	{
		float rc = fabsf( b3Voxel_comp( b->axes[0], c ) ) * b->half.x +
				   fabsf( b3Voxel_comp( b->axes[1], c ) ) * b->half.y +
				   fabsf( b3Voxel_comp( b->axes[2], c ) ) * b->half.z;
		( &r.x )[c] = rc;
	}
	return (b3AABB){ b3Sub( b->center, r ), b3Add( b->center, r ) };
}

static inline void b3VoxelPrepareOBBPairFromRotation( b3ObbPairContext* context, const b3Vec3* axes0, b3Vec3 half0,
													  const b3Vec3* axes1, b3Vec3 half1,
													  float absRotation[3][3] )
{
	context->count = 0;
	for ( int source = 0; source < 2; ++source )
	{
		const b3Vec3* axes = source == 0 ? axes0 : axes1;
		for ( int i = 0; i < 3; ++i )
		{
			int incidentAxis = 0;
			float bestAlignment = -1.0f;
			for ( int j = 0; j < 3; ++j )
			{
				float alignment = source == 0 ? absRotation[i][j] : absRotation[j][i];
				if ( alignment > bestAlignment )
				{
					bestAlignment = alignment;
					incidentAxis = j;
				}
			}

			b3ObbPairAxis* prepared = context->axes + context->count++;
			prepared->axis = axes[i];
			prepared->radius = source == 0 ? b3Voxel_comp( half0, i ) : b3Voxel_comp( half1, i );
			for ( int j = 0; j < 3; ++j )
			{
				prepared->radius += source == 0 ? absRotation[i][j] * b3Voxel_comp( half1, j )
											 : absRotation[j][i] * b3Voxel_comp( half0, j );
			}
			prepared->srcBox = source;
			prepared->isEdge = false;
			prepared->axis0 = (uint8_t)i;
			prepared->axis1 = (uint8_t)incidentAxis;
			prepared->surfaceMask0[0] = prepared->surfaceMask0[1] = 0;
			prepared->surfaceMask1[0] = prepared->surfaceMask1[1] = 0;
		}
	}

	for ( int i = 0; i < 3; ++i )
	{
		for ( int j = 0; j < 3; ++j )
		{
			b3Vec3 axis = b3Cross( axes0[i], axes1[j] );
			float lengthSquared = b3Dot( axis, axis );
			if ( lengthSquared <= 1e-4f )
				continue;
			float invLength = 1.0f / sqrtf( lengthSquared );
			axis = b3MulSV( invLength, axis );
			int i1 = ( i + 1 ) % 3;
			int i2 = ( i + 2 ) % 3;
			int j1 = ( j + 1 ) % 3;
			int j2 = ( j + 2 ) % 3;
			b3ObbPairAxis* prepared = context->axes + context->count++;
			prepared->axis = axis;
			prepared->radius =
				invLength *
				( absRotation[i2][j] * b3Voxel_comp( half0, i1 ) + absRotation[i1][j] * b3Voxel_comp( half0, i2 ) +
				  absRotation[i][j2] * b3Voxel_comp( half1, j1 ) + absRotation[i][j1] * b3Voxel_comp( half1, j2 ) );
			prepared->srcBox = 0;
			prepared->isEdge = true;
			prepared->axis0 = (uint8_t)i;
			prepared->axis1 = (uint8_t)j;
			prepared->surfaceMask0[0] = prepared->surfaceMask0[1] = 0;
			prepared->surfaceMask1[0] = prepared->surfaceMask1[1] = 0;
		}
	}
}

void b3VoxelPrepareOBBPair( b3ObbPairContext* context, const b3VoxelOBB* o0, const b3VoxelOBB* o1 )
{
	float rotation[3][3];
	float absRotation[3][3];
	for ( int i = 0; i < 3; ++i )
	{
		for ( int j = 0; j < 3; ++j )
		{
			rotation[i][j] = b3Dot( o0->axes[i], o1->axes[j] );
			absRotation[i][j] = fabsf( rotation[i][j] );
		}
	}

	b3VoxelPrepareOBBPairFromRotation( context, o0->axes, o0->half, o1->axes, o1->half, absRotation );
}

void b3VoxelPrepareAxisAlignedOBBPair( b3ObbPairContext* context, b3Vec3 half0, const b3VoxelOBB* o1 )
{
	// Voxel/box contacts always express the cell in the voxel's local frame.
	// Specializing that identity basis avoids rebuilding generic dot and cross
	// products for every contact update.
	const b3Vec3 axes0[3] = { b3Vec3_axisX, b3Vec3_axisY, b3Vec3_axisZ };
	float rotation[3][3] = {
		{ o1->axes[0].x, o1->axes[1].x, o1->axes[2].x },
		{ o1->axes[0].y, o1->axes[1].y, o1->axes[2].y },
		{ o1->axes[0].z, o1->axes[1].z, o1->axes[2].z },
	};
	float absRotation[3][3];
	for ( int i = 0; i < 3; ++i )
	{
		for ( int j = 0; j < 3; ++j )
		{
			absRotation[i][j] = fabsf( rotation[i][j] );
		}
	}

	b3VoxelPrepareOBBPairFromRotation( context, axes0, half0, o1->axes, o1->half, absRotation );
}

static uint8_t b3Voxel_directionMask( b3Vec3 direction )
{
	const float epsilon = 1.0e-4f;
	uint8_t mask = 0;
	for ( int axis = 0; axis < 3; ++axis )
	{
		float component = ( &direction.x )[axis];
		if ( fabsf( component ) <= epsilon )
			continue;
		int bit = 2 * axis + ( component > 0.0f );
		mask |= (uint8_t)( 1u << bit );
	}
	return mask;
}

static uint8_t b3Voxel_reverseDirectionMask( uint8_t mask )
{
	return (uint8_t)( ( ( mask & 0x15u ) << 1 ) | ( ( mask & 0x2Au ) >> 1 ) );
}

static void b3Obb_prepareSurfaceMasks( b3ObbPairContext* context, b3Quat rotation0, b3Quat rotation1 )
{
	for ( int i = 0; i < context->count; ++i )
	{
		b3ObbPairAxis* prepared = context->axes + i;
		prepared->surfaceMask0[0] = b3Voxel_directionMask( b3InvRotateVector( rotation0, prepared->axis ) );
		prepared->surfaceMask0[1] = b3Voxel_reverseDirectionMask( prepared->surfaceMask0[0] );
		prepared->surfaceMask1[1] = b3Voxel_directionMask( b3InvRotateVector( rotation1, prepared->axis ) );
		prepared->surfaceMask1[0] = b3Voxel_reverseDirectionMask( prepared->surfaceMask1[1] );
	}
}

static void b3Obb_preparePatchPair( b3ObbPairContext* context, const b3ObbPairContext* source, b3Vec3 half0,
								   b3Vec3 half1, float absRotation[3][3], float edgeInvLength[3][3] )
{
	// Canonical patch OBBs retain the leaf pair's orientations and vary only
	// their centers and half-extents. Reuse its axes, topology, and exposure
	// masks; only the projection radii depend on each patch's half-extents.
	*context = *source;
	for ( int i = 0; i < context->count; ++i )
	{
		b3ObbPairAxis* prepared = context->axes + i;
		int axis0 = prepared->axis0;
		int axis1 = prepared->axis1;
		if ( !prepared->isEdge )
		{
			prepared->radius = prepared->srcBox == 0 ? b3Voxel_comp( half0, axis0 ) : b3Voxel_comp( half1, axis0 );
			for ( int j = 0; j < 3; ++j )
			{
				prepared->radius += prepared->srcBox == 0 ? absRotation[axis0][j] * b3Voxel_comp( half1, j )
												  : absRotation[j][axis0] * b3Voxel_comp( half0, j );
			}
			continue;
		}

		int i1 = ( axis0 + 1 ) % 3;
		int i2 = ( axis0 + 2 ) % 3;
		int j1 = ( axis1 + 1 ) % 3;
		int j2 = ( axis1 + 2 ) % 3;
		prepared->radius =
			edgeInvLength[axis0][axis1] *
			( absRotation[i2][axis1] * b3Voxel_comp( half0, i1 ) +
			  absRotation[i1][axis1] * b3Voxel_comp( half0, i2 ) +
			  absRotation[axis0][j2] * b3Voxel_comp( half1, j1 ) +
			  absRotation[axis0][j1] * b3Voxel_comp( half1, j2 ) );
	}
}

static bool b3Obb_updatePreparedAxis( b3ObbSat* sat, const b3ObbPairAxis* prepared, float projection, float overlap,
									 float contactDist )
{
	if ( overlap + contactDist <= 0.0f )
	{
		return false;
	}
	if ( overlap <= 0.0f )
	{
		if ( sat->sepCount == 0 || fabsf( b3Dot( prepared->axis, sat->sepAxis ) ) < 0.999f )
		{
			sat->sepCount += 1;
			sat->sepAxis = prepared->axis;
			if ( sat->sepCount > 1 )
			{
				return false;
			}
		}
	}

	bool improve = prepared->isEdge ? ( overlap + 1e-4f < sat->minOverlap ) : ( overlap < sat->minOverlap + 1e-4f );
	if ( improve )
	{
		sat->minOverlap = overlap;
		sat->minPrepared = prepared;
		sat->minPositive = projection >= 0.0f;
	}
	return true;
}

static bool b3Obb_testPreparedAxis( b3ObbSat* sat, b3Vec3 delta, const b3ObbPairAxis* prepared, float contactDist )
{
	float projection = b3Dot( prepared->axis, delta );
	float overlap = prepared->radius - fabsf( projection );
	return b3Obb_updatePreparedAxis( sat, prepared, projection, overlap, contactDist );
}

static bool b3Obb_overlapPreparedFrom( b3Vec3 delta, const b3ObbPairContext* pairContext, int firstAxis )
{
	for ( int i = firstAxis; i < pairContext->count; ++i )
	{
		const b3ObbPairAxis* axis = pairContext->axes + i;
		if ( axis->radius - fabsf( b3Dot( axis->axis, delta ) ) <= 0.0f )
		{
			return false;
		}
	}
	return true;
}

static bool b3Obb_overlapPrepared( b3Vec3 delta, const b3ObbPairContext* pairContext )
{
	return b3Obb_overlapPreparedFrom( delta, pairContext, 0 );
}

static void b3Obb_facePolygon( const b3VoxelOBB* b, b3Vec3 d, int faceAxis, b3Vec3 out[4] )
{
	float sign = b3Dot( d, b->axes[faceAxis] ) >= 0.0f ? 1.0f : -1.0f;
	int u = ( faceAxis + 1 ) % 3, w = ( faceAxis + 2 ) % 3;
	float hb = b3Voxel_comp( b->half, faceAxis ), hu = b3Voxel_comp( b->half, u ), hw = b3Voxel_comp( b->half, w );
	b3Vec3 fc = b3MulAdd( b->center, sign * hb, b->axes[faceAxis] );
	b3Vec3 au = b->axes[u], aw = b->axes[w];
	out[0] = b3MulAdd( b3MulAdd( fc, -hu, au ), -hw, aw );
	out[1] = b3MulAdd( b3MulAdd( fc, +hu, au ), -hw, aw );
	out[2] = b3MulAdd( b3MulAdd( fc, +hu, au ), +hw, aw );
	out[3] = b3MulAdd( b3MulAdd( fc, -hu, au ), +hw, aw );
}

static int b3Obb_clipPlane( b3Vec3** polygon, b3Vec3** scratch, int nin, b3Vec3 n, float offset )
{
	if ( nin == 0 )
		return 0;
	b3Vec3* in = *polygon;
	float distance[8];
	int insideCount = 0;
	for ( int i = 0; i < nin; ++i )
	{
		distance[i] = b3Dot( n, in[i] ) - offset;
		insideCount += distance[i] <= 1e-8f;
	}
	if ( insideCount == nin )
		return nin;
	if ( insideCount == 0 )
		return 0;

	b3Vec3* out = *scratch;
	int nout = 0;
	for ( int i = 0; i < nin; ++i )
	{
		int next = i + 1;
		if ( next == nin )
			next = 0;
		b3Vec3 a = in[i];
		b3Vec3 b = in[next];
		float da = distance[i];
		float db = distance[next];
		bool ain = da <= 1e-8f, bin = db <= 1e-8f;
		if ( ain )
			out[nout++] = a;
		if ( ain != bin )
		{
			float denom = da - db;
			if ( fabsf( denom ) > 1e-12f )
			{
				float t = da / denom;
				out[nout++] = b3MulAdd( a, t, b3Sub( b, a ) );
			}
		}
	}
	*polygon = out;
	*scratch = in;
	return nout;
}

static int b3Obb_reduce( const b3VoxelContact* cand, int ncand, int maxContacts, b3VoxelContact* out )
{
	if ( ncand <= maxContacts )
	{
		for ( int i = 0; i < ncand; ++i )
			out[i] = cand[i];
		return ncand;
	}
	bool used[8] = { false };
	int deepest = 0;
	for ( int i = 1; i < ncand; ++i )
		if ( cand[i].initialPenetration > cand[deepest].initialPenetration )
			deepest = i;
	int nsel = 0;
	out[nsel++] = cand[deepest];
	used[deepest] = true;
	while ( nsel < maxContacts )
	{
		int bestIdx = -1;
		float bestMin = -1.0f;
		for ( int i = 0; i < ncand; ++i )
		{
			if ( used[i] )
				continue;
			float minD = FLT_MAX;
			for ( int j = 0; j < nsel; ++j )
			{
				b3Vec3 diff = b3Sub( cand[i].body0Point, out[j].body0Point );
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
		out[nsel++] = cand[bestIdx];
		used[bestIdx] = true;
	}
	return nsel;
}

bool b3VoxelComputeOBBSat( b3Vec3 delta, const b3ObbPairContext* pairContext, float contactDistance, b3ObbSat* sat )
{
	sat->minOverlap = FLT_MAX;
	sat->sepCount = 0;

	for ( int i = 0; i < pairContext->count; ++i )
	{
		if ( !b3Obb_testPreparedAxis( sat, delta, pairContext->axes + i, contactDistance ) )
			return false;
	}
	if ( sat->minOverlap == FLT_MAX )
		return false;

	const b3ObbPairAxis* prepared = sat->minPrepared;
	int positive = sat->minPositive;
	sat->minAxis = positive ? prepared->axis : b3Neg( prepared->axis );
	sat->surfaceMask0 = prepared->surfaceMask0[positive];
	sat->surfaceMask1 = prepared->surfaceMask1[positive];
	sat->srcBox = prepared->srcBox;
	sat->isEdge = prepared->isEdge;
	return true;
}

int b3VoxelManifoldFromSat( const b3VoxelOBB* o0, const b3VoxelOBB* o1, const b3ObbSat* preparedSat, float contactDistance,
							int maxContacts, b3VoxelContact* out )
{
	if ( maxContacts < 1 )
		return 0;
	if ( maxContacts > 4 )
		maxContacts = 4;

	const b3ObbSat* sat = preparedSat;

	if ( sat->isEdge )
	{
		B3_ASSERT( sat->minPrepared != NULL );
		b3Vec3 edge00, edge01, edge10, edge11;
		b3Obb_supportingEdge( o0, sat->minPrepared->axis0, b3Neg( sat->minAxis ), &edge00, &edge01 );
		b3Obb_supportingEdge( o1, sat->minPrepared->axis1, sat->minAxis, &edge10, &edge11 );
		b3SegmentDistanceResult closest = b3SegmentDistance( edge00, edge01, edge10, edge11 );
		out[0].normal = sat->minAxis;
		out[0].body0Point = closest.point1;
		out[0].body1Point = closest.point2;
		out[0].initialPenetration = b3Dot( b3Sub( closest.point2, closest.point1 ), sat->minAxis );
		out[0].penetrationDepth =
			out[0].initialPenetration > 0.0f ? out[0].initialPenetration : out[0].initialPenetration + contactDistance;
		return 1;
	}

	bool refIsBody0 = ( sat->srcBox == 0 );
	const b3VoxelOBB* refBox = refIsBody0 ? o0 : o1;
	const b3VoxelOBB* incBox = refIsBody0 ? o1 : o0;
	b3Vec3 refNormal = refIsBody0 ? b3Neg( sat->minAxis ) : sat->minAxis; // ref face outward normal

	B3_ASSERT( sat->minPrepared != NULL );
	int refAxis = sat->minPrepared->axis0;
	float refSign = b3Dot( refNormal, refBox->axes[refAxis] ) >= 0.0f ? 1.0f : -1.0f;
	b3Vec3 refFaceCenter = b3MulAdd( refBox->center, refSign * b3Voxel_comp( refBox->half, refAxis ), refBox->axes[refAxis] );

	if ( maxContacts == 1 )
	{
		// The incident support point is already on its box. Projecting it onto
		// the reference plane gives a matched witness pair without averaging
		// unrelated tangent coordinates. Most cell pairs land inside the
		// reference face and avoid polygon clipping entirely.
		b3Vec3 incidentPoint = b3VoxelObbSupport( incBox, b3Neg( refNormal ) );
		float signedDistance = b3Dot( b3Sub( incidentPoint, refFaceCenter ), refNormal );
		b3Vec3 referencePoint = b3MulAdd( incidentPoint, -signedDistance, refNormal );
		bool inside = true;
		for ( int axis = 0; axis < 3; ++axis )
		{
			if ( axis == refAxis )
				continue;
			float coordinate = b3Dot( b3Sub( referencePoint, refBox->center ), refBox->axes[axis] );
			float half = b3Voxel_comp( refBox->half, axis );
			if ( fabsf( coordinate ) > half + 8.0f * FLT_EPSILON )
			{
				inside = false;
				break;
			}
		}
		float initialPenetration = -signedDistance;
		if ( inside && initialPenetration + contactDistance > 0.0f )
		{
			out[0].normal = sat->minAxis;
			out[0].initialPenetration = initialPenetration;
			out[0].penetrationDepth = initialPenetration > 0.0f ? initialPenetration : initialPenetration + contactDistance;
			if ( refIsBody0 )
			{
				out[0].body0Point = referencePoint;
				out[0].body1Point = incidentPoint;
			}
			else
			{
				out[0].body0Point = incidentPoint;
				out[0].body1Point = referencePoint;
			}
			return 1;
		}
	}

	b3Vec3 polygons[2][8];
	b3Vec3* poly = polygons[0];
	b3Vec3* tmp = polygons[1];
	b3Obb_facePolygon( incBox, b3Neg( refNormal ), sat->minPrepared->axis1, poly );
	int npoly = 4;
	for ( int i = 0; i < 3; ++i )
	{
		if ( i == refAxis )
			continue;
		float c = b3Dot( refBox->axes[i], refBox->center );
		float h = b3Voxel_comp( refBox->half, i );
		npoly = b3Obb_clipPlane( &poly, &tmp, npoly, refBox->axes[i], c + h );
		npoly = b3Obb_clipPlane( &poly, &tmp, npoly, b3Neg( refBox->axes[i] ), h - c );
	}

	b3VoxelContact cand[8];
	int ncand = 0;
	for ( int i = 0; i < npoly && ncand < 8; ++i )
	{
		b3Vec3 pt = poly[i];
		float sd = b3Dot( b3Sub( pt, refFaceCenter ), refNormal );
		float initPen = -sd;
		if ( initPen + contactDistance <= 0.0f )
			continue;
		b3Vec3 refPt = b3MulAdd( pt, -sd, refNormal );
		b3VoxelContact c;
		c.normal = sat->minAxis;
		c.initialPenetration = initPen;
		c.penetrationDepth = initPen > 0.0f ? initPen : initPen + contactDistance;
		if ( refIsBody0 )
		{
			c.body0Point = refPt; // on box0 (A) ref face
			c.body1Point = pt;	  // on box1 (B) incident face
		}
		else
		{
			c.body0Point = pt;
			c.body1Point = refPt;
		}
		bool dup = false;
		for ( int j = 0; j < ncand; ++j )
		{
			b3Vec3 diff = b3Sub( cand[j].body0Point, c.body0Point );
			if ( b3Dot( diff, diff ) < 1e-12f )
			{
				dup = true;
				break;
			}
		}
		if ( !dup )
			cand[ncand++] = c;
	}

	if ( ncand == 0 )
	{
		// Numerical clipping degeneracy. Preserve the geometric contract: each
		// fallback witness remains on its own box even if their tangent
		// coordinates cannot be paired as cleanly as a clipped face witness.
		b3Obb_makeSupportContact( o0, o1, sat->minAxis, contactDistance, out );
		return 1;
	}

	return b3Obb_reduce( cand, ncand, maxContacts, out );
}

int b3VoxelCollideOBBPrepared( const b3VoxelOBB* o0, const b3VoxelOBB* o1, const b3ObbPairContext* pairContext,
							   float contactDistance, int maxContacts, b3VoxelContact* out )
{
	if ( maxContacts < 1 )
		return 0;
	b3ObbSat sat;
	if ( !b3VoxelComputeOBBSat( b3Sub( o0->center, o1->center ), pairContext, contactDistance, &sat ) )
		return 0;
	return b3VoxelManifoldFromSat( o0, o1, &sat, contactDistance, maxContacts, out );
}

int b3VoxelCollideOBB( const b3VoxelOBB* o0, const b3VoxelOBB* o1, float contactDistance, int maxContacts, b3VoxelContact* out )
{
	b3ObbPairContext pairContext;
	b3VoxelPrepareOBBPair( &pairContext, o0, o1 );
	return b3VoxelCollideOBBPrepared( o0, o1, &pairContext, contactDistance, maxContacts, out );
}

int b3VoxelCollideAABB( const b3AABB* a0, const b3AABB* a1, float contactDistance, int maxContacts, b3VoxelContact* out )
{
	if ( maxContacts < 1 )
		return 0;
	if ( maxContacts > 4 )
		maxContacts = 4;

	float overlap[3], rectMin[3], rectMax[3], c0[3], c1[3];
	int sep = 0;
	for ( int i = 0; i < 3; ++i )
	{
		float lo0 = b3Voxel_comp( a0->lowerBound, i ), hi0 = b3Voxel_comp( a0->upperBound, i );
		float lo1 = b3Voxel_comp( a1->lowerBound, i ), hi1 = b3Voxel_comp( a1->upperBound, i );
		rectMin[i] = b3MaxFloat( lo0, lo1 );
		rectMax[i] = b3MinFloat( hi0, hi1 );
		overlap[i] = rectMax[i] - rectMin[i];
		c0[i] = 0.5f * ( lo0 + hi0 );
		c1[i] = 0.5f * ( lo1 + hi1 );
		if ( overlap[i] + contactDistance <= 0.0f )
			return 0;
		if ( overlap[i] < 0.0f && ++sep > 1 )
			return 0;
	}

	int axis = 0;
	if ( overlap[1] < overlap[axis] )
		axis = 1;
	if ( overlap[2] < overlap[axis] )
		axis = 2;

	float ov = overlap[axis];
	b3Vec3 normal = b3Vec3_zero;
	( &normal.x )[axis] = c0[axis] >= c1[axis] ? 1.0f : -1.0f; // B -> A
	float depth = ov > 0.0f ? ov : ov + contactDistance;
	float planePos = 0.5f * ( rectMin[axis] + rectMax[axis] );

	int t1 = ( axis + 1 ) % 3, t2 = ( axis + 2 ) % 3;

	const float minPos = 1e-6f;
	float t1lo = rectMin[t1], t1hi = b3MaxFloat( rectMax[t1], rectMin[t1] + minPos );
	float t2lo = rectMin[t2], t2hi = b3MaxFloat( rectMax[t2], rectMin[t2] + minPos );
	float corners[4][2] = { { t1lo, t2lo }, { t1hi, t2hi }, { t1lo, t2hi }, { t1hi, t2lo } };

	int n = ( maxContacts < 4 ) ? maxContacts : 4;
	if ( ov > 0.4f * b3MinFloat( overlap[t1] + minPos, overlap[t2] + minPos ) )
	{
		n = 1;
	}
	for ( int i = 0; i < n; ++i )
	{
		b3Vec3 patch;
		( &patch.x )[axis] = planePos;
		( &patch.x )[t1] = ( n == 1 ) ? 0.5f * ( t1lo + t1hi ) : corners[i][0];
		( &patch.x )[t2] = ( n == 1 ) ? 0.5f * ( t2lo + t2hi ) : corners[i][1];
		out[i].normal = normal;
		out[i].initialPenetration = ov;
		out[i].penetrationDepth = depth;
		out[i].body0Point = b3MulAdd( patch, -( ov * 0.5f ), normal );
		out[i].body1Point = b3MulAdd( patch, +( ov * 0.5f ), normal );
	}
	return n;
}

static inline b3Vec3 b3Voxel_xfPoint( b3Transform xf, b3Vec3 local )
{
	return b3Add( xf.p, b3RotateVector( xf.q, local ) );
}

static inline b3Vec3 b3Voxel_invXfPoint( b3Transform xf, b3Vec3 world )
{
	return b3InvRotateVector( xf.q, b3Sub( world, xf.p ) );
}

static b3AABB b3Voxel_mapBounds( b3AABB box, b3Transform xf, bool inverse )
{
	b3AABB r = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
	for ( int i = 0; i < 8; ++i )
	{
		b3Vec3 c = { ( i & 1 ) ? box.upperBound.x : box.lowerBound.x, ( i & 2 ) ? box.upperBound.y : box.lowerBound.y,
					 ( i & 4 ) ? box.upperBound.z : box.lowerBound.z };
		b3Vec3 w = inverse ? b3Voxel_invXfPoint( xf, c ) : b3Voxel_xfPoint( xf, c );
		r.lowerBound.x = b3MinFloat( r.lowerBound.x, w.x );
		r.lowerBound.y = b3MinFloat( r.lowerBound.y, w.y );
		r.lowerBound.z = b3MinFloat( r.lowerBound.z, w.z );
		r.upperBound.x = b3MaxFloat( r.upperBound.x, w.x );
		r.upperBound.y = b3MaxFloat( r.upperBound.y, w.y );
		r.upperBound.z = b3MaxFloat( r.upperBound.z, w.z );
	}
	return r;
}

static inline b3AABB b3Voxel_expandB( b3AABB b, float a )
{
	if ( a <= 0.0f )
		return b;
	b3Vec3 e = { a, a, a };
	return (b3AABB){ b3Sub( b.lowerBound, e ), b3Add( b.upperBound, e ) };
}

static inline bool b3Voxel_isect( b3AABB a, b3AABB b )
{
	return !( a.upperBound.x < b.lowerBound.x || a.lowerBound.x > b.upperBound.x || a.upperBound.y < b.lowerBound.y ||
			  a.lowerBound.y > b.upperBound.y || a.upperBound.z < b.lowerBound.z || a.lowerBound.z > b.upperBound.z );
}

static inline bool b3Voxel_containsAABB( b3AABB outer, b3AABB inner )
{
	return outer.lowerBound.x <= inner.lowerBound.x && outer.lowerBound.y <= inner.lowerBound.y &&
		   outer.lowerBound.z <= inner.lowerBound.z && outer.upperBound.x >= inner.upperBound.x &&
		   outer.upperBound.y >= inner.upperBound.y && outer.upperBound.z >= inner.upperBound.z;
}

static b3VoxelOBB b3Voxel_cellFrame( const b3VoxelData* voxels, b3Transform xf )
{
	b3VoxelOBB o;
	o.center = b3Vec3_zero;
	o.axes[0] = b3RotateVector( xf.q, (b3Vec3){ 1.0f, 0.0f, 0.0f } );
	o.axes[1] = b3RotateVector( xf.q, (b3Vec3){ 0.0f, 1.0f, 0.0f } );
	o.axes[2] = b3RotateVector( xf.q, (b3Vec3){ 0.0f, 0.0f, 1.0f } );
	float h = 0.5f * b3Voxel_GetVoxelSize( voxels );
	o.half = (b3Vec3){ h, h, h };
	return o;
}

static inline b3Vec3 b3Voxel_cellCenterFromGrid( b3Vec3 origin, float voxelSize, b3Vec3i cell )
{
	return b3Add( origin, (b3Vec3){ cell.x * voxelSize, cell.y * voxelSize, cell.z * voxelSize } );
}

static b3VoxelOBB b3Voxel_cellOBB( const b3VoxelOBB* frame, const b3VoxelData* voxels, b3Vec3i cell, b3Transform xf )
{
	b3VoxelOBB obb = *frame;
	obb.center = b3Voxel_xfPoint( xf, b3Voxel_GetCellCenter( voxels, cell ) );
	return obb;
}

static bool b3Voxel_facesExposed( uint8_t exposed0, uint8_t required0, uint8_t exposed1, uint8_t required1 )
{
	return ( exposed0 & required0 ) != 0 && ( exposed1 & required1 ) != 0;
}

static bool b3Voxel_pointOnExposedFace( const b3VoxelOBB* cell, b3Vec3 point, uint8_t exposed, float tolerance )
{
	b3Vec3 offset = b3Sub( point, cell->center );
	float coordinate[3];
	for ( int axis = 0; axis < 3; ++axis )
	{
		coordinate[axis] = b3Dot( offset, cell->axes[axis] );
		if ( fabsf( coordinate[axis] ) > b3Voxel_comp( cell->half, axis ) + tolerance )
			return false;
	}

	for ( int axis = 0; axis < 3; ++axis )
	{
		float half = b3Voxel_comp( cell->half, axis );
		if ( fabsf( fabsf( coordinate[axis] ) - half ) <= tolerance )
		{
			int bit = 2 * axis + ( coordinate[axis] > 0.0f );
			if ( ( exposed & (uint8_t)( 1u << bit ) ) != 0 )
				return true;
		}
	}
	return false;
}

static int b3Voxel_findExposedSatManifold( const b3VoxelOBB* obb0, uint8_t exposed0, const b3VoxelOBB* obb1, uint8_t exposed1,
										   const b3ObbPairContext* pairContext, b3Vec3 delta, float contactDistance,
										   b3VoxelContact out[4], float* overlapOut )
{
	const float witnessTolerance = B3_LINEAR_SLOP + 32.0f * FLT_EPSILON;
	uint16_t tried = 0;
	while ( tried != (uint16_t)( ( 1u << pairContext->count ) - 1u ) )
	{
		int bestAxisIndex = -1;
		int bestPositive = 0;
		float bestOverlap = FLT_MAX;
		for ( int axisIndex = 0; axisIndex < pairContext->count; ++axisIndex )
		{
			uint16_t bit = (uint16_t)( 1u << axisIndex );
			if ( ( tried & bit ) != 0 )
				continue;
			const b3ObbPairAxis* prepared = pairContext->axes + axisIndex;
			float projection = b3Dot( prepared->axis, delta );
			float overlap = prepared->radius - fabsf( projection );
			if ( overlap + contactDistance <= 0.0f )
			{
				tried |= bit;
				continue;
			}
			int positive = projection >= 0.0f;
			if ( !b3Voxel_facesExposed( exposed0, prepared->surfaceMask0[positive], exposed1,
									 prepared->surfaceMask1[positive] ) )
			{
				tried |= bit;
				continue;
			}
			if ( overlap < bestOverlap )
			{
				bestAxisIndex = axisIndex;
				bestPositive = positive;
				bestOverlap = overlap;
			}
		}
		if ( bestAxisIndex < 0 )
			break;
		tried |= (uint16_t)( 1u << bestAxisIndex );
		const b3ObbPairAxis* prepared = pairContext->axes + bestAxisIndex;
		uint8_t surface0 = prepared->surfaceMask0[bestPositive];
		uint8_t surface1 = prepared->surfaceMask1[bestPositive];

		b3ObbSat alternate = {
			.minOverlap = bestOverlap,
			.minAxis = bestPositive ? prepared->axis : b3Neg( prepared->axis ),
			.surfaceMask0 = surface0,
			.surfaceMask1 = surface1,
			.srcBox = prepared->srcBox,
			.isEdge = prepared->isEdge,
			.minPrepared = prepared,
			.minPositive = bestPositive,
		};
		b3VoxelContact candidates[4];
		int candidateCount = b3VoxelManifoldFromSat( obb0, obb1, &alternate, contactDistance, 4, candidates );
		int validCount = 0;
		for ( int k = 0; k < candidateCount; ++k )
		{
			if ( b3Voxel_pointOnExposedFace( obb0, candidates[k].body0Point, exposed0, witnessTolerance ) &&
				 b3Voxel_pointOnExposedFace( obb1, candidates[k].body1Point, exposed1, witnessTolerance ) )
			{
				candidates[validCount++] = candidates[k];
			}
		}
		if ( validCount > 0 )
		{
			memcpy( out, candidates, (size_t)validCount * sizeof( b3VoxelContact ) );
			*overlapOut = bestOverlap;
			return validCount;
		}
	}
	*overlapOut = FLT_MAX;
	return 0;
}

static float b3Voxel_findExposedSatOverlap( uint8_t exposed0, uint8_t exposed1, const b3ObbPairContext* pairContext, b3Vec3 delta,
											float contactDistance )
{
	float bestOverlap = FLT_MAX;
	for ( int i = 0; i < pairContext->count; ++i )
	{
		const b3ObbPairAxis* prepared = pairContext->axes + i;
		float projection = b3Dot( prepared->axis, delta );
		float overlap = prepared->radius - fabsf( projection );
		if ( overlap + contactDistance <= 0.0f )
			continue;
		int positive = projection >= 0.0f;
		if ( b3Voxel_facesExposed( exposed0, prepared->surfaceMask0[positive], exposed1, prepared->surfaceMask1[positive] ) )
		{
			bestOverlap = b3MinFloat( bestOverlap, overlap );
		}
	}
	return bestOverlap;
}

static uint32_t b3Voxel_cellHash( b3Vec3i c )
{
	uint32_t h = (uint32_t)c.x * 73856093u ^ (uint32_t)c.y * 19349663u ^ (uint32_t)c.z * 83492791u;
	h ^= h >> 13;
	h *= 0x5bd1e995u;
	h ^= h >> 15;
	return h;
}

static uint32_t b3Voxel_pairId( b3Vec3i a, b3Vec3i b, int k )
{
	uint32_t h = b3Voxel_cellHash( a ) * 0x9e3779b1u + b3Voxel_cellHash( b );
	h ^= (uint32_t)k * 0x85ebca6bu;
	return h ? h : 1u;
}

static void b3Voxel_order( b3VoxelContact* c, int n )
{
	for ( int i = 1; i < n; ++i )
	{
		b3VoxelContact key = c[i];
		int j = i - 1;
		while ( j >= 0 )
		{
			const b3VoxelContact* a = &c[j];
			bool greater = a->initialPenetration < key.initialPenetration ||
						   ( a->initialPenetration == key.initialPenetration &&
							 ( a->body0Point.x > key.body0Point.x ||
							   ( a->body0Point.x == key.body0Point.x && a->body0Point.y > key.body0Point.y ) ) );
			if ( !greater )
				break;
			c[j + 1] = c[j];
			j--;
		}
		c[j + 1] = key;
	}
}

static int b3Voxel_reduceSmall( const b3VoxelContact* candidates, int count, int capacity, b3VoxelContact* out )
{
	B3_ASSERT( 0 < capacity && capacity <= B3_VOXEL_POINTS_PER_CLUSTER );
	B3_ASSERT( capacity < count && count <= B3_VOXEL_CANDIDATES_PER_CLUSTER );

	bool used[B3_VOXEL_CANDIDATES_PER_CLUSTER] = { false };
	int deepest = 0;
	for ( int i = 1; i < count; ++i )
	{
		if ( candidates[i].initialPenetration > candidates[deepest].initialPenetration )
		{
			deepest = i;
		}
	}

	int selected = 0;
	out[selected++] = candidates[deepest];
	used[deepest] = true;
	while ( selected < capacity )
	{
		int best = -1;
		float bestDistance = -1.0f;
		for ( int i = 0; i < count; ++i )
		{
			if ( used[i] )
				continue;
			float minDistance = FLT_MAX;
			for ( int j = 0; j < selected; ++j )
			{
				b3Vec3 delta = b3Sub( candidates[i].body0Point, out[j].body0Point );
				minDistance = b3MinFloat( minDistance, b3Dot( delta, delta ) );
			}
			if ( minDistance > bestDistance )
			{
				bestDistance = minDistance;
				best = i;
			}
		}
		B3_ASSERT( best >= 0 );
		out[selected++] = candidates[best];
		used[best] = true;
	}
	return selected;
}

void b3VoxelReducer_Init( b3VoxelContactReducer* reducer, int maxContacts, b3VoxelCounters* counters )
{
	reducer->clusterCount = 0;
	reducer->totalPointCount = 0;
	reducer->maxContacts = b3ClampInt( maxContacts, 0, B3_VOXEL_MAX_CONTACTS );
	reducer->counters = counters;
}

static int b3VoxelReducer_FindCluster( const b3VoxelContactReducer* reducer, b3Vec3 normal )
{
	for ( int i = 0; i < reducer->clusterCount; ++i )
	{
		if ( b3Dot( reducer->clusters[i].normal, normal ) > B3_VOXEL_CLUSTER_DOT )
		{
			return i;
		}
	}
	return -1;
}

static bool b3Voxel_preferContact( const b3VoxelContact* candidate, const b3VoxelContact* current )
{
	if ( candidate->initialPenetration != current->initialPenetration )
	{
		return candidate->initialPenetration > current->initialPenetration;
	}
	if ( candidate->featureId != current->featureId )
	{
		return candidate->featureId < current->featureId;
	}
	if ( candidate->body0Point.x != current->body0Point.x )
	{
		return candidate->body0Point.x < current->body0Point.x;
	}
	if ( candidate->body0Point.y != current->body0Point.y )
	{
		return candidate->body0Point.y < current->body0Point.y;
	}
	return candidate->body0Point.z < current->body0Point.z;
}

static void b3Voxel_projectSupports( const b3VoxelContactCluster* cluster, b3Vec3 point,
									 float projections[B3_VOXEL_SUPPORT_DIRECTIONS] )
{
	float u = b3Dot( point, cluster->tangent1 );
	float v = b3Dot( point, cluster->tangent2 );
	projections[0] = u;
	projections[1] = u + v;
	projections[2] = v;
	projections[3] = -u + v;
	projections[4] = -u;
	projections[5] = -u - v;
	projections[6] = -v;
	projections[7] = u - v;
}

static void b3Voxel_initCluster( b3VoxelContactCluster* cluster, const b3VoxelContact* contact )
{
	cluster->normal = contact->normal;
	b3Vec3 reference = fabsf( contact->normal.x ) < 0.8f ? b3Vec3_axisX : b3Vec3_axisY;
	cluster->tangent1 = b3Normalize( b3Cross( contact->normal, reference ) );
	cluster->tangent2 = b3Cross( contact->normal, cluster->tangent1 );
	cluster->candidates[0] = *contact;
	cluster->deepestPenetration = contact->initialPenetration;
	cluster->contactCount = 1;
	float projections[B3_VOXEL_SUPPORT_DIRECTIONS];
	b3Voxel_projectSupports( cluster, contact->body0Point, projections );
	for ( int i = 0; i < B3_VOXEL_SUPPORT_DIRECTIONS; ++i )
	{
		cluster->candidates[i + 1] = *contact;
		cluster->support[i] = projections[i];
	}
}

static void b3Voxel_updateCluster( b3VoxelContactCluster* cluster, const b3VoxelContact* contact )
{
	cluster->contactCount += 1;
	if ( b3Voxel_preferContact( contact, cluster->candidates ) )
	{
		cluster->candidates[0] = *contact;
		cluster->deepestPenetration = contact->initialPenetration;
	}
	float projections[B3_VOXEL_SUPPORT_DIRECTIONS];
	b3Voxel_projectSupports( cluster, contact->body0Point, projections );
	for ( int i = 0; i < B3_VOXEL_SUPPORT_DIRECTIONS; ++i )
	{
		float projection = projections[i];
		if ( projection > cluster->support[i] ||
			 ( projection == cluster->support[i] && b3Voxel_preferContact( contact, cluster->candidates + i + 1 ) ) )
		{
			cluster->support[i] = projection;
			cluster->candidates[i + 1] = *contact;
		}
	}
}

int b3VoxelReducer_AddToCluster( b3VoxelContactReducer* reducer, const b3VoxelContact* contact, int clusterIndex )
{
	if ( reducer->maxContacts == 0 )
		return -1;
	reducer->totalPointCount += 1;

	if ( clusterIndex < 0 )
	{
		int maxClusters = b3MinInt( B3_VOXEL_MAX_CLUSTERS, reducer->maxContacts );
		if ( reducer->clusterCount < maxClusters )
		{
			clusterIndex = reducer->clusterCount++;
			b3Voxel_initCluster( reducer->clusters + clusterIndex, contact );
			return clusterIndex;
		}

		if ( reducer->counters != NULL )
		{
			reducer->counters->reducerOverflowInsertions += 1;
		}
		int weakest = 0;
		for ( int i = 1; i < reducer->clusterCount; ++i )
		{
			if ( reducer->clusters[i].deepestPenetration < reducer->clusters[weakest].deepestPenetration )
			{
				weakest = i;
			}
		}
		b3VoxelContactCluster* cluster = reducer->clusters + weakest;
		if ( contact->initialPenetration <= cluster->deepestPenetration )
			return -1;
		b3Voxel_initCluster( cluster, contact );
		return weakest;
	}

	b3VoxelContactCluster* cluster = reducer->clusters + clusterIndex;
	if ( reducer->counters != NULL && cluster->contactCount >= B3_VOXEL_CANDIDATES_PER_CLUSTER )
	{
		reducer->counters->reducerOverflowInsertions += 1;
	}
	b3Voxel_updateCluster( cluster, contact );
	return clusterIndex;
}

void b3VoxelReducer_Add( b3VoxelContactReducer* reducer, const b3VoxelContact* contact )
{
	int clusterIndex = b3VoxelReducer_FindCluster( reducer, contact->normal );
	b3VoxelReducer_AddToCluster( reducer, contact, clusterIndex );
}

int b3VoxelReducer_Finish( const b3VoxelContactReducer* reducer, b3VoxelContact* out )
{
	int count = 0;
	for ( int i = 0; i < reducer->clusterCount; ++i )
	{
		const b3VoxelContactCluster* cluster = reducer->clusters + i;
		b3VoxelContact unique[B3_VOXEL_CANDIDATES_PER_CLUSTER];
		int uniqueCount = 0;
		for ( int j = 0; j < B3_VOXEL_CANDIDATES_PER_CLUSTER; ++j )
		{
			const b3VoxelContact* candidate = cluster->candidates + j;
			bool duplicate = false;
			for ( int k = 0; k < uniqueCount; ++k )
			{
				if ( unique[k].featureId == candidate->featureId && unique[k].body0Point.x == candidate->body0Point.x &&
					 unique[k].body0Point.y == candidate->body0Point.y && unique[k].body0Point.z == candidate->body0Point.z )
				{
					duplicate = true;
					break;
				}
			}
			if ( !duplicate )
			{
				unique[uniqueCount++] = *candidate;
			}
		}

		int capacity = b3MinInt( B3_VOXEL_POINTS_PER_CLUSTER, reducer->maxContacts - count );
		if ( capacity <= 0 )
		{
			break;
		}
		if ( uniqueCount <= capacity )
		{
			memcpy( out + count, unique, (size_t)uniqueCount * sizeof( b3VoxelContact ) );
			count += uniqueCount;
		}
		else
		{
			count += b3Voxel_reduceSmall( unique, uniqueCount, capacity, out + count );
		}
	}
	b3Voxel_order( out, count );
	return count;
}

bool b3VoxelReducer_ShouldClip( const b3VoxelContactReducer* reducer, const b3VoxelContact* contact, int* clusterIndexOut )
{
	int clusterIndex = b3VoxelReducer_FindCluster( reducer, contact->normal );
	*clusterIndexOut = clusterIndex;
	if ( clusterIndex < 0 )
	{
		return true;
	}
	const b3VoxelContactCluster* cluster = reducer->clusters + clusterIndex;
	if ( contact->initialPenetration > cluster->deepestPenetration + B3_LINEAR_SLOP )
	{
		return true;
	}
	float projections[B3_VOXEL_SUPPORT_DIRECTIONS];
	b3Voxel_projectSupports( cluster, contact->body0Point, projections );
	for ( int i = 0; i < B3_VOXEL_SUPPORT_DIRECTIONS; ++i )
	{
		float projection = projections[i];
		if ( projection > cluster->support[i] + B3_LINEAR_SLOP )
		{
			return true;
		}
	}
	return false;
}

typedef struct b3VoxelVoxelQueryContext
{
	const b3VoxelData* voxels0;
	const b3VoxelData* voxels1;
	b3Transform transform0;
	b3Transform transform1;
	b3Transform transform0To1;
	b3Vec3 origin0;
	b3Vec3 origin1;
	float voxelSize0;
	float voxelSize1;
	float contactDistance;
	b3VoxelOBB frame0;
	b3VoxelOBB frame1;
	b3VoxelOBB frame0In1;
	b3ObbPairContext pairContext;
	b3VoxelContactReducer reducer;
	b3VoxelOBB fallbackObb0;
	b3VoxelOBB fallbackObb1;
	b3Vec3 fallbackDelta;
	b3Vec3i fallbackCell0;
	b3Vec3i fallbackCell1;
	float fallbackOverlap;
	uint8_t fallbackExposed0;
	uint8_t fallbackExposed1;
	bool hasFallback;
	b3Vec3i embeddedSeedCell0;
	b3Vec3i embeddedSeedCell1;
	float embeddedSeedPenetration;
	bool collectEmbeddedSeed;
	bool hasEmbeddedSeed;
	b3VoxelCounters* counters;
} b3VoxelVoxelQueryContext;

typedef struct b3VoxelVoxelInnerContext
{
	b3VoxelVoxelQueryContext* pair;
	b3Vec3i cell0;
	uint8_t exposed0;
	b3VoxelOBB obb0;
} b3VoxelVoxelInnerContext;

static bool b3Voxel_patchPairEligible( b3VoxelTopology topology0, b3VoxelTopology topology1 )
{
	return ( topology0 == b3_voxelVertex && topology1 != b3_voxelInterior ) ||
		   ( topology0 == b3_voxelEdge && ( topology1 == b3_voxelVertex || topology1 == b3_voxelEdge ) ) ||
		   ( topology0 == b3_voxelFace && topology1 == b3_voxelVertex );
}

static bool b3Voxel_collideInnerCell( b3Vec3i cell1, uint8_t exposed1, void* rawContext )
{
	b3VoxelVoxelInnerContext* context = rawContext;
	b3VoxelVoxelQueryContext* pair = context->pair;
	if ( exposed1 == 0 && !pair->collectEmbeddedSeed )
		return true;
	b3Vec3 center1 = b3Voxel_xfPoint( pair->transform1, b3Voxel_cellCenterFromGrid( pair->origin1, pair->voxelSize1, cell1 ) );
	b3VoxelContact contacts[4];
	if ( pair->counters != NULL )
	{
		pair->counters->obbTests += 1;
	}
	// Start with one clipped witness. Once a body-pair normal patch exists,
	// later cell pairs only need to contribute depth/spread candidates. The first
	// pair on a new patch is regenerated from the same SAT result with the full
	// face clip so a single-voxel contact still gets a four-corner support base.
	b3ObbSat sat;
	if ( !b3VoxelComputeOBBSat( b3Sub( context->obb0.center, center1 ), &pair->pairContext, pair->contactDistance, &sat ) )
	{
		return true;
	}
	if ( pair->collectEmbeddedSeed )
	{
		if ( sat.minOverlap > 0.0f && ( !pair->hasEmbeddedSeed || sat.minOverlap > pair->embeddedSeedPenetration ) )
		{
			pair->embeddedSeedCell0 = context->cell0;
			pair->embeddedSeedCell1 = cell1;
			pair->embeddedSeedPenetration = sat.minOverlap;
			pair->hasEmbeddedSeed = true;
		}
		return true;
	}
	if ( !b3Voxel_facesExposed( context->exposed0, sat.surfaceMask0, exposed1, sat.surfaceMask1 ) )
	{
		if ( pair->counters != NULL )
		{
			pair->counters->rawContactPoints += 1;
			pair->counters->surfaceRejects += 1;
		}
		b3VoxelOBB obb1 = pair->frame1;
		obb1.center = center1;
		b3Vec3 fallbackDelta = b3Sub( context->obb0.center, center1 );
		float fallbackOverlap = b3Voxel_findExposedSatOverlap( context->exposed0, exposed1, &pair->pairContext, fallbackDelta,
															   pair->contactDistance );
		if ( fallbackOverlap < FLT_MAX && ( !pair->hasFallback || fallbackOverlap < pair->fallbackOverlap ) )
		{
			pair->fallbackOverlap = fallbackOverlap;
			pair->fallbackObb0 = context->obb0;
			pair->fallbackObb1 = obb1;
			pair->fallbackDelta = fallbackDelta;
			pair->fallbackCell0 = context->cell0;
			pair->fallbackCell1 = cell1;
			pair->fallbackExposed0 = context->exposed0;
			pair->fallbackExposed1 = exposed1;
			pair->hasFallback = true;
		}
		return true;
	}
	b3VoxelOBB obb1 = pair->frame1;
	obb1.center = center1;
	int count = b3VoxelManifoldFromSat( &context->obb0, &obb1, &sat, pair->contactDistance, 1, contacts );
	int clusterIndex = -1;
	const float witnessTolerance = B3_LINEAR_SLOP + 32.0f * FLT_EPSILON;
	bool validSingle = b3Voxel_pointOnExposedFace( &context->obb0, contacts[0].body0Point, context->exposed0,
										 witnessTolerance ) &&
					   b3Voxel_pointOnExposedFace( &obb1, contacts[0].body1Point, exposed1, witnessTolerance );
	bool needsClip = !validSingle;
	if ( validSingle )
	{
		needsClip = b3VoxelReducer_ShouldClip( &pair->reducer, contacts, &clusterIndex );
	}
	if ( needsClip )
	{
		count = b3VoxelManifoldFromSat( &context->obb0, &obb1, &sat, pair->contactDistance, 4, contacts );
		if ( !validSingle )
		{
			clusterIndex = b3VoxelReducer_FindCluster( &pair->reducer, contacts[0].normal );
		}
	}
	else
	{
		if ( pair->counters != NULL )
		{
			pair->counters->rawContactPoints += 1;
		}
		contacts[0].featureId = b3Voxel_pairId( context->cell0, cell1, 0 );
		b3VoxelReducer_AddToCluster( &pair->reducer, contacts, clusterIndex );
		return true;
	}
	if ( pair->counters != NULL )
	{
		pair->counters->rawContactPoints += count;
	}
	for ( int k = 0; k < count; ++k )
	{
		if ( !b3Voxel_pointOnExposedFace( &context->obb0, contacts[k].body0Point, context->exposed0, witnessTolerance ) ||
			 !b3Voxel_pointOnExposedFace( &obb1, contacts[k].body1Point, exposed1, witnessTolerance ) )
		{
			if ( pair->counters != NULL )
			{
				pair->counters->surfaceRejects += 1;
			}
			continue;
		}
		contacts[k].featureId = b3Voxel_pairId( context->cell0, cell1, k );
		clusterIndex = b3VoxelReducer_AddToCluster( &pair->reducer, contacts + k, clusterIndex );
	}
	return true;
}

static bool b3Voxel_collideOuterCell( b3Vec3i cell0, uint8_t exposed0, void* rawContext )
{
	b3VoxelVoxelQueryContext* context = rawContext;
	if ( exposed0 == 0 && !context->collectEmbeddedSeed )
		return true;
	b3Vec3 localCenter0 = b3Voxel_cellCenterFromGrid( context->origin0, context->voxelSize0, cell0 );
	b3VoxelOBB obb0 = context->frame0;
	obb0.center = b3Voxel_xfPoint( context->transform0, localCenter0 );
	b3VoxelOBB localObb0 = context->frame0In1;
	localObb0.center = b3TransformPoint( context->transform0To1, localCenter0 );
	b3AABB query1 = b3Voxel_expandB( b3VoxelOBB_Bounds( &localObb0 ), context->contactDistance );
	b3VoxelVoxelInnerContext inner = { context, cell0, exposed0, obb0 };
	b3Voxel_ForEachCellTracked( context->voxels1, query1, b3Voxel_collideInnerCell, &inner, context->counters );
	return true;
}

typedef struct b3VoxelSupportContext
{
	b3Vec3 direction;
	b3Vec3i cell;
	float projection;
	bool found;
} b3VoxelSupportContext;

static bool b3Voxel_findSupportCell( b3Vec3i cell, uint8_t exposed, void* rawContext )
{
	B3_UNUSED( exposed );
	b3VoxelSupportContext* context = rawContext;
	float projection = cell.x * context->direction.x + cell.y * context->direction.y + cell.z * context->direction.z;
	if ( !context->found || projection > context->projection )
	{
		context->cell = cell;
		context->projection = projection;
		context->found = true;
	}
	return true;
}

static b3Vec3 b3Voxel_supportPoint( const b3VoxelData* voxels, b3Transform transform, b3Vec3 worldDirection )
{
	b3Vec3 localDirection = b3InvRotateVector( transform.q, worldDirection );
	b3VoxelSupportContext context = { .direction = localDirection, .projection = -FLT_MAX };
	b3AABB bounds;
	bool hasBounds = b3Voxel_GetLocalBounds( voxels, &bounds );
	B3_ASSERT( hasBounds );
	if ( !hasBounds )
		return transform.p;
	b3Voxel_ForEachCellTracked( voxels, bounds, b3Voxel_findSupportCell, &context, NULL );
	B3_ASSERT( context.found );
	b3Vec3 point = b3Voxel_GetCellCenter( voxels, context.cell );
	float half = 0.5f * b3Voxel_GetVoxelSize( voxels );
	for ( int axis = 0; axis < 3; ++axis )
	{
		float direction = ( &localDirection.x )[axis];
		if ( direction > 1.0e-8f )
			( &point.x )[axis] += half;
		else if ( direction < -1.0e-8f )
			( &point.x )[axis] -= half;
	}
	return b3Voxel_xfPoint( transform, point );
}

static b3Vec3i b3Voxel_findBoundaryCell( const b3VoxelData* voxels, b3Vec3i seed, int axis, int direction )
{
	b3Vec3i boundary = seed;
	int cellCount = b3Voxel_GetCellCount( voxels );
	for ( int step = 0; step < cellCount; ++step )
	{
		b3Vec3i next = boundary;
		int* coordinate = &( &next.x )[axis];
		if ( ( direction < 0 && *coordinate == INT_MIN ) || ( direction > 0 && *coordinate == INT_MAX ) )
			break;
		*coordinate += direction;
		if ( !b3VoxelData_IsSolid( voxels, next ) )
			break;
		boundary = next;
	}
	return boundary;
}

static void b3Voxel_considerEmbeddedEscape( const b3VoxelVoxelQueryContext* context, int containerIndex, int bit,
											b3VoxelContact* best, bool* hasBest )
{
	const b3VoxelData* container = containerIndex == 0 ? context->voxels0 : context->voxels1;
	const b3VoxelData* other = containerIndex == 0 ? context->voxels1 : context->voxels0;
	b3Transform containerTransform = containerIndex == 0 ? context->transform0 : context->transform1;
	b3Transform otherTransform = containerIndex == 0 ? context->transform1 : context->transform0;
	b3Vec3i seed = containerIndex == 0 ? context->embeddedSeedCell0 : context->embeddedSeedCell1;
	int axis = bit / 2;
	int direction = ( bit & 1 ) != 0 ? 1 : -1;
	b3Vec3i boundary = b3Voxel_findBoundaryCell( container, seed, axis, direction );
	b3Vec3 localOutward = b3Vec3_zero;
	( &localOutward.x )[axis] = (float)direction;
	b3Vec3 outward = b3RotateVector( containerTransform.q, localOutward );
	b3Vec3 normal = containerIndex == 0 ? b3Neg( outward ) : outward;
	b3Vec3 otherDirection = containerIndex == 0 ? normal : b3Neg( normal );
	b3Vec3 otherPoint = b3Voxel_supportPoint( other, otherTransform, otherDirection );
	b3Vec3 localOtherPoint = b3Voxel_invXfPoint( containerTransform, otherPoint );
	b3Vec3 boundaryCenter = b3Voxel_GetCellCenter( container, boundary );
	float half = 0.5f * b3Voxel_GetVoxelSize( container );
	b3Vec3 localBoundaryPoint = localOtherPoint;
	for ( int tangentAxis = 0; tangentAxis < 3; ++tangentAxis )
	{
		float center = ( &boundaryCenter.x )[tangentAxis];
		if ( tangentAxis == axis )
			( &localBoundaryPoint.x )[tangentAxis] = center + direction * half;
		else
			( &localBoundaryPoint.x )[tangentAxis] =
				b3ClampFloat( ( &localBoundaryPoint.x )[tangentAxis], center - half, center + half );
	}
	b3Vec3 boundaryPoint = b3Voxel_xfPoint( containerTransform, localBoundaryPoint );
	b3Vec3 body0Point = containerIndex == 0 ? boundaryPoint : otherPoint;
	b3Vec3 body1Point = containerIndex == 0 ? otherPoint : boundaryPoint;
	float penetration = b3Dot( b3Sub( body1Point, body0Point ), normal );
	uint32_t featureId = b3Voxel_pairId( context->embeddedSeedCell0, context->embeddedSeedCell1, 6 * containerIndex + bit );
	if ( penetration > 0.0f && ( !*hasBest || penetration < best->initialPenetration ||
								 ( penetration == best->initialPenetration && featureId < best->featureId ) ) )
	{
		*best = (b3VoxelContact){
			.normal = normal,
			.initialPenetration = penetration,
			.penetrationDepth = penetration,
			.body0Point = body0Point,
			.body1Point = body1Point,
			.featureId = featureId,
		};
		*hasBest = true;
	}
}

static int b3VoxelCollideImpl( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
							   float contactDistance, int maxContacts, b3VoxelContact* out, b3VoxelCounters* counters )
{
	if ( counters != NULL )
	{
		counters->voxelVoxelCalls += 1;
	}
	if ( maxContacts < 1 )
		return 0;
	if ( maxContacts > B3_VOXEL_MAX_CONTACTS )
		maxContacts = B3_VOXEL_MAX_CONTACTS;

	b3AABB lb0, lb1;
	if ( !b3Voxel_GetLocalBounds( v0, &lb0 ) || !b3Voxel_GetLocalBounds( v1, &lb1 ) )
		return 0;
	if ( contactDistance < 0.0f )
		contactDistance = 0.0f;

	b3AABB wb0 = b3Voxel_mapBounds( lb0, xf0, false );
	b3AABB wb1 = b3Voxel_mapBounds( lb1, xf1, false );
	if ( !b3Voxel_isect( b3Voxel_expandB( wb0, contactDistance ), wb1 ) )
		return 0;
	bool possibleContainment = b3Voxel_containsAABB( wb0, wb1 ) || b3Voxel_containsAABB( wb1, wb0 );

	// Stream complete query results directly into exact leaf collision. This
	// avoids the prototype's count/fill scan pairs and count-sized scratch while
	// preserving deterministic chunk/cell enumeration and uncapped candidates.
	b3AABB query0 = b3Voxel_mapBounds( b3Voxel_expandB( wb1, contactDistance ), xf0, true );
	b3VoxelVoxelQueryContext context = { .voxels0 = v0,
										 .voxels1 = v1,
										 .transform0 = xf0,
										 .transform1 = xf1,
										 .transform0To1 = b3InvMulTransforms( xf1, xf0 ),
										 .origin0 = b3Voxel_GetOrigin( v0 ),
										 .origin1 = b3Voxel_GetOrigin( v1 ),
										 .voxelSize0 = b3Voxel_GetVoxelSize( v0 ),
										 .voxelSize1 = b3Voxel_GetVoxelSize( v1 ),
										 .contactDistance = contactDistance,
										 .counters = counters };
	context.frame0 = b3Voxel_cellFrame( v0, xf0 );
	context.frame1 = b3Voxel_cellFrame( v1, xf1 );
	context.frame0In1 = context.frame0;
	context.frame0In1.center = b3Vec3_zero;
	for ( int i = 0; i < 3; ++i )
	{
		context.frame0In1.axes[i] = b3InvRotateVector( xf1.q, context.frame0.axes[i] );
	}
	b3VoxelPrepareOBBPair( &context.pairContext, &context.frame0, &context.frame1 );
	b3Obb_prepareSurfaceMasks( &context.pairContext, xf0.q, xf1.q );
	b3VoxelReducer_Init( &context.reducer, maxContacts, counters );
	b3Voxel_ForEachCellTracked( v0, query0, b3Voxel_collideOuterCell, &context, counters );
	if ( context.reducer.clusterCount == 0 && context.hasFallback )
	{
		b3VoxelContact contacts[4];
		float fallbackOverlap;
		int count = b3Voxel_findExposedSatManifold( &context.fallbackObb0, context.fallbackExposed0, &context.fallbackObb1,
													context.fallbackExposed1, &context.pairContext, context.fallbackDelta,
													contactDistance, contacts, &fallbackOverlap );
		B3_UNUSED( fallbackOverlap );
		if ( counters != NULL )
		{
			counters->rawContactPoints += count;
		}
		for ( int i = 0; i < count; ++i )
		{
			contacts[i].featureId = b3Voxel_pairId( context.fallbackCell0, context.fallbackCell1, i );
			b3VoxelReducer_Add( &context.reducer, contacts + i );
		}
	}
	if ( context.reducer.clusterCount == 0 && possibleContainment )
	{
		// Surface-only traversal intentionally ignores buried cells. A complete
		// containment can therefore have no surface pair even though occupied
		// cells overlap. Replay only this rare miss to find an overlapping seed,
		// then construct the shallowest escape through either union boundary.
		context.collectEmbeddedSeed = true;
		b3Voxel_ForEachCellTracked( v0, query0, b3Voxel_collideOuterCell, &context, counters );
		context.collectEmbeddedSeed = false;
		if ( context.hasEmbeddedSeed )
		{
			b3VoxelContact escape;
			bool hasEscape = false;
			for ( int containerIndex = 0; containerIndex < 2; ++containerIndex )
			{
				for ( int bit = 0; bit < 6; ++bit )
				{
					b3Voxel_considerEmbeddedEscape( &context, containerIndex, bit, &escape, &hasEscape );
				}
			}
			if ( hasEscape )
			{
				b3VoxelReducer_Add( &context.reducer, &escape );
				if ( counters != NULL )
				{
					counters->deepOverlapFallbacks += 1;
				}
			}
		}
	}
	return b3VoxelReducer_Finish( &context.reducer, out );
}

int b3VoxelCollideWithArena( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
							 float contactDistance, int maxContacts, b3VoxelContact* out, b3Arena* arena,
							 b3VoxelCounters* counters )
{
	B3_ASSERT( arena != NULL );
	if ( !b3VoxelUseCanonicalPatches( v0, xf0.q, v1, xf1.q ) )
	{
		return b3VoxelCollideLeafTracked( v0, xf0, v1, xf1, contactDistance, maxContacts, out, counters );
	}
	return b3VoxelCollideCanonicalWithArena( v0, xf0, v1, xf1, contactDistance, maxContacts, out, arena, counters );
}

static bool b3VoxelOrientationsAreAxisAligned( b3Quat q0, b3Quat q1 )
{
	b3Matrix3 rotation = b3MakeMatrixFromQuat( b3InvMulQuat( q0, q1 ) );
	b3Vec3 columns[3] = { b3Abs( rotation.cx ), b3Abs( rotation.cy ), b3Abs( rotation.cz ) };
	for ( int i = 0; i < 3; ++i )
	{
		float alignment = b3MaxFloat( columns[i].x, b3MaxFloat( columns[i].y, columns[i].z ) );
		if ( alignment < 1.0f - 100.0f * FLT_EPSILON )
		{
			return false;
		}
	}
	return true;
}

bool b3VoxelUseCanonicalPatches( const b3VoxelData* v0, b3Quat q0, const b3VoxelData* v1, b3Quat q1 )
{
	// Rotated medium/medium contacts do not provide enough repeated patch work
	// to repay exact per-key witness validation. Aligned pairs cheaply reuse
	// broad face patches. Very large pairs do too: the measured destruction scene
	// begins above 9k cells on both sides and strongly favors patch compression.
	// Small fragments also favor patches because one side has little discovery
	// work. Keep a wide, measured gap between regimes.
	enum
	{
		b3_leafPairMinCells = 512,
		b3_canonicalLargePairCells = 4096,
	};
	int count0 = b3Voxel_GetCellCount( v0 );
	int count1 = b3Voxel_GetCellCount( v1 );
	int smaller = b3MinInt( count0, count1 );
	int larger = b3MaxInt( count0, count1 );
	bool mediumPair = smaller >= b3_leafPairMinCells && larger < b3_canonicalLargePairCells;
	return !mediumPair || b3VoxelOrientationsAreAxisAligned( q0, q1 );
}

int b3VoxelCollideLeafTracked( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
							  float contactDistance, int maxContacts, b3VoxelContact* out, b3VoxelCounters* counters )
{
	if ( counters != NULL )
	{
		counters->adaptiveLeafPairs += 1;
	}
	return b3VoxelCollideImpl( v0, xf0, v1, xf1, contactDistance, maxContacts, out, counters );
}

int b3VoxelCollide( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1, float contactDistance,
					int maxContacts, b3VoxelContact* out )
{
	return b3VoxelCollideImpl( v0, xf0, v1, xf1, contactDistance, maxContacts, out, NULL );
}

typedef enum b3VoxelCanonicalState
{
	b3_voxelPatchSeparated,
	b3_voxelPatchCandidate,
} b3VoxelCanonicalState;

typedef struct b3VoxelCanonicalAlias
{
	struct b3VoxelCanonicalAlias* next;
	b3Vec3i cell0;
	b3Vec3i cell1;
	uint8_t exposed0;
	uint8_t exposed1;
} b3VoxelCanonicalAlias;

typedef struct b3VoxelCanonicalEntry
{
	b3VoxelPatchManifold manifold;
	// Pseudo witnesses projected onto each voxel body's grid axes from that
	// grid's world-space origin. Alias membership then needs only the integer
	// cell coordinate, rather than rebuilding an OBB and repeating three dots.
	b3Vec3 gridPoint0[B3_VOXEL_POINTS_PER_CLUSTER];
	b3Vec3 gridPoint1[B3_VOXEL_POINTS_PER_CLUSTER];
	b3VoxelCanonicalAlias* firstUnselectedAlias;
	b3VoxelCanonicalAlias* lastUnselectedAlias;
	int representedAliases;
	uint8_t exposed0;
	uint8_t exposed1;
	uint8_t selectedMask;
	uint8_t state;
} b3VoxelCanonicalEntry;

typedef struct b3VoxelCanonicalContext
{
	const b3VoxelData* voxels0;
	const b3VoxelData* voxels1;
	b3Transform transform0;
	b3Transform transform1;
	b3Transform transform0To1;
	b3Transform transform1To0;
	b3Vec3 origin0;
	b3Vec3 origin1;
	b3Vec3 worldOrigin0;
	b3Vec3 worldOrigin1;
	b3Vec3i domainMin0;
	b3Vec3i domainMax0;
	b3Vec3i domainMin1;
	b3Vec3i domainMax1;
	float voxelSize0;
	float voxelSize1;
	float contactDistance;
	b3AABB clip0;
	b3AABB clip1;
	b3VoxelOBB frame0;
	b3VoxelOBB frame1;
	b3VoxelOBB frame0In1;
	b3VoxelOBB frame1In0;
	b3ObbPairContext leafPairContext;
	float absRotation[3][3];
	float edgeInvLength[3][3];
	b3VoxelPatchTable table;
	b3VoxelCanonicalEntry* entries;
	int entryCapacity;
	int scratchBytes;
	b3VoxelCounters* counters;
} b3VoxelCanonicalContext;

typedef struct b3VoxelCanonicalInnerContext
{
	b3VoxelCanonicalContext* pair;
	b3Vec3i outerCell;
	uint8_t outerExposed;
	b3VoxelTopology outerTopology;
	bool swapped;
} b3VoxelCanonicalInnerContext;

static b3AABB b3Voxel_intersectBounds( b3AABB a, b3AABB b )
{
	return (b3AABB){
		{ b3MaxFloat( a.lowerBound.x, b.lowerBound.x ), b3MaxFloat( a.lowerBound.y, b.lowerBound.y ),
		  b3MaxFloat( a.lowerBound.z, b.lowerBound.z ) },
		{ b3MinFloat( a.upperBound.x, b.upperBound.x ), b3MinFloat( a.upperBound.y, b.upperBound.y ),
		  b3MinFloat( a.upperBound.z, b.upperBound.z ) },
	};
}

static bool b3Voxel_makePatchOBB( b3VoxelOBB* obb, b3VoxelPatchRange range, b3Vec3i representativeCell, b3Vec3 origin,
								  float voxelSize, b3AABB clip, const b3VoxelOBB* frame, b3Transform transform )
{
	b3Vec3 lower;
	b3Vec3 upper;
	for ( int axis = 0; axis < 3; ++axis )
	{
		float lo = ( &origin.x )[axis] + voxelSize * (float)( &range.lower.x )[axis];
		float hi = ( &origin.x )[axis] + voxelSize * (float)( &range.upper.x )[axis];
		if ( ( &range.lower.x )[axis] != ( &representativeCell.x )[axis] )
		{
			lo = b3MaxFloat( lo, ( &clip.lowerBound.x )[axis] );
		}
		if ( ( &range.upper.x )[axis] != ( &representativeCell.x )[axis] )
		{
			hi = b3MinFloat( hi, ( &clip.upperBound.x )[axis] );
		}
		if ( !isfinite( lo ) || !isfinite( hi ) || hi < lo )
			return false;
		( &lower.x )[axis] = lo;
		( &upper.x )[axis] = hi;
	}

	*obb = *frame;
	b3Vec3 localCenter = b3MulSV( 0.5f, b3Add( lower, upper ) );
	obb->center = b3Voxel_xfPoint( transform, localCenter );
	obb->half = b3Add( b3MulSV( 0.5f, b3Sub( upper, lower ) ),
					   (b3Vec3){ 0.5f * voxelSize, 0.5f * voxelSize, 0.5f * voxelSize } );
	return obb->half.x > 0.0f && obb->half.y > 0.0f && obb->half.z > 0.0f && b3IsValidVec3( obb->half );
}

static void b3Voxel_growCanonicalEntries( b3VoxelCanonicalContext* context )
{
	int newCapacity = context->table.keyCapacity;
	B3_ASSERT( newCapacity > context->entryCapacity );
	b3VoxelCanonicalEntry* entries = b3Bump( context->table.arena, newCapacity * (int)sizeof( b3VoxelCanonicalEntry ) );
	if ( context->table.count > 1 )
	{
		memcpy( entries, context->entries, (size_t)( context->table.count - 1 ) * sizeof( b3VoxelCanonicalEntry ) );
	}
	context->entries = entries;
	context->entryCapacity = newCapacity;
	context->scratchBytes += newCapacity * (int)sizeof( b3VoxelCanonicalEntry );
}

static uint32_t b3Voxel_patchFeatureId( uint64_t hash, int pointIndex )
{
	uint32_t feature = (uint32_t)hash ^ (uint32_t)( hash >> 32 ) ^ (uint32_t)( pointIndex + 1 ) * 0x9e3779b1u;
	return feature != 0 ? feature : 1u;
}

static b3Vec3 b3Voxel_projectToGridAxes( b3Vec3 point, b3Vec3 worldOrigin, const b3VoxelOBB* frame )
{
	b3Vec3 offset = b3Sub( point, worldOrigin );
	return (b3Vec3){ b3Dot( offset, frame->axes[0] ), b3Dot( offset, frame->axes[1] ),
					 b3Dot( offset, frame->axes[2] ) };
}

static bool b3Voxel_gridPointOnExposedFace( b3Vec3 point, b3Vec3i cell, float voxelSize, uint8_t exposed,
											 float tolerance )
{
	float half = 0.5f * voxelSize;
	b3Vec3 offset = {
		point.x - (float)cell.x * voxelSize,
		point.y - (float)cell.y * voxelSize,
		point.z - (float)cell.z * voxelSize,
	};
	return b3VoxelPointOnExposedCubeFace( offset, half, exposed, tolerance );
}

static int b3Voxel_findCanonicalPseudoManifold( const b3VoxelOBB* obb0, uint8_t exposed0, const b3VoxelOBB* obb1,
										 uint8_t exposed1, const b3ObbPairContext* pairContext, b3Vec3 delta,
										 float contactDistance, b3VoxelContact out[4], bool* separated,
										 bool* minimumIsEdge )
{
	b3ObbSat broadSat = { .minOverlap = FLT_MAX };
	int bestAxisIndex = -1;
	int bestPositive = 0;
	float bestOverlap = FLT_MAX;
	for ( int axisIndex = 0; axisIndex < pairContext->count; ++axisIndex )
	{
		const b3ObbPairAxis* prepared = pairContext->axes + axisIndex;
		float projection = b3Dot( prepared->axis, delta );
		float overlap = prepared->radius - fabsf( projection );
		if ( !b3Obb_updatePreparedAxis( &broadSat, prepared, projection, overlap, contactDistance ) )
		{
			*separated = true;
			*minimumIsEdge = false;
			return 0;
		}
		int positive = projection >= 0.0f;
		if ( b3Voxel_facesExposed( exposed0, prepared->surfaceMask0[positive], exposed1,
								  prepared->surfaceMask1[positive] ) &&
			 overlap < bestOverlap )
		{
			bestAxisIndex = axisIndex;
			bestPositive = positive;
			bestOverlap = overlap;
		}
	}
	if ( broadSat.minOverlap == FLT_MAX )
	{
		*separated = true;
		*minimumIsEdge = false;
		return 0;
	}
	*separated = false;
	*minimumIsEdge = broadSat.minPrepared->isEdge;
	if ( bestAxisIndex < 0 )
		return 0;

	const b3ObbPairAxis* prepared = pairContext->axes + bestAxisIndex;
	b3ObbSat alternate = {
		.minOverlap = bestOverlap,
		.minAxis = bestPositive ? prepared->axis : b3Neg( prepared->axis ),
		.surfaceMask0 = prepared->surfaceMask0[bestPositive],
		.surfaceMask1 = prepared->surfaceMask1[bestPositive],
		.srcBox = prepared->srcBox,
		.isEdge = prepared->isEdge,
		.minPrepared = prepared,
		.minPositive = bestPositive,
	};
	b3VoxelContact raw[8];
	int rawCount = b3VoxelManifoldFromSat( obb0, obb1, &alternate, contactDistance, 1, raw );
	rawCount += b3VoxelManifoldFromSat( obb0, obb1, &alternate, contactDistance, 4, raw + rawCount );
	b3VoxelContact valid[8];
	int validCount = 0;
	for ( int i = 0; i < rawCount; ++i )
	{
		bool duplicate = false;
		for ( int j = 0; j < validCount; ++j )
		{
			if ( b3LengthSquared( b3Sub( valid[j].body0Point, raw[i].body0Point ) ) < 1.0e-12f &&
				 b3LengthSquared( b3Sub( valid[j].body1Point, raw[i].body1Point ) ) < 1.0e-12f )
			{
				duplicate = true;
				break;
			}
		}
		if ( !duplicate )
			valid[validCount++] = raw[i];
	}
	int bestCount = validCount <= 4 ? validCount : b3Obb_reduce( valid, validCount, 4, out );
	if ( validCount <= 4 )
		memcpy( out, valid, (size_t)validCount * sizeof( b3VoxelContact ) );
	return bestCount;
}

static void b3Voxel_buildCanonicalEntry( b3VoxelCanonicalContext* context, int entryIndex, const b3VoxelPatchKey* key,
									 b3Vec3i cell0, uint8_t exposed0, b3Vec3i cell1, uint8_t exposed1 )
{
	b3VoxelCanonicalEntry* entry = context->entries + entryIndex;
	memset( entry, 0, sizeof( *entry ) );
	entry->manifold.key = *key;
	entry->manifold.discoveryOrdinal = entryIndex;
	entry->exposed0 = exposed0;
	entry->exposed1 = exposed1;

	b3VoxelOBB pseudo0;
	b3VoxelOBB pseudo1;
	bool valid0 = b3Voxel_makePatchOBB( &pseudo0, key->patch0, cell0, context->origin0, context->voxelSize0, context->clip0,
									  &context->frame0, context->transform0 );
	bool valid1 = b3Voxel_makePatchOBB( &pseudo1, key->patch1, cell1, context->origin1, context->voxelSize1, context->clip1,
									  &context->frame1, context->transform1 );
	if ( !valid0 || !valid1 )
	{
		entry->state = b3_voxelPatchSeparated;
		if ( context->counters != NULL )
			context->counters->pseudoSeparatedKeys += 1;
		return;
	}

	b3ObbPairContext pairContext;
	b3Obb_preparePatchPair( &pairContext, &context->leafPairContext, pseudo0.half, pseudo1.half,
							context->absRotation, context->edgeInvLength );
	if ( context->counters != NULL )
		context->counters->pseudoSatCalls += 1;
	bool separated;
	bool minimumIsEdge;
	entry->manifold.pointCount = b3Voxel_findCanonicalPseudoManifold(
		&pseudo0, exposed0, &pseudo1, exposed1, &pairContext, b3Sub( pseudo0.center, pseudo1.center ),
		context->contactDistance, entry->manifold.points, &separated, &minimumIsEdge );
	if ( separated )
	{
		entry->state = b3_voxelPatchSeparated;
		if ( context->counters != NULL )
			context->counters->pseudoSeparatedKeys += 1;
		return;
	}
	if ( context->counters != NULL )
	{
		if ( minimumIsEdge )
			context->counters->patchSatEdgeAxes += 1;
		else
			context->counters->patchSatFaceAxes += 1;
	}
	uint64_t featureHash = b3VoxelPatchKeyHash( key );
	for ( int i = 0; i < entry->manifold.pointCount; ++i )
	{
		entry->gridPoint0[i] = b3Voxel_projectToGridAxes( entry->manifold.points[i].body0Point, context->worldOrigin0,
														&context->frame0 );
		entry->gridPoint1[i] = b3Voxel_projectToGridAxes( entry->manifold.points[i].body1Point, context->worldOrigin1,
														&context->frame1 );
		entry->manifold.points[i].featureId = b3Voxel_patchFeatureId( featureHash, i );
	}
	entry->state = b3_voxelPatchCandidate;
}

static void b3Voxel_selectCanonicalAlias( b3VoxelCanonicalContext* context, b3VoxelCanonicalEntry* entry, b3Vec3i cell0,
									   uint8_t exposed0, b3Vec3i cell1, uint8_t exposed1 )
{
	entry->representedAliases += 1;
	if ( context->counters != NULL )
		context->counters->representedLeafPairs += 1;
	if ( entry->state == b3_voxelPatchSeparated )
		return;

	const float witnessTolerance = B3_LINEAR_SLOP + 32.0f * FLT_EPSILON;
	b3VoxelOBB real0;
	b3VoxelOBB real1;
	bool hasRealCells = false;
	uint8_t selectedBefore = entry->selectedMask;
	for ( int i = 0; i < entry->manifold.pointCount; ++i )
	{
		uint8_t bit = (uint8_t)( 1u << i );
		if ( ( entry->selectedMask & bit ) != 0 )
			continue;
		const b3VoxelContact* point = entry->manifold.points + i;
		if ( !b3Voxel_gridPointOnExposedFace( entry->gridPoint0[i], cell0, context->voxelSize0, exposed0,
												 witnessTolerance ) ||
			 !b3Voxel_gridPointOnExposedFace( entry->gridPoint1[i], cell1, context->voxelSize1, exposed1,
												 witnessTolerance ) )
		{
			if ( context->counters != NULL )
				context->counters->pseudoWitnessRejects += 1;
			continue;
		}
		if ( point->initialPenetration > 0.0f )
		{
			if ( !hasRealCells )
			{
				real0 = b3Voxel_cellOBB( &context->frame0, context->voxels0, cell0, context->transform0 );
				real1 = b3Voxel_cellOBB( &context->frame1, context->voxels1, cell1, context->transform1 );
				hasRealCells = true;
			}
			b3Vec3 support0 = b3VoxelObbSupport( &real0, b3Neg( point->normal ) );
			b3Vec3 support1 = b3VoxelObbSupport( &real1, point->normal );
			float realPenetration = b3Dot( b3Sub( support1, support0 ), point->normal );
			float tolerance = B3_LINEAR_SLOP + 64.0f * FLT_EPSILON *
											( 1.0f + context->voxelSize0 + context->voxelSize1 + point->initialPenetration );
			if ( realPenetration + tolerance < point->initialPenetration )
			{
				if ( context->counters != NULL )
					context->counters->pseudoDepthRejects += 1;
				continue;
			}
		}
		entry->selectedMask |= bit;
	}
	if ( selectedBefore == 0 && entry->selectedMask != 0 && entry->representedAliases > 1 && context->counters != NULL )
		context->counters->pseudoLateSelections += 1;
	if ( entry->selectedMask == 0 )
	{
		b3VoxelCanonicalAlias* alias = b3Bump( context->table.arena, sizeof( b3VoxelCanonicalAlias ) );
		*alias = (b3VoxelCanonicalAlias){
			.cell0 = cell0,
			.cell1 = cell1,
			.exposed0 = exposed0,
			.exposed1 = exposed1,
		};
		if ( entry->lastUnselectedAlias != NULL )
			entry->lastUnselectedAlias->next = alias;
		else
			entry->firstUnselectedAlias = alias;
		entry->lastUnselectedAlias = alias;
		context->scratchBytes += sizeof( b3VoxelCanonicalAlias );
	}
}

static void b3Voxel_visitCanonicalPair( b3VoxelCanonicalContext* context, b3Vec3i cell0, uint8_t exposed0,
									 b3VoxelTopology topology0, b3Vec3i cell1, uint8_t exposed1,
									 b3VoxelTopology topology1 )
{
	b3VoxelPatchKey key = {
		.patch0 = b3VoxelCanonicalPatchRange( cell0, exposed0, context->domainMin0, context->domainMax0 ),
		.patch1 = b3VoxelCanonicalPatchRange( cell1, exposed1, context->domainMin1, context->domainMax1 ),
	};
	bool inserted = false;
	int entryIndex = b3VoxelPatchTableFindOrInsert( &context->table, &key, b3VoxelPatchKeyTableHash( &key ), &inserted );
	if ( context->counters != NULL )
	{
		context->counters->patchVisits += 1;
		context->counters->patchEligibleVisits += 1;
		context->counters->patchTopologyPairs[4 * (int)topology0 + (int)topology1] += 1;
		if ( inserted )
		{
			context->counters->patchUniqueKeys += 1;
			context->counters->patchEligibleUniqueKeys += 1;
		}
		else
		{
			context->counters->patchDuplicateVisits += 1;
		}
	}
	if ( inserted )
	{
		if ( context->entryCapacity < context->table.keyCapacity )
			b3Voxel_growCanonicalEntries( context );
		b3Voxel_buildCanonicalEntry( context, entryIndex, &key, cell0, exposed0, cell1, exposed1 );
	}
	b3Voxel_selectCanonicalAlias( context, context->entries + entryIndex, cell0, exposed0, cell1, exposed1 );
}

static bool b3Voxel_canonicalInnerCell( b3Vec3i innerCell, uint8_t innerExposed, void* rawContext )
{
	b3VoxelCanonicalInnerContext* inner = rawContext;
	b3VoxelTopology innerTopology = b3VoxelClassifyTopology( innerExposed );
	if ( inner->swapped )
	{
		if ( inner->outerTopology != b3_voxelVertex || innerTopology != b3_voxelFace )
		{
			if ( inner->pair->counters != NULL )
				inner->pair->counters->topologyPrunedPairs += 1;
			return true;
		}
		b3Voxel_visitCanonicalPair( inner->pair, innerCell, innerExposed, innerTopology, inner->outerCell,
									inner->outerExposed, inner->outerTopology );
	}
	else
	{
		if ( !b3Voxel_patchPairEligible( inner->outerTopology, innerTopology ) )
		{
			if ( inner->pair->counters != NULL )
				inner->pair->counters->topologyPrunedPairs += 1;
			return true;
		}
		b3Voxel_visitCanonicalPair( inner->pair, inner->outerCell, inner->outerExposed, inner->outerTopology, innerCell,
									innerExposed, innerTopology );
	}
	return true;
}

static bool b3Voxel_canonicalOuter0Cell( b3Vec3i cell0, uint8_t exposed0, void* rawContext )
{
	b3VoxelCanonicalContext* context = rawContext;
	b3VoxelTopology topology0 = b3VoxelClassifyTopology( exposed0 );
	if ( topology0 != b3_voxelVertex && topology0 != b3_voxelEdge )
		return true;
	b3VoxelOBB local = context->frame0In1;
	local.center = b3TransformPoint( context->transform0To1, b3Voxel_GetCellCenter( context->voxels0, cell0 ) );
	b3AABB query1 = b3Voxel_expandB( b3VoxelOBB_Bounds( &local ), context->contactDistance );
	b3VoxelCanonicalInnerContext inner = { context, cell0, exposed0, topology0, false };
	b3Voxel_ForEachCellTracked( context->voxels1, query1, b3Voxel_canonicalInnerCell, &inner, context->counters );
	return true;
}

static bool b3Voxel_canonicalOuter1Cell( b3Vec3i cell1, uint8_t exposed1, void* rawContext )
{
	b3VoxelCanonicalContext* context = rawContext;
	b3VoxelTopology topology1 = b3VoxelClassifyTopology( exposed1 );
	if ( topology1 != b3_voxelVertex )
		return true;
	b3VoxelOBB local = context->frame1In0;
	local.center = b3TransformPoint( context->transform1To0, b3Voxel_GetCellCenter( context->voxels1, cell1 ) );
	b3AABB query0 = b3Voxel_expandB( b3VoxelOBB_Bounds( &local ), context->contactDistance );
	b3VoxelCanonicalInnerContext inner = { context, cell1, exposed1, topology1, true };
	b3Voxel_ForEachCellTracked( context->voxels0, query0, b3Voxel_canonicalInnerCell, &inner, context->counters );
	return true;
}

static int b3Voxel_comparePatchKeys( const b3VoxelPatchKey* a, const b3VoxelPatchKey* b )
{
	const b3VoxelPatchRange* rangesA[2] = { &a->patch0, &a->patch1 };
	const b3VoxelPatchRange* rangesB[2] = { &b->patch0, &b->patch1 };
	for ( int range = 0; range < 2; ++range )
	{
		for ( int bound = 0; bound < 2; ++bound )
		{
			const b3Vec3i* va = bound == 0 ? &rangesA[range]->lower : &rangesA[range]->upper;
			const b3Vec3i* vb = bound == 0 ? &rangesB[range]->lower : &rangesB[range]->upper;
			for ( int axis = 0; axis < 3; ++axis )
			{
				int32_t ca = ( &va->x )[axis];
				int32_t cb = ( &vb->x )[axis];
				if ( ca != cb )
					return ca < cb ? -1 : 1;
			}
		}
	}
	return 0;
}

static float b3Voxel_patchDepth( const b3VoxelCanonicalEntry* entry )
{
	float depth = -FLT_MAX;
	for ( int i = 0; i < entry->manifold.pointCount; ++i )
		depth = b3MaxFloat( depth, entry->manifold.points[i].initialPenetration );
	return depth;
}

static int b3Voxel_replayEmptyCanonicalEntry( b3VoxelCanonicalContext* context, b3VoxelCanonicalEntry* entry )
{
	b3VoxelContactReducer reducer;
	b3VoxelReducer_Init( &reducer, B3_VOXEL_POINTS_PER_CLUSTER, NULL );
	for ( b3VoxelCanonicalAlias* alias = entry->firstUnselectedAlias; alias != NULL; alias = alias->next )
	{
		b3VoxelOBB real0 = b3Voxel_cellOBB( &context->frame0, context->voxels0, alias->cell0, context->transform0 );
		b3VoxelOBB real1 = b3Voxel_cellOBB( &context->frame1, context->voxels1, alias->cell1, context->transform1 );
		if ( context->counters != NULL )
		{
			context->counters->obbTests += 1;
			context->counters->patchLeafFallbackTests += 1;
		}
		b3Vec3 delta = b3Sub( real0.center, real1.center );
		b3ObbSat sat;
		if ( !b3VoxelComputeOBBSat( delta, &context->leafPairContext, context->contactDistance, &sat ) )
			continue;
		b3VoxelContact contacts[4];
		float overlap;
		int count = b3Voxel_findExposedSatManifold( &real0, alias->exposed0, &real1, alias->exposed1,
											  &context->leafPairContext, delta, context->contactDistance, contacts, &overlap );
		B3_UNUSED( overlap );
		for ( int point = 0; point < count; ++point )
		{
			contacts[point].featureId = b3Voxel_pairId( alias->cell0, alias->cell1, point );
			b3VoxelReducer_Add( &reducer, contacts + point );
		}
	}
	int count = b3VoxelReducer_Finish( &reducer, entry->manifold.points );
	entry->manifold.pointCount = count;
	if ( count > 0 && context->counters != NULL )
		context->counters->emptyPatchFallbackKeys += 1;
	return count;
}

static int b3Voxel_patchDeepestPoint( const b3VoxelCanonicalEntry* entry )
{
	int deepest = 0;
	for ( int i = 1; i < entry->manifold.pointCount; ++i )
	{
		if ( entry->manifold.points[i].initialPenetration > entry->manifold.points[deepest].initialPenetration )
			deepest = i;
	}
	return deepest;
}

static void b3Voxel_sortPatchIndicesByKey( const b3VoxelCanonicalContext* context, int* indices, int count )
{
	for ( int i = 1; i < count; ++i )
	{
		int value = indices[i];
		int j = i - 1;
		while ( j >= 0 &&
				b3Voxel_comparePatchKeys( &context->entries[indices[j]].manifold.key, &context->entries[value].manifold.key ) > 0 )
		{
			indices[j + 1] = indices[j];
			j -= 1;
		}
		indices[j + 1] = value;
	}
}

static int b3Voxel_reduceCanonicalPatches( b3VoxelCanonicalContext* context, int* candidates, int candidateCount,
										   int selected[B3_VOXEL_MAX_PATCH_MANIFOLDS] )
{
	if ( candidateCount == 0 )
		return 0;
	b3Voxel_sortPatchIndicesByKey( context, candidates, candidateCount );
	if ( candidateCount <= B3_VOXEL_MAX_PATCH_MANIFOLDS )
	{
		memcpy( selected, candidates, (size_t)candidateCount * sizeof( int ) );
		return candidateCount;
	}

	int* groupOf = b3Bump( context->table.arena, context->table.count * (int)sizeof( int ) );
	int* groupRepresentatives = b3Bump( context->table.arena, candidateCount * (int)sizeof( int ) );
	context->scratchBytes += ( context->table.count + candidateCount ) * (int)sizeof( int );
	for ( int i = 0; i < context->table.count; ++i )
		groupOf[i] = -1;
	int groupCount = 0;
	for ( int i = 0; i < candidateCount; ++i )
	{
		int entryIndex = candidates[i];
		b3VoxelCanonicalEntry* entry = context->entries + entryIndex;
		b3Vec3 normal = entry->manifold.points[0].normal;
		int group = -1;
		for ( int j = 0; j < groupCount; ++j )
		{
			if ( b3Dot( normal, context->entries[groupRepresentatives[j]].manifold.points[0].normal ) > B3_VOXEL_CLUSTER_DOT )
			{
				group = j;
				break;
			}
		}
		if ( group < 0 )
		{
			group = groupCount++;
			groupRepresentatives[group] = entryIndex;
		}
		else
		{
			int representative = groupRepresentatives[group];
			float depth = b3Voxel_patchDepth( entry );
			float representativeDepth = b3Voxel_patchDepth( context->entries + representative );
			if ( depth > representativeDepth ||
				 ( depth == representativeDepth && b3Voxel_comparePatchKeys( &entry->manifold.key,
															 &context->entries[representative].manifold.key ) < 0 ) )
			{
				groupRepresentatives[group] = entryIndex;
			}
		}
		groupOf[entryIndex] = group;
	}

	// Deepest normal groups win if the number of genuinely distinct patches
	// alone exceeds the solver budget. Exact keys break every tie.
	for ( int i = 1; i < groupCount; ++i )
	{
		int value = groupRepresentatives[i];
		float valueDepth = b3Voxel_patchDepth( context->entries + value );
		int j = i - 1;
		while ( j >= 0 )
		{
			int current = groupRepresentatives[j];
			float currentDepth = b3Voxel_patchDepth( context->entries + current );
			bool move = currentDepth < valueDepth ||
						( currentDepth == valueDepth &&
						  b3Voxel_comparePatchKeys( &context->entries[current].manifold.key,
											   &context->entries[value].manifold.key ) > 0 );
			if ( !move )
				break;
			groupRepresentatives[j + 1] = current;
			j -= 1;
		}
		groupRepresentatives[j + 1] = value;
	}

	bool* retained = b3Bump( context->table.arena, context->table.count * (int)sizeof( bool ) );
	memset( retained, 0, (size_t)context->table.count * sizeof( bool ) );
	context->scratchBytes += context->table.count * (int)sizeof( bool );
	int selectedCount = b3MinInt( groupCount, B3_VOXEL_MAX_PATCH_MANIFOLDS );
	for ( int i = 0; i < selectedCount; ++i )
	{
		selected[i] = groupRepresentatives[i];
		retained[selected[i]] = true;
	}

	// Fill any remaining slots with deterministic tangent-space extrema from
	// groups whose deepest representative is already retained.
	while ( selectedCount < B3_VOXEL_MAX_PATCH_MANIFOLDS )
	{
		int best = -1;
		float bestSpread = -1.0f;
		for ( int i = 0; i < candidateCount; ++i )
		{
			int entryIndex = candidates[i];
			if ( retained[entryIndex] )
				continue;
			const b3VoxelCanonicalEntry* entry = context->entries + entryIndex;
			int pointIndex = b3Voxel_patchDeepestPoint( entry );
			b3Vec3 point = entry->manifold.points[pointIndex].body0Point;
			b3Vec3 normal = entry->manifold.points[pointIndex].normal;
			float minSpread = FLT_MAX;
			for ( int j = 0; j < selectedCount; ++j )
			{
				int retainedIndex = selected[j];
				if ( groupOf[retainedIndex] != groupOf[entryIndex] )
					continue;
				const b3VoxelCanonicalEntry* other = context->entries + retainedIndex;
				b3Vec3 delta = b3Sub( point, other->manifold.points[b3Voxel_patchDeepestPoint( other )].body0Point );
				delta = b3MulAdd( delta, -b3Dot( delta, normal ), normal );
				minSpread = b3MinFloat( minSpread, b3Dot( delta, delta ) );
			}
			float depth = b3Voxel_patchDepth( entry );
			float bestDepth = best >= 0 ? b3Voxel_patchDepth( context->entries + best ) : -FLT_MAX;
			if ( minSpread > bestSpread ||
				 ( minSpread == bestSpread &&
				   ( depth > bestDepth ||
					 ( depth == bestDepth &&
					   b3Voxel_comparePatchKeys( &entry->manifold.key, &context->entries[best].manifold.key ) < 0 ) ) ) )
			{
				best = entryIndex;
				bestSpread = minSpread;
			}
		}
		if ( best < 0 )
			break;
		selected[selectedCount++] = best;
		retained[best] = true;
	}
	b3Voxel_sortPatchIndicesByKey( context, selected, selectedCount );
	return selectedCount;
}

static void b3Voxel_finishCanonicalCounters( b3VoxelCanonicalContext* context )
{
	if ( context->counters == NULL )
		return;
	b3VoxelCounters* counters = context->counters;
	counters->patchMaxUniqueKeys = b3MaxInt( counters->patchMaxUniqueKeys, context->table.count );
	for ( int i = 0; i < context->table.count; ++i )
	{
		int multiplicity = context->entries[i].representedAliases;
		counters->patchMaxMultiplicity = b3MaxInt( counters->patchMaxMultiplicity, multiplicity );
		int bucket = multiplicity == 1 ? 0 : multiplicity <= 3 ? 1 : multiplicity <= 7 ? 2 : multiplicity <= 15 ? 3 :
					 multiplicity <= 31 ? 4 : 5;
		counters->patchMultiplicityCounts[bucket] += 1;
	}
	counters->patchTableGrowths += context->table.growthCount;
	int scratchBytes = context->table.scratchBytes + context->scratchBytes;
	counters->patchScratchPeakBytes = b3MaxInt( counters->patchScratchPeakBytes, scratchBytes );
}

int b3VoxelBuildCanonicalPatches( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
								  float contactDistance, b3VoxelPatchCollision* result, b3Arena* arena,
								  b3VoxelCounters* counters )
{
	B3_ASSERT( arena != NULL );
	result->manifoldCount = 0;
	result->uniqueKeyCount = 0;
	if ( counters != NULL )
		counters->voxelVoxelCalls += 1;
	b3AABB lb0;
	b3AABB lb1;
	if ( !b3Voxel_GetLocalBounds( v0, &lb0 ) || !b3Voxel_GetLocalBounds( v1, &lb1 ) )
		return 0;
	contactDistance = b3MaxFloat( contactDistance, 0.0f );
	b3AABB wb0 = b3Voxel_mapBounds( lb0, xf0, false );
	b3AABB wb1 = b3Voxel_mapBounds( lb1, xf1, false );
	if ( !b3Voxel_isect( b3Voxel_expandB( wb0, contactDistance ), wb1 ) )
		return 0;
	bool possibleContainment = b3Voxel_containsAABB( wb0, wb1 ) || b3Voxel_containsAABB( wb1, wb0 );

	b3VoxelCanonicalContext context = {
		.voxels0 = v0,
		.voxels1 = v1,
		.transform0 = xf0,
		.transform1 = xf1,
		.transform0To1 = b3InvMulTransforms( xf1, xf0 ),
		.transform1To0 = b3InvMulTransforms( xf0, xf1 ),
		.origin0 = b3Voxel_GetOrigin( v0 ),
		.origin1 = b3Voxel_GetOrigin( v1 ),
		.voxelSize0 = b3Voxel_GetVoxelSize( v0 ),
		.voxelSize1 = b3Voxel_GetVoxelSize( v1 ),
		.contactDistance = contactDistance,
		.counters = counters,
	};
	bool hasDomain0 = b3Voxel_GetDomain( v0, &context.domainMin0, &context.domainMax0 );
	bool hasDomain1 = b3Voxel_GetDomain( v1, &context.domainMin1, &context.domainMax1 );
	B3_ASSERT( hasDomain0 && hasDomain1 );
	B3_UNUSED( hasDomain0 );
	B3_UNUSED( hasDomain1 );
	context.worldOrigin0 = b3Voxel_xfPoint( xf0, context.origin0 );
	context.worldOrigin1 = b3Voxel_xfPoint( xf1, context.origin1 );
	context.frame0 = b3Voxel_cellFrame( v0, xf0 );
	context.frame1 = b3Voxel_cellFrame( v1, xf1 );
	context.frame0In1 = context.frame0;
	context.frame1In0 = context.frame1;
	context.frame0In1.center = b3Vec3_zero;
	context.frame1In0.center = b3Vec3_zero;
	for ( int axis = 0; axis < 3; ++axis )
	{
		context.frame0In1.axes[axis] = b3InvRotateVector( xf1.q, context.frame0.axes[axis] );
		context.frame1In0.axes[axis] = b3InvRotateVector( xf0.q, context.frame1.axes[axis] );
	}
	b3VoxelPrepareOBBPair( &context.leafPairContext, &context.frame0, &context.frame1 );
	b3Obb_prepareSurfaceMasks( &context.leafPairContext, xf0.q, xf1.q );
	for ( int i = 0; i < 3; ++i )
	{
		for ( int j = 0; j < 3; ++j )
		{
			float rotation = b3Dot( context.frame0.axes[i], context.frame1.axes[j] );
			context.absRotation[i][j] = fabsf( rotation );
			b3Vec3 edgeAxis = b3Cross( context.frame0.axes[i], context.frame1.axes[j] );
			float lengthSquared = b3Dot( edgeAxis, edgeAxis );
			context.edgeInvLength[i][j] = lengthSquared > 1e-4f ? 1.0f / sqrtf( lengthSquared ) : 0.0f;
		}
	}
	float margin = 5.0f * ( context.voxelSize0 + context.voxelSize1 ) + contactDistance;
	b3AABB bounds1In0 = b3Voxel_mapBounds( lb1, context.transform1To0, false );
	b3AABB bounds0In1 = b3Voxel_mapBounds( lb0, context.transform0To1, false );
	context.clip0 = b3Voxel_intersectBounds( b3Voxel_expandB( bounds1In0, margin ), b3Voxel_expandB( lb0, margin ) );
	context.clip1 = b3Voxel_intersectBounds( b3Voxel_expandB( bounds0In1, margin ), b3Voxel_expandB( lb1, margin ) );
	b3VoxelPatchTableInit( &context.table, arena );

	b3AABB query0 = b3Voxel_mapBounds( b3Voxel_expandB( wb1, contactDistance ), xf0, true );
	b3Voxel_ForEachCellTracked( v0, query0, b3Voxel_canonicalOuter0Cell, &context, counters );
	b3AABB query1 = b3Voxel_mapBounds( b3Voxel_expandB( wb0, contactDistance ), xf1, true );
	b3Voxel_ForEachCellTracked( v1, query1, b3Voxel_canonicalOuter1Cell, &context, counters );
	result->uniqueKeyCount = context.table.count;

	int* candidates = b3Bump( arena, context.table.count * (int)sizeof( int ) );
	context.scratchBytes += context.table.count * (int)sizeof( int );
	int candidateCount = 0;
	for ( int i = 0; i < context.table.count; ++i )
	{
		b3VoxelCanonicalEntry* entry = context.entries + i;
		if ( entry->state == b3_voxelPatchSeparated )
			continue;
		int write = 0;
		for ( int point = 0; point < entry->manifold.pointCount; ++point )
		{
			if ( ( entry->selectedMask & (uint8_t)( 1u << point ) ) != 0 )
				entry->manifold.points[write++] = entry->manifold.points[point];
		}
		entry->manifold.pointCount = write;
		if ( write == 0 )
		{
			write = b3Voxel_replayEmptyCanonicalEntry( &context, entry );
			if ( write == 0 )
			{
				if ( counters != NULL )
					counters->emptySelectedPatchKeys += 1;
				continue;
			}
		}
		if ( counters != NULL )
			counters->selectedPatchKeys += 1;
		candidates[candidateCount++] = i;
	}

	int selected[B3_VOXEL_MAX_PATCH_MANIFOLDS];
	int selectedCount = b3Voxel_reduceCanonicalPatches( &context, candidates, candidateCount, selected );
	if ( counters != NULL && candidateCount > B3_VOXEL_MAX_PATCH_MANIFOLDS )
		counters->patchBudgetOverflows += candidateCount - B3_VOXEL_MAX_PATCH_MANIFOLDS;
	for ( int i = 0; i < selectedCount; ++i )
		result->manifolds[result->manifoldCount++] = context.entries[selected[i]].manifold;

	if ( result->manifoldCount == 0 && possibleContainment )
	{
		b3VoxelContact escape[B3_VOXEL_POINTS_PER_CLUSTER];
		int count = b3VoxelCollideImpl( v0, xf0, v1, xf1, contactDistance, B3_VOXEL_POINTS_PER_CLUSTER, escape, NULL );
		if ( count > 0 )
		{
			b3VoxelPatchManifold* manifold = result->manifolds;
			memset( manifold, 0, sizeof( *manifold ) );
			manifold->kind = 1;
			manifold->pointCount = count;
			memcpy( manifold->points, escape, (size_t)count * sizeof( b3VoxelContact ) );
			result->manifoldCount = 1;
			if ( counters != NULL )
				counters->deepOverlapFallbacks += 1;
		}
	}
	if ( counters != NULL )
		counters->emittedPatchManifolds += result->manifoldCount;
	b3Voxel_finishCanonicalCounters( &context );
	return result->manifoldCount;
}

int b3VoxelCollideCanonicalWithArena( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1,
								 float contactDistance, int maxContacts, b3VoxelContact* out, b3Arena* arena,
								 b3VoxelCounters* counters )
{
	if ( maxContacts <= 0 )
		return 0;
	maxContacts = b3MinInt( maxContacts, B3_VOXEL_MAX_CONTACTS );
	b3VoxelPatchCollision result;
	b3VoxelBuildCanonicalPatches( v0, xf0, v1, xf1, contactDistance, &result, arena, counters );
	int count = 0;
	for ( int manifoldIndex = 0; manifoldIndex < result.manifoldCount && count < maxContacts; ++manifoldIndex )
	{
		const b3VoxelPatchManifold* manifold = result.manifolds + manifoldIndex;
		int copyCount = b3MinInt( manifold->pointCount, maxContacts - count );
		memcpy( out + count, manifold->points, (size_t)copyCount * sizeof( b3VoxelContact ) );
		count += copyCount;
	}
	return count;
}

typedef struct b3VoxelOverlapPairContext
{
	const b3VoxelData* voxels0;
	const b3VoxelData* voxels1;
	b3Transform transform0;
	b3Transform transform1;
	b3Transform transform0To1;
	b3Vec3 origin0;
	b3Vec3 origin1;
	float voxelSize0;
	float voxelSize1;
	b3VoxelOBB frame0;
	b3VoxelOBB frame1;
	b3VoxelOBB frame0In1;
	b3ObbPairContext pairContext;
	b3VoxelOBB bounds1;
	b3ObbPairContext outerBoundsContext;
	bool hit;
} b3VoxelOverlapPairContext;

typedef struct b3VoxelOverlapInnerContext
{
	b3VoxelOverlapPairContext* pair;
	b3VoxelOBB obb0;
} b3VoxelOverlapInnerContext;

static b3VoxelOBB b3Voxel_makeBoundsOBB( const b3VoxelOBB* frame, b3AABB localBounds, b3Transform transform )
{
	b3VoxelOBB bounds = *frame;
	bounds.center = b3TransformPoint( transform, b3MulSV( 0.5f, b3Add( localBounds.lowerBound, localBounds.upperBound ) ) );
	bounds.half = b3MulSV( 0.5f, b3Sub( localBounds.upperBound, localBounds.lowerBound ) );
	return bounds;
}

static bool b3Voxel_overlapInnerCell( b3Vec3i cell1, uint8_t exposed, void* rawContext )
{
	B3_UNUSED( exposed );
	b3VoxelOverlapInnerContext* context = rawContext;
	b3VoxelOverlapPairContext* pair = context->pair;
	b3VoxelOBB obb1 = pair->frame1;
	obb1.center = b3Voxel_xfPoint( pair->transform1, b3Voxel_cellCenterFromGrid( pair->origin1, pair->voxelSize1, cell1 ) );
	if ( b3Obb_overlapPrepared( b3Sub( context->obb0.center, obb1.center ), &pair->pairContext ) )
	{
		pair->hit = true;
		return false;
	}
	return true;
}

static bool b3Voxel_overlapOuterCell( b3Vec3i cell0, uint8_t exposed, void* rawContext )
{
	B3_UNUSED( exposed );
	b3VoxelOverlapPairContext* pair = rawContext;
	b3Vec3 localCenter0 = b3Voxel_cellCenterFromGrid( pair->origin0, pair->voxelSize0, cell0 );
	b3VoxelOBB obb0 = pair->frame0;
	obb0.center = b3Voxel_xfPoint( pair->transform0, localCenter0 );
	// The occupancy query already proves overlap on frame0's three axes: its
	// AABB is the projection of bounds1 into frame0-local space, expanded by the
	// cell half-width. Test only bounds1's axes and the edge axes here.
	if ( !b3Obb_overlapPreparedFrom( b3Sub( obb0.center, pair->bounds1.center ), &pair->outerBoundsContext, 3 ) )
	{
		return true;
	}
	b3VoxelOBB localObb0 = pair->frame0In1;
	localObb0.center = b3TransformPoint( pair->transform0To1, localCenter0 );
	b3AABB query1 = b3VoxelOBB_Bounds( &localObb0 );
	b3VoxelOverlapInnerContext inner = { pair, obb0 };
	b3Voxel_ForEachCellTracked( pair->voxels1, query1, b3Voxel_overlapInnerCell, &inner, NULL );
	return !pair->hit;
}

static bool b3Voxel_testOverlapImpl( const b3VoxelData* v0, b3Transform xf0, b3AABB local0, const b3VoxelData* v1,
									 b3Transform xf1, b3AABB local1 )
{
	b3VoxelOverlapPairContext context = {
		.voxels0 = v0,
		.voxels1 = v1,
		.transform0 = xf0,
		.transform1 = xf1,
		.transform0To1 = b3InvMulTransforms( xf1, xf0 ),
		.origin0 = b3Voxel_GetOrigin( v0 ),
		.origin1 = b3Voxel_GetOrigin( v1 ),
		.voxelSize0 = b3Voxel_GetVoxelSize( v0 ),
		.voxelSize1 = b3Voxel_GetVoxelSize( v1 ),
	};
	context.frame0 = b3Voxel_cellFrame( v0, xf0 );
	context.frame1 = b3Voxel_cellFrame( v1, xf1 );
	b3VoxelPrepareOBBPair( &context.pairContext, &context.frame0, &context.frame1 );
	context.frame0In1 = context.frame0;
	context.frame0In1.center = b3Vec3_zero;
	for ( int i = 0; i < 3; ++i )
	{
		context.frame0In1.axes[i] = b3InvRotateVector( xf1.q, context.frame0.axes[i] );
	}

	// The local occupied bounds become exact world OBBs. Rejecting separated
	// bounds here avoids walking either occupancy set for broad-phase swept
	// pairs that are separated at this CCD sample.
	b3VoxelOBB bounds0 = b3Voxel_makeBoundsOBB( &context.frame0, local0, xf0 );
	b3VoxelOBB bounds1 = b3Voxel_makeBoundsOBB( &context.frame1, local1, xf1 );
	b3ObbPairContext boundsContext;
	b3VoxelPrepareOBBPair( &boundsContext, &bounds0, &bounds1 );
	if ( !b3Obb_overlapPrepared( b3Sub( bounds0.center, bounds1.center ), &boundsContext ) )
	{
		return false;
	}
	context.bounds1 = bounds1;
	b3VoxelPrepareOBBPair( &context.outerBoundsContext, &context.frame0, &bounds1 );

	b3VoxelOBB localBounds1 = bounds1;
	localBounds1.center =
		b3TransformPoint( b3InvMulTransforms( xf0, xf1 ), b3MulSV( 0.5f, b3Add( local1.lowerBound, local1.upperBound ) ) );
	for ( int i = 0; i < 3; ++i )
	{
		localBounds1.axes[i] = b3InvRotateVector( xf0.q, bounds1.axes[i] );
	}
	b3AABB query0 = b3VoxelOBB_Bounds( &localBounds1 );
	b3Voxel_ForEachCellTracked( v0, query0, b3Voxel_overlapOuterCell, &context, NULL );
	return context.hit;
}

static bool b3Voxel_testOverlap( const b3VoxelData* v0, b3Transform xf0, const b3VoxelData* v1, b3Transform xf1 )
{
	b3AABB local0, local1;
	int count0 = b3Voxel_GetCellCount( v0 );
	int count1 = b3Voxel_GetCellCount( v1 );
	if ( count0 == 0 || count1 == 0 || !b3Voxel_GetLocalBounds( v0, &local0 ) || !b3Voxel_GetLocalBounds( v1, &local1 ) )
	{
		return false;
	}

	// Overlap is symmetric. Iterating the smaller occupancy bounds the number
	// of outer cells without affecting contact ordering or simulation state.
	if ( count1 < count0 )
	{
		return b3Voxel_testOverlapImpl( v1, xf1, local1, v0, xf0, local0 );
	}
	return b3Voxel_testOverlapImpl( v0, xf0, local0, v1, xf1, local1 );
}

enum
{
	b3_voxelCcdMaxPathSegments = 16
};

typedef struct b3VoxelConvexSweepContext
{
	b3Vec3 origin;
	float voxelSize;
	float halfVoxelSize;
	b3ShapeProxy targetProxy;
	b3Sweep targetSweep;
	b3Sweep movingSweep;
	b3TOIOutput output;
	b3AABB targetBounds[b3_voxelCcdMaxPathSegments];
	b3BoxProxySupport targetBoxSupport;
	int pathSegmentCount;
	bool targetIsBox;
	b3VoxelCounters* counters;
} b3VoxelConvexSweepContext;

static float b3Voxel_sweepAngle( b3Quat q1, b3Quat q2 );

static void b3Voxel_makeCellCorners( const b3VoxelData* voxels, b3Vec3i cell, b3Vec3 corners[8] )
{
	b3Vec3 center = b3Voxel_GetCellCenter( voxels, cell );
	float half = 0.5f * b3Voxel_GetVoxelSize( voxels );
	for ( int k = 0; k < 8; ++k )
	{
		corners[k] = (b3Vec3){ center.x + ( ( k & 1 ) ? half : -half ), center.y + ( ( k & 2 ) ? half : -half ),
							   center.z + ( ( k & 4 ) ? half : -half ) };
	}
}

static void b3Voxel_makeCellCornersFromCenter( b3Vec3 center, float half, b3Vec3 corners[8] )
{
	for ( int k = 0; k < 8; ++k )
	{
		corners[k] = (b3Vec3){ center.x + ( ( k & 1 ) ? half : -half ), center.y + ( ( k & 2 ) ? half : -half ),
							   center.z + ( ( k & 4 ) ? half : -half ) };
	}
}

static bool b3Voxel_sweepConvexCell( b3Vec3i cell, uint8_t exposed, void* rawContext )
{
	b3VoxelConvexSweepContext* context = rawContext;
	if ( context->counters != NULL )
	{
		context->counters->ccdCellsVisited += 1;
	}
	// An interior cube cannot be the first feature hit from outside the union.
	// Cavities remain covered because their boundary cells have exposed faces.
	if ( exposed == 0 )
	{
		if ( context->counters != NULL )
		{
			context->counters->ccdInteriorRejects += 1;
		}
		return true;
	}
	b3Vec3 cellCenter = b3Voxel_cellCenterFromGrid( context->origin, context->voxelSize, cell );
	float half = context->halfVoxelSize;
	b3Vec3 halfExtent = { half, half, half };
	b3AABB cellBounds = { b3Sub( cellCenter, halfExtent ), b3Add( cellCenter, halfExtent ) };
	bool insideCorridor = false;
	for ( int i = 0; i < context->pathSegmentCount; ++i )
	{
		if ( b3AABB_Overlaps( cellBounds, context->targetBounds[i] ) )
		{
			insideCorridor = true;
			break;
		}
	}
	if ( !insideCorridor )
	{
		if ( context->counters != NULL )
		{
			context->counters->ccdCorridorRejects += 1;
		}
		return true;
	}
	b3Vec3 corners[8];
	b3Voxel_makeCellCornersFromCenter( cellCenter, half, corners );
	b3TOIInput input = {
		.proxyA = context->targetProxy,
		.proxyB = { corners, 8, 0.0f },
		.sweepA = context->targetSweep,
		.sweepB = context->movingSweep,
		.maxFraction = context->output.fraction,
	};
	if ( context->counters != NULL )
	{
		context->counters->ccdExactToiCalls += 1;
	}
	b3TOIOutput candidate =
		context->targetIsBox ? b3TimeOfImpactBoxes( &input, &context->targetBoxSupport ) : b3TimeOfImpactCellB( &input );
	bool candidateImpact = candidate.state == b3_toiStateHit || candidate.state == b3_toiStateOverlapped;
	bool outputImpact = context->output.state == b3_toiStateHit || context->output.state == b3_toiStateOverlapped;
	if ( candidateImpact && ( !outputImpact || candidate.fraction < context->output.fraction ||
							  ( candidate.fraction == context->output.fraction && candidate.state == b3_toiStateOverlapped ) ) )
	{
		context->output = candidate;
	}
	return true;
}

// Conservative bounds of a convex target's swept volume in the moving voxel
// body's local frame. Each short segment bounds the actual proxy vertices at
// both endpoints. Target and moving-frame rotation can bend their paths away
// from the endpoint box; 2 sin(theta / 2) times the relevant point radius is a
// conservative bound on that departure.
static b3AABB b3Voxel_makeConvexSweepQuery( b3VoxelConvexSweepContext* context, float maxFraction )
{
	const b3ShapeProxy* targetProxy = &context->targetProxy;
	const b3Sweep* targetSweep = &context->targetSweep;
	const b3Sweep* movingSweep = &context->movingSweep;

	float targetPointRadius = 0.0f;
	for ( int i = 0; i < targetProxy->count; ++i )
	{
		targetPointRadius =
			b3MaxFloat( targetPointRadius, b3Length( b3Sub( targetProxy->points[i], targetSweep->localCenter ) ) );
	}

	b3Transform movingEnd = b3GetSweepTransform( movingSweep, maxFraction );
	b3Transform targetEnd = b3GetSweepTransform( targetSweep, maxFraction );
	float movingAngle = b3Voxel_sweepAngle( movingSweep->q1, movingEnd.q );
	float targetAngle = b3Voxel_sweepAngle( targetSweep->q1, targetEnd.q );
	float angle = b3MaxFloat( movingAngle, targetAngle );
	// Even a few degrees becomes a many-metre radius for distant broad-phase
	// hulls. Keep the endpoint-displacement bound, but apply it over short arcs
	// so it does not turn a thin trajectory into a body-wide capsule.
	const float maxSegmentAngle = 0.0872664626f; // five degrees
	context->pathSegmentCount = b3ClampInt( (int)ceilf( angle / maxSegmentAngle ), 1, b3_voxelCcdMaxPathSegments );

	b3Transform movingTransform[b3_voxelCcdMaxPathSegments + 1];
	b3Transform targetTransform[b3_voxelCcdMaxPathSegments + 1];
	movingTransform[0] = b3GetSweepTransform( movingSweep, 0.0f );
	targetTransform[0] = b3GetSweepTransform( targetSweep, 0.0f );
	movingTransform[context->pathSegmentCount] = movingEnd;
	targetTransform[context->pathSegmentCount] = targetEnd;
	for ( int i = 1; i < context->pathSegmentCount; ++i )
	{
		float fraction = maxFraction * (float)i / (float)context->pathSegmentCount;
		movingTransform[i] = b3GetSweepTransform( movingSweep, fraction );
		targetTransform[i] = b3GetSweepTransform( targetSweep, fraction );
	}

	b3AABB query = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
	for ( int i = 0; i < context->pathSegmentCount; ++i )
	{
		b3AABB bounds = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
		float relativeRadius = 0.0f;
		for ( int endpoint = 0; endpoint < 2; ++endpoint )
		{
			int sample = i + endpoint;
			b3Vec3 movingCenter =
				b3Lerp( movingSweep->c1, movingSweep->c2, maxFraction * (float)sample / (float)context->pathSegmentCount );
			for ( int j = 0; j < targetProxy->count; ++j )
			{
				b3Vec3 worldPoint = b3TransformPoint( targetTransform[sample], targetProxy->points[j] );
				b3Vec3 localPoint = b3InvTransformPoint( movingTransform[sample], worldPoint );
				bounds.lowerBound = b3Min( bounds.lowerBound, localPoint );
				bounds.upperBound = b3Max( bounds.upperBound, localPoint );
				relativeRadius = b3MaxFloat( relativeRadius, b3Length( b3Sub( worldPoint, movingCenter ) ) );
			}
		}

		float segmentTargetAngle = b3Voxel_sweepAngle( targetTransform[i].q, targetTransform[i + 1].q );
		float targetCurveBound = 2.0f * sinf( 0.5f * segmentTargetAngle ) * targetPointRadius;
		float segmentMovingAngle = b3Voxel_sweepAngle( movingTransform[i].q, movingTransform[i + 1].q );
		float movingCurveBound = 2.0f * sinf( 0.5f * segmentMovingAngle ) * ( relativeRadius + targetCurveBound );
		float curveBound = targetCurveBound + movingCurveBound;
		float padding = targetProxy->radius + curveBound + B3_LINEAR_SLOP;
		b3Vec3 r = { padding, padding, padding };
		bounds.lowerBound = b3Sub( bounds.lowerBound, r );
		bounds.upperBound = b3Add( bounds.upperBound, r );
		context->targetBounds[i] = bounds;
		query.lowerBound = b3Min( query.lowerBound, bounds.lowerBound );
		query.upperBound = b3Max( query.upperBound, bounds.upperBound );
	}
	return query;
}

typedef struct b3VoxelSampleContext
{
	b3Vec3 origin;
	float voxelSize;
	float halfVoxelSize;
	const b3Shape* target;
	b3Transform targetTransform;
	b3Transform movingTransform;
	b3Vec3 hitPoint;
	bool hit;
} b3VoxelSampleContext;

static bool b3Voxel_sampleCell( b3Vec3i cell, uint8_t exposed, void* rawContext )
{
	B3_UNUSED( exposed );
	b3VoxelSampleContext* context = rawContext;
	b3Vec3 localCorners[8];
	b3Vec3 worldCorners[8];
	b3Vec3 cellCenter = b3Voxel_cellCenterFromGrid( context->origin, context->voxelSize, cell );
	b3Voxel_makeCellCornersFromCenter( cellCenter, context->halfVoxelSize, localCorners );
	for ( int k = 0; k < 8; ++k )
	{
		worldCorners[k] = b3TransformPoint( context->movingTransform, localCorners[k] );
	}
	b3ShapeProxy proxy = { worldCorners, 8, 0.0f };
	if ( b3OverlapShape( context->target, context->targetTransform, &proxy ) )
	{
		context->hit = true;
		context->hitPoint = b3TransformPoint( context->movingTransform, cellCenter );
		return false;
	}
	return true;
}

static bool b3Voxel_sampleShapeOverlap( const b3Shape* target, const b3Sweep* targetSweep, const b3VoxelData* movingVoxel,
										const b3Sweep* movingSweep, float fraction, b3Vec3* hitPoint )
{
	b3Transform targetTransform = b3GetSweepTransform( targetSweep, fraction );
	b3Transform movingTransform = b3GetSweepTransform( movingSweep, fraction );
	if ( target->type == b3_voxelShape )
	{
		bool hit = b3Voxel_testOverlap( target->voxel, targetTransform, movingVoxel, movingTransform );
		if ( hit && hitPoint != NULL )
		{
			*hitPoint = b3TransformPoint( movingTransform, movingSweep->localCenter );
		}
		return hit;
	}

	b3AABB localBounds;
	if ( !b3Voxel_GetLocalBounds( movingVoxel, &localBounds ) )
	{
		return false;
	}
	b3VoxelSampleContext context = {
		.origin = b3Voxel_GetOrigin( movingVoxel ),
		.voxelSize = b3Voxel_GetVoxelSize( movingVoxel ),
		.halfVoxelSize = 0.5f * b3Voxel_GetVoxelSize( movingVoxel ),
		.target = target,
		.targetTransform = targetTransform,
		.movingTransform = movingTransform,
	};
	b3Voxel_ForEachCellTracked( movingVoxel, localBounds, b3Voxel_sampleCell, &context, NULL );
	if ( context.hit && hitPoint != NULL )
	{
		*hitPoint = context.hitPoint;
	}
	return context.hit;
}

static float b3Voxel_sweepAngle( b3Quat q1, b3Quat q2 )
{
	float cosine = b3ClampFloat( fabsf( b3DotQuat( q1, q2 ) ), 0.0f, 1.0f );
	return 2.0f * acosf( cosine );
}

static float b3Voxel_sweepMotionBound( const b3Sweep* sweep, float radius )
{
	return b3Length( b3Sub( sweep->c2, sweep->c1 ) ) + radius * b3Voxel_sweepAngle( sweep->q1, sweep->q2 );
}

b3TOIOutput b3VoxelShapeTimeOfImpact( const b3Shape* target, const b3Sweep* targetSweep, const b3VoxelData* movingVoxel,
									  const b3Sweep* movingSweep, float maxFraction, b3VoxelCounters* counters )
{
	b3TOIOutput output = { 0 };
	output.state = b3_toiStateSeparated;
	output.fraction = maxFraction;
	if ( maxFraction <= 0.0f || b3Voxel_GetCellCount( movingVoxel ) == 0 )
	{
		return output;
	}

	// A convex target has a continuous, exact TOI against every occupied cube.
	// Keeping the cube vertices in body-local space naturally includes arbitrary
	// rotation and an offset voxel origin in the existing Box3D sweep.
	if ( target->type == b3_sphereShape || target->type == b3_capsuleShape || target->type == b3_hullShape )
	{
		if ( counters != NULL )
		{
			counters->ccdConvexTargets += 1;
		}
		b3VoxelConvexSweepContext context = {
			.origin = b3Voxel_GetOrigin( movingVoxel ),
			.voxelSize = b3Voxel_GetVoxelSize( movingVoxel ),
			.halfVoxelSize = 0.5f * b3Voxel_GetVoxelSize( movingVoxel ),
			.targetProxy = b3MakeShapeProxy( target ),
			.targetSweep = *targetSweep,
			.movingSweep = *movingSweep,
			.output = output,
			.counters = counters,
		};
		if ( target->type == b3_hullShape && ( target->flags & b3_boxHull ) != 0 )
		{
			context.targetBoxSupport = b3MakeBoxProxySupport( &context.targetProxy );
			context.targetIsBox = true;
		}
		b3AABB query = b3Voxel_makeConvexSweepQuery( &context, maxFraction );
		b3Voxel_ForEachCellTracked( movingVoxel, query, b3Voxel_sweepConvexCell, &context, NULL );
		return context.output;
	}
	if ( counters != NULL )
	{
		counters->ccdAggregateTargets += 1;
	}

	// Aggregate shapes do not have one convex proxy. Sample at a spacing bounded
	// by one quarter of the moving cell width, including both bodies' rotational
	// point motion, then bisect the first overlap interval. A cell therefore
	// cannot cross a solid voxel/mesh feature between adjacent samples merely
	// because its body origin or centroid missed that feature.
	b3AABB movingBounds;
	if ( !b3Voxel_GetLocalBounds( movingVoxel, &movingBounds ) )
	{
		return output;
	}
	b3Vec3 movingRadiusVector = {
		b3MaxFloat( fabsf( movingBounds.lowerBound.x - movingSweep->localCenter.x ),
					fabsf( movingBounds.upperBound.x - movingSweep->localCenter.x ) ),
		b3MaxFloat( fabsf( movingBounds.lowerBound.y - movingSweep->localCenter.y ),
					fabsf( movingBounds.upperBound.y - movingSweep->localCenter.y ) ),
		b3MaxFloat( fabsf( movingBounds.lowerBound.z - movingSweep->localCenter.z ),
					fabsf( movingBounds.upperBound.z - movingSweep->localCenter.z ) ),
	};
	b3ShapeExtent targetExtent = b3ComputeShapeExtent( target, target->localCentroid );
	float motion = b3Voxel_sweepMotionBound( movingSweep, b3Length( movingRadiusVector ) ) +
				   b3Voxel_sweepMotionBound( targetSweep, b3Length( targetExtent.maxExtent ) );
	float spacing = 0.25f * b3Voxel_GetVoxelSize( movingVoxel );
	if ( target->type == b3_voxelShape )
	{
		spacing = 0.25f * b3MinFloat( b3Voxel_GetVoxelSize( movingVoxel ), b3Voxel_GetVoxelSize( target->voxel ) );
	}
	int sampleCount = b3MaxInt( 1, (int)ceilf( motion / b3MaxFloat( spacing, B3_LINEAR_SLOP ) ) );

	b3Vec3 hitPoint = b3Vec3_zero;
	if ( b3Voxel_sampleShapeOverlap( target, targetSweep, movingVoxel, movingSweep, 0.0f, &hitPoint ) )
	{
		output.state = b3_toiStateOverlapped;
		output.fraction = 0.0f;
		output.point = hitPoint;
		return output;
	}

	float previous = 0.0f;
	for ( int i = 1; i <= sampleCount; ++i )
	{
		float fraction = maxFraction * (float)i / (float)sampleCount;
		if ( !b3Voxel_sampleShapeOverlap( target, targetSweep, movingVoxel, movingSweep, fraction, &hitPoint ) )
		{
			previous = fraction;
			continue;
		}

		float lower = previous;
		float upper = fraction;
		for ( int iteration = 0; iteration < 12; ++iteration )
		{
			float middle = 0.5f * ( lower + upper );
			if ( b3Voxel_sampleShapeOverlap( target, targetSweep, movingVoxel, movingSweep, middle, &hitPoint ) )
			{
				upper = middle;
			}
			else
			{
				lower = middle;
			}
		}

		b3Vec3 relativeMotion = b3Sub( b3Sub( movingSweep->c2, movingSweep->c1 ), b3Sub( targetSweep->c2, targetSweep->c1 ) );
		output.state = b3_toiStateHit;
		output.fraction = upper;
		output.point = hitPoint;
		output.normal = b3LengthSquared( relativeMotion ) > FLT_EPSILON ? b3Normalize( b3Neg( relativeMotion ) ) : b3Vec3_axisX;
		return output;
	}

	return output;
}

// ---------------------------------------------------------------------------
// Ray cast and overlap query (shape-local space).
// ---------------------------------------------------------------------------

// Amanatides-Woo 3D DDA over the cell-centred grid. Cell c occupies
// [c*s - 0.5s, c*s + 0.5s]; the grid coordinate u = p/s + 0.5 puts cell c at
// the integer interval [c, c+1). Reports the fraction (of the ray translation)
// at which the ray first enters a solid cell and the face normal it crossed.
b3CastOutput b3RayCastVoxel( const b3VoxelData* v, const b3RayCastInput* input )
{
	b3CastOutput out = { 0 };

	if ( b3Voxel_GetCellCount( v ) == 0 )
	{
		return out;
	}

	float s = b3Voxel_GetVoxelSize( v );
	float inv = 1.0f / s;
	b3Vec3 gridOrigin = b3Voxel_GetOrigin( v );

	b3Vec3 p0 = input->origin;
	b3Vec3 d = input->translation;

	// Clip the ray to the local bounds to bound the march (and to find the entry t).
	b3AABB bounds;
	b3Voxel_GetLocalBounds( v, &bounds );

	float tmin = 0.0f;
	float tmax = input->maxFraction;
	for ( int a = 0; a < 3; ++a )
	{
		float o = ( &p0.x )[a];
		float dir = ( &d.x )[a];
		float lo = ( &bounds.lowerBound.x )[a];
		float hi = ( &bounds.upperBound.x )[a];
		if ( fabsf( dir ) < 1e-9f )
		{
			if ( o < lo || o > hi )
			{
				return out;
			}
		}
		else
		{
			float t1 = ( lo - o ) / dir;
			float t2 = ( hi - o ) / dir;
			if ( t1 > t2 )
			{
				float tt = t1;
				t1 = t2;
				t2 = tt;
			}
			tmin = b3MaxFloat( tmin, t1 );
			tmax = b3MinFloat( tmax, t2 );
			if ( tmin > tmax )
			{
				return out;
			}
		}
	}

	// Entry point and grid setup.
	b3Vec3 pe = { p0.x + tmin * d.x, p0.y + tmin * d.y, p0.z + tmin * d.z };

	int cell[3];
	int step[3];
	float tMax[3];
	float tDelta[3];
	for ( int a = 0; a < 3; ++a )
	{
		float ue = ( ( &pe.x )[a] - ( &gridOrigin.x )[a] ) * inv + 0.5f;
		float du = ( &d.x )[a] * inv;
		cell[a] = (int)floorf( ue );
		if ( du > 1e-12f )
		{
			step[a] = 1;
			tMax[a] = tmin + ( (float)( cell[a] + 1 ) - ue ) / du;
			tDelta[a] = 1.0f / du;
		}
		else if ( du < -1e-12f )
		{
			step[a] = -1;
			tMax[a] = tmin + ( (float)cell[a] - ue ) / du;
			tDelta[a] = -1.0f / du;
		}
		else
		{
			step[a] = 0;
			tMax[a] = FLT_MAX;
			tDelta[a] = FLT_MAX;
		}
	}

	float tEnter = tmin;
	int lastAxis = -1;
	for ( ;; )
	{
		if ( b3VoxelData_IsSolid( v, (b3Vec3i){ cell[0], cell[1], cell[2] } ) )
		{
			out.hit = true;
			out.fraction = tEnter;
			out.point = (b3Vec3){ p0.x + tEnter * d.x, p0.y + tEnter * d.y, p0.z + tEnter * d.z };
			if ( lastAxis < 0 )
			{
				// Ray began inside a solid cell.
				out.normal = b3Normalize( b3Neg( d ) );
			}
			else
			{
				out.normal = b3Vec3_zero;
				( &out.normal.x )[lastAxis] = (float)( -step[lastAxis] );
			}
			return out;
		}

		int mi = 0;
		if ( tMax[1] < tMax[mi] )
			mi = 1;
		if ( tMax[2] < tMax[mi] )
			mi = 2;

		if ( tMax[mi] > tmax )
		{
			break;
		}

		tEnter = tMax[mi];
		cell[mi] += step[mi];
		lastAxis = mi;
		tMax[mi] += tDelta[mi];
	}

	return out;
}

typedef struct b3VoxelShapeCastContext
{
	const b3VoxelData* voxels;
	const b3ShapeCastInput* input;
	b3Vec3 proxyCenter;
	b3Vec3 corridorRadius;
	float halfVoxelSize;
	b3BitBoxProxySupport boxSupport;
	bool hasBoxSupport;
	bool useCorridor;
	b3CastOutput output;
} b3VoxelShapeCastContext;

// The broad phase queries the union of the proxy's start and end AABBs. For a
// diagonal sweep that union contains cells which the translating AABB never
// overlaps on all three axes at the same time. Intersecting the three 1D time
// intervals is a cheap conservative negative certificate for any fixed point
// cloud (including rounded proxies). Possible hits still run the existing GJK
// shape cast, preserving its output bit-for-bit.
static bool b3Voxel_shapeCastMayHit( const b3VoxelShapeCastContext* context, b3Vec3 cellCenter )
{
	b3Vec3 delta = b3Sub( cellCenter, context->proxyCenter );
	float lower = 0.0f;
	float upper = context->output.fraction;

	for ( int axis = 0; axis < 3; ++axis )
	{
		float projection = ( &delta.x )[axis];
		float speed = ( &context->input->translation.x )[axis];
		float radius = ( &context->corridorRadius.x )[axis];
		radius += 64.0f * FLT_EPSILON *
				  ( 1.0f + fabsf( projection ) + fabsf( speed ) * upper + radius );

		if ( fabsf( speed ) <= FLT_EPSILON )
		{
			if ( fabsf( projection ) > radius )
				return false;
			continue;
		}

		float t1 = ( projection - radius ) / speed;
		float t2 = ( projection + radius ) / speed;
		if ( t1 > t2 )
		{
			float swap = t1;
			t1 = t2;
			t2 = swap;
		}
		lower = b3MaxFloat( lower, t1 );
		upper = b3MinFloat( upper, t2 );
		if ( lower >= upper )
			return false;
	}

	return true;
}

static bool b3Voxel_shapeCastCell( b3Vec3i cell, uint8_t exposed, void* rawContext )
{
	b3VoxelShapeCastContext* context = rawContext;
	b3Vec3 cellCenter = b3Voxel_GetCellCenter( context->voxels, cell );
	if ( context->useCorridor && !b3Voxel_shapeCastMayHit( context, cellCenter ) )
		return true;

	b3Vec3 corners[8];
	if ( context->useCorridor )
	{
		b3Voxel_makeCellCornersFromCenter( cellCenter, context->halfVoxelSize, corners );
	}
	else
	{
		b3Voxel_makeCellCorners( context->voxels, cell, corners );
	}
	b3ShapeCastPairInput pairInput = {
		.proxyA = { corners, 8, 0.0f },
		.proxyB = context->input->proxy,
		.transform = b3Transform_identity,
		.translationB = context->input->translation,
		.maxFraction = context->output.fraction,
		.canEncroach = context->input->canEncroach,
	};
	b3CastOutput candidate = context->useCorridor
							 ? b3ShapeCastCellA( &pairInput, context->hasBoxSupport ? &context->boxSupport : NULL )
							 : b3ShapeCast( &pairInput );
	if ( !candidate.hit )
	{
		return true;
	}

	// A swept convex can touch several adjacent cubes at the same fraction.
	// Only accept a positive-time witness on the boundary of the occupied
	// union, otherwise an internal seam can win based on cell iteration order.
	if ( candidate.fraction > 0.0f )
	{
		b3VoxelOBB frame = b3Voxel_cellFrame( context->voxels, b3Transform_identity );
		b3VoxelOBB cellBox = b3Voxel_cellOBB( &frame, context->voxels, cell, b3Transform_identity );
		if ( !b3Voxel_pointOnExposedFace( &cellBox, candidate.point, exposed,
										 B3_LINEAR_SLOP + 32.0f * FLT_EPSILON ) )
		{
			return true;
		}
	}

	if ( !context->output.hit || candidate.fraction < context->output.fraction )
	{
		context->output = candidate;
		context->output.materialIndex = 0;
	}
	return true;
}

static b3CastOutput b3ShapeCastVoxelImpl( const b3VoxelData* v, const b3ShapeCastInput* input, bool useCorridor )
{
	b3VoxelShapeCastContext context = {
		.voxels = v,
		.input = input,
	};
	context.output.fraction = input->maxFraction;
	context.output.triangleIndex = B3_NULL_INDEX;

	if ( b3Voxel_GetCellCount( v ) == 0 )
	{
		return context.output;
	}

	b3AABB start = b3ComputeProxyAABB( &input->proxy );
	b3Vec3 displacement = b3MulSV( input->maxFraction, input->translation );
	b3AABB end = { b3Add( start.lowerBound, displacement ), b3Add( start.upperBound, displacement ) };
	b3AABB query = { b3Min( start.lowerBound, end.lowerBound ), b3Max( start.upperBound, end.upperBound ) };
	context.useCorridor = useCorridor;
	context.hasBoxSupport = useCorridor && b3TryMakeBitBoxProxySupport( &input->proxy, &context.boxSupport );
	context.proxyCenter = b3MulSV( 0.5f, b3Add( start.lowerBound, start.upperBound ) );
	b3Vec3 proxyHalf = b3MulSV( 0.5f, b3Sub( start.upperBound, start.lowerBound ) );
	float cellHalf = 0.5f * b3Voxel_GetVoxelSize( v );
	context.halfVoxelSize = cellHalf;
	context.corridorRadius = b3Add( proxyHalf, (b3Vec3){ cellHalf + 1.25f * B3_LINEAR_SLOP,
													 cellHalf + 1.25f * B3_LINEAR_SLOP,
													 cellHalf + 1.25f * B3_LINEAR_SLOP } );
	b3Voxel_ForEachCellTracked( v, query, b3Voxel_shapeCastCell, &context, NULL );
	return context.output;
}

b3CastOutput b3ShapeCastVoxel( const b3VoxelData* v, const b3ShapeCastInput* input )
{
	return b3ShapeCastVoxelImpl( v, input, true );
}

b3CastOutput b3ShapeCastVoxelReference( const b3VoxelData* v, const b3ShapeCastInput* input )
{
	return b3ShapeCastVoxelImpl( v, input, false );
}

// Overlap query: is any solid cell within the query proxy? The proxy is mapped
// into shape-local space, then each candidate cell's box is distance-tested
// against it (mirrors b3OverlapHeightField's per-triangle GJK test).
bool b3OverlapVoxel( const b3VoxelData* v, b3Transform xf, const b3ShapeProxy* proxy )
{
	if ( b3Voxel_GetCellCount( v ) == 0 )
	{
		return false;
	}

	b3Vec3 buffer[B3_MAX_SHAPE_CAST_POINTS];
	b3ShapeProxy localProxy = b3MakeLocalProxy( proxy, xf, buffer );
	b3AABB aabb = b3ComputeProxyAABB( &localProxy );

	float s = b3Voxel_GetVoxelSize( v );
	float half = 0.5f * s;

	// Grow the broadphase query by a half cell so a proxy grazing a cell face is
	// still considered; the GJK test below decides the precise result.
	b3AABB query;
	query.lowerBound = (b3Vec3){ aabb.lowerBound.x - half, aabb.lowerBound.y - half, aabb.lowerBound.z - half };
	query.upperBound = (b3Vec3){ aabb.upperBound.x + half, aabb.upperBound.y + half, aabb.upperBound.z + half };

	int n = b3Voxel_QueryCells( v, query, NULL, 0 );
	if ( n == 0 )
	{
		return false;
	}
	b3Vec3i stackCells[256];
	b3Vec3i* cells = stackCells;
	if ( n > 256 )
	{
		cells = (b3Vec3i*)b3Alloc( (size_t)n * sizeof( b3Vec3i ) );
	}
	B3_ASSERT( b3Voxel_QueryCells( v, query, cells, n ) == n );

	b3DistanceInput input;
	input.proxyB = localProxy;
	input.transform = b3Transform_identity;
	input.useRadii = true;

	b3SimplexCache cache = { 0 };
	float tolerance = 0.1f * B3_LINEAR_SLOP;

	bool hit = false;
	for ( int i = 0; i < n; ++i )
	{
		b3Vec3 c = b3Voxel_GetCellCenter( v, cells[i] );
		b3Vec3 corners[8];
		for ( int k = 0; k < 8; ++k )
		{
			corners[k] = (b3Vec3){
				c.x + ( ( k & 1 ) ? half : -half ),
				c.y + ( ( k & 2 ) ? half : -half ),
				c.z + ( ( k & 4 ) ? half : -half ),
			};
		}

		input.proxyA = (b3ShapeProxy){ corners, 8, 0.0f };
		cache.count = 0;

		b3DistanceOutput output = b3ShapeDistance( &input, &cache, NULL, 0 );
		if ( output.distance < tolerance )
		{
			hit = true;
			break;
		}
	}

	if ( cells != stackCells )
	{
		b3Free( cells, (size_t)n * sizeof( b3Vec3i ) );
	}
	return hit;
}
