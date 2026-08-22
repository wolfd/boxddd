// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#include "arena_allocator.h"
#include "shape.h"
#include "test_macros.h"
#include "test_voxel_oracle.h"
#include "voxel_collide.h"

#include "box3d/box3d.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum
{
	PROPERTY_ROOT_COUNT = 4,
	PROPERTY_DISCRETE_PER_ROOT = 2048,
	PROPERTY_SWEEPS_PER_ROOT = 512,
	PROPERTY_MAX_CELLS = 16,
};

typedef struct PropertyRng
{
	uint64_t state;
} PropertyRng;

typedef struct PropertyCensus
{
	uint64_t digest;
	int discreteCases;
	int discreteOverlapScenarios;
	int discreteSeparationScenarios;
	int shallowFaceScenarios;
	int edgeScenarios;
	int deepScenarios;
	int cavityScenarios;
	int disconnectedScenarios;
	int checkerboardScenarios;
	int exactCcdHits;
	int exactCcdMisses;
	int aggregateCcdHits;
	int aggregateCcdMisses;
	int metamorphicCases;
	int canonicalLateSelections;
	int canonicalDepthRejects;
	int canonicalFallbackKeys;
	int canonicalFallbackLeafTests;
	int canonicalConvexCases;
	int canonicalConvexHits;
	int canonicalConvexMisses;
	int canonicalConvexPseudoSat;
	int canonicalConvexFallbackKeys;
} PropertyCensus;

typedef struct PropertyTargets
{
	b3Sphere sphere;
	b3Capsule capsule;
	b3BoxHull box;
	b3HullData* tetra;
	b3HullData* irregular;
	b3Shape shapes[5];
} PropertyTargets;

typedef struct PropertySolid
{
	b3Vec3i cells[PROPERTY_MAX_CELLS];
	TestVoxelSolid solid;
} PropertySolid;

static const uint64_t s_rootSeeds[PROPERTY_ROOT_COUNT] = {
	0x9E3779B97F4A7C15ull,
	0xD1B54A32D192ED03ull,
	0x94D049BB133111EBull,
	0x8538ECB5BD456EA3ull,
};

// Minimized regressions discovered by the deterministic corpus belong here.
// These initial entries cover the two defects found while completing G1--G3.
static const uint64_t s_hardSeeds[] = {
	0x9E3779B97F4A7C15ull,
	0x51A7C0DEull,
	0xBADC0FFEE0DDF00Dull,
};

static uint64_t PropertyNext( PropertyRng* rng )
{
	uint64_t x = rng->state;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	rng->state = x;
	return x * 0x2545F4914F6CDD1Dull;
}

static float PropertyUnit( PropertyRng* rng )
{
	return (float)( PropertyNext( rng ) >> 40 ) * ( 1.0f / 16777216.0f );
}

static float PropertySigned( PropertyRng* rng )
{
	return 2.0f * PropertyUnit( rng ) - 1.0f;
}

static b3Quat PropertyQuat( PropertyRng* rng, float maxAngle )
{
	b3Vec3 axis = { PropertySigned( rng ), PropertySigned( rng ), PropertySigned( rng ) };
	if ( b3LengthSquared( axis ) < 1.0e-6f )
	{
		axis = b3Vec3_axisX;
	}
	return b3MakeQuatFromAxisAngle( b3Normalize( axis ), maxAngle * PropertySigned( rng ) );
}

static bool PropertyHasCell( const PropertySolid* solid, b3Vec3i cell )
{
	for ( int i = 0; i < solid->solid.count; ++i )
	{
		b3Vec3i other = solid->cells[i];
		if ( other.x == cell.x && other.y == cell.y && other.z == cell.z )
		{
			return true;
		}
	}
	return false;
}

static void PropertyAddCell( PropertySolid* solid, b3Vec3i cell )
{
	if ( solid->solid.count < PROPERTY_MAX_CELLS && !PropertyHasCell( solid, cell ) )
	{
		solid->cells[solid->solid.count++] = cell;
	}
}

static void PropertyMakeSolid( PropertySolid* solid, PropertyRng* rng, int topology )
{
	memset( solid, 0, sizeof( *solid ) );
	solid->solid.cells = solid->cells;
	solid->solid.size = 0.3f + 0.7f * PropertyUnit( rng );
	solid->solid.origin = (b3Vec3){ 0.35f * PropertySigned( rng ), 0.35f * PropertySigned( rng ), 0.35f * PropertySigned( rng ) };
	PropertyAddCell( solid, (b3Vec3i){ 0, 0, 0 } );

	if ( topology == 5 )
	{
		for ( int x = -1; x <= 1; ++x )
			for ( int y = -1; y <= 1; ++y )
				for ( int z = -1; z <= 1; ++z )
					if ( ( x != 0 || y != 0 || z != 0 ) && ( ( x + 2 * y + 3 * z ) & 1 ) == 0 )
						PropertyAddCell( solid, (b3Vec3i){ x, y, z } );
		return;
	}
	if ( topology == 6 )
	{
		PropertyAddCell( solid, (b3Vec3i){ 3, 0, 0 } );
		PropertyAddCell( solid, (b3Vec3i){ -3, 1, 0 } );
		PropertyAddCell( solid, (b3Vec3i){ 0, -3, 2 } );
		return;
	}
	if ( topology == 7 )
	{
		for ( int x = -2; x <= 2; ++x )
			for ( int y = -1; y <= 1; ++y )
				if ( ( x + y ) % 2 == 0 )
					PropertyAddCell( solid, (b3Vec3i){ x, y, ( x - y ) & 1 } );
		return;
	}

	int wanted = 2 + (int)( PropertyNext( rng ) % 7 );
	while ( solid->solid.count < wanted )
	{
		b3Vec3i base = solid->cells[PropertyNext( rng ) % (uint64_t)solid->solid.count];
		int axis = (int)( PropertyNext( rng ) % 3 );
		int direction = ( PropertyNext( rng ) & 1 ) != 0 ? 1 : -1;
		if ( axis == 0 )
			base.x += direction;
		else if ( axis == 1 )
			base.y += direction;
		else
			base.z += direction;
		PropertyAddCell( solid, base );
	}
}

static uint32_t PropertyFloatBits( float value )
{
	uint32_t bits;
	memcpy( &bits, &value, sizeof( bits ) );
	return bits;
}

static void PropertyHashU32( PropertyCensus* census, uint32_t value )
{
	census->digest ^= value;
	census->digest *= 1099511628211ull;
}

static void PropertyHashContacts( PropertyCensus* census, const b3VoxelContact* contacts, int count )
{
	PropertyHashU32( census, (uint32_t)count );
	for ( int i = 0; i < count; ++i )
	{
		PropertyHashU32( census, PropertyFloatBits( contacts[i].normal.x ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].normal.y ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].normal.z ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].initialPenetration ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].body0Point.x ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].body0Point.y ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].body0Point.z ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].body1Point.x ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].body1Point.y ) );
		PropertyHashU32( census, PropertyFloatBits( contacts[i].body1Point.z ) );
		PropertyHashU32( census, contacts[i].featureId );
	}
}

static bool PropertyContactsEqual( const b3VoxelContact* a, int countA, const b3VoxelContact* b, int countB )
{
	if ( countA != countB )
		return false;
	for ( int i = 0; i < countA; ++i )
	{
		if ( a[i].normal.x != b[i].normal.x || a[i].normal.y != b[i].normal.y || a[i].normal.z != b[i].normal.z ||
			 a[i].initialPenetration != b[i].initialPenetration || a[i].penetrationDepth != b[i].penetrationDepth ||
			 a[i].body0Point.x != b[i].body0Point.x || a[i].body0Point.y != b[i].body0Point.y ||
			 a[i].body0Point.z != b[i].body0Point.z || a[i].body1Point.x != b[i].body1Point.x ||
			 a[i].body1Point.y != b[i].body1Point.y || a[i].body1Point.z != b[i].body1Point.z ||
			 a[i].featureId != b[i].featureId )
		{
			return false;
		}
	}
	return true;
}

static bool PropertyValidateVoxelWitnesses( const TestVoxelSolid* a, b3Transform transformA, const TestVoxelSolid* b,
											b3Transform transformB, const b3VoxelContact* contacts, int count )
{
	float tolerance = TestVoxelBoundaryTolerance();
	for ( int i = 0; i < count; ++i )
	{
		b3Vec3 delta = b3Sub( contacts[i].body1Point, contacts[i].body0Point );
		if ( !isfinite( contacts[i].initialPenetration ) || b3AbsFloat( b3Length( contacts[i].normal ) - 1.0f ) > 2.0e-4f ||
			 b3AbsFloat( b3Dot( delta, contacts[i].normal ) - contacts[i].initialPenetration ) > 2.0e-4f ||
			 !TestPointOnUnionBoundary( a, transformA, contacts[i].body0Point, tolerance ) ||
			 !TestPointOnUnionBoundary( b, transformB, contacts[i].body1Point, tolerance ) )
		{
			return false;
		}
	}
	return true;
}

static bool PropertyValidateConvexWitnesses( const TestVoxelSolid* voxel, const b3VoxelContact* contacts, int count )
{
	float tolerance = TestVoxelBoundaryTolerance();
	for ( int i = 0; i < count; ++i )
	{
		b3Vec3 delta = b3Sub( contacts[i].body1Point, contacts[i].body0Point );
		if ( !isfinite( contacts[i].initialPenetration ) || b3AbsFloat( b3Length( contacts[i].normal ) - 1.0f ) > 2.0e-4f ||
			 b3AbsFloat( b3Dot( delta, contacts[i].normal ) - contacts[i].initialPenetration ) > 2.0e-4f ||
			 !TestPointOnUnionBoundary( voxel, b3Transform_identity, contacts[i].body0Point, tolerance ) )
		{
			return false;
		}
	}
	return true;
}

static bool PropertyValidateBoxWitnesses( const b3Shape* box, b3Transform transform, const b3VoxelContact* contacts, int count )
{
	const b3Plane* planes = b3GetHullPlanes( box->hull );
	const b3Vec3* points = b3GetHullPoints( box->hull );
	b3Vec3 localAxes[3] = { planes[1].normal, planes[3].normal, planes[5].normal };
	b3Vec3 fromCenter = b3Sub( points[0], box->hull->center );
	b3Vec3 half = { b3Dot( localAxes[0], fromCenter ), b3Dot( localAxes[1], fromCenter ),
				  b3Dot( localAxes[2], fromCenter ) };
	b3Vec3 center = b3TransformPoint( transform, box->hull->center );
	float tolerance = TestVoxelBoundaryTolerance();
	for ( int pointIndex = 0; pointIndex < count; ++pointIndex )
	{
		b3Vec3 relative = b3Sub( contacts[pointIndex].body1Point, center );
		bool onBoundary = false;
		for ( int axis = 0; axis < 3; ++axis )
		{
			b3Vec3 worldAxis = b3RotateVector( transform.q, localAxes[axis] );
			float coordinate = b3AbsFloat( b3Dot( relative, worldAxis ) );
			if ( coordinate > ( &half.x )[axis] + tolerance )
				return false;
			onBoundary = onBoundary || b3AbsFloat( coordinate - ( &half.x )[axis] ) <= tolerance;
		}
		if ( !onBoundary )
			return false;
	}
	return true;
}

static void PropertyPrintSolid( const char* name, const TestVoxelSolid* solid )
{
	printf( "static const b3Vec3i %s_cells[] = {", name );
	for ( int i = 0; i < solid->count; ++i )
	{
		printf( "%s{%d,%d,%d}", i == 0 ? "" : ",", solid->cells[i].x, solid->cells[i].y, solid->cells[i].z );
	}
	printf( "};\n" );
	printf( "TestVoxelSolid %s = {%s_cells,%d,%.9gf,{%.9gf,%.9gf,%.9gf}};\n", name, name, solid->count, solid->size,
			solid->origin.x, solid->origin.y, solid->origin.z );
}

static void PropertyPrintFailure( uint64_t rootSeed, int caseIndex, const char* invariant, const TestVoxelSolid* a,
								  const TestVoxelSolid* b, b3Transform transform )
{
	printf( "G4 failure: root_seed=0x%016llx case=%d invariant=%s epsilon=%.9g engine_band=%.9g\n", (unsigned long long)rootSeed,
			caseIndex, invariant, TestVoxelBoundaryTolerance(), B3_LINEAR_SLOP );
	PropertyPrintSolid( "a", a );
	if ( b != NULL )
		PropertyPrintSolid( "b", b );
	printf( "b3Transform transform = {{%.9gf,%.9gf,%.9gf},{%.9gf,%.9gf,%.9gf,%.9gf}};\n", transform.p.x, transform.p.y,
			transform.p.z, transform.q.v.x, transform.q.v.y, transform.q.v.z, transform.q.s );
}

static void PropertyPrintTarget( const b3Shape* target )
{
	if ( target->type == b3_sphereShape )
	{
		printf( "b3Sphere target = {{%.9gf,%.9gf,%.9gf},%.9gf};\n", target->sphere.center.x, target->sphere.center.y,
				target->sphere.center.z, target->sphere.radius );
	}
	else if ( target->type == b3_capsuleShape )
	{
		printf( "b3Capsule target = {{%.9gf,%.9gf,%.9gf},{%.9gf,%.9gf,%.9gf},%.9gf};\n", target->capsule.center1.x,
				target->capsule.center1.y, target->capsule.center1.z, target->capsule.center2.x, target->capsule.center2.y,
				target->capsule.center2.z, target->capsule.radius );
	}
	else if ( target->type == b3_hullShape )
	{
		const b3Vec3* points = b3GetHullPoints( target->hull );
		printf( "b3Vec3 targetPoints[] = {" );
		for ( int i = 0; i < target->hull->vertexCount; ++i )
		{
			printf( "%s{%.9gf,%.9gf,%.9gf}", i == 0 ? "" : ",", points[i].x, points[i].y, points[i].z );
		}
		printf( "};\n" );
		printf( "b3HullData* target = b3CreateHull(targetPoints,%d,%d);\n", target->hull->vertexCount,
				target->hull->vertexCount );
	}
}

static b3Transform PropertyDiscreteTransform( const PropertySolid* a, const PropertySolid* b, PropertyRng* rng, int scenario )
{
	b3Transform transform = b3Transform_identity;
	if ( scenario == 1 )
	{
		transform.p = (b3Vec3){ 8.0f + PropertyUnit( rng ), -7.0f, 6.0f };
		transform.q = PropertyQuat( rng, 0.8f );
	}
	else if ( scenario == 2 )
	{
		float distance = 0.5f * ( a->solid.size + b->solid.size ) - ( 0.01f + 0.02f * PropertyUnit( rng ) );
		transform.p = b3Sub( b3Add( a->solid.origin, (b3Vec3){ distance, 0.0f, 0.0f } ), b->solid.origin );
	}
	else if ( scenario == 3 )
	{
		transform.p = (b3Vec3){ 0.35f * PropertySigned( rng ), 0.35f * PropertySigned( rng ), 0.35f * PropertySigned( rng ) };
		transform.q = PropertyQuat( rng, 1.2f );
	}
	else
	{
		transform.p = (b3Vec3){ 0.2f * PropertySigned( rng ), 0.2f * PropertySigned( rng ), 0.2f * PropertySigned( rng ) };
		transform.q = scenario == 4 ? PropertyQuat( rng, 0.3f ) : PropertyQuat( rng, 0.7f );
	}
	return transform;
}

static int PropertyConvexFailureKind( const TestVoxelSolid* solid, const b3Shape* target, b3Transform transform )
{
	bool oracle = TestGenericVoxelConvexOverlap( solid, target, transform );
	b3VoxelData* voxel = b3CreateOffsetVoxelData( solid->cells, solid->count, solid->size, solid->origin );
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	int count = b3VoxelCollideConvex( voxel, target, transform, 0.0f, ARRAY_COUNT( contacts ), contacts, NULL, NULL );
	b3DestroyVoxelData( voxel );
	if ( ( count > 0 ) != oracle )
		return 1;
	if ( !PropertyValidateConvexWitnesses( solid, contacts, count ) )
		return 2;
	return 0;
}

static void PropertyRemoveCellRange( PropertySolid* output, const PropertySolid* input, int first, int count )
{
	*output = *input;
	output->solid.cells = output->cells;
	output->solid.count = 0;
	for ( int i = 0; i < input->solid.count; ++i )
	{
		if ( i < first || i >= first + count )
		{
			output->cells[output->solid.count++] = input->cells[i];
		}
	}
}

static PropertySolid PropertyMinimizeConvexFailure( const PropertySolid* original, const b3Shape* target, b3Transform transform,
													int failureKind )
{
	PropertySolid minimized = *original;
	minimized.solid.cells = minimized.cells;
	for ( int chunk = minimized.solid.count / 2; chunk >= 1; chunk /= 2 )
	{
		bool changed;
		do
		{
			changed = false;
			for ( int first = 0; first + chunk <= minimized.solid.count && minimized.solid.count > chunk; ++first )
			{
				PropertySolid candidate;
				PropertyRemoveCellRange( &candidate, &minimized, first, chunk );
				if ( PropertyConvexFailureKind( &candidate.solid, target, transform ) == failureKind )
				{
					minimized = candidate;
					minimized.solid.cells = minimized.cells;
					changed = true;
					break;
				}
			}
		}
		while ( changed );
	}
	return minimized;
}

static int PropertyVoxelFailureKind( const TestVoxelSolid* a, const TestVoxelSolid* b, b3Transform transform )
{
	bool oracle = TestGenericUnionOverlap( a, b3Transform_identity, b, transform );
	b3VoxelData* voxelA = b3CreateOffsetVoxelData( a->cells, a->count, a->size, a->origin );
	b3VoxelData* voxelB = b3CreateOffsetVoxelData( b->cells, b->count, b->size, b->origin );
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	int count = b3VoxelCollide( voxelA, b3Transform_identity, voxelB, transform, 0.0f, ARRAY_COUNT( contacts ), contacts );
	b3DestroyVoxelData( voxelB );
	b3DestroyVoxelData( voxelA );
	if ( ( count > 0 ) != oracle )
		return 1;
	if ( !PropertyValidateVoxelWitnesses( a, b3Transform_identity, b, transform, contacts, count ) )
		return 2;
	return 0;
}

static void PropertyMinimizeVoxelFailure( PropertySolid* a, PropertySolid* b, b3Transform transform, int failureKind )
{
	for ( int side = 0; side < 2; ++side )
	{
		PropertySolid* active = side == 0 ? a : b;
		for ( int chunk = active->solid.count / 2; chunk >= 1; chunk /= 2 )
		{
			bool changed;
			do
			{
				changed = false;
				for ( int first = 0; first + chunk <= active->solid.count && active->solid.count > chunk; ++first )
				{
					PropertySolid candidate;
					PropertyRemoveCellRange( &candidate, active, first, chunk );
					const TestVoxelSolid* candidateA = side == 0 ? &candidate.solid : &a->solid;
					const TestVoxelSolid* candidateB = side == 1 ? &candidate.solid : &b->solid;
					if ( PropertyVoxelFailureKind( candidateA, candidateB, transform ) == failureKind )
					{
						*active = candidate;
						active->solid.cells = active->cells;
						changed = true;
						break;
					}
				}
			}
			while ( changed );
		}
	}
}

static int PropertyDiscreteCase( uint64_t rootSeed, int caseIndex, PropertyRng* rng, const PropertyTargets* targets,
								 PropertyCensus* census )
{
	int scenario = caseIndex & 7;
	PropertySolid a;
	PropertySolid b;
	PropertyMakeSolid( &a, rng, scenario );
	PropertyMakeSolid( &b, rng, ( scenario + 3 ) & 7 );
	b3Transform transform = PropertyDiscreteTransform( &a, &b, rng, scenario );
	int targetKind = caseIndex % 6;
	b3VoxelContact first[B3_VOXEL_MAX_CONTACTS];
	b3VoxelContact second[B3_VOXEL_MAX_CONTACTS];
	int firstCount;
	int secondCount;
	bool oracle;

	b3VoxelData* voxelA = b3CreateOffsetVoxelData( a.solid.cells, a.solid.count, a.solid.size, a.solid.origin );
	if ( targetKind == 0 )
	{
		oracle = TestGenericUnionOverlap( &a.solid, b3Transform_identity, &b.solid, transform );
		b3VoxelData* voxelB = b3CreateOffsetVoxelData( b.solid.cells, b.solid.count, b.solid.size, b.solid.origin );
		firstCount = b3VoxelCollide( voxelA, b3Transform_identity, voxelB, transform, 0.0f, ARRAY_COUNT( first ), first );
		secondCount = b3VoxelCollide( voxelA, b3Transform_identity, voxelB, transform, 0.0f, ARRAY_COUNT( second ), second );
		b3VoxelContact canonical[B3_VOXEL_MAX_CONTACTS];
		b3Arena arena = b3CreateArena( 16384 );
		b3VoxelCounters canonicalCounters = { 0 };
		int canonicalCount = b3VoxelCollideCanonicalWithArena( voxelA, b3Transform_identity, voxelB, transform, 0.0f,
																ARRAY_COUNT( canonical ), canonical, &arena, &canonicalCounters );
		bool canonicalWitnesses =
			PropertyValidateVoxelWitnesses( &a.solid, b3Transform_identity, &b.solid, transform, canonical, canonicalCount );
		census->canonicalLateSelections += canonicalCounters.pseudoLateSelections;
		census->canonicalDepthRejects += canonicalCounters.pseudoDepthRejects;
		census->canonicalFallbackKeys += canonicalCounters.emptyPatchFallbackKeys;
		census->canonicalFallbackLeafTests += canonicalCounters.patchLeafFallbackTests;
		if ( rootSeed == UINT64_C( 0xD1B54A32D192ED03 ) && caseIndex == 4008 )
		{
			ENSURE( canonicalCounters.emptyPatchFallbackKeys > 0 );
			ENSURE( canonicalCounters.patchLeafFallbackTests > 0 );
		}
		b3DestroyArena( &arena );
		b3DestroyVoxelData( voxelB );
		if ( ( canonicalCount > 0 ) != oracle || !canonicalWitnesses )
		{
			PropertyPrintFailure( rootSeed, caseIndex, "voxel_canonical", &a.solid, &b.solid, transform );
			printf( "oracle=%d canonical_count=%d witnesses_valid=%d visits=%d unique=%d pseudo_sat=%d separated=%d "
					"selected=%d empty=%d witness_reject=%d depth_reject=%d pruned=%d represented=%d\n",
					oracle, canonicalCount, canonicalWitnesses, canonicalCounters.patchVisits, canonicalCounters.patchUniqueKeys,
					canonicalCounters.pseudoSatCalls, canonicalCounters.pseudoSeparatedKeys, canonicalCounters.selectedPatchKeys,
					canonicalCounters.emptySelectedPatchKeys, canonicalCounters.pseudoWitnessRejects,
					canonicalCounters.pseudoDepthRejects, canonicalCounters.topologyPrunedPairs,
					canonicalCounters.representedLeafPairs );
			b3DestroyVoxelData( voxelA );
			return 1;
		}
		bool witnessesValid =
			PropertyValidateVoxelWitnesses( &a.solid, b3Transform_identity, &b.solid, transform, first, firstCount );
		if ( ( firstCount > 0 ) != oracle || !witnessesValid )
		{
			PropertyPrintFailure( rootSeed, caseIndex, "voxel_discrete", &a.solid, &b.solid, transform );
			printf( "oracle=%d specialized_count=%d witnesses_valid=%d\n", oracle, firstCount, witnessesValid );
			for ( int i = 0; i < firstCount; ++i )
			{
				printf(
					"contact=%d n=(%.9g,%.9g,%.9g) a=(%.9g,%.9g,%.9g) b=(%.9g,%.9g,%.9g) penetration=%.9g "
					"boundary=(%d,%d)\n",
					i, first[i].normal.x, first[i].normal.y, first[i].normal.z, first[i].body0Point.x, first[i].body0Point.y,
					first[i].body0Point.z, first[i].body1Point.x, first[i].body1Point.y, first[i].body1Point.z,
					first[i].initialPenetration,
					TestPointOnUnionBoundary( &a.solid, b3Transform_identity, first[i].body0Point, TestVoxelBoundaryTolerance() ),
					TestPointOnUnionBoundary( &b.solid, transform, first[i].body1Point, TestVoxelBoundaryTolerance() ) );
			}
			int failureKind = ( firstCount > 0 ) != oracle ? 1 : 2;
			PropertySolid minimizedA = a;
			PropertySolid minimizedB = b;
			minimizedA.solid.cells = minimizedA.cells;
			minimizedB.solid.cells = minimizedB.cells;
			PropertyMinimizeVoxelFailure( &minimizedA, &minimizedB, transform, failureKind );
			printf( "minimized reproducer:\n" );
			PropertyPrintFailure( rootSeed, caseIndex, "voxel_discrete", &minimizedA.solid, &minimizedB.solid, transform );
			printf( "failure_kind=%d\n", failureKind );
			b3DestroyVoxelData( voxelA );
			return 1;
		}
	}
	else
	{
		const b3Shape* target = targets->shapes + targetKind - 1;
		oracle = TestGenericVoxelConvexOverlap( &a.solid, target, transform );
		firstCount = b3VoxelCollideConvex( voxelA, target, transform, 0.0f, ARRAY_COUNT( first ), first, NULL, NULL );
		secondCount = b3VoxelCollideConvex( voxelA, target, transform, 0.0f, ARRAY_COUNT( second ), second, NULL, NULL );
		if ( targetKind == 3 )
		{
			b3Shape canonicalTarget = *target;
			canonicalTarget.flags |= b3_boxHull;
			b3VoxelContact canonicalFirst[B3_VOXEL_MAX_CONTACTS];
			b3VoxelContact canonicalSecond[B3_VOXEL_MAX_CONTACTS];
			b3VoxelCounters canonicalCounters = { 0 };
			b3Arena firstArena = b3CreateArena( 65536 );
			b3Arena secondArena = b3CreateArena( 65536 );
			int canonicalFirstCount = b3VoxelCollideConvexCanonicalWithArena(
				voxelA, &canonicalTarget, transform, 0.0f, ARRAY_COUNT( canonicalFirst ), canonicalFirst, &firstArena,
				&canonicalCounters );
			int canonicalSecondCount = b3VoxelCollideConvexCanonicalWithArena(
				voxelA, &canonicalTarget, transform, 0.0f, ARRAY_COUNT( canonicalSecond ), canonicalSecond, &secondArena, NULL );
			bool canonicalWitnesses = PropertyValidateConvexWitnesses( &a.solid, canonicalFirst, canonicalFirstCount ) &&
									  PropertyValidateBoxWitnesses( &canonicalTarget, transform, canonicalFirst,
															canonicalFirstCount );
			bool canonicalRepeat =
				PropertyContactsEqual( canonicalFirst, canonicalFirstCount, canonicalSecond, canonicalSecondCount );
			b3DestroyArena( &secondArena );
			b3DestroyArena( &firstArena );
			census->canonicalConvexCases += 1;
			census->canonicalConvexHits += canonicalFirstCount > 0;
			census->canonicalConvexMisses += canonicalFirstCount == 0;
			census->canonicalConvexPseudoSat += canonicalCounters.pseudoSatCalls;
			census->canonicalConvexFallbackKeys += canonicalCounters.emptyPatchFallbackKeys;
			if ( ( canonicalFirstCount > 0 ) != oracle || !canonicalWitnesses || !canonicalRepeat )
			{
				PropertyPrintFailure( rootSeed, caseIndex, "convex_canonical", &a.solid, NULL, transform );
				printf( "oracle=%d canonical_count=%d witnesses_valid=%d repeat=%d visits=%d unique=%d pseudo_sat=%d "
						"selected=%d fallback=%d/%d\n",
						oracle, canonicalFirstCount, canonicalWitnesses, canonicalRepeat, canonicalCounters.patchVisits,
						canonicalCounters.patchUniqueKeys, canonicalCounters.pseudoSatCalls,
						canonicalCounters.selectedPatchKeys, canonicalCounters.emptyPatchFallbackKeys,
						canonicalCounters.patchLeafFallbackTests );
				b3DestroyVoxelData( voxelA );
				return 1;
			}
		}
		bool witnessesValid = PropertyValidateConvexWitnesses( &a.solid, first, firstCount );
		if ( ( firstCount > 0 ) != oracle || !witnessesValid )
		{
			PropertyPrintFailure( rootSeed, caseIndex, "convex_discrete", &a.solid, NULL, transform );
			printf( "target_kind=%d oracle=%d specialized_count=%d witnesses_valid=%d\n", targetKind, oracle, firstCount,
					witnessesValid );
			for ( int i = 0; i < firstCount; ++i )
			{
				printf( "contact=%d n=(%.9g,%.9g,%.9g) a=(%.9g,%.9g,%.9g) b=(%.9g,%.9g,%.9g) penetration=%.9g "
						"boundary=%d\n",
						i, first[i].normal.x, first[i].normal.y, first[i].normal.z, first[i].body0Point.x, first[i].body0Point.y,
						first[i].body0Point.z, first[i].body1Point.x, first[i].body1Point.y, first[i].body1Point.z,
						first[i].initialPenetration,
						TestPointOnUnionBoundary( &a.solid, b3Transform_identity, first[i].body0Point,
												  TestVoxelBoundaryTolerance() ) );
			}
			int failureKind = ( firstCount > 0 ) != oracle ? 1 : 2;
			PropertySolid minimized = PropertyMinimizeConvexFailure( &a, target, transform, failureKind );
			printf( "minimized reproducer:\n" );
			PropertyPrintFailure( rootSeed, caseIndex, "convex_discrete", &minimized.solid, NULL, transform );
			printf( "target_kind=%d failure_kind=%d\n", targetKind, failureKind );
			b3DestroyVoxelData( voxelA );
			return 1;
		}
	}
	b3DestroyVoxelData( voxelA );
	if ( !PropertyContactsEqual( first, firstCount, second, secondCount ) )
	{
		PropertyPrintFailure( rootSeed, caseIndex, "repeat_determinism", &a.solid, targetKind == 0 ? &b.solid : NULL, transform );
		return 1;
	}

	PropertyHashU32( census, (uint32_t)scenario );
	PropertyHashU32( census, (uint32_t)targetKind );
	PropertyHashU32( census, oracle );
	PropertyHashContacts( census, first, firstCount );
	census->discreteCases += 1;
	if ( scenario == 0 )
		census->discreteOverlapScenarios += 1;
	else if ( scenario == 1 )
		census->discreteSeparationScenarios += 1;
	else if ( scenario == 2 )
		census->shallowFaceScenarios += 1;
	else if ( scenario == 3 )
		census->edgeScenarios += 1;
	else if ( scenario == 4 )
		census->deepScenarios += 1;
	else if ( scenario == 5 )
		census->cavityScenarios += 1;
	else if ( scenario == 6 )
		census->disconnectedScenarios += 1;
	else
		census->checkerboardScenarios += 1;
	return 0;
}

static b3Sweep PropertySweep( PropertyRng* rng, bool hit, bool rotating, float lateral )
{
	b3Sweep sweep = { 0 };
	sweep.localCenter = b3Vec3_zero;
	sweep.c1 = (b3Vec3){ -4.0f - PropertyUnit( rng ), hit ? lateral : 5.0f + PropertyUnit( rng ), 0.15f * PropertySigned( rng ) };
	sweep.c2 = (b3Vec3){ 4.0f + PropertyUnit( rng ), sweep.c1.y, 0.15f * PropertySigned( rng ) };
	sweep.q1 = rotating ? PropertyQuat( rng, 0.7f ) : b3Quat_identity;
	sweep.q2 = rotating ? PropertyQuat( rng, 1.0f ) : b3Quat_identity;
	return sweep;
}

static int PropertySweepCase( uint64_t rootSeed, int caseIndex, PropertyRng* rng, const PropertyTargets* targets,
							  PropertyCensus* census )
{
	int scenario = caseIndex & 3;
	bool expectedImpact = ( scenario & 1 ) == 0;
	bool aggregate = scenario >= 2;
	bool rotating = ( caseIndex & 4 ) != 0;
	PropertySolid moving;
	PropertyMakeSolid( &moving, rng, caseIndex & 7 );
	moving.solid.origin = b3Vec3_zero;
	// Keeping one to four cells makes the exact minimum oracle cheap while still
	// exercising candidate ordering. Discrete G4 owns randomized grid origins;
	// sweep origins are zeroed so intended hit/miss classes are constructive.
	moving.solid.count = b3MinInt( moving.solid.count, 1 + caseIndex % 4 );
	b3Sweep targetSweep = { .q1 = b3Quat_identity, .q2 = b3Quat_identity };
	if ( ( caseIndex & 8 ) != 0 )
	{
		targetSweep.c1 = (b3Vec3){ 0.25f, -0.1f, 0.15f };
		targetSweep.c2 = (b3Vec3){ -0.2f, 0.12f, -0.1f };
		targetSweep.q1 = PropertyQuat( rng, 0.35f );
		targetSweep.q2 = PropertyQuat( rng, 0.45f );
		rotating = true;
	}
	b3Sweep movingSweep = PropertySweep( rng, expectedImpact, rotating, 0.1f * PropertySigned( rng ) );
	bool oracleFailed;
	b3TOIOutput oracle;
	b3TOIOutput specialized;
	PropertySolid aggregateTarget = { 0 };
	const b3Shape* convexTarget = NULL;
	b3VoxelData* movingData =
		b3CreateOffsetVoxelData( moving.solid.cells, moving.solid.count, moving.solid.size, moving.solid.origin );
	if ( aggregate )
	{
		PropertyMakeSolid( &aggregateTarget, rng, ( caseIndex + 5 ) & 7 );
		aggregateTarget.solid.origin = b3Vec3_zero;
		aggregateTarget.solid.count = b3MinInt( aggregateTarget.solid.count, 1 + ( caseIndex + 1 ) % 4 );
		oracle =
			TestGenericVoxelPairToi( &aggregateTarget.solid, &targetSweep, &moving.solid, &movingSweep, 1.0f, &oracleFailed );
		b3VoxelData* targetData = b3CreateOffsetVoxelData( aggregateTarget.solid.cells, aggregateTarget.solid.count,
														   aggregateTarget.solid.size, aggregateTarget.solid.origin );
		b3Shape targetShape = { .type = b3_voxelShape, .voxel = targetData };
		specialized = b3VoxelShapeTimeOfImpact( &targetShape, &targetSweep, movingData, &movingSweep, 1.0f, NULL );
		b3DestroyVoxelData( targetData );
	}
	else
	{
		convexTarget = targets->shapes + ( caseIndex % ARRAY_COUNT( targets->shapes ) );
		oracle = TestGenericAggregateToi( &moving.solid, convexTarget, &targetSweep, &movingSweep, 1.0f, &oracleFailed );
		specialized = b3VoxelShapeTimeOfImpact( convexTarget, &targetSweep, movingData, &movingSweep, 1.0f, NULL );
	}
	b3DestroyVoxelData( movingData );
	if ( oracleFailed || TestToiIsImpact( oracle.state ) != expectedImpact ||
		 TestToiIsImpact( specialized.state ) != TestToiIsImpact( oracle.state ) )
	{
		PropertyPrintFailure( rootSeed, caseIndex, aggregate ? "aggregate_ccd" : "exact_ccd", &moving.solid, NULL,
							  b3Transform_identity );
		if ( aggregate )
			PropertyPrintSolid( "target", &aggregateTarget.solid );
		else
			PropertyPrintTarget( convexTarget );
		printf( "b3Sweep targetSweep = {.localCenter={%.9gf,%.9gf,%.9gf},.c1={%.9gf,%.9gf,%.9gf},"
				".c2={%.9gf,%.9gf,%.9gf},.q1={{%.9gf,%.9gf,%.9gf},%.9gf},.q2={{%.9gf,%.9gf,%.9gf},%.9gf}};\n",
				targetSweep.localCenter.x, targetSweep.localCenter.y, targetSweep.localCenter.z, targetSweep.c1.x,
				targetSweep.c1.y, targetSweep.c1.z, targetSweep.c2.x, targetSweep.c2.y, targetSweep.c2.z, targetSweep.q1.v.x,
				targetSweep.q1.v.y, targetSweep.q1.v.z, targetSweep.q1.s, targetSweep.q2.v.x, targetSweep.q2.v.y,
				targetSweep.q2.v.z, targetSweep.q2.s );
		printf( "b3Sweep movingSweep = {.localCenter={%.9gf,%.9gf,%.9gf},.c1={%.9gf,%.9gf,%.9gf},"
				".c2={%.9gf,%.9gf,%.9gf},.q1={{%.9gf,%.9gf,%.9gf},%.9gf},.q2={{%.9gf,%.9gf,%.9gf},%.9gf}};\n",
				movingSweep.localCenter.x, movingSweep.localCenter.y, movingSweep.localCenter.z, movingSweep.c1.x,
				movingSweep.c1.y, movingSweep.c1.z, movingSweep.c2.x, movingSweep.c2.y, movingSweep.c2.z, movingSweep.q1.v.x,
				movingSweep.q1.v.y, movingSweep.q1.v.z, movingSweep.q1.s, movingSweep.q2.v.x, movingSweep.q2.v.y,
				movingSweep.q2.v.z, movingSweep.q2.s );
		printf( "maxFraction=1.0f; oracle=(state=%d,fraction=%.9g); specialized=(state=%d,fraction=%.9g); target_kind=%d\n",
				oracle.state, oracle.fraction, specialized.state, specialized.fraction,
				aggregate ? (int)b3_voxelShape : (int)targets->shapes[caseIndex % ARRAY_COUNT( targets->shapes )].type );
		return 1;
	}

	PropertyHashU32( census, (uint32_t)scenario );
	PropertyHashU32( census, (uint32_t)oracle.state );
	PropertyHashU32( census, PropertyFloatBits( oracle.fraction ) );
	PropertyHashU32( census, (uint32_t)specialized.state );
	PropertyHashU32( census, PropertyFloatBits( specialized.fraction ) );
	if ( aggregate )
	{
		if ( expectedImpact )
			census->aggregateCcdHits += 1;
		else
			census->aggregateCcdMisses += 1;
	}
	else if ( expectedImpact )
	{
		census->exactCcdHits += 1;
	}
	else
	{
		census->exactCcdMisses += 1;
	}
	if ( rotating )
		census->metamorphicCases += 1;
	return 0;
}

static int PropertyBuildTargets( PropertyTargets* targets )
{
	memset( targets, 0, sizeof( *targets ) );
	targets->sphere = (b3Sphere){ b3Vec3_zero, 0.38f };
	targets->capsule = (b3Capsule){ { 0.0f, -0.32f, 0.0f }, { 0.0f, 0.32f, 0.0f }, 0.22f };
	targets->box = b3MakeOffsetBoxHull( 0.35f, 0.42f, 0.31f, (b3Vec3){ 0.07f, -0.05f, 0.03f } );
	b3Vec3 tetraPoints[] = {
		{ -0.42f, -0.31f, -0.24f },
		{ 0.45f, -0.28f, -0.19f },
		{ 0.08f, 0.48f, -0.16f },
		{ 0.02f, 0.03f, 0.51f },
	};
	b3Vec3 irregularPoints[] = {
		{ -0.46f, -0.33f, -0.21f }, { 0.49f, -0.26f, -0.17f }, { 0.14f, 0.52f, -0.13f },
		{ -0.25f, 0.07f, 0.47f },	{ 0.31f, 0.13f, 0.41f },
	};
	targets->tetra = b3CreateHull( tetraPoints, ARRAY_COUNT( tetraPoints ), ARRAY_COUNT( tetraPoints ) );
	targets->irregular = b3CreateHull( irregularPoints, ARRAY_COUNT( irregularPoints ), ARRAY_COUNT( irregularPoints ) );
	ENSURE( targets->tetra != NULL && targets->irregular != NULL );
	targets->shapes[0] = (b3Shape){ .type = b3_sphereShape, .sphere = targets->sphere };
	targets->shapes[1] = (b3Shape){ .type = b3_capsuleShape, .capsule = targets->capsule };
	targets->shapes[2] = (b3Shape){ .type = b3_hullShape, .hull = &targets->box.base };
	targets->shapes[3] = (b3Shape){ .type = b3_hullShape, .hull = targets->tetra };
	targets->shapes[4] = (b3Shape){ .type = b3_hullShape, .hull = targets->irregular };
	return 0;
}

static int PropertyRunCorpus( const PropertyTargets* targets, PropertyCensus* census )
{
	memset( census, 0, sizeof( *census ) );
	census->digest = 1469598103934665603ull;
	for ( int root = 0; root < PROPERTY_ROOT_COUNT; ++root )
	{
		PropertyRng rng = { s_rootSeeds[root] };
		for ( int i = 0; i < PROPERTY_DISCRETE_PER_ROOT; ++i )
		{
			int caseIndex = root * PROPERTY_DISCRETE_PER_ROOT + i;
			if ( PropertyDiscreteCase( s_rootSeeds[root], caseIndex, &rng, targets, census ) != 0 )
				return 1;
		}
		for ( int i = 0; i < PROPERTY_SWEEPS_PER_ROOT; ++i )
		{
			int caseIndex = root * PROPERTY_SWEEPS_PER_ROOT + i;
			if ( PropertySweepCase( s_rootSeeds[root], caseIndex, &rng, targets, census ) != 0 )
				return 1;
		}
	}
	return 0;
}

int VoxelPropertyTest( void )
{
	PropertyTargets targets;
	ENSURE( PropertyBuildTargets( &targets ) == 0 );
	PropertyCensus first;
	PropertyCensus second;
	int result = PropertyRunCorpus( &targets, &first );
	if ( result == 0 )
	{
		result = PropertyRunCorpus( &targets, &second );
	}
	b3DestroyHull( targets.tetra );
	b3DestroyHull( targets.irregular );
	ENSURE( result == 0 );
	ENSURE( memcmp( &first, &second, sizeof( first ) ) == 0 );
	printf( "  G4 coverage: discrete=%d overlap=%d separation=%d shallow=%d edge=%d deep=%d cavity=%d disconnected=%d "
			"checkerboard=%d exact_ccd=%d/%d aggregate_ccd=%d/%d metamorphic=%d canonical_late=%d "
			"canonical_depth_reject=%d canonical_fallback=%d/%d canonical_convex=%d/%d/%d pseudo=%d fallback=%d "
			"hard_seeds=%d digest=0x%016llx\n",
			first.discreteCases, first.discreteOverlapScenarios, first.discreteSeparationScenarios, first.shallowFaceScenarios,
			first.edgeScenarios, first.deepScenarios, first.cavityScenarios, first.disconnectedScenarios,
			first.checkerboardScenarios, first.exactCcdHits, first.exactCcdMisses, first.aggregateCcdHits,
			first.aggregateCcdMisses, first.metamorphicCases, first.canonicalLateSelections, first.canonicalDepthRejects,
			first.canonicalFallbackKeys, first.canonicalFallbackLeafTests, first.canonicalConvexCases,
			first.canonicalConvexHits, first.canonicalConvexMisses, first.canonicalConvexPseudoSat,
			first.canonicalConvexFallbackKeys, ARRAY_COUNT( s_hardSeeds ),
			(unsigned long long)first.digest );
	ENSURE( first.discreteCases >= 8192 );
	ENSURE( first.discreteOverlapScenarios >= 512 );
	ENSURE( first.discreteSeparationScenarios >= 512 );
	ENSURE( first.shallowFaceScenarios >= 512 );
	ENSURE( first.edgeScenarios >= 512 );
	ENSURE( first.deepScenarios >= 512 );
	ENSURE( first.cavityScenarios >= 512 );
	ENSURE( first.disconnectedScenarios >= 512 );
	ENSURE( first.checkerboardScenarios >= 512 );
	ENSURE( first.exactCcdHits >= 256 );
	ENSURE( first.exactCcdMisses >= 256 );
	ENSURE( first.aggregateCcdHits >= 256 );
	ENSURE( first.aggregateCcdMisses >= 256 );
	ENSURE( first.metamorphicCases >= 512 );
	ENSURE( first.canonicalLateSelections > 0 );
	ENSURE( first.canonicalFallbackKeys > 0 );
	ENSURE( first.canonicalFallbackLeafTests >= first.canonicalFallbackKeys );
	ENSURE( first.canonicalConvexCases >= 1024 );
	ENSURE( first.canonicalConvexHits > 0 );
	ENSURE( first.canonicalConvexMisses > 0 );
	ENSURE( first.canonicalConvexPseudoSat > 0 );
	return 0;
}
