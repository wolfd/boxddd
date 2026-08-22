// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#include "math_internal.h" // internal cell-support TOI
#include "arena_allocator.h"
#include "contact.h"       // persistent voxel workspace validation
#include "physics_world.h" // test-only contact sparse-array inspection
#include "recording.h"     // save-state envelope corruption checks
#include "shape.h"		   // test-only construction of convex CCD targets
#include "test_macros.h"
#include "test_voxel_oracle.h"
#include "voxel_collide.h" // internal box primitives
#include "voxel_shape.h"   // internal exposure-mask query

#include "box3d/box3d.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct VoxelGeometryCoverage
{
	int analyticPins;
	int boundaryWitnesses;
	int kernelDecisive;
	int kernelOverlaps;
	int kernelSeparations;
	int kernelFaceAxes;
	int kernelEdgeAxes;
	int kernelShallowFaces;
	int kernelDeepOverlaps;
	int kernelAmbiguous;
	int discreteTopologies;
	int discreteDirections;
	int discreteOverlaps;
	int discreteSeparations;
	int discreteUnequalWidths;
	int convexTargetKinds;
	int convexCases;
	int convexOverlaps;
	int convexSeparations;
	int convexBoundaryWitnesses;
	int metamorphicSwaps;
	int metamorphicCommonTransforms;
	int metamorphicOrigins;
	int metamorphicInsertionOrders;
	int metamorphicBuriedCells;
	int metamorphicRepeats;
	int ccdCases;
	int ccdHits;
	int ccdMisses;
	int ccdRotating;
	int ccdClipped;
	int ccdMonotonic;
	float ccdMaxSpatialError;
	int aggregateCcdCases;
	int aggregateCcdHits;
	int aggregateCcdMisses;
	int aggregateCcdRotating;
	int aggregateCcdClipped;
	int aggregateCcdSwaps;
	int aggregateCcdCommonTransforms;
	float aggregateCcdWidestBracket;
	int aggregateSmokeTargetKinds;
	int aggregateSmokeCases;
	int aggregateSmokeHits;
	int aggregateSmokeMisses;
} VoxelGeometryCoverage;

static VoxelGeometryCoverage s_geometryCoverage;
static int s_boundaryFailureReports;
static int s_rawDiagnosticReports;

static int TestCollideSolids( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
							  b3Transform transform1, b3VoxelContact* contacts );

static int VoxelCanonicalPatchIdentity( void )
{
	b3Vec3i domainMin = { -2, -3, -4 };
	b3Vec3i domainMax = { 5, 6, 7 };
	b3Vec3i cell = { 1, 2, 3 };
	for ( int mask = 0; mask < 64; ++mask )
	{
		int expected = ( ( mask & 0x03 ) != 0 ) + ( ( mask & 0x0C ) != 0 ) + ( ( mask & 0x30 ) != 0 );
		ENSURE( b3VoxelClassifyTopology( (uint8_t)mask ) == (b3VoxelTopology)expected );
		ENSURE( b3VoxelClassifyTopology( (uint8_t)( mask | 0xC0 ) ) == (b3VoxelTopology)expected );
		b3VoxelPatchRange range = b3VoxelCanonicalPatchRange( cell, (uint8_t)mask, domainMin, domainMax );
		for ( int axis = 0; axis < 3; ++axis )
		{
			int expectedLower = ( mask & ( 1 << ( 2 * axis ) ) ) != 0 ? ( &cell.x )[axis] : ( &domainMin.x )[axis] - 1;
			int expectedUpper =
				( mask & ( 1 << ( 2 * axis + 1 ) ) ) != 0 ? ( &cell.x )[axis] : ( &domainMax.x )[axis] + 1;
			ENSURE( ( &range.lower.x )[axis] == expectedLower );
			ENSURE( ( &range.upper.x )[axis] == expectedUpper );
		}
	}

	b3VoxelPatchRange interior = b3VoxelCanonicalPatchRange( cell, 0, domainMin, domainMax );
	ENSURE( interior.lower.x == -3 && interior.lower.y == -4 && interior.lower.z == -5 );
	ENSURE( interior.upper.x == 6 && interior.upper.y == 7 && interior.upper.z == 8 );

	uint8_t exposed = (uint8_t)( ( 1u << 0 ) | ( 1u << 3 ) | ( 1u << 4 ) | ( 1u << 5 ) );
	b3VoxelPatchRange mixed = b3VoxelCanonicalPatchRange( cell, exposed, domainMin, domainMax );
	ENSURE( mixed.lower.x == 1 && mixed.lower.y == -4 && mixed.lower.z == 3 );
	ENSURE( mixed.upper.x == 6 && mixed.upper.y == 2 && mixed.upper.z == 3 );

	// Cells on the same planar union face alias, while a parallel face one row
	// away and the opposite orientation remain distinct.
	b3VoxelPatchRange face0 = b3VoxelCanonicalPatchRange( (b3Vec3i){ 1, 2, 3 }, 1u << 2, domainMin, domainMax );
	b3VoxelPatchRange face1 = b3VoxelCanonicalPatchRange( (b3Vec3i){ 4, 2, 6 }, 1u << 2, domainMin, domainMax );
	b3VoxelPatchRange otherRow = b3VoxelCanonicalPatchRange( (b3Vec3i){ 4, 3, 6 }, 1u << 2, domainMin, domainMax );
	b3VoxelPatchRange opposite = b3VoxelCanonicalPatchRange( (b3Vec3i){ 4, 2, 6 }, 1u << 3, domainMin, domainMax );
	b3VoxelPatchKey key0 = { face0, mixed };
	b3VoxelPatchKey key1 = { face1, mixed };
	b3VoxelPatchKey rowKey = { otherRow, mixed };
	b3VoxelPatchKey oppositeKey = { opposite, mixed };
	ENSURE( b3VoxelPatchKeyEqual( &key0, &key1 ) );
	ENSURE( b3VoxelPatchKeyHash( &key0 ) == b3VoxelPatchKeyHash( &key1 ) );
	ENSURE( !b3VoxelPatchKeyEqual( &key0, &rowKey ) );
	ENSURE( !b3VoxelPatchKeyEqual( &key0, &oppositeKey ) );
	b3VoxelPatchKey swapped = b3VoxelSwapPatchKey( key0 );
	ENSURE( !b3VoxelPatchKeyEqual( &key0, &swapped ) );
	swapped = b3VoxelSwapPatchKey( swapped );
	ENSURE( b3VoxelPatchKeyEqual( &key0, &swapped ) );

	// Force every key through one probe chain. Full equality, not hash
	// equality, must preserve distinct keys and discovery order through growth.
	b3Arena arena = b3CreateArena( 1024 );
	b3VoxelPatchTable table;
	b3VoxelPatchTableInit( &table, &arena );
	for ( int i = 0; i < 80; ++i )
	{
		b3VoxelPatchKey key = key0;
		key.patch1.lower.x = i - 40;
		bool inserted = false;
		int index = b3VoxelPatchTableFindOrInsert( &table, &key, UINT64_C( 7 ), &inserted );
		ENSURE( inserted && index == i );
		ENSURE( b3VoxelPatchKeyEqual( table.keys + index, &key ) );
	}
	ENSURE( table.count == 80 );
	for ( int i = 79; i >= 0; --i )
	{
		b3VoxelPatchKey key = key0;
		key.patch1.lower.x = i - 40;
		bool inserted = true;
		int index = b3VoxelPatchTableFindOrInsert( &table, &key, UINT64_C( 7 ), &inserted );
		ENSURE( !inserted && index == i );
	}
	b3DestroyArena( &arena );
	return 0;
}

static int TestValidateUnionContacts( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
									  b3Transform transform1, const b3VoxelContact* contacts, int count )
{
	const float tolerance = TestVoxelBoundaryTolerance();
	for ( int i = 0; i < count; ++i )
	{
		const b3VoxelContact* contact = contacts + i;
		ENSURE( isfinite( contact->normal.x ) && isfinite( contact->normal.y ) && isfinite( contact->normal.z ) );
		ENSURE( isfinite( contact->body0Point.x ) && isfinite( contact->body0Point.y ) && isfinite( contact->body0Point.z ) );
		ENSURE( isfinite( contact->body1Point.x ) && isfinite( contact->body1Point.y ) && isfinite( contact->body1Point.z ) );
		ENSURE( isfinite( contact->initialPenetration ) );
		ENSURE_SMALL( b3Length( contact->normal ) - 1.0f, 2.0e-4f );
		b3Vec3 delta = b3Sub( contact->body1Point, contact->body0Point );
		ENSURE_SMALL( b3Dot( delta, contact->normal ) - contact->initialPenetration, 2.0e-4f );
		if ( !TestPointOnUnionBoundary( solid0, transform0, contact->body0Point, tolerance ) )
		{
			if ( s_boundaryFailureReports < 4 )
			{
				b3Vec3 local = b3InvTransformPoint( transform0, contact->body0Point );
				printf( "union-boundary failure: side=0 contact=%d local=(%.9g,%.9g,%.9g) normal=(%.9g,%.9g,%.9g) "
						"penetration=%.9g\n",
						i, local.x, local.y, local.z, contact->normal.x, contact->normal.y, contact->normal.z,
						contact->initialPenetration );
			}
			s_boundaryFailureReports += 1;
			return 1;
		}
		if ( !TestPointOnUnionBoundary( solid1, transform1, contact->body1Point, tolerance ) )
		{
			if ( s_boundaryFailureReports < 4 )
			{
				b3Vec3 local = b3InvTransformPoint( transform1, contact->body1Point );
				printf( "union-boundary failure: side=1 contact=%d local=(%.9g,%.9g,%.9g) normal=(%.9g,%.9g,%.9g) "
						"penetration=%.9g\n",
						i, local.x, local.y, local.z, contact->normal.x, contact->normal.y, contact->normal.z,
						contact->initialPenetration );
			}
			s_boundaryFailureReports += 1;
			return 1;
		}
		s_geometryCoverage.boundaryWitnesses += 2;
	}
	return 0;
}

static b3VoxelOBB MakeOBB( b3Vec3 center, b3Quat q, b3Vec3 half )
{
	b3VoxelOBB o;
	o.center = center;
	o.half = half;
	o.axes[0] = b3RotateVector( q, (b3Vec3){ 1.0f, 0.0f, 0.0f } );
	o.axes[1] = b3RotateVector( q, (b3Vec3){ 0.0f, 1.0f, 0.0f } );
	o.axes[2] = b3RotateVector( q, (b3Vec3){ 0.0f, 0.0f, 1.0f } );
	return o;
}

static float ProjR( const b3VoxelOBB* b, b3Vec3 a )
{
	return fabsf( b3Dot( a, b->axes[0] ) ) * b->half.x + fabsf( b3Dot( a, b->axes[1] ) ) * b->half.y +
		   fabsf( b3Dot( a, b->axes[2] ) ) * b->half.z;
}

static float OverlapAlong( const b3VoxelOBB* o0, const b3VoxelOBB* o1, b3Vec3 a )
{
	b3Vec3 d = b3Sub( o0->center, o1->center );
	return ProjR( o0, a ) + ProjR( o1, a ) - fabsf( b3Dot( a, d ) );
}

static float RefMinOverlap( const b3VoxelOBB* o0, const b3VoxelOBB* o1, b3Vec3* axisOut )
{
	b3Vec3 axes[15];
	int na = 0;
	for ( int i = 0; i < 3; ++i )
		axes[na++] = o0->axes[i];
	for ( int i = 0; i < 3; ++i )
		axes[na++] = o1->axes[i];
	for ( int i = 0; i < 3; ++i )
		for ( int j = 0; j < 3; ++j )
			axes[na++] = b3Cross( o0->axes[i], o1->axes[j] );

	b3Vec3 d = b3Sub( o0->center, o1->center );
	float minOv = FLT_MAX;
	b3Vec3 minAxis = { 0.0f, 0.0f, 1.0f };
	for ( int k = 0; k < na; ++k )
	{
		b3Vec3 a = axes[k];
		float l = b3Dot( a, a );
		if ( l <= 1e-9f )
			continue;
		a = b3MulSV( 1.0f / sqrtf( l ), a );
		float proj = b3Dot( a, d );
		float ov = ProjR( o0, a ) + ProjR( o1, a ) - fabsf( proj );
		if ( ov < minOv )
		{
			minOv = ov;
			minAxis = proj >= 0.0f ? a : b3Neg( a );
		}
	}
	*axisOut = minAxis;
	return minOv;
}

static uint32_t s_rng = 0x9e3779b9u;
static float Frand( void )
{
	s_rng ^= s_rng << 13;
	s_rng ^= s_rng >> 17;
	s_rng ^= s_rng << 5;
	return (float)( ( s_rng >> 8 ) & 0xFFFFFF ) / (float)0x1000000; // [0,1)
}
static float Frand2( void )
{
	return Frand() * 2.0f - 1.0f;
}
static b3Quat RandQuat( void )
{
	b3Vec3 axis = { Frand2(), Frand2(), Frand2() };
	float len = b3Length( axis );
	if ( len < 0.1f )
		axis = (b3Vec3){ 0.0f, 0.0f, 1.0f };
	else
		axis = b3MulSV( 1.0f / len, axis );
	float ang = Frand() * B3_PI;
	float s = sinf( ang * 0.5f );
	return (b3Quat){ { axis.x * s, axis.y * s, axis.z * s }, cosf( ang * 0.5f ) };
}

static int VoxelObbKnownCases( void )
{
	b3Quat qi = b3Quat_identity;

	b3VoxelOBB a = MakeOBB( (b3Vec3){ 0.8f, 0.0f, 0.0f }, qi, (b3Vec3){ 0.5f, 0.5f, 0.5f } );
	b3VoxelOBB b = MakeOBB( (b3Vec3){ 0.0f, 0.0f, 0.0f }, qi, (b3Vec3){ 0.5f, 0.5f, 0.5f } );
	b3VoxelContact out[4];
	int n = b3VoxelCollideOBB( &a, &b, 0.0f, 4, out );
	ENSURE( n >= 1 );
	ENSURE_SMALL( out[0].normal.x - 1.0f, 1e-4f ); // B->A points +x (A at +x)
	ENSURE_SMALL( out[0].normal.y, 1e-4f );
	ENSURE_SMALL( out[0].normal.z, 1e-4f );
	ENSURE_SMALL( out[0].initialPenetration - 0.2f, 1e-4f );
	ENSURE( n == 4 );

	b3VoxelOBB c = MakeOBB( (b3Vec3){ 3.0f, 0.0f, 0.0f }, qi, (b3Vec3){ 0.5f, 0.5f, 0.5f } );
	ENSURE( b3VoxelCollideOBB( &c, &b, 0.0f, 4, out ) == 0 );

	b3VoxelOBB e = MakeOBB( (b3Vec3){ 1.1f, 0.0f, 0.0f }, qi, (b3Vec3){ 0.5f, 0.5f, 0.5f } );
	n = b3VoxelCollideOBB( &e, &b, 0.2f, 4, out );
	ENSURE( n >= 1 );
	ENSURE( out[0].initialPenetration < 0.0f ); // gap -> negative penetration
	ENSURE_SMALL( out[0].initialPenetration - ( -0.1f ), 1e-4f );
	return 0;
}

static int VoxelObbFuzz( void )
{
	s_rng = 0x1234567u;
	for ( int iter = 0; iter < 40000; ++iter )
	{
		b3VoxelOBB o0 =
			MakeOBB( (b3Vec3){ 0.0f, 0.0f, 0.0f }, RandQuat(), (b3Vec3){ 0.3f + Frand(), 0.3f + Frand(), 0.3f + Frand() } );
		b3VoxelOBB o1 = MakeOBB( (b3Vec3){ Frand2() * 2.0f, Frand2() * 2.0f, Frand2() * 2.0f }, RandQuat(),
								 (b3Vec3){ 0.3f + Frand(), 0.3f + Frand(), 0.3f + Frand() } );

		b3Vec3 refAxis;
		float refOv = RefMinOverlap( &o0, &o1, &refAxis );

		b3VoxelContact out[4];
		int n = b3VoxelCollideOBB( &o0, &o1, 0.0f, 4, out );

		if ( refOv < -1e-3f )
		{
			ENSURE( n == 0 );
			s_geometryCoverage.kernelDecisive += 1;
			s_geometryCoverage.kernelSeparations += 1;
		}
		else if ( refOv > 1e-3f )
		{
			ENSURE( n >= 1 );
			b3ObbPairContext pairContext;
			b3VoxelPrepareOBBPair( &pairContext, &o0, &o1 );
			b3ObbSat sat;
			ENSURE( b3VoxelComputeOBBSat( b3Sub( o0.center, o1.center ), &pairContext, 0.0f, &sat ) );
			float ovN = OverlapAlong( &o0, &o1, out[0].normal );
			ENSURE_SMALL( ovN - refOv, 3e-3f );
			b3Vec3 diff = b3Sub( out[0].body1Point, out[0].body0Point );
			ENSURE( b3Dot( diff, out[0].normal ) > -1e-3f );
			float maxPen = out[0].initialPenetration;
			for ( int i = 1; i < n; ++i )
				maxPen = b3MaxFloat( maxPen, out[i].initialPenetration );
			ENSURE( maxPen > 0.0f );
			s_geometryCoverage.kernelDecisive += 1;
			s_geometryCoverage.kernelOverlaps += 1;
			if ( sat.isEdge )
			{
				s_geometryCoverage.kernelEdgeAxes += 1;
			}
			else
			{
				s_geometryCoverage.kernelFaceAxes += 1;
				if ( refOv < 0.1f )
				{
					s_geometryCoverage.kernelShallowFaces += 1;
				}
			}
			if ( refOv > 0.5f )
			{
				s_geometryCoverage.kernelDeepOverlaps += 1;
			}
		}
		else
		{
			s_geometryCoverage.kernelAmbiguous += 1;
		}
	}
	printf( "  G1 coverage: decisive=%d overlap=%d separation=%d face=%d edge=%d shallow_face=%d deep=%d ambiguous=%d\n",
			s_geometryCoverage.kernelDecisive, s_geometryCoverage.kernelOverlaps, s_geometryCoverage.kernelSeparations,
			s_geometryCoverage.kernelFaceAxes, s_geometryCoverage.kernelEdgeAxes, s_geometryCoverage.kernelShallowFaces,
			s_geometryCoverage.kernelDeepOverlaps, s_geometryCoverage.kernelAmbiguous );
	ENSURE( s_geometryCoverage.kernelDecisive >= 39000 );
	ENSURE( s_geometryCoverage.kernelOverlaps >= 20000 );
	ENSURE( s_geometryCoverage.kernelSeparations >= 14000 );
	ENSURE( s_geometryCoverage.kernelFaceAxes >= 10000 );
	ENSURE( s_geometryCoverage.kernelEdgeAxes >= 10000 );
	ENSURE( s_geometryCoverage.kernelShallowFaces >= 900 );
	ENSURE( s_geometryCoverage.kernelDeepOverlaps >= 10000 );
	return 0;
}

static int VoxelAabbFuzz( void )
{
	s_rng = 0x89abcdefu;
	for ( int iter = 0; iter < 40000; ++iter )
	{
		b3Vec3 c0 = { 0.0f, 0.0f, 0.0f }, c1 = { Frand2() * 2.0f, Frand2() * 2.0f, Frand2() * 2.0f };
		b3Vec3 h0 = { 0.3f + Frand(), 0.3f + Frand(), 0.3f + Frand() };
		b3Vec3 h1 = { 0.3f + Frand(), 0.3f + Frand(), 0.3f + Frand() };
		b3AABB a0 = { b3Sub( c0, h0 ), b3Add( c0, h0 ) };
		b3AABB a1 = { b3Sub( c1, h1 ), b3Add( c1, h1 ) };

		float ovx = b3MinFloat( a0.upperBound.x, a1.upperBound.x ) - b3MaxFloat( a0.lowerBound.x, a1.lowerBound.x );
		float ovy = b3MinFloat( a0.upperBound.y, a1.upperBound.y ) - b3MaxFloat( a0.lowerBound.y, a1.lowerBound.y );
		float ovz = b3MinFloat( a0.upperBound.z, a1.upperBound.z ) - b3MaxFloat( a0.lowerBound.z, a1.lowerBound.z );
		float minOv = b3MinFloat( ovx, b3MinFloat( ovy, ovz ) );

		b3VoxelContact out[4];
		int n = b3VoxelCollideAABB( &a0, &a1, 0.0f, 4, out );

		if ( minOv < -1e-3f )
			ENSURE( n == 0 );
		else if ( minOv > 1e-3f )
		{
			ENSURE( n >= 1 );
			ENSURE_SMALL( out[0].initialPenetration - minOv, 1e-4f );
			float nlen = b3Length( out[0].normal );
			ENSURE_SMALL( nlen - 1.0f, 1e-4f );
			b3Vec3 diff = b3Sub( out[0].body1Point, out[0].body0Point );
			ENSURE( b3Dot( diff, out[0].normal ) > -1e-3f );
		}
	}
	return 0;
}

static b3VoxelOBB CellOBB( b3Vec3i cell, float vs, b3Transform xf )
{
	b3Vec3 lc = { cell.x * vs, cell.y * vs, cell.z * vs };
	b3VoxelOBB o;
	o.center = b3Add( xf.p, b3RotateVector( xf.q, lc ) );
	o.axes[0] = b3RotateVector( xf.q, (b3Vec3){ 1.0f, 0.0f, 0.0f } );
	o.axes[1] = b3RotateVector( xf.q, (b3Vec3){ 0.0f, 1.0f, 0.0f } );
	o.axes[2] = b3RotateVector( xf.q, (b3Vec3){ 0.0f, 0.0f, 1.0f } );
	float h = 0.5f * vs;
	o.half = (b3Vec3){ h, h, h };
	return o;
}

static int OraclePairs( const b3Vec3i* c0, int n0, b3Transform xf0, const b3Vec3i* c1, int n1, b3Transform xf1, float* maxPen )
{
	int count = 0;
	float mp = -FLT_MAX;
	for ( int a = 0; a < n0; ++a )
	{
		b3VoxelOBB o0 = CellOBB( c0[a], 1.0f, xf0 );
		for ( int b = 0; b < n1; ++b )
		{
			b3VoxelOBB o1 = CellOBB( c1[b], 1.0f, xf1 );
			b3VoxelContact c[4];
			if ( b3VoxelCollideOBB( &o0, &o1, 0.0f, 1, c ) > 0 )
			{
				count++;
				if ( c[0].initialPenetration > mp )
					mp = c[0].initialPenetration;
			}
		}
	}
	*maxPen = mp;
	return count;
}

static b3AABB WorldBounds( const b3Vec3i* cells, int n, b3Transform xf )
{
	b3AABB r = { { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
	for ( int i = 0; i < n; ++i )
	{
		for ( int k = 0; k < 8; ++k )
		{
			b3Vec3 corner = { cells[i].x + ( ( k & 1 ) ? 0.5f : -0.5f ), cells[i].y + ( ( k & 2 ) ? 0.5f : -0.5f ),
							  cells[i].z + ( ( k & 4 ) ? 0.5f : -0.5f ) };
			b3Vec3 w = b3Add( xf.p, b3RotateVector( xf.q, corner ) );
			r.lowerBound.x = b3MinFloat( r.lowerBound.x, w.x );
			r.lowerBound.y = b3MinFloat( r.lowerBound.y, w.y );
			r.lowerBound.z = b3MinFloat( r.lowerBound.z, w.z );
			r.upperBound.x = b3MaxFloat( r.upperBound.x, w.x );
			r.upperBound.y = b3MaxFloat( r.upperBound.y, w.y );
			r.upperBound.z = b3MaxFloat( r.upperBound.z, w.z );
		}
	}
	return r;
}

static bool InAABB( b3Vec3 p, b3AABB b, float m )
{
	return p.x >= b.lowerBound.x - m && p.x <= b.upperBound.x + m && p.y >= b.lowerBound.y - m && p.y <= b.upperBound.y + m &&
		   p.z >= b.lowerBound.z - m && p.z <= b.upperBound.z + m;
}

static b3VoxelOBB TestSolidCellObb( const TestVoxelSolid* solid, b3Vec3i cell, b3Transform transform )
{
	b3VoxelOBB obb;
	obb.center = b3TransformPoint( transform, TestCellCenter( solid, cell ) );
	obb.axes[0] = b3RotateVector( transform.q, b3Vec3_axisX );
	obb.axes[1] = b3RotateVector( transform.q, b3Vec3_axisY );
	obb.axes[2] = b3RotateVector( transform.q, b3Vec3_axisZ );
	float half = 0.5f * solid->size;
	obb.half = (b3Vec3){ half, half, half };
	return obb;
}

static void TestDiagnoseRawCellWitnesses( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
										  b3Transform transform1, const b3VoxelContact* aggregate, int aggregateCount )
{
	const float tolerance = TestVoxelBoundaryTolerance();
	for ( int i = 0; i < solid0->count; ++i )
	{
		b3VoxelOBB obb0 = TestSolidCellObb( solid0, solid0->cells[i], transform0 );
		for ( int j = 0; j < solid1->count; ++j )
		{
			b3VoxelOBB obb1 = TestSolidCellObb( solid1, solid1->cells[j], transform1 );
			b3VoxelContact raw[4];
			int rawCount = b3VoxelCollideOBB( &obb0, &obb1, 0.0f, ARRAY_COUNT( raw ), raw );
			for ( int k = 0; k < rawCount; ++k )
			{
				bool boundary0 = TestPointOnUnionBoundary( solid0, transform0, raw[k].body0Point, tolerance );
				bool boundary1 = TestPointOnUnionBoundary( solid1, transform1, raw[k].body1Point, tolerance );
				bool selected = false;
				for ( int a = 0; a < aggregateCount; ++a )
				{
					if ( b3LengthSquared( b3Sub( raw[k].body0Point, aggregate[a].body0Point ) ) < 1.0e-10f &&
						 b3LengthSquared( b3Sub( raw[k].body1Point, aggregate[a].body1Point ) ) < 1.0e-10f )
					{
						selected = true;
						break;
					}
				}
				if ( selected || !boundary0 || !boundary1 )
				{
					b3Vec3 local0 = b3InvTransformPoint( transform0, raw[k].body0Point );
					b3Vec3 local1 = b3InvTransformPoint( transform1, raw[k].body1Point );
					printf( "  raw cell witness: A=(%d,%d,%d) B=(%d,%d,%d) point=%d selected=%d boundary=(%d,%d) "
							"localA=(%.9g,%.9g,%.9g) localB=(%.9g,%.9g,%.9g)\n",
							solid0->cells[i].x, solid0->cells[i].y, solid0->cells[i].z, solid1->cells[j].x, solid1->cells[j].y,
							solid1->cells[j].z, k, selected, boundary0, boundary1, local0.x, local0.y, local0.z, local1.x,
							local1.y, local1.z );
				}
			}
		}
	}
}

static int TestRunVoxelPair( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
							 b3Transform transform1, bool expectedOverlap, b3Vec3 expectedNormal, bool pinNormal )
{
	b3VoxelData* voxel0 = b3CreateOffsetVoxelData( solid0->cells, solid0->count, solid0->size, solid0->origin );
	b3VoxelData* voxel1 = b3CreateOffsetVoxelData( solid1->cells, solid1->count, solid1->size, solid1->origin );
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	int count = b3VoxelCollide( voxel0, transform0, voxel1, transform1, 0.0f, ARRAY_COUNT( contacts ), contacts );
	b3VoxelContact canonical[B3_VOXEL_MAX_CONTACTS];
	b3Arena arena = b3CreateArena( 16384 );
	int canonicalCount = b3VoxelCollideCanonicalWithArena( voxel0, transform0, voxel1, transform1, 0.0f,
														  ARRAY_COUNT( canonical ), canonical, &arena, NULL );
	if ( expectedOverlap )
	{
		ENSURE( count > 0 );
		ENSURE( canonicalCount > 0 );
		if ( pinNormal )
		{
			ENSURE( b3Dot( contacts[0].normal, expectedNormal ) > 1.0f - 1.0e-4f );
		}
		ENSURE( TestValidateUnionContacts( solid0, transform0, solid1, transform1, contacts, count ) == 0 );
		ENSURE( TestValidateUnionContacts( solid0, transform0, solid1, transform1, canonical, canonicalCount ) == 0 );
	}
	else
	{
		ENSURE( count == 0 );
		ENSURE( canonicalCount == 0 );
	}
	b3DestroyArena( &arena );
	b3DestroyVoxelData( voxel0 );
	b3DestroyVoxelData( voxel1 );
	return 0;
}

static int VoxelUnionBoundaryPins( void )
{
	b3Transform identity = b3Transform_identity;
	b3Vec3i oneCell[] = { { 0, 0, 0 } };
	b3Vec3i seamCells[] = { { 0, 0, 0 }, { 1, 0, 0 } };
	TestVoxelSolid seam = { seamCells, ARRAY_COUNT( seamCells ), 1.0f, b3Vec3_zero };
	TestVoxelSolid one = { oneCell, ARRAY_COUNT( oneCell ), 1.0f, b3Vec3_zero };

	// A target straddles the x seam, but its valid contact is the exposed +y
	// union face. A witness justified only by the shared x face fails the
	// independent boundary test.
	b3Transform aboveSeam = { { 0.5f, 0.9f, 0.0f }, b3Quat_identity };
	ENSURE( TestRunVoxelPair( &seam, identity, &one, aboveSeam, true, b3Neg( b3Vec3_axisY ), true ) == 0 );
	s_geometryCoverage.analyticPins += 1;

	// The same two-cell union approached from both outer x faces pins that the
	// shared face is not substituted for the exterior boundary.
	b3Transform fromRight = { { 1.9f, 0.0f, 0.0f }, b3Quat_identity };
	ENSURE( TestRunVoxelPair( &seam, identity, &one, fromRight, true, b3Neg( b3Vec3_axisX ), true ) == 0 );
	b3Transform fromLeft = { { -0.9f, 0.0f, 0.0f }, b3Quat_identity };
	ENSURE( TestRunVoxelPair( &seam, identity, &one, fromLeft, true, b3Vec3_axisX, true ) == 0 );
	s_geometryCoverage.analyticPins += 2;

	// Local origins are part of the represented solid. Compensating body
	// translation aligns the two non-zero origins with a known 0.2 overlap.
	TestVoxelSolid offset0 = { oneCell, 1, 0.75f, { 3.25f, -2.0f, 1.5f } };
	TestVoxelSolid offset1 = { oneCell, 1, 0.75f, { -4.0f, 1.25f, -3.5f } };
	b3Vec3 wantedCenter1 = b3Add( offset0.origin, (b3Vec3){ 0.55f, 0.0f, 0.0f } );
	b3Transform offsetTransform1 = { b3Sub( wantedCenter1, offset1.origin ), b3Quat_identity };
	ENSURE( TestRunVoxelPair( &offset0, identity, &offset1, offsetTransform1, true, b3Neg( b3Vec3_axisX ), true ) == 0 );
	s_geometryCoverage.analyticPins += 1;

	// A sealed 3x3x3 shell has an empty center. A smaller target at the center
	// is separated; offsetting it into the +x inner wall produces a cavity-face
	// contact whose shell witness must lie on that exposed inner surface.
	b3Vec3i shellCells[26];
	int shellCount = 0;
	for ( int x = -1; x <= 1; ++x )
		for ( int y = -1; y <= 1; ++y )
			for ( int z = -1; z <= 1; ++z )
				if ( x != 0 || y != 0 || z != 0 )
					shellCells[shellCount++] = (b3Vec3i){ x, y, z };
	TestVoxelSolid shell = { shellCells, shellCount, 1.0f, b3Vec3_zero };
	TestVoxelSolid cavityTarget = { oneCell, 1, 0.5f, b3Vec3_zero };
	ENSURE( TestRunVoxelPair( &shell, identity, &cavityTarget, identity, false, b3Vec3_zero, false ) == 0 );
	b3Transform intoInnerWall = { { 0.4f, 0.0f, 0.0f }, b3Quat_identity };
	ENSURE( TestRunVoxelPair( &shell, identity, &cavityTarget, intoInnerWall, true, b3Vec3_axisX, true ) == 0 );
	s_geometryCoverage.analyticPins += 2;

	printf( "  G0 coverage: analytic_pins=%d boundary_witnesses=%d\n", s_geometryCoverage.analyticPins,
			s_geometryCoverage.boundaryWitnesses );
	ENSURE( s_geometryCoverage.analyticPins >= 6 );
	ENSURE( s_geometryCoverage.boundaryWitnesses >= 12 );
	return 0;
}

static int VoxelVoxelInternalAxisFallbackPin( void )
{
	b3Vec3i cellsA[] = { { 0, 0, 0 }, { 1, 0, 0 } };
	b3Vec3i cellB = { 0, 0, 0 };
	TestVoxelSolid solidA = {
		cellsA,
		ARRAY_COUNT( cellsA ),
		0.881066799f,
		{ 0.108119465f, 0.0098132249f, 0.0362815782f },
	};
	TestVoxelSolid solidB = {
		&cellB,
		1,
		0.916291177f,
		{ -0.307008535f, 0.0371976532f, 0.0221963935f },
	};
	b3Transform transformB = { { 1.29437864f, -0.0273844283f, 0.0140851848f }, b3Quat_identity };
	ENSURE( TestRunVoxelPair( &solidA, b3Transform_identity, &solidB, transformB, true, b3Vec3_zero, false ) == 0 );
	return 0;
}

typedef struct TestOwnedVoxelSolid
{
	b3Vec3i cells[32];
	TestVoxelSolid solid;
} TestOwnedVoxelSolid;

#define TEST_TOPOLOGY_COUNT 16

static void TestMakeTopology( TestOwnedVoxelSolid* owned, int kind )
{
	memset( owned, 0, sizeof( *owned ) );
	owned->solid.cells = owned->cells;
	owned->solid.size = 0.8f;
	owned->solid.origin = b3Vec3_zero;
	switch ( kind )
	{
		case 0: // single
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 0, 0 };
			break;
		case 1: // face-connected beam
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 1, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 2, 0, 0 };
			break;
		case 2: // L
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 1, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 1, 0 };
			break;
		case 3: // stair
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 1, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 1, 1, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 2, 1, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 2, 2, 0 };
			break;
		case 4: // disconnected islands and a long empty span
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 3, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ -3, 1, 0 };
			break;
		case 5: // concave planar notch
			for ( int x = 0; x < 3; ++x )
				for ( int y = 0; y < 3; ++y )
					if ( x != 1 || y != 1 )
						owned->cells[owned->solid.count++] = (b3Vec3i){ x, y, 0 };
			break;
		case 6: // sealed cavity / hollow shell
			for ( int x = -1; x <= 1; ++x )
				for ( int y = -1; y <= 1; ++y )
					for ( int z = -1; z <= 1; ++z )
						if ( x != 0 || y != 0 || z != 0 )
							owned->cells[owned->solid.count++] = (b3Vec3i){ x, y, z };
			break;
		case 7: // edge-connected pair
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 1, 1, 0 };
			break;
		case 8: // corner-connected pair
			owned->cells[owned->solid.count++] = (b3Vec3i){ 0, 0, 0 };
			owned->cells[owned->solid.count++] = (b3Vec3i){ 1, 1, 1 };
			break;
		case 9: // solid block
			for ( int x = 0; x < 2; ++x )
				for ( int y = 0; y < 2; ++y )
					for ( int z = 0; z < 2; ++z )
						owned->cells[owned->solid.count++] = (b3Vec3i){ x, y, z };
			break;
		case 10: // thick slab
			for ( int x = 0; x < 3; ++x )
				for ( int y = 0; y < 2; ++y )
					for ( int z = 0; z < 3; ++z )
						owned->cells[owned->solid.count++] = (b3Vec3i){ x, y, z };
			break;
		case 11: // one-cell-thick wall
			for ( int y = -1; y <= 1; ++y )
				for ( int z = -1; z <= 1; ++z )
					owned->cells[owned->solid.count++] = (b3Vec3i){ 0, y, z };
			break;
		case 12: // one-cell-thick plate
			for ( int x = -1; x <= 1; ++x )
				for ( int z = -1; z <= 1; ++z )
					owned->cells[owned->solid.count++] = (b3Vec3i){ x, 0, z };
			break;
		case 13: // T
			for ( int x = -2; x <= 2; ++x )
				owned->cells[owned->solid.count++] = (b3Vec3i){ x, 1, 0 };
			for ( int y = -1; y <= 0; ++y )
				owned->cells[owned->solid.count++] = (b3Vec3i){ 0, y, 0 };
			break;
		case 14: // open cavity
			for ( int x = -1; x <= 1; ++x )
				for ( int y = -1; y <= 1; ++y )
					for ( int z = -1; z <= 1; ++z )
						if ( ( x != 0 || y != 0 || z != 0 ) && ( x != 1 || y != 0 || z != 0 ) )
							owned->cells[owned->solid.count++] = (b3Vec3i){ x, y, z };
			break;
		default: // sparse checkerboard
			for ( int x = 0; x < 4; ++x )
				for ( int y = 0; y < 4; ++y )
					for ( int z = 0; z < 2; ++z )
						if ( ( x + y + z ) % 2 == 0 )
							owned->cells[owned->solid.count++] = (b3Vec3i){ x, y, z };
			break;
	}
}

static int TestDiscreteDirection( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
								  b3Transform transform1, bool oracleOverlap )
{
	b3VoxelData* voxel0 = b3CreateOffsetVoxelData( solid0->cells, solid0->count, solid0->size, solid0->origin );
	b3VoxelData* voxel1 = b3CreateOffsetVoxelData( solid1->cells, solid1->count, solid1->size, solid1->origin );
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	int count = b3VoxelCollide( voxel0, transform0, voxel1, transform1, 0.0f, ARRAY_COUNT( contacts ), contacts );
	b3VoxelContact canonical[B3_VOXEL_MAX_CONTACTS];
	b3Arena arena = b3CreateArena( 16384 );
	int canonicalCount = b3VoxelCollideCanonicalWithArena( voxel0, transform0, voxel1, transform1, 0.0f,
														  ARRAY_COUNT( canonical ), canonical, &arena, NULL );
	int result = 0;
	if ( ( count > 0 ) != oracleOverlap )
	{
		printf( "G2 classification mismatch: specialized=%s oracle=%s\n", count > 0 ? "overlap" : "separated",
				oracleOverlap ? "overlap" : "separated" );
		result = 1;
	}
	if ( ( canonicalCount > 0 ) != oracleOverlap )
	{
		printf( "G2 canonical classification mismatch: canonical=%s oracle=%s old=%s\n",
				canonicalCount > 0 ? "overlap" : "separated", oracleOverlap ? "overlap" : "separated",
				count > 0 ? "overlap" : "separated" );
		result = 1;
	}
	if ( canonicalCount > 0 )
	{
		result |= TestValidateUnionContacts( solid0, transform0, solid1, transform1, canonical, canonicalCount );
	}
	if ( count > 0 )
	{
		s_geometryCoverage.discreteOverlaps += 1;
		int validation = TestValidateUnionContacts( solid0, transform0, solid1, transform1, contacts, count );
		if ( validation != 0 && s_rawDiagnosticReports < 2 )
		{
			TestDiagnoseRawCellWitnesses( solid0, transform0, solid1, transform1, contacts, count );
			s_rawDiagnosticReports += 1;
		}
		result |= validation;
	}
	else
	{
		s_geometryCoverage.discreteSeparations += 1;
	}
	s_geometryCoverage.discreteDirections += 1;
	b3DestroyArena( &arena );
	b3DestroyVoxelData( voxel0 );
	b3DestroyVoxelData( voxel1 );
	return result;
}

static int VoxelAggregateDiscreteDifferential( void )
{
	TestOwnedVoxelSolid targetOwned;
	TestMakeTopology( &targetOwned, 1 );
	int failureCount = 0;
	for ( int topology = 0; topology < TEST_TOPOLOGY_COUNT; ++topology )
	{
		TestOwnedVoxelSolid owned;
		TestMakeTopology( &owned, topology );
		const TestVoxelSolid* solid0 = &owned.solid;
		const TestVoxelSolid* solid1 = &targetOwned.solid;
		b3Transform transform0 = { { -0.15f, 0.1f, -0.05f },
								   b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 0.7f, -0.2f } ), 0.18f ) };
		b3Transform placements[3] = {
			{ { 0.08f, 0.03f, 0.11f }, b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ -0.4f, 0.2f, 0.8f } ), 0.31f ) },
			{ { 0.55f, -0.12f, 0.16f }, b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.5f, 0.6f, 0.1f } ), -0.24f ) },
			{ { 12.0f, -9.0f, 7.0f }, b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, -0.9f, 0.3f } ), 0.73f ) },
		};
		for ( int placement = 0; placement < ARRAY_COUNT( placements ); ++placement )
		{
			b3Transform transform1 = placements[placement];
			bool oracle = TestGenericUnionOverlap( solid0, transform0, solid1, transform1 );
			int result = TestDiscreteDirection( solid0, transform0, solid1, transform1, oracle );
			if ( result != 0 )
			{
				if ( failureCount < 4 )
					printf( "G2 fixture: topology=%d placement=%d direction=forward oracle=%s\n", topology, placement,
							oracle ? "overlap" : "separated" );
				failureCount += 1;
			}
			// Explicitly reverse arguments so neither outer traversal direction can
			// hide an asymmetric pruning or exposure defect.
			result = TestDiscreteDirection( solid1, transform1, solid0, transform0, oracle );
			if ( result != 0 )
			{
				if ( failureCount < 4 )
					printf( "G2 fixture: topology=%d placement=%d direction=reverse oracle=%s\n", topology, placement,
							oracle ? "overlap" : "separated" );
				failureCount += 1;
			}
		}
		s_geometryCoverage.discreteTopologies += 1;
	}

	// Unequal widths use different non-zero origins and nearly parallel body
	// axes. Run an overlapping containment/seam case and a clear separation in
	// both argument orders so outer traversal cannot assume a shared grid scale.
	b3Vec3i fineCells[] = { { -1, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 } };
	b3Vec3i coarseCell = { 0, 0, 0 };
	TestVoxelSolid fine = { fineCells, ARRAY_COUNT( fineCells ), 0.45f, { 0.17f, -0.11f, 0.08f } };
	TestVoxelSolid coarse = { &coarseCell, 1, 1.1f, { -0.23f, 0.19f, -0.14f } };
	b3Transform fineTransform = { { -0.12f, 0.07f, -0.04f },
								  b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, 0.8f, -0.3f } ), 0.015f ) };
	b3Transform coarseTransforms[] = {
		{ { 0.18f, -0.06f, 0.09f }, b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.21f, 0.79f, -0.31f } ), 0.017f ) },
		{ { 8.0f, -6.0f, 5.0f }, b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ -0.4f, 0.3f, 0.7f } ), 0.42f ) },
	};
	for ( int placement = 0; placement < ARRAY_COUNT( coarseTransforms ); ++placement )
	{
		bool oracle = TestGenericUnionOverlap( &fine, fineTransform, &coarse, coarseTransforms[placement] );
		failureCount += TestDiscreteDirection( &fine, fineTransform, &coarse, coarseTransforms[placement], oracle );
		failureCount += TestDiscreteDirection( &coarse, coarseTransforms[placement], &fine, fineTransform, oracle );
		s_geometryCoverage.discreteUnequalWidths += 2;
	}

	printf( "  G2 coverage: topologies=%d directions=%d overlap=%d separation=%d unequal_width=%d witness_failures=%d\n",
			s_geometryCoverage.discreteTopologies, s_geometryCoverage.discreteDirections, s_geometryCoverage.discreteOverlaps,
			s_geometryCoverage.discreteSeparations, s_geometryCoverage.discreteUnequalWidths, failureCount );
	ENSURE( s_geometryCoverage.discreteTopologies >= TEST_TOPOLOGY_COUNT );
	ENSURE( s_geometryCoverage.discreteDirections >= 6 * TEST_TOPOLOGY_COUNT );
	ENSURE( s_geometryCoverage.discreteOverlaps >= 4 * TEST_TOPOLOGY_COUNT );
	ENSURE( s_geometryCoverage.discreteSeparations >= 2 * TEST_TOPOLOGY_COUNT );
	ENSURE( s_geometryCoverage.discreteUnequalWidths >= 4 );
	ENSURE( failureCount == 0 );
	return 0;
}

static int TestValidateVoxelConvexContacts( const TestVoxelSolid* solid, const b3VoxelContact* contacts, int count )
{
	float tolerance = TestVoxelBoundaryTolerance();
	for ( int i = 0; i < count; ++i )
	{
		const b3VoxelContact* contact = contacts + i;
		ENSURE( isfinite( contact->normal.x ) && isfinite( contact->normal.y ) && isfinite( contact->normal.z ) );
		ENSURE( isfinite( contact->body0Point.x ) && isfinite( contact->body0Point.y ) && isfinite( contact->body0Point.z ) );
		ENSURE( isfinite( contact->body1Point.x ) && isfinite( contact->body1Point.y ) && isfinite( contact->body1Point.z ) );
		ENSURE( isfinite( contact->initialPenetration ) );
		ENSURE_SMALL( b3Length( contact->normal ) - 1.0f, 2.0e-4f );
		b3Vec3 delta = b3Sub( contact->body1Point, contact->body0Point );
		ENSURE_SMALL( b3Dot( delta, contact->normal ) - contact->initialPenetration, 2.0e-4f );
		ENSURE( TestPointOnUnionBoundary( solid, b3Transform_identity, contact->body0Point, tolerance ) );
		s_geometryCoverage.convexBoundaryWitnesses += 1;
	}
	return 0;
}

static int TestDiscreteConvexCase( const TestVoxelSolid* solid, const b3Shape* target, b3Transform transformTargetToVoxel,
								   bool expectedOverlap )
{
	bool oracle = TestGenericVoxelConvexOverlap( solid, target, transformTargetToVoxel );
	ENSURE( oracle == expectedOverlap );
	b3VoxelData* voxel = b3CreateOffsetVoxelData( solid->cells, solid->count, solid->size, solid->origin );
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	int count =
		b3VoxelCollideConvex( voxel, target, transformTargetToVoxel, 0.0f, ARRAY_COUNT( contacts ), contacts, NULL, NULL );
	b3DestroyVoxelData( voxel );
	ENSURE( ( count > 0 ) == oracle );
	if ( count > 0 )
	{
		int validation = TestValidateVoxelConvexContacts( solid, contacts, count );
		if ( validation != 0 )
		{
			printf( "voxel/convex witness failure: target_type=%d cells=%d transform=(%.9g,%.9g,%.9g) count=%d\n",
					(int)target->type, solid->count, transformTargetToVoxel.p.x, transformTargetToVoxel.p.y,
					transformTargetToVoxel.p.z, count );
			for ( int i = 0; i < count; ++i )
			{
				b3Vec3 delta = b3Sub( contacts[i].body1Point, contacts[i].body0Point );
				printf( "  contact=%d n=(%.9g,%.9g,%.9g) a=(%.9g,%.9g,%.9g) b=(%.9g,%.9g,%.9g) penetration=%.9g "
						"projected=%.9g\n",
						i, contacts[i].normal.x, contacts[i].normal.y, contacts[i].normal.z, contacts[i].body0Point.x,
						contacts[i].body0Point.y, contacts[i].body0Point.z, contacts[i].body1Point.x, contacts[i].body1Point.y,
						contacts[i].body1Point.z, contacts[i].initialPenetration, b3Dot( delta, contacts[i].normal ) );
			}
			return 1;
		}
		s_geometryCoverage.convexOverlaps += 1;
	}
	else
	{
		s_geometryCoverage.convexSeparations += 1;
	}
	s_geometryCoverage.convexCases += 1;
	return 0;
}

static int VoxelConvexWitnessPins( void )
{
	b3Vec3i cell = { 0, 0, 0 };
	TestVoxelSolid solid = { &cell, 1, 0.8f, b3Vec3_zero };
	b3Shape sphere = { .type = b3_sphereShape, .sphere = { b3Vec3_zero, 0.35f } };
	b3Transform sphereToVoxel = { { -0.45f, 0.0f, 0.0f }, b3Quat_identity };
	ENSURE( TestDiscreteConvexCase( &solid, &sphere, sphereToVoxel, true ) == 0 );
	return 0;
}

static int VoxelConvexInternalSeamWitnessPin( void )
{
	b3Vec3i cells[] = { { 1, 0, 0 }, { 1, 0, -1 } };
	TestVoxelSolid solid = {
		cells,
		ARRAY_COUNT( cells ),
		0.342321455f,
		{ -0.0202023555f, -0.337713152f, -0.213618875f },
	};
	b3Shape capsule = {
		.type = b3_capsuleShape,
		.capsule = { { 0.0f, -0.32f, 0.0f }, { 0.0f, 0.32f, 0.0f }, 0.22f },
	};
	b3Transform capsuleToVoxel = { { 0.277794182f, -0.0888428092f, -0.316016674f }, b3Quat_identity };
	ENSURE( TestDiscreteConvexCase( &solid, &capsule, capsuleToVoxel, true ) == 0 );
	return 0;
}

static int VoxelConvexDeepSphereWitnessPin( void )
{
	b3Vec3i cell = { 0, 0, 0 };
	TestVoxelSolid solid = {
		&cell,
		1,
		0.750906646f,
		{ -0.0455468521f, -0.317672223f, -0.281858236f },
	};
	b3Shape sphere = { .type = b3_sphereShape, .sphere = { b3Vec3_zero, 0.38f } };
	b3Transform sphereToVoxel = {
		{ -0.167110831f, -0.146906286f, -0.0771540925f },
		{ { 0.250596732f, -0.157846421f, 0.156177536f }, 0.942281425f },
	};
	ENSURE( TestDiscreteConvexCase( &solid, &sphere, sphereToVoxel, true ) == 0 );
	return 0;
}

static int VoxelConvexEmbeddedEscapePin( void )
{
	b3Vec3i cells[27];
	int count = 0;
	for ( int x = -1; x <= 1; ++x )
	{
		for ( int y = -1; y <= 1; ++y )
		{
			for ( int z = -1; z <= 1; ++z )
			{
				cells[count++] = (b3Vec3i){ x, y, z };
			}
		}
	}
	TestVoxelSolid solid = { cells, count, 1.0f, b3Vec3_zero };
	b3BoxHull box = b3MakeBoxHull( 0.2f, 0.2f, 0.2f );
	b3Shape target = { .type = b3_hullShape, .hull = &box.base };
	ENSURE( TestDiscreteConvexCase( &solid, &target, b3Transform_identity, true ) == 0 );
	return 0;
}

static int VoxelVoxelEmbeddedEscapePin( void )
{
	b3Vec3i containerCells[27];
	int containerCount = 0;
	for ( int x = -1; x <= 1; ++x )
	{
		for ( int y = -1; y <= 1; ++y )
		{
			for ( int z = -1; z <= 1; ++z )
			{
				containerCells[containerCount++] = (b3Vec3i){ x, y, z };
			}
		}
	}
	b3Vec3i embeddedCell = { 0, 0, 0 };
	TestVoxelSolid container = { containerCells, containerCount, 1.0f, b3Vec3_zero };
	TestVoxelSolid embedded = { &embeddedCell, 1, 0.2f, b3Vec3_zero };
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	b3Transform containerTransforms[] = {
		b3Transform_identity,
		{ { 0.4f, -0.3f, 0.2f }, b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 0.7f, -0.2f } ), 0.38f ) },
	};
	b3Transform embeddedRelative[] = {
		b3Transform_identity,
		{ { 0.11f, -0.07f, 0.09f }, b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ -0.4f, 0.2f, 0.8f } ), 0.31f ) },
	};
	for ( int i = 0; i < ARRAY_COUNT( containerTransforms ); ++i )
	{
		b3Transform containerTransform = containerTransforms[i];
		b3Transform embeddedTransform = b3MulTransforms( containerTransform, embeddedRelative[i] );
		int count = TestCollideSolids( &container, containerTransform, &embedded, embeddedTransform, contacts );
		ENSURE( count > 0 );
		ENSURE( TestValidateUnionContacts( &container, containerTransform, &embedded, embeddedTransform, contacts, count ) == 0 );

		count = TestCollideSolids( &embedded, embeddedTransform, &container, containerTransform, contacts );
		ENSURE( count > 0 );
		ENSURE( TestValidateUnionContacts( &embedded, embeddedTransform, &container, containerTransform, contacts, count ) == 0 );
	}
	return 0;
}

static int VoxelBoxInternalSeamWitnessPin( void )
{
	b3Vec3i cells[] = { { 0, 0, 0 }, { 0, -1, 0 } };
	TestVoxelSolid solid = {
		cells,
		ARRAY_COUNT( cells ),
		0.892696798f,
		{ 0.196078882f, 0.303839773f, 0.225865766f },
	};
	b3BoxHull box = b3MakeOffsetBoxHull( 0.35f, 0.42f, 0.31f, (b3Vec3){ 0.07f, -0.05f, 0.03f } );
	b3Shape target = { .type = b3_hullShape, .hull = &box.base };
	b3Transform boxToVoxel = {
		{ -0.00745275011f, 0.0634740144f, -0.0440412536f },
		{ { 0.193514377f, -0.136878744f, -0.139407903f }, 0.961447835f },
	};
	ENSURE( TestDiscreteConvexCase( &solid, &target, boxToVoxel, true ) == 0 );
	return 0;
}

static int VoxelAggregateConvexDiscreteDifferential( void )
{
	b3Sphere sphere = { b3Vec3_zero, 0.35f };
	b3Capsule capsule = { { 0.0f, -0.3f, 0.0f }, { 0.0f, 0.3f, 0.0f }, 0.22f };
	b3BoxHull box = b3MakeBoxHull( 0.32f, 0.28f, 0.36f );
	b3BoxHull offsetBox = b3MakeOffsetBoxHull( 0.29f, 0.34f, 0.27f, (b3Vec3){ 0.08f, -0.04f, 0.06f } );
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
	b3HullData* tetra = b3CreateHull( tetraPoints, ARRAY_COUNT( tetraPoints ), ARRAY_COUNT( tetraPoints ) );
	b3HullData* irregular = b3CreateHull( irregularPoints, ARRAY_COUNT( irregularPoints ), ARRAY_COUNT( irregularPoints ) );
	ENSURE( tetra != NULL && irregular != NULL );
	b3Shape targets[] = {
		{ .type = b3_sphereShape, .sphere = sphere }, { .type = b3_capsuleShape, .capsule = capsule },
		{ .type = b3_hullShape, .hull = &box.base },  { .type = b3_hullShape, .hull = &offsetBox.base },
		{ .type = b3_hullShape, .hull = tetra },	  { .type = b3_hullShape, .hull = irregular },
	};
	b3Quat rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ -0.3f, 0.7f, 0.4f } ), 0.27f );
	for ( int topology = 0; topology < 5; ++topology )
	{
		TestOwnedVoxelSolid owned;
		TestMakeTopology( &owned, topology );
		for ( int targetIndex = 0; targetIndex < ARRAY_COUNT( targets ); ++targetIndex )
		{
			b3Transform overlap = { { -0.45f, 0.03f, -0.02f }, rotation };
			ENSURE( TestDiscreteConvexCase( &owned.solid, targets + targetIndex, overlap, true ) == 0 );
			b3Transform separated = { { -8.0f, 6.0f, -5.0f }, rotation };
			ENSURE( TestDiscreteConvexCase( &owned.solid, targets + targetIndex, separated, false ) == 0 );
		}
	}
	s_geometryCoverage.convexTargetKinds = ARRAY_COUNT( targets );
	b3DestroyHull( tetra );
	b3DestroyHull( irregular );
	printf( "  convex G2 coverage: targets=%d cases=%d overlap=%d separation=%d boundary_witnesses=%d\n",
			s_geometryCoverage.convexTargetKinds, s_geometryCoverage.convexCases, s_geometryCoverage.convexOverlaps,
			s_geometryCoverage.convexSeparations, s_geometryCoverage.convexBoundaryWitnesses );
	ENSURE( s_geometryCoverage.convexTargetKinds >= 6 );
	ENSURE( s_geometryCoverage.convexCases >= 60 );
	ENSURE( s_geometryCoverage.convexOverlaps >= 30 );
	ENSURE( s_geometryCoverage.convexSeparations >= 30 );
	ENSURE( s_geometryCoverage.convexBoundaryWitnesses >= 30 );
	return 0;
}

static int TestCollideSolids( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
							  b3Transform transform1, b3VoxelContact* contacts )
{
	b3VoxelData* voxel0 = b3CreateOffsetVoxelData( solid0->cells, solid0->count, solid0->size, solid0->origin );
	b3VoxelData* voxel1 = b3CreateOffsetVoxelData( solid1->cells, solid1->count, solid1->size, solid1->origin );
	int count = b3VoxelCollide( voxel0, transform0, voxel1, transform1, 0.0f, B3_VOXEL_MAX_CONTACTS, contacts );
	b3DestroyVoxelData( voxel0 );
	b3DestroyVoxelData( voxel1 );
	return count;
}

static bool TestVec3Near( b3Vec3 a, b3Vec3 b, float tolerance )
{
	return b3LengthSquared( b3Sub( a, b ) ) <= tolerance * tolerance;
}

static bool TestContactScalarsNear( const b3VoxelContact* a, const b3VoxelContact* b, float tolerance )
{
	return b3AbsFloat( a->initialPenetration - b->initialPenetration ) <= tolerance &&
		   b3AbsFloat( a->penetrationDepth - b->penetrationDepth ) <= tolerance;
}

static bool TestContactsMatchRigidTransform( const b3VoxelContact* source, int sourceCount, const b3VoxelContact* transformed,
											 int transformedCount, b3Transform transform )
{
	if ( sourceCount != transformedCount )
	{
		printf( "rigid contact-count mismatch: source=%d transformed=%d\n", sourceCount, transformedCount );
		return false;
	}
	bool used[B3_VOXEL_MAX_CONTACTS] = { false };
	const float tolerance = 5.0e-4f;
	for ( int i = 0; i < sourceCount; ++i )
	{
		b3Vec3 normal = b3RotateVector( transform.q, source[i].normal );
		b3Vec3 point0 = b3TransformPoint( transform, source[i].body0Point );
		b3Vec3 point1 = b3TransformPoint( transform, source[i].body1Point );
		bool found = false;
		for ( int j = 0; j < transformedCount; ++j )
		{
			if ( !used[j] && TestVec3Near( normal, transformed[j].normal, tolerance ) &&
				 TestVec3Near( point0, transformed[j].body0Point, tolerance ) &&
				 TestVec3Near( point1, transformed[j].body1Point, tolerance ) &&
				 TestContactScalarsNear( source + i, transformed + j, tolerance ) )
			{
				used[j] = true;
				found = true;
				break;
			}
		}
		if ( !found )
		{
			printf( "rigid contact mismatch: source=%d expected_n=(%.9g,%.9g,%.9g) expected_a=(%.9g,%.9g,%.9g) "
					"expected_b=(%.9g,%.9g,%.9g) penetration=%.9g\n",
					i, normal.x, normal.y, normal.z, point0.x, point0.y, point0.z, point1.x, point1.y, point1.z,
					source[i].initialPenetration );
			for ( int j = 0; j < transformedCount; ++j )
			{
				printf( "  actual=%d n=(%.9g,%.9g,%.9g) a=(%.9g,%.9g,%.9g) b=(%.9g,%.9g,%.9g) penetration=%.9g\n", j,
						transformed[j].normal.x, transformed[j].normal.y, transformed[j].normal.z, transformed[j].body0Point.x,
						transformed[j].body0Point.y, transformed[j].body0Point.z, transformed[j].body1Point.x,
						transformed[j].body1Point.y, transformed[j].body1Point.z, transformed[j].initialPenetration );
			}
			return false;
		}
	}
	return true;
}

static bool TestContactsMatchSwap( const b3VoxelContact* forward, int forwardCount, const b3VoxelContact* reverse,
								   int reverseCount )
{
	if ( forwardCount != reverseCount )
		return false;
	bool used[B3_VOXEL_MAX_CONTACTS] = { false };
	const float tolerance = 5.0e-4f;
	for ( int i = 0; i < forwardCount; ++i )
	{
		bool found = false;
		for ( int j = 0; j < reverseCount; ++j )
		{
			if ( !used[j] && TestVec3Near( forward[i].normal, b3Neg( reverse[j].normal ), tolerance ) &&
				 TestVec3Near( forward[i].body0Point, reverse[j].body1Point, tolerance ) &&
				 TestVec3Near( forward[i].body1Point, reverse[j].body0Point, tolerance ) &&
				 TestContactScalarsNear( forward + i, reverse + j, tolerance ) )
			{
				used[j] = true;
				found = true;
				break;
			}
		}
		if ( !found )
			return false;
	}
	return true;
}

static bool TestContactsPreserveRigidGeometry( const b3VoxelContact* source, int sourceCount, const b3VoxelContact* transformed,
											   int transformedCount, b3Transform transform )
{
	if ( sourceCount == 0 || transformedCount == 0 )
		return sourceCount == transformedCount;
	int deepest = 0;
	for ( int i = 1; i < sourceCount; ++i )
	{
		if ( source[i].initialPenetration > source[deepest].initialPenetration )
			deepest = i;
	}
	b3Vec3 expectedNormal = b3RotateVector( transform.q, source[deepest].normal );
	for ( int i = 0; i < transformedCount; ++i )
	{
		if ( b3Dot( expectedNormal, transformed[i].normal ) > 1.0f - 1.0e-4f &&
			 b3AbsFloat( source[deepest].initialPenetration - transformed[i].initialPenetration ) <= 1.0e-3f )
		{
			return true;
		}
	}
	return false;
}

static bool TestContactsBitEqual( const b3VoxelContact* a, int countA, const b3VoxelContact* b, int countB )
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

static int VoxelAggregateMetamorphic( void )
{
	TestOwnedVoxelSolid targetOwned;
	TestMakeTopology( &targetOwned, 1 );
	b3Transform transform0 = { { -0.15f, 0.1f, -0.05f },
							   b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 0.7f, -0.2f } ), 0.18f ) };
	b3Transform transform1 = { { 0.08f, 0.03f, 0.11f },
							   b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ -0.4f, 0.2f, 0.8f } ), 0.31f ) };
	b3Transform common = { { 2.25f, -1.75f, 0.625f },
						   b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, -0.6f, 0.7f } ), 0.53f ) };

	for ( int topology = 0; topology < 4; ++topology )
	{
		TestOwnedVoxelSolid owned;
		TestMakeTopology( &owned, topology );
		const TestVoxelSolid* solid0 = &owned.solid;
		const TestVoxelSolid* solid1 = &targetOwned.solid;
		b3VoxelContact baseline[B3_VOXEL_MAX_CONTACTS];
		int baselineCount = TestCollideSolids( solid0, transform0, solid1, transform1, baseline );
		ENSURE( baselineCount > 0 );
		ENSURE( TestValidateUnionContacts( solid0, transform0, solid1, transform1, baseline, baselineCount ) == 0 );

		b3VoxelContact repeated[B3_VOXEL_MAX_CONTACTS];
		int repeatedCount = TestCollideSolids( solid0, transform0, solid1, transform1, repeated );
		ENSURE( TestContactsBitEqual( baseline, baselineCount, repeated, repeatedCount ) );
		s_geometryCoverage.metamorphicRepeats += 1;

		b3VoxelContact reverse[B3_VOXEL_MAX_CONTACTS];
		int reverseCount = TestCollideSolids( solid1, transform1, solid0, transform0, reverse );
		ENSURE( TestContactsMatchSwap( baseline, baselineCount, reverse, reverseCount ) );
		s_geometryCoverage.metamorphicSwaps += 1;

		b3Transform common0 = b3MulTransforms( common, transform0 );
		b3Transform common1 = b3MulTransforms( common, transform1 );
		b3VoxelContact transformed[B3_VOXEL_MAX_CONTACTS];
		int transformedCount = TestCollideSolids( solid0, common0, solid1, common1, transformed );
		ENSURE( TestContactsPreserveRigidGeometry( baseline, baselineCount, transformed, transformedCount, common ) );
		ENSURE( TestValidateUnionContacts( solid0, common0, solid1, common1, transformed, transformedCount ) == 0 );
		s_geometryCoverage.metamorphicCommonTransforms += 1;

		TestVoxelSolid shifted = *solid0;
		b3Vec3 originDelta = { 1.25f, -0.75f, 0.5f };
		shifted.origin = b3Add( shifted.origin, originDelta );
		b3Transform shiftedTransform = transform0;
		shiftedTransform.p = b3Sub( shiftedTransform.p, b3RotateVector( shiftedTransform.q, originDelta ) );
		b3VoxelContact originContacts[B3_VOXEL_MAX_CONTACTS];
		int originCount = TestCollideSolids( &shifted, shiftedTransform, solid1, transform1, originContacts );
		ENSURE( TestContactsMatchRigidTransform( baseline, baselineCount, originContacts, originCount, b3Transform_identity ) );
		ENSURE( TestValidateUnionContacts( &shifted, shiftedTransform, solid1, transform1, originContacts, originCount ) == 0 );
		s_geometryCoverage.metamorphicOrigins += 1;

		b3Vec3i reversed0[32];
		b3Vec3i reversed1[32];
		for ( int i = 0; i < solid0->count; ++i )
			reversed0[i] = solid0->cells[solid0->count - 1 - i];
		for ( int i = 0; i < solid1->count; ++i )
			reversed1[i] = solid1->cells[solid1->count - 1 - i];
		TestVoxelSolid shuffled0 = { reversed0, solid0->count, solid0->size, solid0->origin };
		TestVoxelSolid shuffled1 = { reversed1, solid1->count, solid1->size, solid1->origin };
		b3VoxelContact insertionContacts[B3_VOXEL_MAX_CONTACTS];
		int insertionCount = TestCollideSolids( &shuffled0, transform0, &shuffled1, transform1, insertionContacts );
		ENSURE( TestContactsBitEqual( baseline, baselineCount, insertionContacts, insertionCount ) );
		s_geometryCoverage.metamorphicInsertionOrders += 1;
	}

	b3Vec3i fullCells[27];
	b3Vec3i shellCells[26];
	int fullCount = 0;
	int shellCount = 0;
	for ( int x = -1; x <= 1; ++x )
		for ( int y = -1; y <= 1; ++y )
			for ( int z = -1; z <= 1; ++z )
			{
				b3Vec3i cell = { x, y, z };
				fullCells[fullCount++] = cell;
				if ( x != 0 || y != 0 || z != 0 )
					shellCells[shellCount++] = cell;
			}
	TestVoxelSolid full = { fullCells, fullCount, 0.8f, b3Vec3_zero };
	TestVoxelSolid shell = { shellCells, shellCount, 0.8f, b3Vec3_zero };
	b3Vec3i targetCell = { 0, 0, 0 };
	TestVoxelSolid target = { &targetCell, 1, 0.8f, b3Vec3_zero };
	b3Transform targetTransform = { { 1.35f, 0.0f, 0.0f }, b3Quat_identity };
	b3VoxelContact fullContacts[B3_VOXEL_MAX_CONTACTS];
	b3VoxelContact shellContacts[B3_VOXEL_MAX_CONTACTS];
	int fullContactCount = TestCollideSolids( &full, b3Transform_identity, &target, targetTransform, fullContacts );
	int shellContactCount = TestCollideSolids( &shell, b3Transform_identity, &target, targetTransform, shellContacts );
	ENSURE( fullContactCount > 0 );
	ENSURE( TestContactsMatchRigidTransform( fullContacts, fullContactCount, shellContacts, shellContactCount,
											 b3Transform_identity ) );
	ENSURE( TestValidateUnionContacts( &full, b3Transform_identity, &target, targetTransform, fullContacts, fullContactCount ) ==
			0 );
	ENSURE( TestValidateUnionContacts( &shell, b3Transform_identity, &target, targetTransform, shellContacts,
									   shellContactCount ) == 0 );
	s_geometryCoverage.metamorphicBuriedCells += 1;

	printf( "  metamorphic coverage: swap=%d common=%d origin=%d insertion=%d buried=%d repeat=%d\n",
			s_geometryCoverage.metamorphicSwaps, s_geometryCoverage.metamorphicCommonTransforms,
			s_geometryCoverage.metamorphicOrigins, s_geometryCoverage.metamorphicInsertionOrders,
			s_geometryCoverage.metamorphicBuriedCells, s_geometryCoverage.metamorphicRepeats );
	ENSURE( s_geometryCoverage.metamorphicSwaps >= 4 );
	ENSURE( s_geometryCoverage.metamorphicCommonTransforms >= 4 );
	ENSURE( s_geometryCoverage.metamorphicOrigins >= 4 );
	ENSURE( s_geometryCoverage.metamorphicInsertionOrders >= 4 );
	ENSURE( s_geometryCoverage.metamorphicBuriedCells >= 1 );
	ENSURE( s_geometryCoverage.metamorphicRepeats >= 4 );
	return 0;
}

static b3VoxelData* MakeVoxelBox( int x0, int x1, int y0, int y1, int z0, int z1, float s );

static int VoxelDriverBasic( void )
{
	b3Vec3i c[] = { { 0, 0, 0 } };
	b3VoxelData* v0 = b3CreateVoxelData( c, 1, 1.0f );
	b3VoxelData* v1 = b3CreateVoxelData( c, 1, 1.0f );
	b3Transform x1 = { { 0.0f, 0.0f, 0.0f }, b3Quat_identity };
	b3VoxelContact out[64];

	b3Transform x0 = { { 0.8f, 0.0f, 0.0f }, b3Quat_identity };
	int n = b3VoxelCollide( v0, x0, v1, x1, 0.0f, 64, out );
	ENSURE( n >= 1 );
	ENSURE_SMALL( out[0].normal.x - 1.0f, 1e-3f );
	ENSURE_SMALL( out[0].initialPenetration - 0.2f, 1e-3f );
	b3Arena arena = b3CreateArena( 4096 );
	b3VoxelCounters counters = { 0 };
	int trackedCount = b3VoxelCollideWithArena( v0, x0, v1, x1, 0.0f, 64, out, &arena, &counters );
	ENSURE( trackedCount == n );
	ENSURE( counters.voxelVoxelCalls == 1 );
	ENSURE( counters.patchVisits == 1 );
	ENSURE( counters.patchUniqueKeys == 1 );
	ENSURE( counters.patchDuplicateVisits == 0 );
	ENSURE( counters.patchEligibleVisits == 1 );
	ENSURE( counters.patchEligibleUniqueKeys == 1 );
	ENSURE( counters.patchMaxMultiplicity == 1 );
	ENSURE( counters.patchMaxUniqueKeys == 1 );
	ENSURE( counters.patchMultiplicityCounts[0] == 1 );
	ENSURE( counters.patchTopologyPairs[15] == 1 );
	ENSURE( counters.patchSatFaceAxes == 1 );
	ENSURE( counters.patchSatEdgeAxes == 0 );
	ENSURE( counters.patchSatSeparations == 0 );
	ENSURE( counters.patchTableGrowths == 2 );
	ENSURE( counters.patchScratchPeakBytes >= 1024 );
	b3ArenaSync( &arena );
	counters = (b3VoxelCounters){ 0 };
	int canonicalCount = b3VoxelCollideCanonicalWithArena( v0, x0, v1, x1, 0.0f, 64, out, &arena, &counters );
	ENSURE( canonicalCount >= 1 );
	ENSURE( counters.patchVisits == 1 );
	ENSURE( counters.patchUniqueKeys == 1 );
	ENSURE( counters.pseudoSatCalls == 1 );
	ENSURE( counters.selectedPatchKeys == 1 );
	ENSURE( counters.emittedPatchManifolds == 1 );
	b3DestroyArena( &arena );

	b3Transform xf = { { 3.0f, 0.0f, 0.0f }, b3Quat_identity };
	ENSURE( b3VoxelCollide( v0, xf, v1, x1, 0.0f, 64, out ) == 0 );
	ENSURE( b3VoxelUseCanonicalPatches( v0, b3Quat_identity, v1, b3Quat_identity ) );
	b3VoxelData* large0 = MakeVoxelBox( 0, 7, 0, 7, 0, 7, 1.0f );
	b3VoxelData* large1 = MakeVoxelBox( 0, 7, 0, 7, 0, 7, 1.0f );
	ENSURE( b3VoxelUseCanonicalPatches( large0, b3Quat_identity, large1, b3Quat_identity ) );
	b3Quat adaptiveRotation = b3MakeQuatFromAxisAngle( (b3Vec3){ 0.0f, 1.0f, 0.0f }, 0.29f );
	ENSURE( b3VoxelUseCanonicalPatches( large0, b3Quat_identity, large1, adaptiveRotation ) == false );
	b3Arena adaptiveArena = b3CreateArena( 16384 );
	b3VoxelCounters adaptiveCounters = { 0 };
	b3Transform largeXf = { { 0.0f, 7.8f, 0.0f }, adaptiveRotation };
	ENSURE( b3VoxelCollideWithArena( large0, b3Transform_identity, large1, largeXf, 0.0f, 64, out, &adaptiveArena,
									  &adaptiveCounters ) > 0 );
	ENSURE( adaptiveCounters.adaptiveLeafPairs == 1 );
	ENSURE( adaptiveCounters.voxelVoxelCalls == 1 );
	b3DestroyArena( &adaptiveArena );
	b3DestroyVoxelData( large1 );
	b3DestroyVoxelData( large0 );

	b3DestroyVoxelData( v0 );
	b3DestroyVoxelData( v1 );
	return 0;
}

static int VoxelCanonicalPatchBudget( void )
{
	enum
	{
		patchCount = 24,
	};
	b3Vec3i cells[patchCount];
	b3Vec3i reversed[patchCount];
	for ( int i = 0; i < patchCount; ++i )
	{
		cells[i] = (b3Vec3i){ 3 * i, 0, 0 };
		reversed[patchCount - 1 - i] = cells[i];
	}
	TestVoxelSolid solid = { cells, patchCount, 1.0f, b3Vec3_zero };
	b3VoxelData* voxel0 = b3CreateVoxelData( cells, patchCount, 1.0f );
	b3VoxelData* voxel1 = b3CreateVoxelData( cells, patchCount, 1.0f );
	b3VoxelData* reordered = b3CreateVoxelData( reversed, patchCount, 1.0f );
	b3Transform transform0 = b3Transform_identity;
	b3Transform transform1 = { { 0.0f, 0.8f, 0.0f }, b3Quat_identity };
	b3Arena arena = b3CreateArena( 65536 );

	b3VoxelPatchCollision first;
	b3VoxelCounters firstCounters = { 0 };
	int count = b3VoxelBuildCanonicalPatches( voxel0, transform0, voxel1, transform1, 0.0f, &first, &arena, &firstCounters );
	ENSURE( count == B3_VOXEL_MAX_PATCH_MANIFOLDS );
	ENSURE( first.uniqueKeyCount == patchCount );
	ENSURE( firstCounters.patchVisits == patchCount );
	ENSURE( firstCounters.patchUniqueKeys == patchCount );
	ENSURE( firstCounters.selectedPatchKeys == patchCount );
	ENSURE( firstCounters.patchBudgetOverflows == patchCount - B3_VOXEL_MAX_PATCH_MANIFOLDS );

	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	int contactCount = 0;
	for ( int i = 0; i < first.manifoldCount; ++i )
	{
		const b3VoxelPatchManifold* manifold = first.manifolds + i;
		ENSURE( manifold->pointCount > 0 );
		ENSURE( contactCount + manifold->pointCount <= ARRAY_COUNT( contacts ) );
		memcpy( contacts + contactCount, manifold->points,
				(size_t)manifold->pointCount * sizeof( b3VoxelContact ) );
		contactCount += manifold->pointCount;
	}
	ENSURE( TestValidateUnionContacts( &solid, transform0, &solid, transform1, contacts, contactCount ) == 0 );

	b3ArenaSync( &arena );
	b3VoxelPatchCollision repeated;
	b3VoxelCounters repeatedCounters = { 0 };
	ENSURE( b3VoxelBuildCanonicalPatches( voxel0, transform0, voxel1, transform1, 0.0f, &repeated, &arena,
										   &repeatedCounters ) == count );
	ENSURE( memcmp( &first, &repeated, sizeof( first ) ) == 0 );
	ENSURE( memcmp( &firstCounters, &repeatedCounters, sizeof( firstCounters ) ) == 0 );

	b3ArenaSync( &arena );
	b3VoxelPatchCollision insertionOrder;
	b3VoxelCounters insertionCounters = { 0 };
	ENSURE( b3VoxelBuildCanonicalPatches( reordered, transform0, voxel1, transform1, 0.0f, &insertionOrder, &arena,
										   &insertionCounters ) == count );
	ENSURE( memcmp( &first, &insertionOrder, sizeof( first ) ) == 0 );
	ENSURE( memcmp( &firstCounters, &insertionCounters, sizeof( firstCounters ) ) == 0 );

	b3ArenaSync( &arena );
	b3VoxelPatchCollision swapped;
	ENSURE( b3VoxelBuildCanonicalPatches( voxel1, transform1, voxel0, transform0, 0.0f, &swapped, &arena, NULL ) == count );
	contactCount = 0;
	for ( int i = 0; i < swapped.manifoldCount; ++i )
	{
		const b3VoxelPatchManifold* manifold = swapped.manifolds + i;
		ENSURE( contactCount + manifold->pointCount <= ARRAY_COUNT( contacts ) );
		memcpy( contacts + contactCount, manifold->points,
				(size_t)manifold->pointCount * sizeof( b3VoxelContact ) );
		contactCount += manifold->pointCount;
	}
	ENSURE( TestValidateUnionContacts( &solid, transform1, &solid, transform0, contacts, contactCount ) == 0 );

	b3DestroyArena( &arena );
	b3DestroyVoxelData( reordered );
	b3DestroyVoxelData( voxel1 );
	b3DestroyVoxelData( voxel0 );
	return 0;
}

static int VoxelExposureMask( void )
{
	b3Vec3i cells[] = { { 0, 0, 0 }, { -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 } };
	b3VoxelData* voxel = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 1.0f );

	ENSURE( !b3Voxel_IsDirectionExposed( voxel, (b3Vec3i){ 0, 0, 0 }, (b3Vec3){ 1.0f, 0.0f, 0.0f } ) );
	ENSURE( !b3Voxel_IsDirectionExposed( voxel, (b3Vec3i){ 0, 0, 0 }, (b3Vec3){ 1.0f, 1.0f, 1.0f } ) );
	ENSURE( b3Voxel_IsDirectionExposed( voxel, (b3Vec3i){ 1, 0, 0 }, (b3Vec3){ 1.0f, 0.0f, 0.0f } ) );
	ENSURE( !b3Voxel_IsDirectionExposed( voxel, (b3Vec3i){ 1, 0, 0 }, (b3Vec3){ -1.0f, 0.0f, 0.0f } ) );

	b3DestroyVoxelData( voxel );
	return 0;
}

static int VoxelNegativeChunkCoordinates( void )
{
	b3Vec3i cells[] = { { -17, 0, 0 }, { -1, 0, 0 }, { 0, 0, 0 } };
	b3VoxelData* voxel = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 1.0f );
	b3AABB query = { { -17.5f, -0.5f, -0.5f }, { 0.5f, 0.5f, 0.5f } };
	b3Vec3i found[3] = { 0 };
	ENSURE( b3Voxel_QueryCells( voxel, query, found, ARRAY_COUNT( found ) ) == 3 );
	ENSURE( found[0].x == -17 && found[1].x == -1 && found[2].x == 0 );
	b3MassData mass = b3Voxel_ComputeMass( voxel, 1.0f );
	ENSURE( mass.mass == 3.0f );
	ENSURE( isfinite( mass.center.x ) && mass.center.x < 0.0f );
	b3DestroyVoxelData( voxel );
	return 0;
}

static int VoxelHullPublicManifold( void )
{
	b3Vec3i cell = { 0, 0, 0 };
	b3VoxelData* voxel = b3CreateOffsetVoxelData( &cell, 1, 1.0f, (b3Vec3){ 0.5f, 0.5f, 0.5f } );
	b3BoxHull hull = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
	b3Transform hullToVoxel = { { 1.3f, 0.5f, 0.5f }, b3Quat_identity };
	b3LocalManifoldPoint points[8];
	b3LocalManifold manifold = { 0 };
	manifold.points = points;

	b3CollideVoxelAndHull( &manifold, 8, voxel, &hull.base, hullToVoxel );

	ENSURE( manifold.pointCount > 0 );
	ENSURE_SMALL( manifold.normal.x - 1.0f, 1e-4f );
	ENSURE_SMALL( manifold.normal.y, 1e-4f );
	ENSURE_SMALL( manifold.normal.z, 1e-4f );
	ENSURE_SMALL( manifold.points[0].separation - ( -0.2f ), 1e-4f );

	b3DestroyVoxelData( voxel );
	return 0;
}

static int VoxelDriverSlab( void )
{
	b3Vec3i c[9];
	int n = 0;
	for ( int x = 0; x < 3; ++x )
		for ( int z = 0; z < 3; ++z )
			c[n++] = (b3Vec3i){ x, 0, z };
	b3VoxelData* v0 = b3CreateVoxelData( c, 9, 1.0f );
	b3VoxelData* v1 = b3CreateVoxelData( c, 9, 1.0f );
	b3Transform x0 = { { 0.0f, 0.0f, 0.0f }, b3Quat_identity };
	b3Transform x1 = { { 0.0f, 0.9f, 0.0f }, b3Quat_identity };
	b3VoxelContact out[64];
	int cnt = b3VoxelCollide( v0, x0, v1, x1, 0.0f, 64, out );
	ENSURE( cnt >= 4 );									// a spread resting manifold, not a single point
	ENSURE_SMALL( out[0].normal.y - ( -1.0f ), 1e-3f ); // B (above) -> A (below) = -y
	ENSURE_SMALL( out[0].initialPenetration - 0.1f, 1e-3f );
	b3AABB wb0 = WorldBounds( c, 9, x0 ), wb1 = WorldBounds( c, 9, x1 );
	for ( int i = 0; i < cnt; ++i )
	{
		ENSURE( InAABB( out[i].body0Point, wb0, 0.02f ) );
		ENSURE( InAABB( out[i].body1Point, wb1, 0.02f ) );
	}
	b3DestroyVoxelData( v0 );
	b3DestroyVoxelData( v1 );
	return 0;
}

static int VoxelDriverOracle( void )
{
	s_rng = 0xC0FFEEu;
	b3Vec3i c0[27];
	int n0 = 0;
	for ( int x = 0; x < 3; ++x )
		for ( int y = 0; y < 3; ++y )
			for ( int z = 0; z < 3; ++z )
				c0[n0++] = (b3Vec3i){ x, y, z };
	b3Vec3i c1[8];
	int n1 = 0;
	for ( int x = 0; x < 2; ++x )
		for ( int y = 0; y < 2; ++y )
			for ( int z = 0; z < 2; ++z )
				c1[n1++] = (b3Vec3i){ x, y, z };
	b3VoxelData* v0 = b3CreateVoxelData( c0, n0, 1.0f );
	b3VoxelData* v1 = b3CreateVoxelData( c1, n1, 1.0f );

	b3Transform x0 = { { 0.0f, 0.0f, 0.0f }, b3Quat_identity };
	int shallowChecked = 0;
	for ( int iter = 0; iter < 4000; ++iter )
	{
		bool rotate = ( iter & 1 ) != 0;
		b3Transform x1 = { { 0.5f + Frand2() * 2.5f, Frand2() * 3.0f, Frand2() * 3.0f }, rotate ? RandQuat() : b3Quat_identity };

		float mp;
		int op = OraclePairs( c0, n0, x0, c1, n1, x1, &mp );

		b3VoxelContact out[64];
		int n = b3VoxelCollide( v0, x0, v1, x1, 0.0f, 64, out );

		if ( op == 0 )
		{
			ENSURE( n == 0 ); // no cell pair overlaps -> no contacts
		}
		else if ( mp > 0.02f && mp < 0.4f )
		{
			ENSURE( n >= 1 ); // shallow surface overlap -> at least one contact survives culling
			b3AABB wb0 = WorldBounds( c0, n0, x0 ), wb1 = WorldBounds( c1, n1, x1 );
			b3AABB wbU = { { b3MinFloat( wb0.lowerBound.x, wb1.lowerBound.x ), b3MinFloat( wb0.lowerBound.y, wb1.lowerBound.y ),
							 b3MinFloat( wb0.lowerBound.z, wb1.lowerBound.z ) },
						   { b3MaxFloat( wb0.upperBound.x, wb1.upperBound.x ), b3MaxFloat( wb0.upperBound.y, wb1.upperBound.y ),
							 b3MaxFloat( wb0.upperBound.z, wb1.upperBound.z ) } };
			for ( int i = 0; i < n; ++i )
			{
				ENSURE_SMALL( b3Length( out[i].normal ) - 1.0f, 1e-3f );
				ENSURE( InAABB( out[i].body0Point, wbU, 0.1f ) );
				ENSURE( InAABB( out[i].body1Point, wbU, 0.1f ) );
				b3Vec3 diff = b3Sub( out[i].body1Point, out[i].body0Point );
				ENSURE_SMALL( b3Dot( diff, out[i].normal ) - out[i].initialPenetration, 1e-4f );
			}
			shallowChecked++;
		}
	}
	ENSURE( shallowChecked > 200 ); // the fuzz actually hit the shallow-overlap regime
	b3DestroyVoxelData( v0 );
	b3DestroyVoxelData( v1 );
	return 0;
}

static b3VoxelData* MakeVoxelBox( int x0, int x1, int y0, int y1, int z0, int z1, float s )
{
	int nx = x1 - x0 + 1, ny = y1 - y0 + 1, nz = z1 - z0 + 1;
	int count = nx * ny * nz;
	b3Vec3i* cells = (b3Vec3i*)malloc( (size_t)count * sizeof( b3Vec3i ) );
	int k = 0;
	for ( int x = x0; x <= x1; ++x )
		for ( int y = y0; y <= y1; ++y )
			for ( int z = z0; z <= z1; ++z )
				cells[k++] = (b3Vec3i){ x, y, z };
	b3VoxelData* v = b3CreateVoxelData( cells, count, s );
	free( cells );
	return v;
}

static int VoxelConvexCanonicalDispatchThreshold( void )
{
	b3VoxelData* large = MakeVoxelBox( 0, 7, 0, 7, 0, 7, 1.0f );
	b3VoxelData* small = MakeVoxelBox( 0, 6, 0, 7, 0, 7, 1.0f );
	b3BoxHull box = b3MakeBoxHull( 0.5f, 3.8f, 3.8f );
	b3Shape target = { .type = b3_hullShape, .flags = b3_boxHull, .hull = &box.base };
	b3VoxelContact contacts[B3_VOXEL_MAX_CONTACTS];
	b3Arena arena = b3CreateArena( 65536 );
	b3VoxelCounters counters = { 0 };

	b3Transform largeTransform = { { 7.9f, 3.5f, 3.5f }, b3Quat_identity };
	int exactLargeCount =
		b3VoxelCollideConvex( large, &target, largeTransform, 0.0f, ARRAY_COUNT( contacts ), contacts, NULL, NULL );
	int canonicalLargeCount = b3VoxelCollideConvex( large, &target, largeTransform, 0.0f, ARRAY_COUNT( contacts ), contacts,
											 &arena, &counters );
	ENSURE( exactLargeCount > 0 );
	ENSURE( canonicalLargeCount > 0 );
	ENSURE( counters.pseudoSatCalls > 0 );
	ENSURE( counters.patchVisits > counters.patchUniqueKeys );

	b3ArenaSync( &arena );
	counters = (b3VoxelCounters){ 0 };
	b3Transform smallTransform = { { 6.9f, 3.5f, 3.5f }, b3Quat_identity };
	int exactSmallCount =
		b3VoxelCollideConvex( small, &target, smallTransform, 0.0f, ARRAY_COUNT( contacts ), contacts, NULL, NULL );
	int thresholdSmallCount = b3VoxelCollideConvex( small, &target, smallTransform, 0.0f, ARRAY_COUNT( contacts ), contacts,
											  &arena, &counters );
	ENSURE( exactSmallCount > 0 );
	ENSURE( thresholdSmallCount == exactSmallCount );
	ENSURE( counters.pseudoSatCalls == 0 );
	ENSURE( counters.patchVisits == 0 );

	b3DestroyArena( &arena );
	b3DestroyVoxelData( small );
	b3DestroyVoxelData( large );
	return 0;
}

static int VoxelBodyRest( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	b3WorldId worldId = b3CreateWorld( &worldDef );
	ENSURE( b3World_IsValid( worldId ) );

	b3VoxelData* ground = MakeVoxelBox( -5, 5, 0, 0, -5, 5, 1.0f );
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.position = (b3Pos){ 0.0f, 0.0f, 0.0f };
	b3BodyId groundId = b3CreateBody( worldId, &groundDef );
	b3ShapeDef groundShapeDef = b3DefaultShapeDef();
	b3CreateVoxelShape( groundId, &groundShapeDef, ground );

	b3VoxelData* cube = MakeVoxelBox( -1, 1, -1, 1, -1, 1, 1.0f );
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ 0.0f, 5.0f, 0.0f };
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	b3CreateVoxelShape( bodyId, &shapeDef, cube );

	ENSURE( b3Body_GetMass( bodyId ) > 0.0f );

	float timeStep = 1.0f / 60.0f;
	for ( int i = 0; i < 240; ++i )
	{
		b3World_Step( worldId, timeStep, 4 );
	}

	b3Pos position = b3Body_GetPosition( bodyId );
	b3Quat rotation = b3Body_GetRotation( bodyId );

	b3DestroyWorld( worldId );
	b3DestroyVoxelData( ground );
	b3DestroyVoxelData( cube );

	ENSURE( position.y > 1.5f );			 // did not tunnel through the slab
	ENSURE_SMALL( position.y - 2.0f, 0.2f ); // came to rest on top
	ENSURE_SMALL( rotation.v.x, 0.05f );	 // did not tumble or explode
	ENSURE_SMALL( rotation.v.z, 0.05f );
	return 0;
}

static b3Contact* TestFindLiveVoxelWorkspace( b3World* world )
{
	for ( int i = 0; i < world->contacts.count; ++i )
	{
		b3Contact* contact = world->contacts.data + i;
		if ( contact->contactId == i && ( contact->flags & b3_simVoxelContact ) && contact->manifoldCount > 0 &&
			 contact->voxelContact.states.count == contact->manifoldCount )
		{
			return contact;
		}
	}
	return NULL;
}

static int VoxelPatchWorkspacePersistence( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	b3WorldId worldId = b3CreateWorld( &worldDef );

	b3VoxelData* ground = MakeVoxelBox( -4, 4, 0, 0, -4, 4, 1.0f );
	b3BodyDef groundDef = b3DefaultBodyDef();
	groundDef.enableContactRecycling = false;
	b3BodyId groundId = b3CreateBody( worldId, &groundDef );
	b3ShapeDef groundShapeDef = b3DefaultShapeDef();
	b3CreateVoxelShape( groundId, &groundShapeDef, ground );

	b3VoxelData* cube = MakeVoxelBox( -1, 1, -1, 1, -1, 1, 1.0f );
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.enableSleep = false;
	bodyDef.enableContactRecycling = false;
	bodyDef.position = (b3Pos){ 0.0f, 2.05f, 0.0f };
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	b3CreateVoxelShape( bodyId, &shapeDef, cube );

	const float timeStep = 1.0f / 60.0f;
	for ( int i = 0; i < 90; ++i )
		b3World_Step( worldId, timeStep, 4 );

	b3World* world = b3GetWorldFromId( worldId );
	b3Contact* contact = TestFindLiveVoxelWorkspace( world );
	ENSURE( contact != NULL );
	ENSURE( contact->voxelContact.states.count > 0 );
	int contactId = contact->contactId;
	int stateCount = contact->voxelContact.states.count;
	b3VoxelManifoldState savedStates[B3_VOXEL_MAX_CONTACTS];
	ENSURE( stateCount <= B3_VOXEL_MAX_CONTACTS );
	memcpy( savedStates, contact->voxelContact.states.data, (size_t)stateCount * sizeof( b3VoxelManifoldState ) );

	int imageSize = 0;
	uint8_t* image = b3World_SaveState( worldId, &imageSize );
	ENSURE( image != NULL && imageSize > 0 );
	b3RecHeader* recordHeader = (b3RecHeader*)image;
	ENSURE( recordHeader->snapshotSize > 16 );

	// Envelope and snapshot-version failures are rejected before mutating the
	// destination world. These pin the v3 compatibility boundary and the two
	// integer-overflow/truncation guards that protect workspace deserialization.
	b3WorldId rejectId = b3CreateWorld( &worldDef );
	uint32_t* snapshotVersion = (uint32_t*)( image + sizeof( b3RecHeader ) + sizeof( uint32_t ) );
	uint32_t savedVersion = *snapshotVersion;
	*snapshotVersion = savedVersion - 1;
	ENSURE( b3World_LoadState( rejectId, image, imageSize ) == false );
	*snapshotVersion = savedVersion;
	uint64_t savedSnapshotSize = recordHeader->snapshotSize;
	recordHeader->snapshotSize = UINT64_MAX;
	ENSURE( b3World_LoadState( rejectId, image, imageSize ) == false );
	recordHeader->snapshotSize = savedSnapshotSize;
	int truncatedSize = (int)sizeof( b3RecHeader ) + (int)savedSnapshotSize - 1;
	ENSURE( b3World_LoadState( rejectId, image, truncatedSize ) == false );
	b3DestroyWorld( rejectId );

	b3WorldId restoredId = b3CreateWorld( &worldDef );
	// Loading over a populated shell must release its prior objects and own the
	// restored workspace exactly once.
	b3BodyDef dummyDef = b3DefaultBodyDef();
	b3CreateBody( restoredId, &dummyDef );
	ENSURE( b3World_LoadState( restoredId, image, imageSize ) );
	b3FreeSaveState( image, imageSize );

	b3World* restored = b3GetWorldFromId( restoredId );
	ENSURE( contactId < restored->contacts.count );
	b3Contact* restoredContact = restored->contacts.data + contactId;
	ENSURE( restoredContact->voxelContact.states.count == stateCount );
	ENSURE( restoredContact->manifoldCount == stateCount );
	ENSURE( memcmp( restoredContact->voxelContact.states.data, savedStates,
				   (size_t)stateCount * sizeof( b3VoxelManifoldState ) ) == 0 );
	ENSURE( memcmp( restoredContact->manifolds, contact->manifolds, (size_t)stateCount * sizeof( b3Manifold ) ) == 0 );

	b3World_Step( worldId, timeStep, 4 );
	b3World_Step( restoredId, timeStep, 4 );
	b3Pos position = b3Body_GetPosition( bodyId );
	b3BodyId restoredBodyId = { bodyId.index1, (uint16_t)( restoredId.index1 - 1 ), bodyId.generation };
	b3Pos restoredPosition = b3Body_GetPosition( restoredBodyId );
	ENSURE( position.x == restoredPosition.x && position.y == restoredPosition.y && position.z == restoredPosition.z );
	b3Counters restoredCounters = b3World_GetCounters( restoredId );
	ENSURE( restoredCounters.voxel.workspaceKeyHits > 0 );
	ENSURE( restoredCounters.voxel.exactPatchPersistedPoints > 0 );

	// Corrupt only the persistent identity, not geometry. The next update must
	// miss rather than transferring impulses by a matching normal; the emitter
	// then rewrites the authoritative key and the following update hits again.
	contact = world->contacts.data + contactId;
	contact->voxelContact.states.data[0].key.patch0.lower.x += 17;
	b3World_Step( worldId, timeStep, 4 );
	b3Counters missCounters = b3World_GetCounters( worldId );
	ENSURE( missCounters.voxel.workspaceKeyMisses > 0 );
	b3World_Step( worldId, timeStep, 4 );
	b3Counters hitCounters = b3World_GetCounters( worldId );
	ENSURE( hitCounters.voxel.workspaceKeyHits > 0 );

	b3DestroyWorld( restoredId );
	b3DestroyWorld( worldId );
	b3DestroyVoxelData( ground );
	b3DestroyVoxelData( cube );
	return 0;
}

static int VoxelContactSpread( void )
{
	b3Vec3i c[25];
	int n = 0;
	for ( int x = 0; x < 5; ++x )
		for ( int z = 0; z < 5; ++z )
			c[n++] = (b3Vec3i){ x, 0, z };
	b3VoxelData* v0 = b3CreateVoxelData( c, 25, 1.0f );
	b3VoxelData* v1 = b3CreateVoxelData( c, 25, 1.0f );
	b3Transform x0 = { { 0.0f, 0.0f, 0.0f }, b3Quat_identity };
	b3Transform x1 = { { 0.0f, 0.95f, 0.0f }, b3Quat_identity }; // 0.05 overlap in y

	b3VoxelContact out[64];
	int cnt = b3VoxelCollide( v0, x0, v1, x1, 0.0f, 64, out );
	ENSURE( cnt >= 4 );

	float minX = FLT_MAX, maxX = -FLT_MAX, minZ = FLT_MAX, maxZ = -FLT_MAX;
	for ( int i = 0; i < cnt; ++i )
	{
		minX = b3MinFloat( minX, out[i].body0Point.x );
		maxX = b3MaxFloat( maxX, out[i].body0Point.x );
		minZ = b3MinFloat( minZ, out[i].body0Point.z );
		maxZ = b3MaxFloat( maxZ, out[i].body0Point.z );
	}
	ENSURE( maxX - minX > 4.5f );
	ENSURE( maxZ - minZ > 4.5f );

	b3DestroyVoxelData( v0 );
	b3DestroyVoxelData( v1 );
	return 0;
}

static uint32_t HashBytes( uint32_t h, const void* data, size_t bytes )
{
	const uint8_t* p = (const uint8_t*)data;
	for ( size_t i = 0; i < bytes; ++i )
	{
		h ^= p[i];
		h *= 16777619u; // FNV-1a
	}
	return h;
}

static uint32_t HashVoxelScene( int workerCount, int steps )
{
	b3WorldDef wd = b3DefaultWorldDef();
	wd.workerCount = workerCount;
	wd.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	b3WorldId world = b3CreateWorld( &wd );

	b3VoxelData* ground = MakeVoxelBox( -10, 10, 0, 0, -10, 10, 1.0f );
	b3BodyDef gd = b3DefaultBodyDef();
	b3BodyId gid = b3CreateBody( world, &gd );
	b3ShapeDef gsd = b3DefaultShapeDef();
	b3CreateVoxelShape( gid, &gsd, ground );

	enum
	{
		NB = 4 * 4 * 2
	};
	b3VoxelData* datas[NB];
	b3BodyId bodies[NB];
	int nb = 0;
	for ( int gx = 0; gx < 4; ++gx )
		for ( int gz = 0; gz < 4; ++gz )
			for ( int level = 0; level < 2; ++level )
			{
				b3VoxelData* cube = MakeVoxelBox( -1, 1, -1, 1, -1, 1, 1.0f );
				datas[nb] = cube;
				b3BodyDef bd = b3DefaultBodyDef();
				bd.type = b3_dynamicBody;
				bd.position = (b3Pos){ (float)( gx * 4 - 6 ), 2.0f + (float)level * 3.2f, (float)( gz * 4 - 6 ) };
				b3BodyId body = b3CreateBody( world, &bd );
				b3ShapeDef sd = b3DefaultShapeDef();
				sd.density = 1.0f;
				b3CreateVoxelShape( body, &sd, cube );
				bodies[nb] = body;
				nb++;
			}

	for ( int i = 0; i < steps; ++i )
		b3World_Step( world, 1.0f / 60.0f, 4 );

	uint32_t h = 2166136261u;
	for ( int i = 0; i < nb; ++i )
	{
		b3Pos p = b3Body_GetPosition( bodies[i] );
		b3Quat q = b3Body_GetRotation( bodies[i] );
		h = HashBytes( h, &p, sizeof( p ) );
		h = HashBytes( h, &q, sizeof( q ) );
	}

	b3DestroyWorld( world );
	for ( int i = 0; i < nb; ++i )
		b3DestroyVoxelData( datas[i] );
	b3DestroyVoxelData( ground );
	return h;
}

static int VoxelDeterminism( void )
{
	uint32_t h1 = HashVoxelScene( 1, 200 );
	uint32_t h1b = HashVoxelScene( 1, 200 );
	uint32_t h2 = HashVoxelScene( 2, 200 );
	uint32_t h4 = HashVoxelScene( 4, 200 );

	ENSURE( h1 == h1b ); // reproducible run-to-run
	ENSURE( h1 == h2 );	 // independent of worker count
	ENSURE( h1 == h4 );
	return 0;
}

static float DropConvexOnVoxel( int shapeKind )
{
	b3WorldDef wd = b3DefaultWorldDef();
	wd.gravity = (b3Vec3){ 0.0f, -10.0f, 0.0f };
	b3WorldId world = b3CreateWorld( &wd );

	b3VoxelData* ground = MakeVoxelBox( -5, 5, 0, 0, -5, 5, 1.0f ); // top face at y = 0.5
	b3BodyDef gd = b3DefaultBodyDef();
	b3BodyId gid = b3CreateBody( world, &gd );
	b3ShapeDef gsd = b3DefaultShapeDef();
	b3CreateVoxelShape( gid, &gsd, ground );

	b3BodyDef bd = b3DefaultBodyDef();
	bd.type = b3_dynamicBody;
	bd.position = (b3Pos){ 0.0f, 4.0f, 0.0f };
	b3BodyId body = b3CreateBody( world, &bd );
	b3ShapeDef sd = b3DefaultShapeDef();
	sd.density = 1.0f;

	if ( shapeKind == 0 )
	{
		b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.5f };
		b3CreateSphereShape( body, &sd, &sphere );
	}
	else if ( shapeKind == 1 )
	{
		b3BoxHull hull = b3MakeBoxHull( 0.5f, 0.5f, 0.5f );
		b3CreateHullShape( body, &sd, &hull.base );
	}
	else
	{
		b3Capsule cap = { { -0.5f, 0.0f, 0.0f }, { 0.5f, 0.0f, 0.0f }, 0.5f }; // lying flat
		b3CreateCapsuleShape( body, &sd, &cap );
	}

	for ( int i = 0; i < 240; ++i )
		b3World_Step( world, 1.0f / 60.0f, 4 );

	float y = (float)b3Body_GetPosition( body ).y;
	b3DestroyWorld( world );
	b3DestroyVoxelData( ground );
	return y;
}

static int VoxelConvexRest( void )
{
	for ( int kind = 0; kind < 3; ++kind )
	{
		float y = DropConvexOnVoxel( kind );
		ENSURE( y > 0.6f );				// did not tunnel through the voxel ground
		ENSURE_SMALL( y - 1.0f, 0.2f ); // came to rest on top
	}
	return 0;
}

static int VoxelConvexFastHitDiag( void )
{
	float speeds[] = { 5.0f, 20.0f, 50.0f, 100.0f, 200.0f };
	for ( int s = 0; s < 5; ++s )
	{
		b3WorldDef wd = b3DefaultWorldDef();
		wd.gravity = (b3Vec3){ 0.0f, 0.0f, 0.0f };
		b3WorldId world = b3CreateWorld( &wd );

		b3VoxelData* wall = MakeVoxelBox( 0, 2, -3, 3, -3, 3, 1.0f ); // 3 thick in x
		b3BodyDef wallDef = b3DefaultBodyDef();
		b3BodyId wallId = b3CreateBody( world, &wallDef );
		b3ShapeDef wsd = b3DefaultShapeDef();
		b3CreateVoxelShape( wallId, &wsd, wall );

		b3BodyDef bd = b3DefaultBodyDef();
		bd.type = b3_dynamicBody;
		bd.position = (b3Pos){ -8.0f, 0.0f, 0.0f };
		bd.linearVelocity = (b3Vec3){ speeds[s], 0.0f, 0.0f };
		b3BodyId body = b3CreateBody( world, &bd );
		b3ShapeDef sd = b3DefaultShapeDef();
		sd.density = 1.0f;
		b3Sphere sphere = { { 0.0f, 0.0f, 0.0f }, 0.5f };
		b3CreateSphereShape( body, &sd, &sphere );

		for ( int i = 0; i < 90; ++i )
			b3World_Step( world, 1.0f / 60.0f, 4 );

		b3Pos p = b3Body_GetPosition( body );
		b3Vec3 v = b3Body_GetLinearVelocity( body );
		printf( "  speed=%6.1f -> finalX=%10.3f  y=%8.3f z=%8.3f  |v|=%10.3f\n", speeds[s], (float)p.x, (float)p.y, (float)p.z,
				b3Length( v ) );
		float finalX = (float)p.x;

		b3DestroyWorld( world );
		b3DestroyVoxelData( wall );

		ENSURE( finalX < 3.0f );
	}
	return 0;
}

static float FastMovingVoxelHit( bool voxelTarget )
{
	b3WorldDef wd = b3DefaultWorldDef();
	wd.gravity = b3Vec3_zero;
	b3WorldId world = b3CreateWorld( &wd );

	b3VoxelData* wall = NULL;
	b3BodyDef wallDef = b3DefaultBodyDef();
	b3BodyId wallId = b3CreateBody( world, &wallDef );
	b3ShapeDef wallShapeDef = b3DefaultShapeDef();
	if ( voxelTarget )
	{
		wall = MakeVoxelBox( 0, 0, -4, 4, -4, 4, 1.0f );
		b3CreateVoxelShape( wallId, &wallShapeDef, wall );
	}
	else
	{
		b3BoxHull wallHull = b3MakeBoxHull( 0.5f, 4.5f, 4.5f );
		b3CreateHullShape( wallId, &wallShapeDef, &wallHull.base );
	}

	// An asymmetric aggregate and arbitrary starting rotation make a centroid
	// ray insufficient: CCD has to sweep the occupied cubes themselves.
	b3Vec3i movingCells[] = { { -1, -1, 0 }, { -1, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
	b3VoxelData* moving = b3CreateVoxelData( movingCells, ARRAY_COUNT( movingCells ), 0.75f );
	b3BodyDef bodyDef = b3DefaultBodyDef();
	bodyDef.type = b3_dynamicBody;
	bodyDef.position = (b3Pos){ -4.0f, 0.2f, 0.1f };
	bodyDef.rotation = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 0.7f, 0.2f } ), 0.47f );
	bodyDef.linearVelocity = (b3Vec3){ 400.0f, 0.0f, 0.0f };
	bodyDef.angularVelocity = (b3Vec3){ 3.0f, -5.0f, 2.0f };
	b3BodyId body = b3CreateBody( world, &bodyDef );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	shapeDef.density = 1.0f;
	b3CreateVoxelShape( body, &shapeDef, moving );

	b3World_Step( world, 1.0f / 60.0f, 4 );
	b3Counters counters = b3World_GetCounters( world );
	b3Pos position = b3Body_GetPosition( body );

	b3DestroyWorld( world );
	b3DestroyVoxelData( moving );
	if ( wall != NULL )
	{
		b3DestroyVoxelData( wall );
	}

	ENSURE( counters.voxel.ccdEncounters > 0 );
	return (float)position.x;
}

static int VoxelCellToiSupportDifferential( void )
{
	s_rng = 0x51A7C0DEu;
	for ( int iteration = 0; iteration < 4096; ++iteration )
	{
		b3Vec3 halfA = { 0.1f + 1.5f * Frand(), 0.1f + 1.5f * Frand(), 0.1f + 1.5f * Frand() };
		b3Transform localA = { { Frand2(), Frand2(), Frand2() }, RandQuat() };
		b3BoxHull hullA = b3MakeTransformedBoxHull( halfA.x, halfA.y, halfA.z, localA );

		b3Vec3 centerB = { 2.0f * Frand2(), 2.0f * Frand2(), 2.0f * Frand2() };
		float halfB = 0.1f + Frand();
		b3Vec3 cornersB[8];
		for ( int k = 0; k < 8; ++k )
		{
			cornersB[k] = (b3Vec3){ centerB.x + ( ( k & 1 ) ? halfB : -halfB ), centerB.y + ( ( k & 2 ) ? halfB : -halfB ),
									centerB.z + ( ( k & 4 ) ? halfB : -halfB ) };
		}

		b3Sweep sweepA = {
			.localCenter = { 0.25f * Frand2(), 0.25f * Frand2(), 0.25f * Frand2() },
			.c1 = { 3.0f * Frand2(), 3.0f * Frand2(), 3.0f * Frand2() },
			.q1 = RandQuat(),
			.q2 = RandQuat(),
		};
		sweepA.c2 = b3Add( sweepA.c1, (b3Vec3){ 2.0f * Frand2(), 2.0f * Frand2(), 2.0f * Frand2() } );
		b3Sweep sweepB = {
			.localCenter = { 0.25f * Frand2(), 0.25f * Frand2(), 0.25f * Frand2() },
			.c1 = { 3.0f * Frand2(), 3.0f * Frand2(), 3.0f * Frand2() },
			.q1 = RandQuat(),
			.q2 = RandQuat(),
		};
		sweepB.c2 = b3Add( sweepB.c1, (b3Vec3){ 2.0f * Frand2(), 2.0f * Frand2(), 2.0f * Frand2() } );
		b3TOIInput input = {
			.proxyA = { b3GetHullPoints( &hullA.base ), 8, 0.0f },
			.proxyB = { cornersB, 8, 0.0f },
			.sweepA = sweepA,
			.sweepB = sweepB,
			.maxFraction = 0.25f + 0.75f * Frand(),
		};

		b3TOIOutput generic = b3TimeOfImpact( &input );
		b3TOIOutput cell = b3TimeOfImpactCellB( &input );
		b3BoxProxySupport supportA = b3MakeBoxProxySupport( &input.proxyA );
		b3TOIOutput boxes = b3TimeOfImpactBoxes( &input, &supportA );
		ENSURE( cell.state == generic.state );
		ENSURE( cell.fraction == generic.fraction );
		ENSURE( cell.distance == generic.distance );
		ENSURE( cell.point.x == generic.point.x && cell.point.y == generic.point.y && cell.point.z == generic.point.z );
		ENSURE( cell.normal.x == generic.normal.x && cell.normal.y == generic.normal.y && cell.normal.z == generic.normal.z );
		ENSURE( cell.distanceIterations == generic.distanceIterations );
		ENSURE( cell.pushBackIterations == generic.pushBackIterations );
		ENSURE( cell.rootIterations == generic.rootIterations );
		ENSURE( cell.usedFallback == generic.usedFallback );
		ENSURE( boxes.state == generic.state );
		ENSURE( boxes.fraction == generic.fraction );
		ENSURE( boxes.distance == generic.distance );
		ENSURE( boxes.point.x == generic.point.x && boxes.point.y == generic.point.y && boxes.point.z == generic.point.z );
		ENSURE( boxes.normal.x == generic.normal.x && boxes.normal.y == generic.normal.y && boxes.normal.z == generic.normal.z );
		ENSURE( boxes.distanceIterations == generic.distanceIterations );
		ENSURE( boxes.pushBackIterations == generic.pushBackIterations );
		ENSURE( boxes.rootIterations == generic.rootIterations );
		ENSURE( boxes.usedFallback == generic.usedFallback );
	}
	return 0;
}

static int TestCompareAggregateToi( const TestVoxelSolid* moving, const b3Shape* target, const b3Sweep* targetSweep,
									const b3Sweep* movingSweep, float maxFraction, b3TOIOutput* aggregateOut )
{
	bool oracleFailed;
	b3TOIOutput oracle = TestGenericAggregateToi( moving, target, targetSweep, movingSweep, maxFraction, &oracleFailed );
	ENSURE( !oracleFailed );
	b3VoxelData* voxel = b3CreateOffsetVoxelData( moving->cells, moving->count, moving->size, moving->origin );
	b3TOIOutput aggregate = b3VoxelShapeTimeOfImpact( target, targetSweep, voxel, movingSweep, maxFraction, NULL );
	b3DestroyVoxelData( voxel );
	bool oracleImpact = TestToiIsImpact( oracle.state );
	bool aggregateImpact = TestToiIsImpact( aggregate.state );
	if ( aggregateImpact != oracleImpact )
	{
		printf( "aggregate TOI mismatch: target=%d oracle=(state=%d fraction=%.9g distance=%.9g) "
				"specialized=(state=%d fraction=%.9g distance=%.9g) maxFraction=%.9g\n",
				target->type, oracle.state, oracle.fraction, oracle.distance, aggregate.state, aggregate.fraction,
				aggregate.distance, maxFraction );
	}
	ENSURE( aggregateImpact == oracleImpact );
	if ( oracleImpact )
	{
		ENSURE( aggregate.state == oracle.state );
		float motion =
			TestProxySweepMotion( b3MakeShapeProxy( target ), targetSweep ) + TestVoxelSweepMotion( moving, movingSweep );
		float spatialError = b3AbsFloat( aggregate.fraction - oracle.fraction ) * b3MaxFloat( motion, 1.0f );
		float coordinateScale = 1.0f + b3MaxFloat( b3Length( movingSweep->c1 ), b3Length( targetSweep->c1 ) );
		float tolerance = TestVoxelCcdTolerance( coordinateScale );
		ENSURE( spatialError <= tolerance );
		s_geometryCoverage.ccdMaxSpatialError = b3MaxFloat( s_geometryCoverage.ccdMaxSpatialError, spatialError );
		s_geometryCoverage.ccdHits += 1;
	}
	else
	{
		s_geometryCoverage.ccdMisses += 1;
	}
	s_geometryCoverage.ccdCases += 1;
	*aggregateOut = aggregate;
	return 0;
}

static b3Sweep TestMakeSweep( b3Vec3 localCenter, b3Vec3 c1, b3Vec3 c2, b3Quat q1, b3Quat q2 )
{
	return (b3Sweep){ .localCenter = localCenter, .c1 = c1, .c2 = c2, .q1 = q1, .q2 = q2 };
}

static int TestCcdMotionCase( const TestVoxelSolid* moving, const b3Shape* target, b3Sweep targetSweep, b3Sweep movingSweep,
							  bool rotating, bool testClipping, bool expectedImpact )
{
	b3TOIOutput full;
	ENSURE( TestCompareAggregateToi( moving, target, &targetSweep, &movingSweep, 1.0f, &full ) == 0 );
	ENSURE( TestToiIsImpact( full.state ) == expectedImpact );
	if ( rotating )
		s_geometryCoverage.ccdRotating += 1;
	if ( !TestToiIsImpact( full.state ) || !testClipping || full.fraction <= 0.02f || full.fraction >= 0.9f )
		return 0;

	float before = 0.5f * full.fraction;
	b3TOIOutput clipped;
	ENSURE( TestCompareAggregateToi( moving, target, &targetSweep, &movingSweep, before, &clipped ) == 0 );
	ENSURE( !TestToiIsImpact( clipped.state ) );
	s_geometryCoverage.ccdClipped += 1;

	float after = b3MinFloat( 1.0f, full.fraction + 0.15f );
	b3TOIOutput extended;
	ENSURE( TestCompareAggregateToi( moving, target, &targetSweep, &movingSweep, after, &extended ) == 0 );
	ENSURE( TestToiIsImpact( extended.state ) );
	float motion =
		TestProxySweepMotion( b3MakeShapeProxy( target ), &targetSweep ) + TestVoxelSweepMotion( moving, &movingSweep );
	ENSURE( b3AbsFloat( extended.fraction - full.fraction ) * b3MaxFloat( motion, 1.0f ) <= 2.0f * B3_LINEAR_SLOP );
	s_geometryCoverage.ccdMonotonic += 1;
	return 0;
}

static int VoxelConvexCcdInitialOverlapPin( void )
{
	b3Vec3i cell = { 0, 0, 0 };
	TestVoxelSolid voxel = { &cell, 1, 1.0f, b3Vec3_zero };
	b3Sphere sphere = { b3Vec3_zero, 0.25f };
	b3Shape target = { .type = b3_sphereShape, .sphere = sphere };
	b3Sweep stationary = TestMakeSweep( b3Vec3_zero, b3Vec3_zero, b3Vec3_zero, b3Quat_identity, b3Quat_identity );
	b3TOIOutput output;
	ENSURE( TestCompareAggregateToi( &voxel, &target, &stationary, &stationary, 1.0f, &output ) == 0 );
	ENSURE( output.state == b3_toiStateOverlapped );
	ENSURE( output.fraction == 0.0f );
	return 0;
}

static int VoxelAggregateConvexCcdDifferential( void )
{
	b3Vec3i movingCells[] = { { -1, -1, 0 }, { -1, 0, 0 }, { 0, 0, 0 }, { 0, 1, 0 }, { 1, 1, 0 } };
	TestVoxelSolid moving = { movingCells, ARRAY_COUNT( movingCells ), 0.7f, { 0.23f, -0.17f, 0.31f } };

	b3Sphere sphere = { { 0.17f, -0.08f, 0.11f }, 0.38f };
	b3Capsule capsule = { { -0.35f, -0.25f, 0.0f }, { 0.35f, 0.25f, 0.0f }, 0.24f };
	b3BoxHull box = b3MakeOffsetBoxHull( 0.42f, 0.55f, 0.31f, (b3Vec3){ -0.13f, 0.09f, -0.07f } );
	b3Vec3 irregularPoints[] = {
		{ -0.52f, -0.31f, -0.22f }, { 0.61f, -0.24f, -0.18f }, { 0.13f, 0.67f, -0.11f },
		{ -0.29f, 0.08f, 0.58f },	{ 0.34f, 0.16f, 0.47f },
	};
	b3HullData* irregular = b3CreateHull( irregularPoints, ARRAY_COUNT( irregularPoints ), ARRAY_COUNT( irregularPoints ) );
	ENSURE( irregular != NULL );
	b3Shape targets[4] = {
		{ .type = b3_sphereShape, .sphere = sphere },
		{ .type = b3_capsuleShape, .capsule = capsule },
		{ .type = b3_hullShape, .hull = &box.base },
		{ .type = b3_hullShape, .hull = irregular },
	};

	b3Quat movingStart = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 0.7f, 0.2f } ), 0.41f );
	b3Quat movingEnd = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ -0.6f, 0.1f, 0.5f } ), 1.07f );
	b3Quat targetStart = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, -0.8f, 0.3f } ), -0.27f );
	b3Quat targetEnd = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.7f, 0.2f, -0.4f } ), 0.38f );
	for ( int targetIndex = 0; targetIndex < ARRAY_COUNT( targets ); ++targetIndex )
	{
		b3Sweep stationaryTarget =
			TestMakeSweep( (b3Vec3){ 0.11f, -0.07f, 0.05f }, b3Vec3_zero, b3Vec3_zero, b3Quat_identity, b3Quat_identity );
		b3Sweep translatingVoxel = TestMakeSweep( (b3Vec3){ -0.09f, 0.13f, -0.04f }, (b3Vec3){ -4.5f, 0.0f, 0.0f },
												  (b3Vec3){ 4.5f, 0.0f, 0.0f }, b3Quat_identity, b3Quat_identity );
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, stationaryTarget, translatingVoxel, false, true, true ) == 0 );

		b3Sweep movingTarget = TestMakeSweep( (b3Vec3){ 0.11f, -0.07f, 0.05f }, (b3Vec3){ 0.55f, -0.1f, 0.2f },
											  (b3Vec3){ -0.45f, 0.25f, -0.15f }, targetStart, targetEnd );
		b3Sweep rotatingVoxel = TestMakeSweep( (b3Vec3){ -0.09f, 0.13f, -0.04f }, (b3Vec3){ -4.2f, 0.35f, -0.2f },
											   (b3Vec3){ 4.0f, -0.15f, 0.25f }, movingStart, movingEnd );
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, movingTarget, rotatingVoxel, true, true, true ) == 0 );

		b3Sweep missVoxel = rotatingVoxel;
		missVoxel.c1.y += 6.0f;
		missVoxel.c2.y += 6.0f;
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, movingTarget, missVoxel, true, false, false ) == 0 );

		// Fraction-zero overlap and a zero-length separated sweep pin the two
		// stationary endpoints of the CCD contract for every target family.
		b3Sweep overlapTarget = TestMakeSweep( b3Vec3_zero, b3Vec3_zero, b3Vec3_zero, b3Quat_identity, b3Quat_identity );
		b3Sweep overlapVoxel = overlapTarget;
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, overlapTarget, overlapVoxel, false, false, true ) == 0 );
		b3Sweep staticMissVoxel = overlapVoxel;
		staticMissVoxel.c1 = (b3Vec3){ 7.0f, 6.0f, 5.0f };
		staticMissVoxel.c2 = staticMissVoxel.c1;
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, overlapTarget, staticMissVoxel, false, false, false ) == 0 );

		// Reverse candidate order, rotation, and common diagonal motion exercise
		// face-, edge-, and corner-first paths without assuming cell traversal order.
		b3Sweep reverseVoxel = TestMakeSweep( (b3Vec3){ -0.09f, 0.13f, -0.04f }, (b3Vec3){ 4.5f, 0.0f, 0.0f },
											  (b3Vec3){ -4.5f, 0.0f, 0.0f }, movingEnd, movingStart );
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, stationaryTarget, reverseVoxel, true, true, true ) == 0 );

		b3Sweep diagonalTarget = TestMakeSweep( (b3Vec3){ 0.11f, -0.07f, 0.05f }, (b3Vec3){ 0.35f, -0.2f, 0.15f },
												(b3Vec3){ -0.3f, 0.2f, -0.1f }, targetStart, targetEnd );
		b3Sweep diagonalVoxel = TestMakeSweep( (b3Vec3){ -0.09f, 0.13f, -0.04f }, (b3Vec3){ -4.0f, -4.0f, -3.5f },
											   (b3Vec3){ 4.0f, 4.0f, 3.5f }, movingStart, movingEnd );
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, diagonalTarget, diagonalVoxel, true, true, true ) == 0 );
		b3Sweep grazingMiss = diagonalVoxel;
		grazingMiss.c1.y += 6.5f;
		grazingMiss.c2.y += 6.5f;
		ENSURE( TestCcdMotionCase( &moving, targets + targetIndex, diagonalTarget, grazingMiss, false, false, false ) == 0 );
	}
	b3DestroyHull( irregular );
	printf( "  G3 coverage: cases=%d hits=%d misses=%d rotating=%d clipped=%d monotonic=%d max_spatial_error=%.9g\n",
			s_geometryCoverage.ccdCases, s_geometryCoverage.ccdHits, s_geometryCoverage.ccdMisses, s_geometryCoverage.ccdRotating,
			s_geometryCoverage.ccdClipped, s_geometryCoverage.ccdMonotonic, s_geometryCoverage.ccdMaxSpatialError );
	ENSURE( s_geometryCoverage.ccdCases >= 64 );
	ENSURE( s_geometryCoverage.ccdHits >= 24 );
	ENSURE( s_geometryCoverage.ccdMisses >= 24 );
	ENSURE( s_geometryCoverage.ccdRotating >= 16 );
	ENSURE( s_geometryCoverage.ccdClipped >= 16 );
	ENSURE( s_geometryCoverage.ccdMonotonic >= 16 );
	return 0;
}

static int TestCompareConservativeVoxelToi( const TestVoxelSolid* target, const b3Sweep* targetSweep,
											const TestVoxelSolid* moving, const b3Sweep* movingSweep, float maxFraction,
											bool expectedImpact )
{
	bool oracleFailed;
	b3TOIOutput oracle = TestGenericVoxelPairToi( target, targetSweep, moving, movingSweep, maxFraction, &oracleFailed );
	ENSURE( !oracleFailed );
	b3VoxelData* targetData = b3CreateOffsetVoxelData( target->cells, target->count, target->size, target->origin );
	b3VoxelData* movingData = b3CreateOffsetVoxelData( moving->cells, moving->count, moving->size, moving->origin );
	b3Shape targetShape = { .type = b3_voxelShape, .voxel = targetData };
	b3TOIOutput specialized = b3VoxelShapeTimeOfImpact( &targetShape, targetSweep, movingData, movingSweep, maxFraction, NULL );
	b3DestroyVoxelData( movingData );
	b3DestroyVoxelData( targetData );

	bool oracleImpact = TestToiIsImpact( oracle.state );
	bool specializedImpact = TestToiIsImpact( specialized.state );
	if ( specializedImpact != oracleImpact )
	{
		printf( "aggregate voxel TOI mismatch: oracle=(state=%d fraction=%.9g) specialized=(state=%d fraction=%.9g) "
				"maxFraction=%.9g targetCells=%d movingCells=%d\n",
				oracle.state, oracle.fraction, specialized.state, specialized.fraction, maxFraction, target->count,
				moving->count );
	}
	ENSURE( oracleImpact == expectedImpact );
	ENSURE( specializedImpact == oracleImpact );

	float motion = TestVoxelSweepMotion( target, targetSweep ) + TestVoxelSweepMotion( moving, movingSweep );
	float coordinateScale = 1.0f + b3MaxFloat( b3Length( movingSweep->c1 ), b3Length( targetSweep->c1 ) );
	float tolerance = TestVoxelCcdTolerance( coordinateScale );
	if ( oracleImpact )
	{
		if ( oracle.fraction == 0.0f )
		{
			ENSURE( specialized.fraction == 0.0f );
		}
		else
		{
			float spacing = 0.25f * b3MinFloat( target->size, moving->size );
			int sampleCount = b3MaxInt( 1, (int)ceilf( motion / b3MaxFloat( spacing, B3_LINEAR_SLOP ) ) );
			float sampleStep = maxFraction / (float)sampleCount;
			int upperIndex = b3ClampInt( (int)ceilf( specialized.fraction / sampleStep - 1.0e-5f ), 1, sampleCount );
			float lower = (float)( upperIndex - 1 ) * sampleStep;
			float upper = (float)upperIndex * sampleStep;
			float fractionTolerance = tolerance / b3MaxFloat( motion, 1.0f );
			ENSURE( lower - fractionTolerance <= oracle.fraction );
			ENSURE( oracle.fraction <= upper + fractionTolerance );
			float spatialWidth = ( upper - lower ) * motion;
			ENSURE( spatialWidth <= spacing + 2.0f * B3_LINEAR_SLOP );
			s_geometryCoverage.aggregateCcdWidestBracket =
				b3MaxFloat( s_geometryCoverage.aggregateCcdWidestBracket, spatialWidth );
		}
		s_geometryCoverage.aggregateCcdHits += 1;
	}
	else
	{
		s_geometryCoverage.aggregateCcdMisses += 1;
	}
	s_geometryCoverage.aggregateCcdCases += 1;
	return 0;
}

static b3Sweep TestTransformSweep( b3Sweep sweep, b3Transform transform )
{
	sweep.c1 = b3TransformPoint( transform, sweep.c1 );
	sweep.c2 = b3TransformPoint( transform, sweep.c2 );
	sweep.q1 = b3MulQuat( transform.q, sweep.q1 );
	sweep.q2 = b3MulQuat( transform.q, sweep.q2 );
	return sweep;
}

static int TestConservativeVoxelCcdCase( const TestVoxelSolid* target, b3Sweep targetSweep, const TestVoxelSolid* moving,
										 b3Sweep movingSweep, bool expectedImpact, bool rotating, bool testClipping,
										 bool testMetamorphisms )
{
	ENSURE( TestCompareConservativeVoxelToi( target, &targetSweep, moving, &movingSweep, 1.0f, expectedImpact ) == 0 );
	if ( rotating )
		s_geometryCoverage.aggregateCcdRotating += 1;

	bool failed;
	b3TOIOutput exact = TestGenericVoxelPairToi( target, &targetSweep, moving, &movingSweep, 1.0f, &failed );
	ENSURE( !failed );
	if ( testClipping && expectedImpact && exact.fraction > 0.02f && exact.fraction < 0.9f )
	{
		float before = 0.5f * exact.fraction;
		ENSURE( TestCompareConservativeVoxelToi( target, &targetSweep, moving, &movingSweep, before, false ) == 0 );
		float after = b3MinFloat( 1.0f, exact.fraction + 0.15f );
		ENSURE( TestCompareConservativeVoxelToi( target, &targetSweep, moving, &movingSweep, after, true ) == 0 );
		s_geometryCoverage.aggregateCcdClipped += 1;
	}

	if ( testMetamorphisms )
	{
		ENSURE( TestCompareConservativeVoxelToi( moving, &movingSweep, target, &targetSweep, 1.0f, expectedImpact ) == 0 );
		s_geometryCoverage.aggregateCcdSwaps += 1;
		b3Transform common = { { 2.3f, -1.7f, 0.6f },
							   b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, -0.7f, 0.4f } ), 0.47f ) };
		b3Sweep transformedTarget = TestTransformSweep( targetSweep, common );
		b3Sweep transformedMoving = TestTransformSweep( movingSweep, common );
		ENSURE( TestCompareConservativeVoxelToi( target, &transformedTarget, moving, &transformedMoving, 1.0f, expectedImpact ) ==
				0 );
		s_geometryCoverage.aggregateCcdCommonTransforms += 1;
	}
	return 0;
}

static int VoxelAggregateCcdContract( void )
{
	b3Vec3i oneCell = { 0, 0, 0 };
	b3Vec3i beamCells[] = { { -3, 0, 0 }, { -2, 0, 0 }, { -1, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 }, { 2, 0, 0 }, { 3, 0, 0 } };
	b3Vec3i wallCells[] = { { 0, -1, -1 }, { 0, -1, 0 }, { 0, -1, 1 }, { 0, 0, -1 }, { 0, 0, 0 },
							{ 0, 0, 1 },   { 0, 1, -1 }, { 0, 1, 0 },  { 0, 1, 1 } };
	TestVoxelSolid fine = { &oneCell, 1, 0.3f, { 0.11f, -0.07f, 0.05f } };
	TestVoxelSolid coarse = { &oneCell, 1, 0.8f, { -0.09f, 0.13f, -0.04f } };
	TestVoxelSolid beam = { beamCells, ARRAY_COUNT( beamCells ), 0.32f, { 0.08f, -0.12f, 0.03f } };
	TestVoxelSolid wall = { wallCells, ARRAY_COUNT( wallCells ), 0.26f, { -0.06f, 0.04f, -0.02f } };
	b3Sweep stationary = TestMakeSweep( b3Vec3_zero, b3Vec3_zero, b3Vec3_zero, b3Quat_identity, b3Quat_identity );

	// Initial overlap and zero-length separation pin fraction zero and the no-motion path.
	ENSURE( TestConservativeVoxelCcdCase( &fine, stationary, &coarse, stationary, true, false, false, true ) == 0 );
	b3Sweep staticMiss = stationary;
	staticMiss.c1 = (b3Vec3){ 4.0f, 3.0f, -2.0f };
	staticMiss.c2 = staticMiss.c1;
	ENSURE( TestConservativeVoxelCcdCase( &fine, stationary, &coarse, staticMiss, false, false, false, true ) == 0 );

	// A small cell crosses a thin wall in one frame; a nearby parallel path is a decisive miss.
	b3Sweep fast = TestMakeSweep( b3Vec3_zero, (b3Vec3){ -50.0f, 0.0f, 0.0f }, (b3Vec3){ 50.0f, 0.0f, 0.0f }, b3Quat_identity,
								  b3Quat_identity );
	ENSURE( TestConservativeVoxelCcdCase( &wall, stationary, &fine, fast, true, false, true, true ) == 0 );
	b3Sweep fastMiss = fast;
	fastMiss.c1.y += 2.0f;
	fastMiss.c2.y += 2.0f;
	ENSURE( TestConservativeVoxelCcdCase( &wall, stationary, &fine, fastMiss, false, false, false, true ) == 0 );

	// Unequal cells approach a face near its edge. The second path has a clear
	// gap, keeping true tangency out of the decisive corpus.
	b3Sweep grazingHit = TestMakeSweep( b3Vec3_zero, (b3Vec3){ -4.0f, 0.45f, 0.0f }, (b3Vec3){ 4.0f, 0.45f, 0.0f },
										b3Quat_identity, b3Quat_identity );
	ENSURE( TestConservativeVoxelCcdCase( &coarse, stationary, &fine, grazingHit, true, false, true, true ) == 0 );
	b3Sweep grazingMiss = grazingHit;
	grazingMiss.c1.y = 0.8f;
	grazingMiss.c2.y = 0.8f;
	ENSURE( TestConservativeVoxelCcdCase( &coarse, stationary, &fine, grazingMiss, false, false, false, true ) == 0 );

	// Long aggregates rotate and translate simultaneously on both bodies.
	b3Quat q0 = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.3f, 0.8f, -0.2f } ), -0.45f );
	b3Quat q1 = b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ -0.6f, 0.2f, 0.7f } ), 0.82f );
	b3Sweep movingWall = TestMakeSweep( (b3Vec3){ 0.07f, -0.05f, 0.03f }, (b3Vec3){ 0.4f, -0.2f, 0.1f },
										(b3Vec3){ -0.35f, 0.25f, -0.1f }, q0, q1 );
	b3Sweep movingBeam = TestMakeSweep( (b3Vec3){ -0.04f, 0.09f, -0.02f }, (b3Vec3){ -4.0f, -0.4f, 0.2f },
										(b3Vec3){ 4.0f, 0.3f, -0.15f }, q1, q0 );
	ENSURE( TestConservativeVoxelCcdCase( &wall, movingWall, &beam, movingBeam, true, true, true, true ) == 0 );
	b3Sweep rotatingMiss = movingBeam;
	rotatingMiss.c1.y += 4.0f;
	rotatingMiss.c2.y += 4.0f;
	ENSURE( TestConservativeVoxelCcdCase( &wall, movingWall, &beam, rotatingMiss, false, true, false, true ) == 0 );

	printf( "  aggregate CCD coverage: cases=%d hits=%d misses=%d rotating=%d clipped=%d swaps=%d common=%d "
			"widest_bracket=%.9g\n",
			s_geometryCoverage.aggregateCcdCases, s_geometryCoverage.aggregateCcdHits, s_geometryCoverage.aggregateCcdMisses,
			s_geometryCoverage.aggregateCcdRotating, s_geometryCoverage.aggregateCcdClipped, s_geometryCoverage.aggregateCcdSwaps,
			s_geometryCoverage.aggregateCcdCommonTransforms, s_geometryCoverage.aggregateCcdWidestBracket );
	ENSURE( s_geometryCoverage.aggregateCcdCases >= 28 );
	ENSURE( s_geometryCoverage.aggregateCcdHits >= 12 );
	ENSURE( s_geometryCoverage.aggregateCcdMisses >= 12 );
	ENSURE( s_geometryCoverage.aggregateCcdRotating >= 2 );
	ENSURE( s_geometryCoverage.aggregateCcdClipped >= 3 );
	ENSURE( s_geometryCoverage.aggregateCcdSwaps >= 8 );
	ENSURE( s_geometryCoverage.aggregateCcdCommonTransforms >= 8 );
	return 0;
}

static int TestAggregateTargetCcdSmoke( const b3Shape* target, b3Sweep targetSweep, const TestVoxelSolid* moving,
										b3Sweep movingSweep, bool expectedImpact )
{
	b3VoxelData* movingData = b3CreateOffsetVoxelData( moving->cells, moving->count, moving->size, moving->origin );
	b3TOIOutput output = b3VoxelShapeTimeOfImpact( target, &targetSweep, movingData, &movingSweep, 1.0f, NULL );
	b3DestroyVoxelData( movingData );
	ENSURE( TestToiIsImpact( output.state ) == expectedImpact );
	s_geometryCoverage.aggregateSmokeCases += 1;
	if ( expectedImpact )
		s_geometryCoverage.aggregateSmokeHits += 1;
	else
		s_geometryCoverage.aggregateSmokeMisses += 1;
	return 0;
}

static int VoxelAggregateTargetCcdSmoke( void )
{
	b3Vec3i cell = { 0, 0, 0 };
	TestVoxelSolid moving = { &cell, 1, 0.3f, { 0.04f, -0.03f, 0.02f } };
	b3Sweep stationary = TestMakeSweep( b3Vec3_zero, b3Vec3_zero, b3Vec3_zero, b3Quat_identity, b3Quat_identity );

	b3MeshData* meshData = b3CreateBoxMesh( b3Vec3_zero, (b3Vec3){ 0.6f, 0.6f, 0.6f }, true );
	ENSURE( meshData != NULL );
	b3Shape mesh = { .type = b3_meshShape, .mesh = { meshData, b3Vec3_one } };
	b3Sweep meshHit = TestMakeSweep( b3Vec3_zero, (b3Vec3){ -3.0f, 0.0f, 0.0f }, (b3Vec3){ 3.0f, 0.0f, 0.0f }, b3Quat_identity,
									 b3Quat_identity );
	ENSURE( TestAggregateTargetCcdSmoke( &mesh, stationary, &moving, meshHit, true ) == 0 );
	meshHit.c1.y = 3.0f;
	meshHit.c2.y = 3.0f;
	ENSURE( TestAggregateTargetCcdSmoke( &mesh, stationary, &moving, meshHit, false ) == 0 );
	b3DestroyMesh( meshData );
	s_geometryCoverage.aggregateSmokeTargetKinds += 1;

	b3HeightFieldData* heightField = b3CreateGrid( 4, 4, b3Vec3_one, false );
	ENSURE( heightField != NULL );
	b3Shape height = { .type = b3_heightShape, .heightField = heightField };
	b3Sweep heightHit = TestMakeSweep( b3Vec3_zero, (b3Vec3){ 1.5f, 3.0f, 1.5f }, (b3Vec3){ 1.5f, -3.0f, 1.5f }, b3Quat_identity,
									   b3Quat_identity );
	ENSURE( TestAggregateTargetCcdSmoke( &height, stationary, &moving, heightHit, true ) == 0 );
	heightHit.c1.x = -2.0f;
	heightHit.c2.x = -2.0f;
	ENSURE( TestAggregateTargetCcdSmoke( &height, stationary, &moving, heightHit, false ) == 0 );
	b3DestroyHeightField( heightField );
	s_geometryCoverage.aggregateSmokeTargetKinds += 1;

	b3SurfaceMaterial material = b3DefaultSurfaceMaterial();
	b3CompoundSphereDef sphere = { .sphere = { b3Vec3_zero, 0.6f }, .material = material };
	b3CompoundDef compoundDef = { .spheres = &sphere, .sphereCount = 1 };
	b3CompoundData* compoundData = b3CreateCompound( &compoundDef );
	ENSURE( compoundData != NULL );
	b3Shape compound = { .type = b3_compoundShape, .compound = compoundData };
	b3Sweep compoundHit = TestMakeSweep( b3Vec3_zero, (b3Vec3){ 0.0f, -3.0f, 0.0f }, (b3Vec3){ 0.0f, 3.0f, 0.0f },
										 b3Quat_identity, b3Quat_identity );
	ENSURE( TestAggregateTargetCcdSmoke( &compound, stationary, &moving, compoundHit, true ) == 0 );
	compoundHit.c1.z = 3.0f;
	compoundHit.c2.z = 3.0f;
	ENSURE( TestAggregateTargetCcdSmoke( &compound, stationary, &moving, compoundHit, false ) == 0 );
	b3DestroyCompound( compoundData );
	s_geometryCoverage.aggregateSmokeTargetKinds += 1;

	printf( "  aggregate target smoke: kinds=%d cases=%d hits=%d misses=%d\n", s_geometryCoverage.aggregateSmokeTargetKinds,
			s_geometryCoverage.aggregateSmokeCases, s_geometryCoverage.aggregateSmokeHits,
			s_geometryCoverage.aggregateSmokeMisses );
	ENSURE( s_geometryCoverage.aggregateSmokeTargetKinds >= 3 );
	ENSURE( s_geometryCoverage.aggregateSmokeCases >= 6 );
	ENSURE( s_geometryCoverage.aggregateSmokeHits >= 3 );
	ENSURE( s_geometryCoverage.aggregateSmokeMisses >= 3 );
	return 0;
}

static int VoxelGeometryCoverageGate( void )
{
	printf( "  voxel geometry coverage: G0 pins=%d witnesses=%d; G2 topologies=%d directions=%d overlap=%d separation=%d "
			"unequal_width=%d; "
			"G3 cases=%d hits=%d misses=%d rotating=%d clipped=%d monotonic=%d\n",
			s_geometryCoverage.analyticPins, s_geometryCoverage.boundaryWitnesses, s_geometryCoverage.discreteTopologies,
			s_geometryCoverage.discreteDirections, s_geometryCoverage.discreteOverlaps, s_geometryCoverage.discreteSeparations,
			s_geometryCoverage.discreteUnequalWidths, s_geometryCoverage.ccdCases, s_geometryCoverage.ccdHits,
			s_geometryCoverage.ccdMisses, s_geometryCoverage.ccdRotating, s_geometryCoverage.ccdClipped,
			s_geometryCoverage.ccdMonotonic );
	ENSURE( s_geometryCoverage.analyticPins >= 6 );
	ENSURE( s_geometryCoverage.boundaryWitnesses >= 40 );
	ENSURE( s_geometryCoverage.discreteTopologies >= 16 );
	ENSURE( s_geometryCoverage.discreteDirections >= 100 );
	ENSURE( s_geometryCoverage.discreteOverlaps >= 66 );
	ENSURE( s_geometryCoverage.discreteSeparations >= 34 );
	ENSURE( s_geometryCoverage.discreteUnequalWidths >= 4 );
	ENSURE( s_geometryCoverage.convexTargetKinds >= 6 );
	ENSURE( s_geometryCoverage.convexCases >= 60 );
	ENSURE( s_geometryCoverage.metamorphicSwaps >= 4 );
	ENSURE( s_geometryCoverage.metamorphicCommonTransforms >= 4 );
	ENSURE( s_geometryCoverage.metamorphicOrigins >= 4 );
	ENSURE( s_geometryCoverage.metamorphicInsertionOrders >= 4 );
	ENSURE( s_geometryCoverage.metamorphicBuriedCells >= 1 );
	ENSURE( s_geometryCoverage.metamorphicRepeats >= 4 );
	ENSURE( s_geometryCoverage.ccdCases >= 64 );
	ENSURE( s_geometryCoverage.ccdHits >= 24 );
	ENSURE( s_geometryCoverage.ccdMisses >= 24 );
	ENSURE( s_geometryCoverage.ccdRotating >= 16 );
	ENSURE( s_geometryCoverage.ccdClipped >= 16 );
	ENSURE( s_geometryCoverage.ccdMonotonic >= 16 );
	return 0;
}

static void MakeBitCornerBoxProxy( b3Vec3 center, b3Vec3 half, b3Quat rotation, b3Vec3 points[8] )
{
	for ( int i = 0; i < 8; ++i )
	{
		b3Vec3 corner = {
			( i & 1 ) != 0 ? half.x : -half.x,
			( i & 2 ) != 0 ? half.y : -half.y,
			( i & 4 ) != 0 ? half.z : -half.z,
		};
		points[i] = b3Add( center, b3RotateVector( rotation, corner ) );
	}
}

static int CompareVoxelShapeCastOutputs( const b3CastOutput* reference, const b3CastOutput* optimized )
{
	ENSURE( optimized->hit == reference->hit );
	ENSURE( optimized->fraction == reference->fraction );
	ENSURE( optimized->iterations == reference->iterations );
	ENSURE( optimized->triangleIndex == reference->triangleIndex );
	ENSURE( optimized->childIndex == reference->childIndex );
	ENSURE( optimized->materialIndex == reference->materialIndex );
	ENSURE( optimized->point.x == reference->point.x && optimized->point.y == reference->point.y &&
			optimized->point.z == reference->point.z );
	ENSURE( optimized->normal.x == reference->normal.x && optimized->normal.y == reference->normal.y &&
			optimized->normal.z == reference->normal.z );
	return 0;
}

static int VoxelShapeCastCorridorDifferential( void )
{
	enum
	{
		boxSamples = 20000,
		sphereSamples = 10000,
	};
	b3Vec3i cells[51];
	int cellCount = 0;
	for ( int y = -3; y <= 3; ++y )
	{
		for ( int z = -3; z <= 3; ++z )
			cells[cellCount++] = (b3Vec3i){ 0, y, z };
	}
	cells[cellCount++] = (b3Vec3i){ 2, -2, 1 };
	cells[cellCount++] = (b3Vec3i){ -2, 2, -1 };
	ENSURE( cellCount == ARRAY_COUNT( cells ) );
	b3VoxelData* voxel = b3CreateOffsetVoxelData( cells, cellCount, 0.35f, (b3Vec3){ 0.07f, -0.11f, 0.05f } );
	ENSURE( voxel != NULL );

	int hits = 0;
	int misses = 0;
	s_rng = UINT32_C( 0xB05CA57 );
	for ( int sample = 0; sample < boxSamples; ++sample )
	{
		b3Vec3 center = { 3.5f * Frand2(), 2.5f * Frand2(), 2.5f * Frand2() };
		b3Vec3 half = { 0.06f + 0.5f * Frand(), 0.06f + 0.5f * Frand(), 0.06f + 0.5f * Frand() };
		b3Quat rotation = RandQuat();
		b3Vec3 points[8];
		MakeBitCornerBoxProxy( center, half, rotation, points );
		b3ShapeCastInput input = {
			.proxy = { points, 8, 0.0f },
			.translation = { 5.0f * Frand2(), 4.0f * Frand2(), 4.0f * Frand2() },
			.maxFraction = sample % 3 == 0 ? 0.2f : ( sample % 3 == 1 ? 0.65f : 1.0f ),
			.canEncroach = sample % 11 == 0,
		};
		b3CastOutput reference = b3ShapeCastVoxelReference( voxel, &input );
		b3CastOutput optimized = b3ShapeCastVoxel( voxel, &input );
		ENSURE( CompareVoxelShapeCastOutputs( &reference, &optimized ) == 0 );
		if ( reference.hit )
			hits += 1;
		else
			misses += 1;
	}

	// Pin a deep initial overlap and a nearly grazing, nearly parallel sweep.
	b3Vec3 points[8];
	MakeBitCornerBoxProxy( (b3Vec3){ 0.07f, -0.11f, 0.05f }, (b3Vec3){ 0.2f, 0.17f, 0.13f }, RandQuat(), points );
	b3ShapeCastInput overlap = {
		.proxy = { points, 8, 0.0f },
		.translation = { 2.0f, 0.01f, -0.02f },
		.maxFraction = 1.0f,
	};
	b3CastOutput reference = b3ShapeCastVoxelReference( voxel, &overlap );
	b3CastOutput optimized = b3ShapeCastVoxel( voxel, &overlap );
	ENSURE( reference.hit && reference.fraction == 0.0f );
	ENSURE( CompareVoxelShapeCastOutputs( &reference, &optimized ) == 0 );
	hits += 1;

	MakeBitCornerBoxProxy( (b3Vec3){ -2.0f, 1.22f, 1.21f }, (b3Vec3){ 0.13f, 0.11f, 0.09f },
					   b3MakeQuatFromAxisAngle( b3Normalize( (b3Vec3){ 0.2f, 0.9f, -0.3f } ), 0.017f ), points );
	b3ShapeCastInput grazing = {
		.proxy = { points, 8, 0.0f },
		.translation = { 4.0f, -0.003f, 0.002f },
		.maxFraction = 0.9f,
	};
	reference = b3ShapeCastVoxelReference( voxel, &grazing );
	optimized = b3ShapeCastVoxel( voxel, &grazing );
	ENSURE( CompareVoxelShapeCastOutputs( &reference, &optimized ) == 0 );
	if ( reference.hit )
		hits += 1;
	else
		misses += 1;

	for ( int sample = 0; sample < sphereSamples; ++sample )
	{
		b3Vec3 center = { 3.5f * Frand2(), 2.5f * Frand2(), 2.5f * Frand2() };
		b3ShapeCastInput sphere = {
			.proxy = { &center, 1, 0.04f + 0.35f * Frand() },
			.translation = { 5.0f * Frand2(), 4.0f * Frand2(), 4.0f * Frand2() },
			.maxFraction = sample % 2 == 0 ? 0.4f : 1.0f,
			.canEncroach = sample % 11 == 0,
		};
		reference = b3ShapeCastVoxelReference( voxel, &sphere );
		optimized = b3ShapeCastVoxel( voxel, &sphere );
		ENSURE( CompareVoxelShapeCastOutputs( &reference, &optimized ) == 0 );
		if ( reference.hit )
			hits += 1;
		else
			misses += 1;
	}

	printf( "  voxel shape-cast corridor differential: samples=%d hits=%d misses=%d\n", boxSamples + sphereSamples + 2,
			hits, misses );
	ENSURE( hits >= 100 );
	ENSURE( misses >= 100 );
	b3DestroyVoxelData( voxel );
	return 0;
}

static int VoxelMovingCcd( void )
{
	ENSURE( FastMovingVoxelHit( true ) < 1.0f );
	ENSURE( FastMovingVoxelHit( false ) < 1.0f );
	return 0;
}

int VoxelCollideTest( void )
{
	RUN_SUBTEST( VoxelCanonicalPatchIdentity );
	memset( &s_geometryCoverage, 0, sizeof( s_geometryCoverage ) );
	RUN_SUBTEST( VoxelUnionBoundaryPins );
	RUN_SUBTEST( VoxelVoxelInternalAxisFallbackPin );
	RUN_SUBTEST( VoxelConvexCcdInitialOverlapPin );
	RUN_SUBTEST( VoxelAggregateConvexCcdDifferential );
	RUN_SUBTEST( VoxelAggregateCcdContract );
	RUN_SUBTEST( VoxelAggregateTargetCcdSmoke );
	RUN_SUBTEST( VoxelAggregateDiscreteDifferential );
	RUN_SUBTEST( VoxelConvexWitnessPins );
	RUN_SUBTEST( VoxelConvexInternalSeamWitnessPin );
	RUN_SUBTEST( VoxelConvexDeepSphereWitnessPin );
	RUN_SUBTEST( VoxelConvexEmbeddedEscapePin );
	RUN_SUBTEST( VoxelVoxelEmbeddedEscapePin );
	RUN_SUBTEST( VoxelBoxInternalSeamWitnessPin );
	RUN_SUBTEST( VoxelAggregateConvexDiscreteDifferential );
	RUN_SUBTEST( VoxelAggregateMetamorphic );
	RUN_SUBTEST( VoxelGeometryCoverageGate );
	RUN_SUBTEST( VoxelShapeCastCorridorDifferential );
	RUN_SUBTEST( VoxelCellToiSupportDifferential );
	RUN_SUBTEST( VoxelMovingCcd );
	RUN_SUBTEST( VoxelConvexFastHitDiag );
	RUN_SUBTEST( VoxelConvexRest );
	RUN_SUBTEST( VoxelObbKnownCases );
	RUN_SUBTEST( VoxelObbFuzz );
	RUN_SUBTEST( VoxelAabbFuzz );
	RUN_SUBTEST( VoxelDriverBasic );
	RUN_SUBTEST( VoxelConvexCanonicalDispatchThreshold );
	RUN_SUBTEST( VoxelCanonicalPatchBudget );
	RUN_SUBTEST( VoxelExposureMask );
	RUN_SUBTEST( VoxelNegativeChunkCoordinates );
	RUN_SUBTEST( VoxelHullPublicManifold );
	RUN_SUBTEST( VoxelDriverSlab );
	RUN_SUBTEST( VoxelDriverOracle );
	RUN_SUBTEST( VoxelContactSpread );
	RUN_SUBTEST( VoxelDeterminism );
	RUN_SUBTEST( VoxelBodyRest );
	RUN_SUBTEST( VoxelPatchWorkspacePersistence );
	return 0;
}
