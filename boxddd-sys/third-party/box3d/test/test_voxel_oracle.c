// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#include "test_voxel_oracle.h"

#include "math_internal.h"

#include <float.h>
#include <math.h>

bool TestCellEqual( b3Vec3i a, b3Vec3i b )
{
	return a.x == b.x && a.y == b.y && a.z == b.z;
}

// Deliberately linear and independent of b3VoxelData's chunks, hash table,
// cached exposure masks, and traversal order. These fixtures are tiny.
bool TestSolidContainsCell( const TestVoxelSolid* solid, b3Vec3i cell )
{
	for ( int i = 0; i < solid->count; ++i )
	{
		if ( TestCellEqual( solid->cells[i], cell ) )
		{
			return true;
		}
	}
	return false;
}

b3Vec3 TestCellCenter( const TestVoxelSolid* solid, b3Vec3i cell )
{
	return b3Add( solid->origin,
				  (b3Vec3){ solid->size * (float)cell.x, solid->size * (float)cell.y, solid->size * (float)cell.z } );
}

void TestMakeCellCorners( const TestVoxelSolid* solid, b3Vec3i cell, b3Vec3 corners[8] )
{
	b3Vec3 center = TestCellCenter( solid, cell );
	float half = 0.5f * solid->size;
	for ( int k = 0; k < 8; ++k )
	{
		corners[k] = (b3Vec3){ center.x + ( ( k & 1 ) ? half : -half ), center.y + ( ( k & 2 ) ? half : -half ),
							   center.z + ( ( k & 4 ) ? half : -half ) };
	}
}

float TestVoxelBoundaryTolerance( void )
{
	return B3_LINEAR_SLOP + 64.0f * FLT_EPSILON;
}

float TestVoxelCcdTolerance( float coordinateScale )
{
	return 2.0f * B3_LINEAR_SLOP + 128.0f * FLT_EPSILON * coordinateScale;
}

bool TestPointOnUnionBoundary( const TestVoxelSolid* solid, b3Transform transform, b3Vec3 worldPoint, float tolerance )
{
	b3Vec3 point = b3InvTransformPoint( transform, worldPoint );
	float half = 0.5f * solid->size;
	static const int directions[6][3] = {
		{ -1, 0, 0 }, { 1, 0, 0 }, { 0, -1, 0 }, { 0, 1, 0 }, { 0, 0, -1 }, { 0, 0, 1 },
	};
	for ( int i = 0; i < solid->count; ++i )
	{
		b3Vec3i cell = solid->cells[i];
		b3Vec3 center = TestCellCenter( solid, cell );
		for ( int face = 0; face < 6; ++face )
		{
			int axis = face >> 1;
			float side = ( face & 1 ) != 0 ? half : -half;
			if ( b3AbsFloat( ( &point.x )[axis] - ( &center.x )[axis] - side ) > tolerance )
			{
				continue;
			}
			int tangent0 = ( axis + 1 ) % 3;
			int tangent1 = ( axis + 2 ) % 3;
			if ( b3AbsFloat( ( &point.x )[tangent0] - ( &center.x )[tangent0] ) > half + tolerance ||
				 b3AbsFloat( ( &point.x )[tangent1] - ( &center.x )[tangent1] ) > half + tolerance )
			{
				continue;
			}
			b3Vec3i neighbor = { cell.x + directions[face][0], cell.y + directions[face][1], cell.z + directions[face][2] };
			if ( !TestSolidContainsCell( solid, neighbor ) )
			{
				return true;
			}
		}
	}
	return false;
}

// Exhaustive generic-hull reference. This does not call the voxel OBB kernel,
// voxel chunk query, exposure cache, traversal selector, or contact reducer.
bool TestGenericUnionOverlap( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
							  b3Transform transform1 )
{
	b3Transform transform1To0 = b3InvMulTransforms( transform0, transform1 );
	float half0 = 0.5f * solid0->size;
	float half1 = 0.5f * solid1->size;
	for ( int i = 0; i < solid0->count; ++i )
	{
		b3BoxHull hull0 = b3MakeOffsetBoxHull( half0, half0, half0, TestCellCenter( solid0, solid0->cells[i] ) );
		for ( int j = 0; j < solid1->count; ++j )
		{
			b3BoxHull hull1 = b3MakeOffsetBoxHull( half1, half1, half1, TestCellCenter( solid1, solid1->cells[j] ) );
			b3LocalManifoldPoint points[8];
			b3LocalManifold manifold = { .points = points };
			b3SATCache cache = { 0 };
			b3CollideHulls( &manifold, 8, &hull0.base, &hull1.base, transform1To0, &cache );
			for ( int k = 0; k < manifold.pointCount; ++k )
			{
				// G2 places decisive fixtures well outside the speculative band.
				if ( manifold.points[k].separation < -1.0e-4f )
				{
					return true;
				}
			}
		}
	}
	return false;
}

bool TestGenericVoxelConvexOverlap( const TestVoxelSolid* solid, const b3Shape* convex, b3Transform transformConvexToVoxel )
{
	float half = 0.5f * solid->size;
	for ( int i = 0; i < solid->count; ++i )
	{
		b3BoxHull cell = b3MakeOffsetBoxHull( half, half, half, TestCellCenter( solid, solid->cells[i] ) );
		b3LocalManifoldPoint points[8];
		b3LocalManifold manifold = { .points = points };
		switch ( convex->type )
		{
			case b3_sphereShape:
			{
				b3SimplexCache cache = { 0 };
				b3CollideHullAndSphere( &manifold, 8, &cell.base, &convex->sphere, transformConvexToVoxel, &cache );
				break;
			}
			case b3_capsuleShape:
			{
				b3SimplexCache cache = { 0 };
				b3CollideHullAndCapsule( &manifold, 8, &cell.base, &convex->capsule, transformConvexToVoxel, &cache );
				break;
			}
			case b3_hullShape:
			{
				b3SATCache cache = { 0 };
				b3CollideHulls( &manifold, 8, &cell.base, convex->hull, transformConvexToVoxel, &cache );
				break;
			}
			default:
				return false;
		}
		for ( int j = 0; j < manifold.pointCount; ++j )
		{
			if ( manifold.points[j].separation < -1.0e-4f )
			{
				return true;
			}
		}
	}
	return false;
}

bool TestToiIsImpact( b3TOIState state )
{
	return state == b3_toiStateHit || state == b3_toiStateOverlapped;
}

b3TOIOutput TestGenericAggregateToi( const TestVoxelSolid* moving, const b3Shape* target, const b3Sweep* targetSweep,
									 const b3Sweep* movingSweep, float maxFraction, bool* failed )
{
	b3TOIOutput best = { 0 };
	best.state = b3_toiStateSeparated;
	best.fraction = maxFraction;
	*failed = false;
	b3ShapeProxy targetProxy = b3MakeShapeProxy( target );
	for ( int i = 0; i < moving->count; ++i )
	{
		b3Vec3 corners[8];
		TestMakeCellCorners( moving, moving->cells[i], corners );
		b3TOIInput input = {
			.proxyA = targetProxy,
			.proxyB = { corners, 8, 0.0f },
			.sweepA = *targetSweep,
			.sweepB = *movingSweep,
			.maxFraction = maxFraction,
		};
		b3TOIOutput candidate = b3TimeOfImpact( &input );
		if ( candidate.state == b3_toiStateFailed || candidate.state == b3_toiStateUnknown )
		{
			*failed = true;
			continue;
		}
		if ( TestToiIsImpact( candidate.state ) &&
			 ( !TestToiIsImpact( best.state ) || candidate.fraction < best.fraction ||
			   ( candidate.fraction == best.fraction && candidate.state == b3_toiStateOverlapped ) ) )
		{
			best = candidate;
		}
	}
	return best;
}

b3TOIOutput TestGenericVoxelPairToi( const TestVoxelSolid* target, const b3Sweep* targetSweep, const TestVoxelSolid* moving,
									 const b3Sweep* movingSweep, float maxFraction, bool* failed )
{
	b3TOIOutput best = { 0 };
	best.state = b3_toiStateSeparated;
	best.fraction = maxFraction;
	*failed = false;
	for ( int i = 0; i < target->count; ++i )
	{
		b3Vec3 targetCorners[8];
		TestMakeCellCorners( target, target->cells[i], targetCorners );
		for ( int j = 0; j < moving->count; ++j )
		{
			b3Vec3 movingCorners[8];
			TestMakeCellCorners( moving, moving->cells[j], movingCorners );
			b3TOIInput input = {
				.proxyA = { targetCorners, 8, 0.0f },
				.proxyB = { movingCorners, 8, 0.0f },
				.sweepA = *targetSweep,
				.sweepB = *movingSweep,
				.maxFraction = maxFraction,
			};
			b3TOIOutput candidate = b3TimeOfImpact( &input );
			if ( candidate.state == b3_toiStateFailed || candidate.state == b3_toiStateUnknown )
			{
				*failed = true;
				continue;
			}
			if ( TestToiIsImpact( candidate.state ) &&
				 ( !TestToiIsImpact( best.state ) || candidate.fraction < best.fraction ||
				   ( candidate.fraction == best.fraction && candidate.state == b3_toiStateOverlapped ) ) )
			{
				best = candidate;
			}
		}
	}
	return best;
}

static float TestQuatAngle( b3Quat q1, b3Quat q2 )
{
	float cosine = b3ClampFloat( b3AbsFloat( b3DotQuat( q1, q2 ) ), 0.0f, 1.0f );
	return 2.0f * acosf( cosine );
}

float TestProxySweepMotion( b3ShapeProxy proxy, const b3Sweep* sweep )
{
	float radius = proxy.radius;
	for ( int i = 0; i < proxy.count; ++i )
	{
		radius = b3MaxFloat( radius, b3Length( b3Sub( proxy.points[i], sweep->localCenter ) ) + proxy.radius );
	}
	return b3Length( b3Sub( sweep->c2, sweep->c1 ) ) + radius * TestQuatAngle( sweep->q1, sweep->q2 );
}

float TestVoxelSweepMotion( const TestVoxelSolid* solid, const b3Sweep* sweep )
{
	float radius = 0.0f;
	for ( int i = 0; i < solid->count; ++i )
	{
		b3Vec3 corners[8];
		TestMakeCellCorners( solid, solid->cells[i], corners );
		for ( int k = 0; k < 8; ++k )
		{
			radius = b3MaxFloat( radius, b3Length( b3Sub( corners[k], sweep->localCenter ) ) );
		}
	}
	return b3Length( b3Sub( sweep->c2, sweep->c1 ) ) + radius * TestQuatAngle( sweep->q1, sweep->q2 );
}
