// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#pragma once

#include "shape.h"

typedef struct TestVoxelSolid
{
	const b3Vec3i* cells;
	int count;
	float size;
	b3Vec3 origin;
} TestVoxelSolid;

bool TestCellEqual( b3Vec3i a, b3Vec3i b );
bool TestSolidContainsCell( const TestVoxelSolid* solid, b3Vec3i cell );
b3Vec3 TestCellCenter( const TestVoxelSolid* solid, b3Vec3i cell );
void TestMakeCellCorners( const TestVoxelSolid* solid, b3Vec3i cell, b3Vec3 corners[8] );

float TestVoxelBoundaryTolerance( void );
float TestVoxelCcdTolerance( float coordinateScale );
bool TestPointOnUnionBoundary( const TestVoxelSolid* solid, b3Transform transform, b3Vec3 worldPoint, float tolerance );

bool TestGenericUnionOverlap( const TestVoxelSolid* solid0, b3Transform transform0, const TestVoxelSolid* solid1,
							  b3Transform transform1 );
bool TestGenericVoxelConvexOverlap( const TestVoxelSolid* solid, const b3Shape* convex, b3Transform transformConvexToVoxel );
bool TestToiIsImpact( b3TOIState state );
b3TOIOutput TestGenericAggregateToi( const TestVoxelSolid* moving, const b3Shape* target, const b3Sweep* targetSweep,
									 const b3Sweep* movingSweep, float maxFraction, bool* failed );
b3TOIOutput TestGenericVoxelPairToi( const TestVoxelSolid* target, const b3Sweep* targetSweep, const TestVoxelSolid* moving,
									 const b3Sweep* movingSweep, float maxFraction, bool* failed );
float TestProxySweepMotion( b3ShapeProxy proxy, const b3Sweep* sweep );
float TestVoxelSweepMotion( const TestVoxelSolid* solid, const b3Sweep* sweep );
