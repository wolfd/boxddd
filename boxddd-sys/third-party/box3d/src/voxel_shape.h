// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#pragma once

#include "box3d/math_functions.h"
#include "box3d/types.h"
#include "box3d/voxel.h"

#include <stdbool.h>
#include <stdint.h>

float b3Voxel_GetVoxelSize( const b3VoxelData* v );
b3Vec3 b3Voxel_GetOrigin( const b3VoxelData* v );
b3Vec3 b3Voxel_GetCellCenter( const b3VoxelData* v, b3Vec3i cell );
int b3Voxel_GetCellCount( const b3VoxelData* v );
uint64_t b3Voxel_GetHash( const b3VoxelData* v );
bool b3Voxel_IsDirectionExposed( const b3VoxelData* v, b3Vec3i cell, b3Vec3 direction );
static inline bool b3Voxel_ExposureMaskContains( uint8_t mask, b3Vec3 direction )
{
	const float epsilon = 1.0e-4f;
	bool considered = false;
	for ( int axis = 0; axis < 3; ++axis )
	{
		float component = ( &direction.x )[axis];
		if ( b3AbsFloat( component ) <= epsilon )
			continue;
		considered = true;
		int bit = 2 * axis + ( component > 0.0f );
		if ( ( mask & ( 1u << bit ) ) != 0 )
			return true;
	}
	return !considered;
}

// Canonical pointer-free representation used by recording and world snapshots.
// The returned bytes are allocated with b3Alloc and owned by the caller.
uint8_t* b3Voxel_Serialize( const b3VoxelData* v, int* byteCount );
b3VoxelData* b3Voxel_Deserialize( const uint8_t* bytes, int byteCount );

bool b3Voxel_GetLocalBounds( const b3VoxelData* v, b3AABB* out );
bool b3Voxel_GetDomain( const b3VoxelData* v, b3Vec3i* lower, b3Vec3i* upper );

int b3Voxel_QueryCells( const b3VoxelData* v, b3AABB queryLocal, b3Vec3i* out, int cap );
int b3Voxel_QueryCellsTracked( const b3VoxelData* v, b3AABB queryLocal, b3Vec3i* out, int cap, b3VoxelCounters* counters );
// Return false to stop enumeration after the current occupied cell.
typedef bool b3VoxelQueryCallback( b3Vec3i cell, uint8_t exposed, void* context );
int b3Voxel_ForEachCellTracked( const b3VoxelData* v, b3AABB queryLocal, b3VoxelQueryCallback* callback, void* context,
								b3VoxelCounters* counters );

int b3Voxel_GetCells( const b3VoxelData* v, b3Vec3i* out, int cap );

b3MassData b3Voxel_ComputeMass( const b3VoxelData* v, float density );
float b3Voxel_ComputeSurfaceArea( const b3VoxelData* v );
float b3Voxel_ComputeProjectedArea( const b3VoxelData* v, b3Vec3 planeNormal );

b3CastOutput b3RayCastVoxel( const b3VoxelData* v, const b3RayCastInput* input );
b3CastOutput b3ShapeCastVoxel( const b3VoxelData* v, const b3ShapeCastInput* input );
// Unfiltered per-cell oracle for native differential tests. Production queries
// use b3ShapeCastVoxel's conservative time-coherent AABB corridor first.
b3CastOutput b3ShapeCastVoxelReference( const b3VoxelData* v, const b3ShapeCastInput* input );

bool b3OverlapVoxel( const b3VoxelData* v, b3Transform xf, const b3ShapeProxy* proxy );
