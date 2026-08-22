// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#pragma once

#include "base.h"
#include "id.h"
#include "math_functions.h"

#include <stdbool.h>

typedef struct b3VoxelData b3VoxelData;

B3_API b3VoxelData* b3CreateVoxelData( const b3Vec3i* cells, int count, float voxelSize );

/// Create voxel data whose integer cell centers are translated by localOrigin.
B3_API b3VoxelData* b3CreateOffsetVoxelData( const b3Vec3i* cells, int count, float voxelSize, b3Vec3 localOrigin );

B3_API void b3DestroyVoxelData( b3VoxelData* voxels );

B3_API int b3VoxelData_GetCellCount( const b3VoxelData* voxels );

B3_API float b3VoxelData_GetVoxelSize( const b3VoxelData* voxels );

B3_API b3Vec3 b3VoxelData_GetOrigin( const b3VoxelData* voxels );

B3_API b3AABB b3VoxelData_GetBounds( const b3VoxelData* voxels );

B3_API bool b3VoxelData_IsSolid( const b3VoxelData* voxels, b3Vec3i cell );

B3_API int b3VoxelData_GetCells( const b3VoxelData* voxels, b3Vec3i* cells, int capacity );

/**@}*/
