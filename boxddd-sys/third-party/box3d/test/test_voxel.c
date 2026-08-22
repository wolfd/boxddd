// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#include "box3d/box3d.h"

#include "core.h"
#include "voxel_shape.h" // internal query layer

#include "test_macros.h"

#include <math.h>
#include <limits.h>

static int VoxelBuildAndPointQuery( void )
{
	b3Vec3i cells[] = {
		{ 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 5, 5, 5 }, { -1, -1, -1 }, { 16, 0, 0 },
	};
	b3VoxelData* v = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 1.0f );
	ENSURE( v != NULL );
	ENSURE( b3VoxelData_GetCellCount( v ) == 6 );
	ENSURE( b3VoxelData_GetVoxelSize( v ) == 1.0f );

	ENSURE( b3VoxelData_IsSolid( v, ( b3Vec3i ){ 0, 0, 0 } ) );
	ENSURE( b3VoxelData_IsSolid( v, ( b3Vec3i ){ 1, 0, 0 } ) );
	ENSURE( b3VoxelData_IsSolid( v, ( b3Vec3i ){ -1, -1, -1 } ) ); // negative chunk
	ENSURE( b3VoxelData_IsSolid( v, ( b3Vec3i ){ 16, 0, 0 } ) );   // chunk 1
	ENSURE( b3VoxelData_IsSolid( v, ( b3Vec3i ){ 2, 0, 0 } ) == false );
	ENSURE( b3VoxelData_IsSolid( v, ( b3Vec3i ){ 15, 0, 0 } ) == false ); // in chunk 0 but empty

	b3AABB b = b3VoxelData_GetBounds( v );
	ENSURE_SMALL( b.lowerBound.x - ( -1.5f ), 1e-6f );
	ENSURE_SMALL( b.lowerBound.y - ( -1.5f ), 1e-6f );
	ENSURE_SMALL( b.lowerBound.z - ( -1.5f ), 1e-6f );
	ENSURE_SMALL( b.upperBound.x - ( 16.5f ), 1e-6f );
	ENSURE_SMALL( b.upperBound.y - ( 5.5f ), 1e-6f );
	ENSURE_SMALL( b.upperBound.z - ( 5.5f ), 1e-6f );

	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelDedup( void )
{
	b3Vec3i cells[] = { { 1, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 }, { 0, 0, 0 }, { 1, 0, 0 } };
	b3VoxelData* v = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 1.0f );
	ENSURE( v != NULL );
	ENSURE( b3VoxelData_GetCellCount( v ) == 2 );
	b3Vec3i canonical[2];
	ENSURE( b3VoxelData_GetCells( v, canonical, 2 ) == 2 );
	ENSURE( canonical[0].x == 0 && canonical[1].x == 1 );
	uint64_t hash = b3Voxel_GetHash( v );

	b3Vec3i reversed[] = { { 0, 0, 0 }, { 1, 0, 0 } };
	b3VoxelData* v2 = b3CreateVoxelData( reversed, ARRAY_COUNT( reversed ), 1.0f );
	ENSURE( v2 != NULL );
	ENSURE( b3Voxel_GetHash( v2 ) == hash );

	int byteCount = 0;
	uint8_t* bytes = b3Voxel_Serialize( v, &byteCount );
	ENSURE( bytes != NULL && byteCount > 0 );
	b3VoxelData* restored = b3Voxel_Deserialize( bytes, byteCount );
	ENSURE( restored != NULL );
	ENSURE( b3Voxel_GetHash( restored ) == hash );
	b3Free( bytes, (size_t)byteCount );
	b3DestroyVoxelData( restored );
	b3DestroyVoxelData( v2 );
	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelValidation( void )
{
	b3Vec3i cell = { 0, 0, 0 };
	ENSURE( b3CreateVoxelData( NULL, 1, 1.0f ) == NULL );
	ENSURE( b3CreateVoxelData( &cell, 0, 1.0f ) == NULL );
	ENSURE( b3CreateVoxelData( &cell, 1, 0.0f ) == NULL );
	ENSURE( b3CreateVoxelData( &cell, 1, NAN ) == NULL );
	b3Vec3i outside = { INT_MAX, 0, 0 };
	ENSURE( b3CreateVoxelData( &outside, 1, 1.0f ) == NULL );
	return 0;
}

static int VoxelOffsetRoundTrip( void )
{
	b3Vec3i cell = { 0, 0, 0 };
	b3Vec3 origin = { 0.125f, 0.125f, 0.125f };
	b3VoxelData* v = b3CreateOffsetVoxelData( &cell, 1, 0.25f, origin );
	ENSURE( v != NULL );
	b3Vec3 gotOrigin = b3VoxelData_GetOrigin( v );
	ENSURE_SMALL( gotOrigin.x - origin.x, 1e-6f );
	ENSURE_SMALL( gotOrigin.y - origin.y, 1e-6f );
	ENSURE_SMALL( gotOrigin.z - origin.z, 1e-6f );
	b3AABB bounds = b3VoxelData_GetBounds( v );
	ENSURE_SMALL( b3Length( bounds.lowerBound ), 1e-6f );
	ENSURE_SMALL( bounds.upperBound.x - 0.25f, 1e-6f );
	ENSURE_SMALL( bounds.upperBound.y - 0.25f, 1e-6f );
	ENSURE_SMALL( bounds.upperBound.z - 0.25f, 1e-6f );

	int byteCount = 0;
	uint8_t* bytes = b3Voxel_Serialize( v, &byteCount );
	b3VoxelData* restored = b3Voxel_Deserialize( bytes, byteCount );
	ENSURE( restored != NULL );
	ENSURE( b3Voxel_GetHash( restored ) == b3Voxel_GetHash( v ) );
	gotOrigin = b3VoxelData_GetOrigin( restored );
	ENSURE_SMALL( gotOrigin.x - origin.x, 1e-6f );
	ENSURE_SMALL( gotOrigin.y - origin.y, 1e-6f );
	ENSURE_SMALL( gotOrigin.z - origin.z, 1e-6f );
	b3Free( bytes, (size_t)byteCount );
	b3DestroyVoxelData( restored );
	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelIntegerDomain( void )
{
	b3Vec3i cells[] = {
		{ 7, -5, 11 }, { -3, 9, 4 }, { 7, -5, 11 }, { 2, 1, -8 },
	};
	b3Vec3 origin = { 12.5f, -7.25f, 0.125f };
	b3VoxelData* v = b3CreateOffsetVoxelData( cells, ARRAY_COUNT( cells ), 0.25f, origin );
	ENSURE( v != NULL );
	b3Vec3i lower;
	b3Vec3i upper;
	ENSURE( b3Voxel_GetDomain( v, &lower, &upper ) );
	ENSURE( lower.x == -3 && lower.y == -5 && lower.z == -8 );
	ENSURE( upper.x == 7 && upper.y == 9 && upper.z == 11 );

	int byteCount = 0;
	uint8_t* bytes = b3Voxel_Serialize( v, &byteCount );
	ENSURE( bytes != NULL );
	b3VoxelData* restored = b3Voxel_Deserialize( bytes, byteCount );
	ENSURE( restored != NULL );
	b3Vec3i restoredLower;
	b3Vec3i restoredUpper;
	ENSURE( b3Voxel_GetDomain( restored, &restoredLower, &restoredUpper ) );
	ENSURE( restoredLower.x == lower.x && restoredLower.y == lower.y && restoredLower.z == lower.z );
	ENSURE( restoredUpper.x == upper.x && restoredUpper.y == upper.y && restoredUpper.z == upper.z );
	b3Free( bytes, (size_t)byteCount );
	b3DestroyVoxelData( restored );
	b3DestroyVoxelData( v );

	// The representable grid extremes still leave one integer for each
	// canonical unbounded-direction sentinel.
	b3Vec3i extremes[] = { { -16777200, 0, 0 }, { 16777215, 0, 0 } };
	v = b3CreateVoxelData( extremes, ARRAY_COUNT( extremes ), 1.0f );
	ENSURE( v != NULL );
	ENSURE( b3Voxel_GetDomain( v, &lower, &upper ) );
	ENSURE( lower.x == -16777200 && upper.x == 16777215 );
	ENSURE( lower.x > INT_MIN && upper.x < INT_MAX );
	b3DestroyVoxelData( v );
	return 0;
}

static bool QueryHas( const b3VoxelData* v, b3AABB q, b3Vec3i want )
{
	b3Vec3i out[64];
	int n = b3Voxel_QueryCells( v, q, out, 64 );
	for ( int i = 0; i < n; ++i )
		if ( out[i].x == want.x && out[i].y == want.y && out[i].z == want.z )
			return true;
	return false;
}

static int VoxelRangeQuery( void )
{
	b3Vec3i cells[] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } };
	b3VoxelData* v = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 1.0f );
	b3Vec3i out[64];

	b3AABB q0 = { { -0.4f, -0.4f, -0.4f }, { 0.4f, 0.4f, 0.4f } };
	ENSURE( b3Voxel_QueryCells( v, q0, out, 64 ) == 1 );
	ENSURE( QueryHas( v, q0, ( b3Vec3i ){ 0, 0, 0 } ) );

	b3AABB q1 = { { -0.4f, -0.4f, -0.4f }, { 1.4f, 0.4f, 0.4f } };
	ENSURE( b3Voxel_QueryCells( v, q1, out, 64 ) == 2 );
	ENSURE( QueryHas( v, q1, ( b3Vec3i ){ 0, 0, 0 } ) );
	ENSURE( QueryHas( v, q1, ( b3Vec3i ){ 1, 0, 0 } ) );

	b3AABB qall = { { -10, -10, -10 }, { 10, 10, 10 } };
	ENSURE( b3Voxel_QueryCells( v, qall, out, 64 ) == 3 );

	b3AABB qfar = { { 100, 100, 100 }, { 101, 101, 101 } };
	ENSURE( b3Voxel_QueryCells( v, qfar, out, 64 ) == 0 );

	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelTrackedRangeQuery( void )
{
	b3Vec3i cells[] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } };
	b3VoxelData* v = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 1.0f );
	b3Vec3i out[2];
	b3VoxelCounters counters = { 0 };
	b3AABB query = { { -0.4f, -0.4f, -0.4f }, { 1.4f, 0.4f, 0.4f } };

	ENSURE( b3Voxel_QueryCellsTracked( v, query, out, ARRAY_COUNT( out ), &counters ) == 2 );
	ENSURE( counters.queryCalls == 1 );
	ENSURE( counters.chunksVisited == 1 );
	ENSURE( counters.occupiedEntriesScanned == 2 );
	ENSURE( counters.cellsReturned == 2 );
	ENSURE( counters.voxelVoxelCalls == 0 );
	ENSURE( counters.voxelConvexCalls == 0 );

	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelFaceBoundary( void )
{
	b3Vec3i cells[] = { { 0, 0, 0 } };
	b3VoxelData* v = b3CreateVoxelData( cells, 1, 1.0f );
	b3Vec3i out[8];

	b3AABB inside = { { 0.49f, -0.1f, -0.1f }, { 0.6f, 0.1f, 0.1f } };
	ENSURE( b3Voxel_QueryCells( v, inside, out, 8 ) == 1 );

	b3AABB touch = { { 0.5f, -0.1f, -0.1f }, { 0.6f, 0.1f, 0.1f } };
	ENSURE( b3Voxel_QueryCells( v, touch, out, 8 ) == 1 );

	b3AABB past = { { 0.51f, -0.1f, -0.1f }, { 0.6f, 0.1f, 0.1f } };
	ENSURE( b3Voxel_QueryCells( v, past, out, 8 ) == 0 );

	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelScaledSize( void )
{
	b3Vec3i cells[] = { { 0, 0, 0 }, { 1, 0, 0 } };
	b3VoxelData* v = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 2.0f );
	b3Vec3i out[8];

	b3AABB b = b3VoxelData_GetBounds( v );
	ENSURE_SMALL( b.lowerBound.x - ( -1.0f ), 1e-6f ); // cell 0 min = -1
	ENSURE_SMALL( b.upperBound.x - ( 3.0f ), 1e-6f );  // cell 1 max = 3

	b3AABB q = { { 1.5f, -0.5f, -0.5f }, { 2.5f, 0.5f, 0.5f } };
	ENSURE( b3Voxel_QueryCells( v, q, out, 8 ) == 1 );
	ENSURE( out[0].x == 1 );

	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelUnboundedQueries( void )
{
	b3Vec3i cells[300];
	for ( int i = 0; i < ARRAY_COUNT( cells ); ++i )
		cells[i] = (b3Vec3i){ i, 0, 0 };
	b3VoxelData* v = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 1.0f );
	b3AABB all = b3VoxelData_GetBounds( v );
	b3Vec3i small[4];
	ENSURE( b3Voxel_QueryCells( v, all, small, ARRAY_COUNT( small ) ) == ARRAY_COUNT( cells ) );
	b3DestroyVoxelData( v );

	b3Vec3i distant[] = { { 0, 0, 0 }, { 5000, 0, 0 } };
	v = b3CreateVoxelData( distant, ARRAY_COUNT( distant ), 1.0f );
	b3RayCastInput input = { { 1.0f, 0.0f, 0.0f }, { 5000.0f, 0.0f, 0.0f }, 1.0f };
	b3CastOutput hit = b3RayCastVoxel( v, &input );
	ENSURE( hit.hit );
	ENSURE( hit.fraction > 0.9f );
	b3DestroyVoxelData( v );
	return 0;
}

static int VoxelSurfaceArea( void )
{
	b3Vec3i cells[] = { { 0, 0, 0 }, { 1, 0, 0 } };
	b3VoxelData* v = b3CreateVoxelData( cells, ARRAY_COUNT( cells ), 2.0f );
	ENSURE_SMALL( b3Voxel_ComputeSurfaceArea( v ) - 40.0f, 1e-6f );
	ENSURE_SMALL( b3Voxel_ComputeProjectedArea( v, (b3Vec3){ 1.0f, 0.0f, 0.0f } ) - 4.0f, 1e-6f );
	ENSURE_SMALL( b3Voxel_ComputeProjectedArea( v, (b3Vec3){ 0.0f, 1.0f, 0.0f } ) - 8.0f, 1e-6f );
	b3DestroyVoxelData( v );
	return 0;
}

typedef struct VoxelMoverResult
{
	int count;
	b3PlaneResult first;
} VoxelMoverResult;

static bool VoxelMoverCallback( b3ShapeId shapeId, const b3PlaneResult* planes, int count, void* context )
{
	B3_UNUSED( shapeId );
	VoxelMoverResult* result = context;
	if ( result->count == 0 && count > 0 )
	{
		result->first = planes[0];
	}
	result->count += count;
	return true;
}

static int VoxelMoverQuery( void )
{
	b3WorldDef worldDef = b3DefaultWorldDef();
	worldDef.gravity = b3Vec3_zero;
	b3WorldId worldId = b3CreateWorld( &worldDef );
	b3BodyDef bodyDef = b3DefaultBodyDef();
	b3BodyId bodyId = b3CreateBody( worldId, &bodyDef );
	b3Vec3i cell = { 0, 0, 0 };
	b3VoxelData* voxels = b3CreateVoxelData( &cell, 1, 1.0f );
	ENSURE( voxels != NULL );
	b3ShapeDef shapeDef = b3DefaultShapeDef();
	b3CreateVoxelShape( bodyId, &shapeDef, voxels );

	// A player-style vertical capsule overlaps the cell's negative-X face.
	b3Capsule mover = { { -0.6f, -0.2f, 0.0f }, { -0.6f, 0.2f, 0.0f }, 0.2f };
	VoxelMoverResult result = { 0 };
	b3World_CollideMover( worldId, b3Pos_zero, &mover, b3DefaultQueryFilter(), VoxelMoverCallback, &result );
	ENSURE( result.count == 1 );
	ENSURE( result.first.plane.normal.x < -0.99f );
	ENSURE_SMALL( result.first.point.x + 0.5f, 1.0e-5f );
	ENSURE( result.first.plane.offset > 0.09f && result.first.plane.offset < 0.11f );

	// The same query outside the radius produces no plane.
	mover.center1.x = -0.8f;
	mover.center2.x = -0.8f;
	result = (VoxelMoverResult){ 0 };
	b3World_CollideMover( worldId, b3Pos_zero, &mover, b3DefaultQueryFilter(), VoxelMoverCallback, &result );
	ENSURE( result.count == 0 );

	b3DestroyWorld( worldId );
	b3DestroyVoxelData( voxels );
	return 0;
}

int VoxelTest( void )
{
	RUN_SUBTEST( VoxelBuildAndPointQuery );
	RUN_SUBTEST( VoxelDedup );
	RUN_SUBTEST( VoxelValidation );
	RUN_SUBTEST( VoxelOffsetRoundTrip );
	RUN_SUBTEST( VoxelIntegerDomain );
	RUN_SUBTEST( VoxelRangeQuery );
	RUN_SUBTEST( VoxelTrackedRangeQuery );
	RUN_SUBTEST( VoxelFaceBoundary );
	RUN_SUBTEST( VoxelScaledSize );
	RUN_SUBTEST( VoxelUnboundedQueries );
	RUN_SUBTEST( VoxelSurfaceArea );
	RUN_SUBTEST( VoxelMoverQuery );
	return 0;
}
