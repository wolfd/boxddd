// SPDX-FileCopyrightText: 2026 Tribulla
// SPDX-FileCopyrightText: 2026 Danny Wolf
// SPDX-License-Identifier: MIT

#include "voxel_shape.h"

#include "core.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define B3_VOXEL_CHUNK_BITS 4
#define B3_VOXEL_CHUNK_SIZE 16	  // 1 << B3_VOXEL_CHUNK_BITS
#define B3_VOXEL_CHUNK_MASK 15	  // B3_VOXEL_CHUNK_SIZE - 1
#define B3_VOXEL_CHUNK_CELLS 4096 // 16^3
#define B3_VOXEL_CHUNK_COORD_LIMIT ( ( 1 << 20 ) - 1 )
#define B3_VOXEL_DIRECT_QUERY_CELLS 512
#define B3_VOXEL_EXPOSED_MASK 0x3Fu
#define B3_VOXEL_OCCUPIED_BIT 0x80u

typedef struct b3VoxelChunk
{
	// The high bit stores occupancy; the low six bits store local-axis exposure.
	uint8_t exposed[B3_VOXEL_CHUNK_CELLS];
	uint16_t* occupied; // packed 12-bit solid local-cell indices
	int occupiedCount;
	int occupiedCapacity;
	b3Vec3i position;
	b3AABB solidBounds; // shape-local world-unit bounds of this chunk's solid cells
} b3VoxelChunk;

typedef struct b3VoxelChunkSlot
{
	uint64_t key; // 0 = empty (real keys are never 0, see b3Voxel_packChunk)
	int index;
} b3VoxelChunkSlot;

struct b3VoxelData
{
	float voxelSize;
	float invVoxelSize;
	b3Vec3 origin;
	int cellCount; // total solid cells
	int cellCapacity;
	uint64_t hash;
	b3Vec3i* cells; // canonical lexicographic order, duplicates removed

	b3AABB localBounds; // shape-local world-unit bounds of all solid cells
	b3Vec3i domainMin; // inclusive occupied-cell domain, independent of local origin
	b3Vec3i domainMax;
	bool hasBounds;

	b3VoxelChunk* chunks; // dense array of chunks
	int chunkCount;
	int chunkCapacity;
	b3VoxelChunkSlot* slots; // hash: chunkKey -> chunks index
	int slotCap;			 // power of two
};

#define B3_VOXEL_BLOB_MAGIC 0x32584F56u // "VOX2" in little-endian

typedef struct b3VoxelBlobHeader
{
	uint32_t magic;
	uint32_t cellCount;
	float voxelSize;
	b3Vec3 origin;
} b3VoxelBlobHeader;

_Static_assert( sizeof( b3VoxelBlobHeader ) == 24, "voxel blob header must be 24 bytes" );

static inline b3Vec3 b3Voxel_cellCenterFromGrid( b3Vec3 origin, float voxelSize, b3Vec3i cell )
{
	return b3Add( origin, (b3Vec3){ cell.x * voxelSize, cell.y * voxelSize, cell.z * voxelSize } );
}

static int b3Voxel_compareCells( const void* a, const void* b )
{
	const b3Vec3i* ca = a;
	const b3Vec3i* cb = b;
	if ( ca->x != cb->x )
		return ca->x < cb->x ? -1 : 1;
	if ( ca->y != cb->y )
		return ca->y < cb->y ? -1 : 1;
	if ( ca->z != cb->z )
		return ca->z < cb->z ? -1 : 1;
	return 0;
}

static inline b3Vec3i b3Voxel_chunkOf( b3Vec3i cell )
{
	return (b3Vec3i){ cell.x >> B3_VOXEL_CHUNK_BITS, cell.y >> B3_VOXEL_CHUNK_BITS, cell.z >> B3_VOXEL_CHUNK_BITS };
}

static bool b3Voxel_cellIsRepresentable( b3Vec3i cell )
{
	b3Vec3i chunk = b3Voxel_chunkOf( cell );
	return -B3_VOXEL_CHUNK_COORD_LIMIT <= chunk.x && chunk.x <= B3_VOXEL_CHUNK_COORD_LIMIT &&
		   -B3_VOXEL_CHUNK_COORD_LIMIT <= chunk.y && chunk.y <= B3_VOXEL_CHUNK_COORD_LIMIT &&
		   -B3_VOXEL_CHUNK_COORD_LIMIT <= chunk.z && chunk.z <= B3_VOXEL_CHUNK_COORD_LIMIT;
}

static inline int b3Voxel_localIndex( int lx, int ly, int lz )
{
	return ( lx << 8 ) | ( ly << 4 ) | lz;
}

static uint64_t b3Voxel_packChunk( b3Vec3i c )
{
	const int64_t OFF = 1 << 20;
	uint64_t x = (uint64_t)( (int64_t)c.x + OFF );
	uint64_t y = (uint64_t)( (int64_t)c.y + OFF );
	uint64_t z = (uint64_t)( (int64_t)c.z + OFF );
	return ( x << 42 ) | ( y << 21 ) | z; // never 0 for the representable range
}

static uint32_t b3Voxel_mixU64( uint64_t k )
{
	k ^= k >> 33;
	k *= 0xff51afd7ed558ccdULL;
	k ^= k >> 33;
	k *= 0xc4ceb9fe1a85ec53ULL;
	k ^= k >> 33;
	return (uint32_t)k;
}

static void b3Voxel_growSlots( b3VoxelData* v )
{
	int newCap = v->slotCap ? 2 * v->slotCap : 16;
	b3VoxelChunkSlot* old = v->slots;
	int oldCap = v->slotCap;
	v->slots = (b3VoxelChunkSlot*)b3AllocZeroed( (size_t)newCap * sizeof( b3VoxelChunkSlot ) );
	v->slotCap = newCap;
	uint32_t mask = (uint32_t)( newCap - 1 );
	for ( int i = 0; i < oldCap; ++i )
	{
		if ( old[i].key == 0 )
			continue;
		uint32_t s = b3Voxel_mixU64( old[i].key ) & mask;
		while ( v->slots[s].key != 0 )
			s = ( s + 1 ) & mask;
		v->slots[s] = old[i];
	}
	if ( old )
		b3Free( old, (size_t)oldCap * sizeof( b3VoxelChunkSlot ) );
}

static int b3Voxel_findChunk( const b3VoxelData* v, uint64_t key )
{
	if ( v->slotCap == 0 )
		return -1;
	uint32_t mask = (uint32_t)( v->slotCap - 1 );
	uint32_t s = b3Voxel_mixU64( key ) & mask;
	while ( v->slots[s].key != 0 )
	{
		if ( v->slots[s].key == key )
			return v->slots[s].index;
		s = ( s + 1 ) & mask;
	}
	return -1;
}

static bool b3Voxel_isSolidInternal( const b3VoxelData* v, b3Vec3i cell )
{
	int ci = b3Voxel_findChunk( v, b3Voxel_packChunk( b3Voxel_chunkOf( cell ) ) );
	if ( ci < 0 )
		return false;
	int li = b3Voxel_localIndex( cell.x & B3_VOXEL_CHUNK_MASK, cell.y & B3_VOXEL_CHUNK_MASK, cell.z & B3_VOXEL_CHUNK_MASK );
	return ( v->chunks[ci].exposed[li] & B3_VOXEL_OCCUPIED_BIT ) != 0;
}

static int b3Voxel_getOrCreateChunk( b3VoxelData* v, b3Vec3i chunkPos, uint64_t key )
{
	if ( v->slotCap == 0 || 10 * ( v->chunkCount + 1 ) >= 7 * v->slotCap )
		b3Voxel_growSlots( v );

	uint32_t mask = (uint32_t)( v->slotCap - 1 );
	uint32_t s = b3Voxel_mixU64( key ) & mask;
	while ( v->slots[s].key != 0 )
	{
		if ( v->slots[s].key == key )
			return v->slots[s].index;
		s = ( s + 1 ) & mask;
	}

	if ( v->chunkCount == v->chunkCapacity )
	{
		int nc = v->chunkCapacity ? 2 * v->chunkCapacity : 8;
		b3VoxelChunk* grown = (b3VoxelChunk*)b3Alloc( (size_t)nc * sizeof( b3VoxelChunk ) );
		if ( v->chunks )
		{
			memcpy( grown, v->chunks, (size_t)v->chunkCount * sizeof( b3VoxelChunk ) );
			b3Free( v->chunks, (size_t)v->chunkCapacity * sizeof( b3VoxelChunk ) );
		}
		v->chunks = grown;
		v->chunkCapacity = nc;
	}
	int idx = v->chunkCount++;
	b3VoxelChunk* chunk = v->chunks + idx;
	memset( chunk->exposed, 0, sizeof( chunk->exposed ) );
	chunk->occupied = NULL;
	chunk->occupiedCount = 0;
	chunk->occupiedCapacity = 0;
	chunk->position = chunkPos;
	chunk->solidBounds = (b3AABB){ { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };

	v->slots[s].key = key;
	v->slots[s].index = idx;
	return idx;
}

b3VoxelData* b3CreateVoxelData( const b3Vec3i* cells, int count, float voxelSize )
{
	return b3CreateOffsetVoxelData( cells, count, voxelSize, b3Vec3_zero );
}

b3VoxelData* b3CreateOffsetVoxelData( const b3Vec3i* cells, int count, float voxelSize, b3Vec3 origin )
{
	if ( cells == NULL || count <= 0 || !( voxelSize > 0.0f ) || !isfinite( voxelSize ) || !b3IsValidVec3( origin ) )
		return NULL;

	b3Vec3i* canonical = (b3Vec3i*)b3Alloc( (size_t)count * sizeof( b3Vec3i ) );
	memcpy( canonical, cells, (size_t)count * sizeof( b3Vec3i ) );
	bool isCanonical = true;
	for ( int i = 0; i < count; ++i )
	{
		if ( !b3Voxel_cellIsRepresentable( cells[i] ) )
		{
			b3Free( canonical, (size_t)count * sizeof( b3Vec3i ) );
			return NULL;
		}
		if ( i > 0 && b3Voxel_compareCells( cells + i - 1, cells + i ) >= 0 )
		{
			// Equal cells also take the canonicalization path so duplicate
			// removal retains its existing behavior.
			isCanonical = false;
		}
	}
	if ( !isCanonical )
	{
		qsort( canonical, (size_t)count, sizeof( b3Vec3i ), b3Voxel_compareCells );
	}

	b3VoxelData* v = (b3VoxelData*)b3AllocZeroed( sizeof( b3VoxelData ) );
	v->voxelSize = voxelSize;
	v->invVoxelSize = 1.0f / voxelSize;
	v->origin = origin;
	v->cells = canonical;
	v->cellCapacity = count;
	v->localBounds = (b3AABB){ { FLT_MAX, FLT_MAX, FLT_MAX }, { -FLT_MAX, -FLT_MAX, -FLT_MAX } };
	v->domainMin = (b3Vec3i){ INT_MAX, INT_MAX, INT_MAX };
	v->domainMax = (b3Vec3i){ INT_MIN, INT_MIN, INT_MIN };

	float half = 0.5f * voxelSize;
	uint64_t recentChunkKeys[8] = { 0 };
	int recentChunkIndices[8];

	for ( int i = 0; i < count; ++i )
	{
		b3Vec3i cell = canonical[i];
		if ( !isCanonical && i > 0 && b3Voxel_compareCells( canonical + i - 1, canonical + i ) == 0 )
			continue;

		canonical[v->cellCount] = cell;
		b3Vec3i chunkPos = b3Voxel_chunkOf( cell );
		uint64_t chunkKey = b3Voxel_packChunk( chunkPos );
		uint32_t recentSlot = (uint32_t)chunkKey & 7u;
		int ci;
		if ( recentChunkKeys[recentSlot] == chunkKey )
		{
			ci = recentChunkIndices[recentSlot];
		}
		else
		{
			ci = b3Voxel_getOrCreateChunk( v, chunkPos, chunkKey );
			recentChunkKeys[recentSlot] = chunkKey;
			recentChunkIndices[recentSlot] = ci;
		}
		b3VoxelChunk* chunk = v->chunks + ci;

		int lx = cell.x & B3_VOXEL_CHUNK_MASK;
		int ly = cell.y & B3_VOXEL_CHUNK_MASK;
		int lz = cell.z & B3_VOXEL_CHUNK_MASK;
		int li = b3Voxel_localIndex( lx, ly, lz );
		if ( ( chunk->exposed[li] & B3_VOXEL_OCCUPIED_BIT ) != 0 )
			continue; // duplicate cell
		chunk->exposed[li] = B3_VOXEL_OCCUPIED_BIT;

		if ( chunk->occupiedCount == chunk->occupiedCapacity )
		{
			int nc = chunk->occupiedCapacity ? 2 * chunk->occupiedCapacity : 32;
			uint16_t* grown = (uint16_t*)b3Alloc( (size_t)nc * sizeof( uint16_t ) );
			if ( chunk->occupied )
			{
				memcpy( grown, chunk->occupied, (size_t)chunk->occupiedCount * sizeof( uint16_t ) );
				b3Free( chunk->occupied, (size_t)chunk->occupiedCapacity * sizeof( uint16_t ) );
			}
			chunk->occupied = grown;
			chunk->occupiedCapacity = nc;
		}
		chunk->occupied[chunk->occupiedCount++] = (uint16_t)li;
		v->domainMin.x = cell.x < v->domainMin.x ? cell.x : v->domainMin.x;
		v->domainMin.y = cell.y < v->domainMin.y ? cell.y : v->domainMin.y;
		v->domainMin.z = cell.z < v->domainMin.z ? cell.z : v->domainMin.z;
		v->domainMax.x = cell.x > v->domainMax.x ? cell.x : v->domainMax.x;
		v->domainMax.y = cell.y > v->domainMax.y ? cell.y : v->domainMax.y;
		v->domainMax.z = cell.z > v->domainMax.z ? cell.z : v->domainMax.z;

		b3Vec3 center = b3Add( origin, (b3Vec3){ cell.x * voxelSize, cell.y * voxelSize, cell.z * voxelSize } );
		b3Vec3 lo = { center.x - half, center.y - half, center.z - half };
		b3Vec3 hi = { center.x + half, center.y + half, center.z + half };
		b3AABB* cb = &chunk->solidBounds;
		cb->lowerBound.x = b3MinFloat( cb->lowerBound.x, lo.x );
		cb->lowerBound.y = b3MinFloat( cb->lowerBound.y, lo.y );
		cb->lowerBound.z = b3MinFloat( cb->lowerBound.z, lo.z );
		cb->upperBound.x = b3MaxFloat( cb->upperBound.x, hi.x );
		cb->upperBound.y = b3MaxFloat( cb->upperBound.y, hi.y );
		cb->upperBound.z = b3MaxFloat( cb->upperBound.z, hi.z );
		b3AABB* sb = &v->localBounds;
		sb->lowerBound.x = b3MinFloat( sb->lowerBound.x, lo.x );
		sb->lowerBound.y = b3MinFloat( sb->lowerBound.y, lo.y );
		sb->lowerBound.z = b3MinFloat( sb->lowerBound.z, lo.z );
		sb->upperBound.x = b3MaxFloat( sb->upperBound.x, hi.x );
		sb->upperBound.y = b3MaxFloat( sb->upperBound.y, hi.y );
		sb->upperBound.z = b3MaxFloat( sb->upperBound.z, hi.z );

		v->cellCount++;
	}

	// Cache the six exposed local faces once. Collision filtering asks this for
	// every exact leaf point, so repeating hash lookups there is needlessly hot.
	for ( int ci = 0; ci < v->chunkCount; ++ci )
	{
		b3VoxelChunk* chunk = v->chunks + ci;
		int base[3] = { chunk->position.x * B3_VOXEL_CHUNK_SIZE, chunk->position.y * B3_VOXEL_CHUNK_SIZE,
						chunk->position.z * B3_VOXEL_CHUNK_SIZE };
		for ( int i = 0; i < chunk->occupiedCount; ++i )
		{
			int li = chunk->occupied[i];
			int lx = ( li >> 8 ) & B3_VOXEL_CHUNK_MASK;
			int ly = ( li >> 4 ) & B3_VOXEL_CHUNK_MASK;
			int lz = li & B3_VOXEL_CHUNK_MASK;
			uint8_t mask = 0;
			bool solid = lx > 0 ? ( chunk->exposed[li - 256] & B3_VOXEL_OCCUPIED_BIT ) != 0
								: b3Voxel_isSolidInternal( v, (b3Vec3i){ base[0] - 1, base[1] + ly, base[2] + lz } );
			mask |= !solid ? 1u << 0 : 0;
			solid = lx < B3_VOXEL_CHUNK_MASK
						? ( chunk->exposed[li + 256] & B3_VOXEL_OCCUPIED_BIT ) != 0
						: b3Voxel_isSolidInternal( v, (b3Vec3i){ base[0] + B3_VOXEL_CHUNK_SIZE, base[1] + ly, base[2] + lz } );
			mask |= !solid ? 1u << 1 : 0;
			solid = ly > 0 ? ( chunk->exposed[li - 16] & B3_VOXEL_OCCUPIED_BIT ) != 0
						   : b3Voxel_isSolidInternal( v, (b3Vec3i){ base[0] + lx, base[1] - 1, base[2] + lz } );
			mask |= !solid ? 1u << 2 : 0;
			solid = ly < B3_VOXEL_CHUNK_MASK
						? ( chunk->exposed[li + 16] & B3_VOXEL_OCCUPIED_BIT ) != 0
						: b3Voxel_isSolidInternal( v, (b3Vec3i){ base[0] + lx, base[1] + B3_VOXEL_CHUNK_SIZE, base[2] + lz } );
			mask |= !solid ? 1u << 3 : 0;
			solid = lz > 0 ? ( chunk->exposed[li - 1] & B3_VOXEL_OCCUPIED_BIT ) != 0
						   : b3Voxel_isSolidInternal( v, (b3Vec3i){ base[0] + lx, base[1] + ly, base[2] - 1 } );
			mask |= !solid ? 1u << 4 : 0;
			solid = lz < B3_VOXEL_CHUNK_MASK
						? ( chunk->exposed[li + 1] & B3_VOXEL_OCCUPIED_BIT ) != 0
						: b3Voxel_isSolidInternal( v, (b3Vec3i){ base[0] + lx, base[1] + ly, base[2] + B3_VOXEL_CHUNK_SIZE } );
			mask |= !solid ? 1u << 5 : 0;

			chunk->exposed[li] = (uint8_t)( B3_VOXEL_OCCUPIED_BIT | mask );
		}
	}

	uint32_t voxelSizeBits = 0;
	memcpy( &voxelSizeBits, &voxelSize, sizeof( voxelSizeBits ) );
	struct
	{
		uint64_t cellsHash;
		uint32_t voxelSizeBits;
		uint32_t originBits[3];
		uint32_t cellCount;
	} descriptor = { 0 };
	descriptor.cellsHash = b3Hash64NonZero( (const uint8_t*)canonical, v->cellCount * (int)sizeof( b3Vec3i ) );
	descriptor.voxelSizeBits = voxelSizeBits;
	memcpy( descriptor.originBits, &origin, sizeof( descriptor.originBits ) );
	descriptor.cellCount = (uint32_t)v->cellCount;
	v->hash = b3Hash64NonZero( (const uint8_t*)&descriptor, (int)sizeof( descriptor ) );
	v->hasBounds = v->cellCount > 0;
	if ( !v->hasBounds )
		v->localBounds = (b3AABB){ b3Vec3_zero, b3Vec3_zero };

	return v;
}

void b3DestroyVoxelData( b3VoxelData* v )
{
	if ( v == NULL )
		return;
	for ( int i = 0; i < v->chunkCount; ++i )
	{
		b3VoxelChunk* chunk = v->chunks + i;
		if ( chunk->occupied )
			b3Free( chunk->occupied, (size_t)chunk->occupiedCapacity * sizeof( uint16_t ) );
	}
	if ( v->chunks )
		b3Free( v->chunks, (size_t)v->chunkCapacity * sizeof( b3VoxelChunk ) );
	if ( v->slots )
		b3Free( v->slots, (size_t)v->slotCap * sizeof( b3VoxelChunkSlot ) );
	if ( v->cells )
		b3Free( v->cells, (size_t)v->cellCapacity * sizeof( b3Vec3i ) );
	b3Free( v, sizeof( b3VoxelData ) );
}

int b3VoxelData_GetCellCount( const b3VoxelData* v )
{
	return v ? v->cellCount : 0;
}

float b3VoxelData_GetVoxelSize( const b3VoxelData* v )
{
	return v ? v->voxelSize : 0.0f;
}

b3Vec3 b3VoxelData_GetOrigin( const b3VoxelData* v )
{
	return v ? v->origin : b3Vec3_zero;
}

b3AABB b3VoxelData_GetBounds( const b3VoxelData* v )
{
	if ( v == NULL || v->cellCount == 0 )
		return (b3AABB){ b3Vec3_zero, b3Vec3_zero };
	return v->localBounds;
}

bool b3VoxelData_IsSolid( const b3VoxelData* v, b3Vec3i cell )
{
	if ( v == NULL )
		return false;
	return b3Voxel_isSolidInternal( v, cell );
}

int b3VoxelData_GetCells( const b3VoxelData* v, b3Vec3i* cells, int capacity )
{
	if ( v == NULL || cells == NULL || capacity <= 0 )
		return 0;
	int count = b3MinInt( v->cellCount, capacity );
	memcpy( cells, v->cells, (size_t)count * sizeof( b3Vec3i ) );
	return count;
}

float b3Voxel_GetVoxelSize( const b3VoxelData* v )
{
	return v->voxelSize;
}

b3Vec3 b3Voxel_GetOrigin( const b3VoxelData* v )
{
	return v->origin;
}

b3Vec3 b3Voxel_GetCellCenter( const b3VoxelData* v, b3Vec3i cell )
{
	return b3Voxel_cellCenterFromGrid( v->origin, v->voxelSize, cell );
}

int b3Voxel_GetCellCount( const b3VoxelData* v )
{
	return v->cellCount;
}

uint64_t b3Voxel_GetHash( const b3VoxelData* v )
{
	return v->hash;
}

bool b3Voxel_IsDirectionExposed( const b3VoxelData* v, b3Vec3i cell, b3Vec3 direction )
{
	if ( v == NULL )
		return false;
	int ci = b3Voxel_findChunk( v, b3Voxel_packChunk( b3Voxel_chunkOf( cell ) ) );
	if ( ci < 0 )
		return false;
	int li = b3Voxel_localIndex( cell.x & B3_VOXEL_CHUNK_MASK, cell.y & B3_VOXEL_CHUNK_MASK, cell.z & B3_VOXEL_CHUNK_MASK );
	if ( ( v->chunks[ci].exposed[li] & B3_VOXEL_OCCUPIED_BIT ) == 0 )
		return false;
	return b3Voxel_ExposureMaskContains( v->chunks[ci].exposed[li] & B3_VOXEL_EXPOSED_MASK, direction );
}

uint8_t* b3Voxel_Serialize( const b3VoxelData* v, int* byteCount )
{
	if ( byteCount != NULL )
	{
		*byteCount = 0;
	}
	if ( v == NULL || v->cellCount <= 0 ||
		 v->cellCount > ( INT_MAX - (int)sizeof( b3VoxelBlobHeader ) ) / (int)sizeof( b3Vec3i ) )
	{
		return NULL;
	}

	int size = (int)sizeof( b3VoxelBlobHeader ) + v->cellCount * (int)sizeof( b3Vec3i );
	uint8_t* bytes = b3Alloc( (size_t)size );
	b3VoxelBlobHeader header = { B3_VOXEL_BLOB_MAGIC, (uint32_t)v->cellCount, v->voxelSize, v->origin };
	memcpy( bytes, &header, sizeof( header ) );
	memcpy( bytes + sizeof( header ), v->cells, (size_t)v->cellCount * sizeof( b3Vec3i ) );
	if ( byteCount != NULL )
	{
		*byteCount = size;
	}
	return bytes;
}

b3VoxelData* b3Voxel_Deserialize( const uint8_t* bytes, int byteCount )
{
	if ( bytes == NULL || byteCount < (int)sizeof( b3VoxelBlobHeader ) )
	{
		return NULL;
	}
	b3VoxelBlobHeader header;
	memcpy( &header, bytes, sizeof( header ) );
	if ( header.magic != B3_VOXEL_BLOB_MAGIC || header.cellCount == 0 ||
		 header.cellCount > (uint32_t)( ( INT_MAX - (int)sizeof( header ) ) / (int)sizeof( b3Vec3i ) ) )
	{
		return NULL;
	}
	int expected = (int)sizeof( header ) + (int)header.cellCount * (int)sizeof( b3Vec3i );
	if ( byteCount != expected )
	{
		return NULL;
	}
	return b3CreateOffsetVoxelData( (const b3Vec3i*)( bytes + sizeof( header ) ), (int)header.cellCount, header.voxelSize,
									header.origin );
}

bool b3Voxel_GetLocalBounds( const b3VoxelData* v, b3AABB* out )
{
	if ( v->cellCount == 0 )
		return false;
	*out = v->localBounds;
	return true;
}

bool b3Voxel_GetDomain( const b3VoxelData* v, b3Vec3i* lower, b3Vec3i* upper )
{
	if ( v->cellCount == 0 )
		return false;
	*lower = v->domainMin;
	*upper = v->domainMax;
	return true;
}

static inline bool b3Voxel_aabbIntersects( const b3AABB* a, const b3AABB* b )
{
	if ( a->upperBound.x < b->lowerBound.x || a->lowerBound.x > b->upperBound.x )
		return false;
	if ( a->upperBound.y < b->lowerBound.y || a->lowerBound.y > b->upperBound.y )
		return false;
	if ( a->upperBound.z < b->lowerBound.z || a->lowerBound.z > b->upperBound.z )
		return false;
	return true;
}

static int b3Voxel_queryCellsImpl( const b3VoxelData* v, b3AABB queryLocal, b3Vec3i* out, int cap, b3VoxelQueryCallback* callback,
								   void* context, b3VoxelCounters* counters )
{
	if ( counters != NULL )
	{
		counters->queryCalls += 1;
	}
	if ( v->cellCount == 0 )
		return 0;

	b3AABB clip;
	clip.lowerBound.x = b3MaxFloat( queryLocal.lowerBound.x, v->localBounds.lowerBound.x );
	clip.lowerBound.y = b3MaxFloat( queryLocal.lowerBound.y, v->localBounds.lowerBound.y );
	clip.lowerBound.z = b3MaxFloat( queryLocal.lowerBound.z, v->localBounds.lowerBound.z );
	clip.upperBound.x = b3MinFloat( queryLocal.upperBound.x, v->localBounds.upperBound.x );
	clip.upperBound.y = b3MinFloat( queryLocal.upperBound.y, v->localBounds.upperBound.y );
	clip.upperBound.z = b3MinFloat( queryLocal.upperBound.z, v->localBounds.upperBound.z );
	if ( clip.lowerBound.x > clip.upperBound.x || clip.lowerBound.y > clip.upperBound.y || clip.lowerBound.z > clip.upperBound.z )
		return 0;

	const float eps = 1e-6f;
	float inv = v->invVoxelSize;
	float minX = ( clip.lowerBound.x - v->origin.x ) * inv - 0.5f - eps;
	float minY = ( clip.lowerBound.y - v->origin.y ) * inv - 0.5f - eps;
	float minZ = ( clip.lowerBound.z - v->origin.z ) * inv - 0.5f - eps;
	int gMinX = (int)minX;
	int gMinY = (int)minY;
	int gMinZ = (int)minZ;
	gMinX += (float)gMinX < minX;
	gMinY += (float)gMinY < minY;
	gMinZ += (float)gMinZ < minZ;
	float maxX = ( clip.upperBound.x - v->origin.x ) * inv + 0.5f + eps;
	float maxY = ( clip.upperBound.y - v->origin.y ) * inv + 0.5f + eps;
	float maxZ = ( clip.upperBound.z - v->origin.z ) * inv + 0.5f + eps;
	int gMaxX = (int)maxX;
	int gMaxY = (int)maxY;
	int gMaxZ = (int)maxZ;
	gMaxX -= (float)gMaxX > maxX;
	gMaxY -= (float)gMaxY > maxY;
	gMaxZ -= (float)gMaxZ > maxZ;
	if ( gMinX > gMaxX || gMinY > gMaxY || gMinZ > gMaxZ )
		return 0;

	int cMinX = gMinX >> B3_VOXEL_CHUNK_BITS, cMaxX = gMaxX >> B3_VOXEL_CHUNK_BITS;
	int cMinY = gMinY >> B3_VOXEL_CHUNK_BITS, cMaxY = gMaxY >> B3_VOXEL_CHUNK_BITS;
	int cMinZ = gMinZ >> B3_VOXEL_CHUNK_BITS, cMaxZ = gMaxZ >> B3_VOXEL_CHUNK_BITS;

	// Narrow collision queries usually cover only a handful of grid positions
	// inside a dense chunk. Probe those occupancy bytes directly instead of
	// scanning the chunk's entire compact solid list. Broad sparse queries keep
	// the list path below. Both paths preserve the same chunk and cell order.
	int64_t nx = (int64_t)gMaxX - gMinX + 1;
	int64_t ny = (int64_t)gMaxY - gMinY + 1;
	int64_t nz = (int64_t)gMaxZ - gMinZ + 1;
	if ( nx * ny * nz <= B3_VOXEL_DIRECT_QUERY_CELLS )
	{
		int n = 0;
		for ( int cz = cMinZ; cz <= cMaxZ; ++cz )
		{
			for ( int cy = cMinY; cy <= cMaxY; ++cy )
			{
				for ( int cx = cMinX; cx <= cMaxX; ++cx )
				{
					if ( counters != NULL )
					{
						counters->chunksVisited += 1;
					}
					int ci = b3Voxel_findChunk( v, b3Voxel_packChunk( (b3Vec3i){ cx, cy, cz } ) );
					if ( ci < 0 )
						continue;

					int baseX = cx * B3_VOXEL_CHUNK_SIZE;
					int baseY = cy * B3_VOXEL_CHUNK_SIZE;
					int baseZ = cz * B3_VOXEL_CHUNK_SIZE;
					int loX = b3MaxInt( gMinX, baseX ), hiX = b3MinInt( gMaxX, baseX + B3_VOXEL_CHUNK_MASK );
					int loY = b3MaxInt( gMinY, baseY ), hiY = b3MinInt( gMaxY, baseY + B3_VOXEL_CHUNK_MASK );
					int loZ = b3MaxInt( gMinZ, baseZ ), hiZ = b3MinInt( gMaxZ, baseZ + B3_VOXEL_CHUNK_MASK );
					const b3VoxelChunk* chunk = v->chunks + ci;
					if ( counters != NULL )
					{
						counters->occupiedEntriesScanned += ( hiX - loX + 1 ) * ( hiY - loY + 1 ) * ( hiZ - loZ + 1 );
					}
					for ( int gx = loX; gx <= hiX; ++gx )
					{
						for ( int gy = loY; gy <= hiY; ++gy )
						{
							for ( int gz = loZ; gz <= hiZ; ++gz )
							{
								int li = b3Voxel_localIndex( gx & B3_VOXEL_CHUNK_MASK, gy & B3_VOXEL_CHUNK_MASK,
															 gz & B3_VOXEL_CHUNK_MASK );
								uint8_t cellData = chunk->exposed[li];
								if ( ( cellData & B3_VOXEL_OCCUPIED_BIT ) == 0 )
									continue;
								b3Vec3i cell = { gx, gy, gz };
								if ( out != NULL && n < cap )
									out[n] = cell;
								n += 1;
								if ( callback != NULL && callback( cell, cellData & B3_VOXEL_EXPOSED_MASK, context ) == false )
								{
									if ( counters != NULL )
									{
										counters->cellsReturned += n;
									}
									return n;
								}
							}
						}
					}
				}
			}
		}
		if ( counters != NULL )
		{
			counters->cellsReturned += n;
		}
		return n;
	}

	int n = 0;
	for ( int cz = cMinZ; cz <= cMaxZ; ++cz )
	{
		for ( int cy = cMinY; cy <= cMaxY; ++cy )
		{
			for ( int cx = cMinX; cx <= cMaxX; ++cx )
			{
				if ( counters != NULL )
				{
					counters->chunksVisited += 1;
				}
				int ci = b3Voxel_findChunk( v, b3Voxel_packChunk( (b3Vec3i){ cx, cy, cz } ) );
				if ( ci < 0 )
					continue;
				const b3VoxelChunk* chunk = v->chunks + ci;
				if ( !b3Voxel_aabbIntersects( &chunk->solidBounds, &clip ) )
					continue;
				if ( counters != NULL )
				{
					counters->occupiedEntriesScanned += chunk->occupiedCount;
				}
				int base[3] = { cx * B3_VOXEL_CHUNK_SIZE, cy * B3_VOXEL_CHUNK_SIZE, cz * B3_VOXEL_CHUNK_SIZE };
				for ( int k = 0; k < chunk->occupiedCount; ++k )
				{
					int li = chunk->occupied[k];
					b3Vec3i lc = { ( li >> 8 ) & 15, ( li >> 4 ) & 15, li & 15 };
					b3Vec3i g = { base[0] + lc.x, base[1] + lc.y, base[2] + lc.z };
					if ( g.x < gMinX || g.x > gMaxX || g.y < gMinY || g.y > gMaxY || g.z < gMinZ || g.z > gMaxZ )
						continue;
					if ( out != NULL && n < cap )
						out[n] = g;
					n++;
					if ( callback != NULL && callback( g, chunk->exposed[li] & B3_VOXEL_EXPOSED_MASK, context ) == false )
					{
						if ( counters != NULL )
						{
							counters->cellsReturned += n;
						}
						return n;
					}
				}
			}
		}
	}
	if ( counters != NULL )
	{
		counters->cellsReturned += n;
	}
	return n;
}

int b3Voxel_QueryCellsTracked( const b3VoxelData* v, b3AABB queryLocal, b3Vec3i* out, int cap, b3VoxelCounters* counters )
{
	return b3Voxel_queryCellsImpl( v, queryLocal, out, cap, NULL, NULL, counters );
}

int b3Voxel_ForEachCellTracked( const b3VoxelData* v, b3AABB queryLocal, b3VoxelQueryCallback* callback, void* context,
								b3VoxelCounters* counters )
{
	B3_ASSERT( callback != NULL );
	return b3Voxel_queryCellsImpl( v, queryLocal, NULL, 0, callback, context, counters );
}

int b3Voxel_QueryCells( const b3VoxelData* v, b3AABB queryLocal, b3Vec3i* out, int cap )
{
	return b3Voxel_QueryCellsTracked( v, queryLocal, out, cap, NULL );
}

static b3Vec3i b3Voxel_unpackChunk( uint64_t key )
{
	const int64_t OFF = 1 << 20;
	const uint64_t M = ( 1ull << 21 ) - 1;
	return (b3Vec3i){ (int)( (int64_t)( ( key >> 42 ) & M ) - OFF ), (int)( (int64_t)( ( key >> 21 ) & M ) - OFF ),
					  (int)( (int64_t)( key & M ) - OFF ) };
}

int b3Voxel_GetCells( const b3VoxelData* v, b3Vec3i* out, int cap )
{
	if ( v == NULL || out == NULL || cap <= 0 )
		return 0;
	int count = b3MinInt( v->cellCount, cap );
	memcpy( out, v->cells, (size_t)count * sizeof( b3Vec3i ) );
	return count;
}

b3MassData b3Voxel_ComputeMass( const b3VoxelData* v, float density )
{
	b3MassData md = { 0.0f, b3Vec3_zero, b3Mat3_zero };
	if ( v == NULL || v->cellCount == 0 || density <= 0.0f )
		return md;

	float s = v->voxelSize;
	float cellMass = density * s * s * s;
	float cubeInertia = ( 1.0f / 6.0f ) * cellMass * s * s; // solid cube about its centre, per axis

	double M = 0.0;
	b3Vec3 com = b3Vec3_zero;
	for ( int slot = 0; slot < v->slotCap; ++slot )
	{
		if ( v->slots[slot].key == 0 )
			continue;
		b3Vec3i cp = b3Voxel_unpackChunk( v->slots[slot].key );
		const b3VoxelChunk* chunk = v->chunks + v->slots[slot].index;
		int base[3] = { cp.x * B3_VOXEL_CHUNK_SIZE, cp.y * B3_VOXEL_CHUNK_SIZE, cp.z * B3_VOXEL_CHUNK_SIZE };
		for ( int k = 0; k < chunk->occupiedCount; ++k )
		{
			int li = chunk->occupied[k];
			b3Vec3i lc = { ( li >> 8 ) & 15, ( li >> 4 ) & 15, li & 15 };
			b3Vec3 c = b3Voxel_GetCellCenter( v, (b3Vec3i){ base[0] + lc.x, base[1] + lc.y, base[2] + lc.z } );
			M += cellMass;
			com = b3MulAdd( com, cellMass, c );
		}
	}
	com = b3MulSV( 1.0f / (float)M, com );

	b3Matrix3 I = b3Mat3_zero;
	for ( int slot = 0; slot < v->slotCap; ++slot )
	{
		if ( v->slots[slot].key == 0 )
			continue;
		b3Vec3i cp = b3Voxel_unpackChunk( v->slots[slot].key );
		const b3VoxelChunk* chunk = v->chunks + v->slots[slot].index;
		int base[3] = { cp.x * B3_VOXEL_CHUNK_SIZE, cp.y * B3_VOXEL_CHUNK_SIZE, cp.z * B3_VOXEL_CHUNK_SIZE };
		for ( int k = 0; k < chunk->occupiedCount; ++k )
		{
			int li = chunk->occupied[k];
			b3Vec3i lc = { ( li >> 8 ) & 15, ( li >> 4 ) & 15, li & 15 };
			b3Vec3 c = b3Voxel_GetCellCenter( v, (b3Vec3i){ base[0] + lc.x, base[1] + lc.y, base[2] + lc.z } );
			b3Vec3 r = b3Sub( c, com );
			float rr = b3Dot( r, r );
			I.cx.x += cubeInertia + cellMass * ( rr - r.x * r.x );
			I.cy.y += cubeInertia + cellMass * ( rr - r.y * r.y );
			I.cz.z += cubeInertia + cellMass * ( rr - r.z * r.z );
			I.cx.y += -cellMass * r.x * r.y;
			I.cx.z += -cellMass * r.x * r.z;
			I.cy.z += -cellMass * r.y * r.z;
		}
	}
	I.cy.x = I.cx.y;
	I.cz.x = I.cx.z;
	I.cz.y = I.cy.z;

	md.mass = (float)M;
	md.center = com;
	md.inertia = I;
	return md;
}

float b3Voxel_ComputeSurfaceArea( const b3VoxelData* v )
{
	if ( v == NULL )
		return 0.0f;
	int exposedFaceCount = 0;
	for ( int i = 0; i < v->cellCount; ++i )
	{
		b3Vec3i c = v->cells[i];
		exposedFaceCount += !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x - 1, c.y, c.z } );
		exposedFaceCount += !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x + 1, c.y, c.z } );
		exposedFaceCount += !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y - 1, c.z } );
		exposedFaceCount += !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y + 1, c.z } );
		exposedFaceCount += !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y, c.z - 1 } );
		exposedFaceCount += !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y, c.z + 1 } );
	}
	return (float)exposedFaceCount * v->voxelSize * v->voxelSize;
}

float b3Voxel_ComputeProjectedArea( const b3VoxelData* v, b3Vec3 planeNormal )
{
	if ( v == NULL )
		return 0.0f;
	float weights[3] = { 0.5f * b3AbsFloat( planeNormal.x ), 0.5f * b3AbsFloat( planeNormal.y ),
						 0.5f * b3AbsFloat( planeNormal.z ) };
	float weightedFaces = 0.0f;
	for ( int i = 0; i < v->cellCount; ++i )
	{
		b3Vec3i c = v->cells[i];
		weightedFaces += weights[0] * ( !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x - 1, c.y, c.z } ) +
										!b3VoxelData_IsSolid( v, (b3Vec3i){ c.x + 1, c.y, c.z } ) );
		weightedFaces += weights[1] * ( !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y - 1, c.z } ) +
										!b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y + 1, c.z } ) );
		weightedFaces += weights[2] * ( !b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y, c.z - 1 } ) +
										!b3VoxelData_IsSolid( v, (b3Vec3i){ c.x, c.y, c.z + 1 } ) );
	}
	return weightedFaces * v->voxelSize * v->voxelSize;
}
