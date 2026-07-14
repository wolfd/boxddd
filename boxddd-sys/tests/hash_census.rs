//! SCRATCH DIAGNOSTIC (uncommitted): collision census of the OLD djb2-fold
//! content hash vs the NEW multiply-rotate hash, over REAL baked b3BoxHull
//! bytes, replicating exactly what the hull database sees.
//!
//! Run: cargo test -p boxddd-sys --release --test hash_census -- --ignored --nocapture

use boxddd_sys::ffi;

/// EXACT replica of the pre-fix b3Hash (folded djb2, little-endian host).
fn old_djb2_fold(mut h: u32, data: &[u8]) -> u32 {
    let mut i = 0;
    while i + 8 <= data.len() {
        let word = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
        h = (h << 5).wrapping_add(h).wrapping_add(word as u32);
        h = (h << 5).wrapping_add(h).wrapping_add((word >> 32) as u32);
        i += 8;
    }
    while i < data.len() {
        h = (h << 5).wrapping_add(h).wrapping_add(data[i] as u32);
        i += 1;
    }
    h
}

/// EXACT replica of the post-fix b3Hash (xxHash32-style rounds).
fn new_mul_rot(mut h: u32, data: &[u8]) -> u32 {
    let mut i = 0;
    while i + 8 <= data.len() {
        let word = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
        h = h.wrapping_add((word as u32).wrapping_mul(2246822519));
        h = h.rotate_left(13).wrapping_mul(2654435761);
        h = h.wrapping_add(((word >> 32) as u32).wrapping_mul(2246822519));
        h = h.rotate_left(13).wrapping_mul(2654435761);
        i += 8;
    }
    while i < data.len() {
        h = h.wrapping_add((data[i] as u32).wrapping_mul(374761393));
        h = h.rotate_left(11).wrapping_mul(2654435761);
        i += 1;
    }
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    h = h.wrapping_mul(3266489917);
    h ^= h >> 16;
    h
}

/// Bake a real box hull through the C library and return its identity bytes
/// exactly as b3AddHullToDatabase hashes them: full struct, hash field
/// (offset 12, after u64 version + i32 byteCount) zeroed.
fn baked_bytes(hx: f32, hy: f32, hz: f32, off: [f32; 3]) -> Vec<u8> {
    let hull = unsafe {
        ffi::b3MakeOffsetBoxHull(
            hx,
            hy,
            hz,
            ffi::b3Vec3 {
                x: off[0],
                y: off[1],
                z: off[2],
            },
        )
    };
    let count = hull.base.byteCount as usize;
    assert_eq!(count, std::mem::size_of::<ffi::b3BoxHull>());
    let mut bytes =
        unsafe { std::slice::from_raw_parts(&hull as *const _ as *const u8, count) }.to_vec();
    bytes[12..16].fill(0); // hash field zeroed during baking
    bytes
}

const B3_HASH_INIT: u32 = 5381;
const FIB: u64 = 0x9E3779B97F4A7C15;

fn non_zero(h: u32) -> u32 {
    if h != 0 { h } else { 1 }
}

fn census(label: &str, hulls: &[Vec<u8>], f: fn(u32, &[u8]) -> u32) {
    // 32-bit content hashes as baked into hull->hash.
    let h32: Vec<u32> = hulls.iter().map(|b| non_zero(f(B3_HASH_INIT, b))).collect();
    let distinct: std::collections::HashSet<u32> = h32.iter().copied().collect();

    // What verstable sees: 64-bit b3HashHullData = h32 * Fibonacci;
    // home bucket = hash & mask (LOW bits), fragment = top 4 bits.
    let mask: u64 = 8192 - 1; // realistic bucket count for ~6k entries
    let mut buckets: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
    for h in &h32 {
        let h64 = (*h as u64).wrapping_mul(FIB);
        buckets.entry(h64 & mask).or_default().push(h64 >> 60);
    }
    let mut sizes: Vec<usize> = buckets.values().map(|v| v.len()).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let biggest = buckets.values().max_by_key(|v| v.len()).unwrap();
    let frags: std::collections::HashSet<u64> = biggest.iter().copied().collect();
    // Expected probes for one lookup/insert = average chain walked.
    let avg_chain: f64 =
        sizes.iter().map(|s| (s * s) as f64).sum::<f64>() / hulls.len() as f64;

    println!("{label}:");
    println!("  {} hulls -> {} distinct 32-bit hashes", hulls.len(), distinct.len());
    println!(
        "  home buckets used: {} of 8192, top chains: {:?}",
        buckets.len(),
        &sizes[..sizes.len().min(8)]
    );
    println!(
        "  avg probes per operation: {avg_chain:.1}; biggest chain has {} entries with {} distinct fragments",
        biggest.len(),
        frags.len()
    );
}

#[test]
#[ignore = "manual diagnostic"]
fn collision_census_quantized_vs_smooth() {
    const V: f32 = 0.25;
    let mut quantized = Vec::new();
    let mut smooth = Vec::new();
    // Same population as the spawn_micro bench: 2 survivors' worth of
    // distinct hulls (3000 each), voxel-quantized vs mantissa-diverse.
    for it in 0..2u32 {
        for i in 0..3000u32 {
            let (x, y, z) = (i % 15, (i / 15) % 20, i / 300);
            let h = 1 + (it * 3000 + i) % 8; // vary dims like merged voxel spans
            quantized.push(baked_bytes(
                V * 0.5,
                V * 0.5 * h as f32,
                V * 0.5,
                [x as f32 * V, y as f32 * V * h as f32, z as f32 * V],
            ));
            smooth.push(baked_bytes(
                0.05,
                0.05,
                0.05,
                [
                    x as f32 * 0.1 + (it * 3000 + i) as f32 * 1e-4,
                    y as f32 * 0.1,
                    z as f32 * 0.1,
                ],
            ));
        }
    }

    census("OLD djb2 fold / quantized", &quantized, old_djb2_fold);
    census("OLD djb2 fold / smooth   ", &smooth, old_djb2_fold);
    census("NEW mul-rot   / quantized", &quantized, new_mul_rot);
    census("NEW mul-rot   / smooth   ", &smooth, new_mul_rot);

    // Option A check: keep the old djb2 32-bit values, but replace
    // b3HashHullData's bare Fibonacci multiply with murmur3's fmix64.
    // If this fixes bucketing, the minimal upstream patch is viable.
    for (label, hulls) in [("quantized", &quantized), ("smooth", &smooth)] {
        let mask: u64 = 8192 - 1;
        let mut buckets: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
        for b in hulls.iter() {
            let mut h = non_zero(old_djb2_fold(B3_HASH_INIT, b)) as u64;
            h ^= h >> 33;
            h = h.wrapping_mul(0xFF51AFD7ED558CCD);
            h ^= h >> 33;
            h = h.wrapping_mul(0xC4CEB9FE1A85EC53);
            h ^= h >> 33;
            *buckets.entry(h & mask).or_default() += 1;
        }
        let mut sizes: Vec<usize> = buckets.values().copied().collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let avg: f64 = sizes.iter().map(|s| (s * s) as f64).sum::<f64>() / hulls.len() as f64;
        println!(
            "OPTION A (old djb2 + fmix64) / {label}: buckets {} of 8192, top chains {:?}, avg probes {avg:.1}",
            buckets.len(),
            &sizes[..sizes.len().min(6)]
        );
    }

    // Bonus: show the low-bit degeneracy directly — how many distinct values
    // do the LOW 13 bits of the old 32-bit hash take?
    for (label, hulls) in [("quantized", &quantized), ("smooth", &smooth)] {
        let lows: std::collections::HashSet<u32> = hulls
            .iter()
            .map(|b| non_zero(old_djb2_fold(B3_HASH_INIT, b)) & 0x1FFF)
            .collect();
        println!("OLD hash low-13-bits distinct values ({label}): {}", lows.len());
    }
}
