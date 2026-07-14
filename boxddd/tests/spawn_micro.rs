//! Micro-bench for many-shape dynamic-body spawn cost (run manually with
//! `cargo test -p boxddd --release --test spawn_micro -- --ignored --nocapture`,
//! attach `sample <pid>` to see where the C time goes).

use boxddd::{BodyDef, BodyType, BoxHull, ShapeDef, Vec3, World, WorldDef};
use std::time::Instant;

#[test]
#[ignore = "manual profiling harness"]
fn spawn_3000_shape_body_repeatedly() {
    let mut world = World::new(WorldDef::default()).unwrap();
    let ground = world.create_body(BodyDef::builder().body_type(BodyType::Static).build());
    world.create_hull_shape(ground, &ShapeDef::default(), &BoxHull::new(100.0, 0.5, 100.0));

    let def = ShapeDef::builder()
        .density(1.0)
        .update_body_mass(false)
        .build();

    // Context like the sim's collapse scene: ~600 small dynamic bodies (a
    // few dozen DISTINCT hulls each, ~38k shapes total in the hull
    // database), settled into a pile.
    for b in 0..600u32 {
        let (bx, bz) = ((b % 25) as f32 * 1.2 - 15.0, (b / 25) as f32 * 1.2 - 15.0);
        let body = world.create_body(
            BodyDef::builder()
                .body_type(BodyType::Dynamic)
                .position([bx, 0.8 + (b % 3) as f32 * 0.4, bz])
                .build(),
        );
        for i in 0..64u32 {
            let (x, y, z) = (i % 4, (i / 4) % 4, i / 16);
            let hull = BoxHull::offset(
                0.05,
                0.05,
                0.05,
                // Vary by body so hulls are DISTINCT database entries.
                Vec3::new(
                    x as f32 * 0.1 + b as f32 * 1e-4,
                    y as f32 * 0.1,
                    z as f32 * 0.1,
                ),
            );
            world.create_hull_shape(body, &def, &hull);
        }
        world.try_apply_mass_from_shapes(body).unwrap();
    }
    for _ in 0..120 {
        world.step(1.0 / 60.0, 4);
    }

    // Match the sim: spawned survivors are bullets (CCD parity with rapier)
    // and the scene never sleeps. Env toggles so A/B runs need no rebuild.
    let bullet = std::env::var_os("BULLET").is_some();
    if std::env::var_os("AWAKE").is_some() {
        world.enable_sleeping(false);
    }

    let t0 = Instant::now();
    let mut create_ms = 0.0f64;
    let iters = 50u32;
    for it in 0..iters {
        let body = world.create_body(
            BodyDef::builder()
                .body_type(BodyType::Dynamic)
                .position([0.0, 1.0, 0.0]) // inside the pile volume
                .bullet(bullet)
                .build(),
        );
        // ~3000 box hulls, DISTINCT per iteration (like each new survivor).
        // QUANTIZED=1 uses voxel-grid coordinates exactly like the sim
        // (multiples of 0.25, distinct via integer dims), probing whether the
        // hull database's djb2 content hash clusters on structured floats;
        // default keeps the mantissa-diverse smooth offsets.
        let quantized = std::env::var_os("QUANTIZED").is_some();
        let c0 = Instant::now();
        for i in 0..3000u32 {
            let (x, y, z) = (i % 15, (i / 15) % 20, i / 300);
            let hull = if quantized {
                const V: f32 = 0.25;
                // Vary height in whole voxels per iteration so every
                // (iteration, cell) hull is a distinct database entry while
                // every float stays a small multiple of the voxel size.
                let h = 1 + (it % 8);
                BoxHull::offset(
                    V * 0.5,
                    V * 0.5 * h as f32,
                    V * 0.5,
                    Vec3::new(x as f32 * V, y as f32 * V * h as f32, z as f32 * V),
                )
            } else {
                BoxHull::offset(
                    0.05,
                    0.05,
                    0.05,
                    Vec3::new(
                        x as f32 * 0.1 + it as f32 * 1e-4,
                        y as f32 * 0.1,
                        z as f32 * 0.1,
                    ),
                )
            };
            world.create_hull_shape(body, &def, &hull);
        }
        create_ms += c0.elapsed().as_secs_f64() * 1e3;
        world.try_apply_mass_from_shapes(body).unwrap();
        world.step(1.0 / 60.0, 4);
        world.destroy_body(body);
    }
    let per = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
    println!(
        "spawn(3000 distinct)+step+destroy inside pile: {per:.2} ms/iter (create loop {:.2} ms/iter)",
        create_ms / iters as f64
    );
}
