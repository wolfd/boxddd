//! `b3Shape_GetFatAABB` is the bound the broad-phase tree stores for a shape's
//! proxy, so a caller may re-test a widened overlap query's results against a
//! narrower AABB and reproduce the tree's own leaf test exactly.
//!
//! Two properties are pinned here, at every mutation site that can move a
//! proxy (creation, `b3Body_SetTransform` -> `MoveProxy`, and a step that grows
//! a moving shape's bound -> `EnlargeProxy`):
//!
//! 1. the fat AABB contains the tight AABB (`try_shape_aabb`);
//! 2. the fat AABB IS the tree's leaf bound — a degenerate query at a fat
//!    corner reports the shape (Box3D's overlap test is inclusive), and the
//!    same query nudged just outside does not.
//!
//! Property 2 is the load-bearing one: it fails if a future revision enlarges
//! tree bounds without writing `shape->fatAABB` back.

use boxddd::{Aabb, BodyType, BoxHull, Foundation, QueryFilter, ShapeId, Vec3, World};

/// Degenerate (point) query AABB — Box3D's overlap test is inclusive, so a
/// point exactly on a bound counts as overlapping it.
fn point_query(p: Vec3) -> Aabb {
    Aabb {
        lower_bound: p,
        upper_bound: p,
    }
}

fn reports(world: &World, aabb: Aabb, shape: ShapeId) -> bool {
    world
        .overlap_aabb(aabb, QueryFilter::default())
        .unwrap()
        .iter()
        .any(|hit| hit.shape_id == shape)
}

/// The tree leaf test uses exactly `fat`: its corners are inside, and a nudge
/// past them on every axis is outside.
fn assert_fat_is_the_tree_bound(world: &World, shape: ShapeId, label: &str) {
    let tight = world.shape_aabb(shape).unwrap();
    let fat = world.shape_fat_aabb(shape).unwrap();

    assert!(
        fat.lower_bound.x <= tight.lower_bound.x
            && fat.lower_bound.y <= tight.lower_bound.y
            && fat.lower_bound.z <= tight.lower_bound.z
            && fat.upper_bound.x >= tight.upper_bound.x
            && fat.upper_bound.y >= tight.upper_bound.y
            && fat.upper_bound.z >= tight.upper_bound.z,
        "{label}: fat {fat:?} must contain tight {tight:?}"
    );

    // A margin is what makes the two distinguishable; without one this test
    // would pass vacuously against `b3Shape_GetAABB`.
    assert!(
        fat.upper_bound.x > tight.upper_bound.x,
        "{label}: expected a non-zero broad-phase margin, fat {fat:?} tight {tight:?}"
    );

    for corner in [fat.lower_bound, fat.upper_bound] {
        assert!(
            reports(world, point_query(corner), shape),
            "{label}: the tree must report the shape at its fat corner {corner:?} \
             (fat {fat:?})"
        );
    }

    // One margin's worth clear of the fat bound on every axis at once: far
    // enough that float slop cannot explain a hit, still nowhere near any
    // other shape.
    let slack = (fat.upper_bound.x - tight.upper_bound.x).max(1e-3);
    let outside_hi = Vec3::new(
        fat.upper_bound.x + slack,
        fat.upper_bound.y + slack,
        fat.upper_bound.z + slack,
    );
    let outside_lo = Vec3::new(
        fat.lower_bound.x - slack,
        fat.lower_bound.y - slack,
        fat.lower_bound.z - slack,
    );
    for corner in [outside_lo, outside_hi] {
        assert!(
            !reports(world, point_query(corner), shape),
            "{label}: the tree must NOT report the shape past its fat bound at \
             {corner:?} (fat {fat:?}) — the stored leaf bound is wider than \
             b3Shape_GetFatAABB reports"
        );
    }
}

#[test]
fn fat_aabb_is_the_broad_phase_bound_through_create_move_and_enlarge() {
    let foundation = Foundation::initialize_default().unwrap();
    let mut world = foundation.create_world(foundation.world_def()).unwrap();

    // Static: proxy written once, at creation.
    let ground = world
        .create_body(
            foundation
                .body_def_builder()
                .body_type(BodyType::Static)
                .position([0.0, -20.0, 0.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    let ground_shape = world
        .create_hull_shape(
            ground,
            &foundation.shape_def_builder().density(1.0).build().unwrap(),
            &BoxHull::cube(0.5).unwrap(),
        )
        .unwrap();
    assert_fat_is_the_tree_bound(&world, ground_shape, "static, after create");

    // Dynamic: creation, then a teleport (MoveProxy), then motion (EnlargeProxy).
    let body = world
        .create_body(
            foundation
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position([0.0, 5.0, 0.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    let shape = world
        .create_hull_shape(
            body,
            &foundation.shape_def_builder().density(1.0).build().unwrap(),
            &BoxHull::cube(0.25).unwrap(),
        )
        .unwrap();
    assert_fat_is_the_tree_bound(&world, shape, "dynamic, after create");

    world
        .set_body_transform(body, [3.0, 5.0, -2.0], boxddd::Quat::IDENTITY)
        .unwrap();
    assert_fat_is_the_tree_bound(&world, shape, "dynamic, after set_transform");

    // Free fall grows the bound past its margin within a few steps, which is
    // the EnlargeProxy path.
    for _ in 0..30 {
        world.step(1.0 / 60.0, 4).unwrap();
        assert_fat_is_the_tree_bound(&world, shape, "dynamic, while stepping");
    }
    assert_fat_is_the_tree_bound(&world, ground_shape, "static, after stepping");
}
