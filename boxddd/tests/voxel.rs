use boxddd::{
    BodySnapshotId, BodyType, BoxHull, Foundation, Hull, Quat, Recording, ShapeType, Vec3,
    VoxelCell, VoxelData, World,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn voxel_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn foundation() -> &'static Foundation {
    Foundation::initialize_default().unwrap()
}

fn world_with_gravity(gravity: impl Into<Vec3>) -> World {
    let foundation = foundation();
    foundation
        .create_world(
            foundation
                .world_def_builder()
                .gravity(gravity)
                .build()
                .unwrap(),
        )
        .unwrap()
}

fn default_world() -> World {
    let foundation = foundation();
    foundation.create_world(foundation.world_def()).unwrap()
}

fn filled_box(
    x: std::ops::RangeInclusive<i32>,
    y: std::ops::RangeInclusive<i32>,
    z: std::ops::RangeInclusive<i32>,
) -> Vec<VoxelCell> {
    let mut cells = Vec::new();
    for x in x {
        for y in y.clone() {
            for z in z.clone() {
                cells.push(VoxelCell::new(x, y, z));
            }
        }
    }
    cells
}

#[test]
fn voxel_data_is_canonical_and_queryable() {
    let _guard = voxel_test_lock();
    let _foundation = foundation();
    let voxel = VoxelData::new([[1, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 2, 3]], 2.0).unwrap();

    assert_eq!(voxel.cell_count(), 3);
    assert_eq!(voxel.voxel_size(), 2.0);
    assert_eq!(
        voxel.cells(),
        vec![
            VoxelCell::new(-1, 2, 3),
            VoxelCell::new(0, 0, 0),
            VoxelCell::new(1, 0, 0),
        ]
    );
    assert!(voxel.is_solid([0, 0, 0]));
    assert!(!voxel.is_solid([0, 1, 0]));
    assert_eq!(voxel.bounds().lower_bound, Vec3::new(-3.0, -1.0, -1.0));
    assert_eq!(voxel.bounds().upper_bound, Vec3::new(3.0, 5.0, 7.0));

    assert!(VoxelData::new(Vec::<VoxelCell>::new(), 1.0).is_err());
    assert!(VoxelData::new([[0, 0, 0]], f32::NAN).is_err());
}

#[test]
fn voxel_data_origin_offsets_geometry_and_round_trips() {
    let _guard = voxel_test_lock();
    let _foundation = foundation();
    let voxel = VoxelData::new_with_origin([[0, 0, 0]], 0.25, [0.125, 0.125, 0.125]).unwrap();
    assert_eq!(voxel.origin(), Vec3::new(0.125, 0.125, 0.125));
    assert_eq!(voxel.bounds().lower_bound, Vec3::ZERO);
    assert_eq!(voxel.bounds().upper_bound, Vec3::new(0.25, 0.25, 0.25));
}

#[test]
fn arbitrarily_rotated_voxel_bodies_generate_contacts() {
    let _guard = voxel_test_lock();
    let mut world = world_with_gravity([0.0, 0.0, 0.0]);

    let static_body = world.create_body(foundation().body_def()).unwrap();
    world
        .create_voxel_shape(
            static_body,
            &foundation().shape_def(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();

    let half_angle = std::f32::consts::FRAC_PI_8;
    let rotation = Quat::new(Vec3::new(0.0, half_angle.sin(), 0.0), half_angle.cos());
    let dynamic_body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position([0.7, 0.0, 0.0])
                .rotation(rotation)
                .build()
                .unwrap(),
        )
        .unwrap();
    let shape = world
        .create_voxel_shape(
            dynamic_body,
            &foundation()
                .shape_def_builder()
                .density(1.0)
                .build()
                .unwrap(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();

    world.step(1.0 / 60.0, 4).unwrap();
    assert_eq!(world.shape_type(shape).unwrap(), ShapeType::Voxel);
    assert!(!world.body_contacts(dynamic_body).unwrap().is_empty());

    let counters = world.counters().unwrap().voxel;
    assert_eq!(counters.voxel_voxel_calls, 1, "{counters:#?}");
    assert_eq!(counters.voxel_convex_calls, 0, "{counters:#?}");
    assert_eq!(counters.query_calls, 4, "{counters:#?}");
    assert_eq!(counters.count_fill_rescans, 0, "{counters:#?}");
    assert_eq!(counters.obb_tests, 0, "{counters:#?}");
    assert_eq!(counters.convex_leaf_tests, 0, "{counters:#?}");
    assert_eq!(counters.hull_sat_calls, 0, "{counters:#?}");
    assert_eq!(counters.raw_contact_points, 0, "{counters:#?}");
    assert_eq!(counters.patch_visits, 1, "{counters:#?}");
    assert_eq!(counters.patch_unique_keys, 1, "{counters:#?}");
    assert_eq!(counters.pseudo_sat_calls, 1, "{counters:#?}");
    assert_eq!(counters.selected_patch_keys, 1, "{counters:#?}");
    assert_eq!(counters.emitted_points, 4, "{counters:#?}");
    assert_eq!(counters.scalar_contacts, 0, "{counters:#?}");
    assert_eq!(counters.single_manifold_contacts, 1, "{counters:#?}");
}

#[test]
fn voxel_contact_migrates_between_wide_and_scalar_solver_lanes() {
    let _guard = voxel_test_lock();
    let mut world = world_with_gravity([0.0, 0.0, 0.0]);

    let mut corner = filled_box(-2..=2, 0..=0, -2..=2);
    corner.extend(filled_box(0..=0, 0..=2, -2..=2));
    let static_body = world.create_body(foundation().body_def()).unwrap();
    world
        .create_voxel_shape(
            static_body,
            &foundation().shape_def(),
            VoxelData::new(corner, 1.0).unwrap(),
        )
        .unwrap();

    let dynamic_body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position([1.2, 1.0, 0.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    world
        .create_voxel_shape(
            dynamic_body,
            &foundation()
                .shape_def_builder()
                .density(1.0)
                .build()
                .unwrap(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();

    world.step(1.0 / 60.0, 4).unwrap();
    let flat = world.counters().unwrap().voxel;
    assert_eq!(flat.single_manifold_contacts, 1, "{flat:#?}");
    assert_eq!(flat.scalar_contacts, 0, "{flat:#?}");
    assert_eq!(flat.solver_class_changes, 0, "{flat:#?}");

    world
        .set_body_transform(dynamic_body, [0.9, 0.9, 0.0], Quat::IDENTITY)
        .unwrap();
    world.step(1.0 / 60.0, 4).unwrap();
    let corner = world.counters().unwrap().voxel;
    assert_eq!(corner.single_manifold_contacts, 0, "{corner:#?}");
    assert_eq!(corner.scalar_contacts, 1, "{corner:#?}");
    assert_eq!(corner.solver_class_changes, 1, "{corner:#?}");

    world
        .set_body_transform(dynamic_body, [1.2, 1.0, 0.0], Quat::IDENTITY)
        .unwrap();
    world.step(1.0 / 60.0, 4).unwrap();
    let flat_again = world.counters().unwrap().voxel;
    assert_eq!(flat_again.single_manifold_contacts, 1, "{flat_again:#?}");
    assert_eq!(flat_again.scalar_contacts, 0, "{flat_again:#?}");
    assert_eq!(flat_again.solver_class_changes, 1, "{flat_again:#?}");
}

#[test]
fn overflow_voxel_contact_refreshes_its_scalar_manifold_budget() {
    let _guard = voxel_test_lock();
    let mut world = world_with_gravity([0.0, 0.0, 0.0]);

    let mut corner = filled_box(-2..=2, 0..=0, -2..=2);
    corner.extend(filled_box(0..=0, 0..=2, -2..=2));

    // A dynamic body can occupy the 22 ordinary static-contact colors once.
    // The 23rd identical contact is therefore stored in the overflow color.
    for _ in 0..23 {
        let static_body = world.create_body(foundation().body_def()).unwrap();
        world
            .create_voxel_shape(
                static_body,
                &foundation().shape_def(),
                VoxelData::new(corner.clone(), 1.0).unwrap(),
            )
            .unwrap();
    }

    let dynamic_body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position([1.2, 1.0, 0.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    world
        .create_voxel_shape(
            dynamic_body,
            &foundation()
                .shape_def_builder()
                .density(1.0)
                .build()
                .unwrap(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();

    world.step(1.0 / 60.0, 4).unwrap();
    let flat = world.counters().unwrap().voxel;
    assert_eq!(flat.single_manifold_contacts, 23, "{flat:#?}");
    assert_eq!(flat.scalar_contacts, 0, "{flat:#?}");

    world
        .set_body_transform(dynamic_body, [0.9, 0.9, 0.0], Quat::IDENTITY)
        .unwrap();
    world.step(1.0 / 60.0, 4).unwrap();

    let corner = world.counters().unwrap().voxel;
    assert_eq!(corner.single_manifold_contacts, 0, "{corner:#?}");
    assert_eq!(corner.scalar_contacts, 23, "{corner:#?}");
    // 22 contacts migrate from ordinary wide colors. The overflow contact
    // was already in scalar storage and only needs its budget refreshed.
    assert_eq!(corner.solver_class_changes, 22, "{corner:#?}");
}

#[test]
fn embedded_convex_uses_one_deterministic_escape_contact() {
    let _guard = voxel_test_lock();
    let mut world = world_with_gravity([0.0, 0.0, 0.0]);

    let voxel_body = world.create_body(foundation().body_def()).unwrap();
    world
        .create_voxel_shape(
            voxel_body,
            &foundation().shape_def(),
            VoxelData::new(filled_box(-1..=1, -1..=1, -1..=1), 1.0).unwrap(),
        )
        .unwrap();

    let convex_body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .build()
                .unwrap(),
        )
        .unwrap();
    world
        .create_hull_shape(
            convex_body,
            &foundation()
                .shape_def_builder()
                .density(1.0)
                .build()
                .unwrap(),
            &BoxHull::new(0.2, 0.2, 0.2).unwrap(),
        )
        .unwrap();

    world.step(1.0 / 60.0, 1).unwrap();

    assert!(!world.body_contacts(convex_body).unwrap().is_empty());
    let counters = world.counters().unwrap().voxel;
    assert_eq!(counters.voxel_convex_calls, 1, "{counters:#?}");
    assert!(counters.obb_tests > 0, "{counters:#?}");
    assert_eq!(counters.hull_sat_calls, 0, "{counters:#?}");
    assert!(counters.surface_rejects > 0, "{counters:#?}");
    assert_eq!(counters.deep_overlap_fallbacks, 1, "{counters:#?}");
    assert_eq!(counters.emitted_points, 1, "{counters:#?}");
}

#[test]
fn non_box_voxel_hull_contacts_keep_the_generic_sat_fallback() {
    let _guard = voxel_test_lock();
    let mut world = world_with_gravity([0.0, 0.0, 0.0]);

    let voxel_body = world.create_body(foundation().body_def()).unwrap();
    world
        .create_voxel_shape(
            voxel_body,
            &foundation().shape_def(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();

    let hull = Hull::from_points(
        [
            Vec3::new(-0.3, -0.3, -0.3),
            Vec3::new(0.3, -0.3, -0.3),
            Vec3::new(0.0, 0.3, -0.3),
            Vec3::new(0.0, 0.0, 0.3),
        ],
        4,
    )
    .unwrap();
    let hull_body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position([0.35, 0.0, 0.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    world
        .create_created_hull_shape(
            hull_body,
            &foundation()
                .shape_def_builder()
                .density(1.0)
                .build()
                .unwrap(),
            &hull,
        )
        .unwrap();

    world.step(1.0 / 60.0, 1).unwrap();

    assert!(!world.body_contacts(hull_body).unwrap().is_empty());
    let counters = world.counters().unwrap().voxel;
    assert_eq!(counters.voxel_convex_calls, 1, "{counters:#?}");
    assert_eq!(counters.obb_tests, 0, "{counters:#?}");
    assert!(counters.hull_sat_calls > 0, "{counters:#?}");
}

#[test]
fn voxel_recording_replays_shape_creation() {
    let _guard = voxel_test_lock();
    let mut world = default_world();
    let mut recording = Recording::new().unwrap();
    {
        let mut session = world.record(&mut recording).unwrap();
        let world = session.world();
        let body = world
            .create_body(
                foundation()
                    .body_def_builder()
                    .body_type(BodyType::Dynamic)
                    .position([0.0, 2.0, 0.0])
                    .build()
                    .unwrap(),
            )
            .unwrap();
        world
            .create_voxel_shape(
                body,
                &foundation()
                    .shape_def_builder()
                    .density(1.0)
                    .build()
                    .unwrap(),
                VoxelData::new_with_origin(
                    filled_box(-1..=1, -1..=1, -1..=1),
                    0.5,
                    [0.25, 0.25, 0.25],
                )
                .unwrap(),
            )
            .unwrap();
        for _ in 0..4 {
            world.step(1.0 / 60.0, 4).unwrap();
        }
        session.finish().unwrap();
    }
    let bytes = recording.bytes().unwrap();
    drop(world);
    assert!(foundation().validate_replay(bytes, 1).unwrap());
    let mut player = foundation().create_replay_player(bytes, 1).unwrap();
    player.seek_frame(player.frame_count().unwrap()).unwrap();
    assert!(!player.has_diverged().unwrap());
}

#[test]
fn voxel_collision_has_no_legacy_8192_candidate_ceiling() {
    let _guard = voxel_test_lock();
    const SIDE: i32 = 92;
    let slab = filled_box(0..=SIDE - 1, 0..=0, 0..=SIDE - 1);
    assert!(slab.len() > 8192);

    let mut world = world_with_gravity([0.0, 0.0, 0.0]);
    let ground = world.create_body(foundation().body_def()).unwrap();
    world
        .create_voxel_shape(
            ground,
            &foundation().shape_def(),
            VoxelData::new(slab.clone(), 1.0).unwrap(),
        )
        .unwrap();

    let body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position([0.0, 0.9, 0.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    world
        .create_voxel_shape(
            body,
            &foundation()
                .shape_def_builder()
                .density(1.0)
                .build()
                .unwrap(),
            VoxelData::new(slab, 1.0).unwrap(),
        )
        .unwrap();

    world.step(1.0 / 60.0, 1).unwrap();
    assert!(!world.body_contacts(body).unwrap().is_empty());
}

#[test]
fn nonquiescent_voxel_world_snapshot_resumes_exactly() {
    let _guard = voxel_test_lock();
    let mut original = world_with_gravity([0.0, -10.0, 0.0]);
    let ground = original.create_body(foundation().body_def()).unwrap();
    original
        .create_hull_shape(
            ground,
            &foundation().shape_def(),
            &BoxHull::new(10.0, 0.1, 10.0).unwrap(),
        )
        .unwrap();
    let cells = filled_box(0..=2, 0..=2, 0..=2);
    let mut bodies = Vec::new();
    for position in [[0.0, 0.6, 0.0], [0.1, 1.4, 0.05]] {
        let body = original
            .create_body(
                foundation()
                    .body_def_builder()
                    .body_type(BodyType::Dynamic)
                    .position(position)
                    .rotation(Quat::IDENTITY)
                    .build()
                    .unwrap(),
            )
            .unwrap();
        original
            .create_voxel_shape(
                body,
                &foundation()
                    .shape_def_builder()
                    .density(1.0)
                    .friction(0.6)
                    .build()
                    .unwrap(),
                VoxelData::new_with_origin(cells.clone(), 0.25, [0.125; 3]).unwrap(),
            )
            .unwrap();
        bodies.push(body);
    }
    for _ in 0..20 {
        original.step(1.0 / 60.0, 8).unwrap();
    }

    let body_snapshot_ids: Vec<BodySnapshotId> = bodies
        .iter()
        .map(|&body| original.body_snapshot_id(body).unwrap())
        .collect();
    let image = original.save_state().unwrap();
    let mut restored = default_world();
    restored.load_state(&image).unwrap();
    let restored_bodies: Vec<_> = body_snapshot_ids
        .into_iter()
        .map(|body| restored.resolve_body_snapshot_id(body).unwrap())
        .collect();

    for (&a, &b) in bodies.iter().zip(&restored_bodies) {
        assert_eq!(original.body_transform(a), restored.body_transform(b));
        assert_eq!(
            original.body_linear_velocity(a),
            restored.body_linear_velocity(b)
        );
        assert_eq!(
            original.body_angular_velocity(a),
            restored.body_angular_velocity(b)
        );
        let contacts_a = original.body_contacts(a).unwrap();
        let contacts_b = restored.body_contacts(b).unwrap();
        assert_eq!(contacts_a.len(), contacts_b.len());
        for (contact_a, contact_b) in contacts_a.iter().zip(&contacts_b) {
            assert_eq!(contact_a.manifolds, contact_b.manifolds);
        }
    }

    for _ in 0..10 {
        original.step(1.0 / 60.0, 8).unwrap();
        restored.step(1.0 / 60.0, 8).unwrap();
        for (&a, &b) in bodies.iter().zip(&restored_bodies) {
            assert_eq!(original.body_transform(a), restored.body_transform(b));
            assert_eq!(
                original.body_linear_velocity(a),
                restored.body_linear_velocity(b)
            );
            assert_eq!(
                original.body_angular_velocity(a),
                restored.body_angular_velocity(b)
            );
        }
    }
}
