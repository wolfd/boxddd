use boxddd::error::HandleKind;
use boxddd::{
    Aabb, BodyType, BoxHull, Capsule, Error, Filter, Foundation, QueryFilter, Recording,
    ShapeCastInput, ShapeProxy, ShapeType, Sphere, Vec3, VoxelCell, VoxelData, World,
};
use static_assertions::assert_not_impl_any;
use std::sync::{Mutex, MutexGuard, OnceLock};

assert_not_impl_any!(VoxelData: Send, Sync);

const DT: f32 = 1.0 / 60.0;

fn voxel_test_lock() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn foundation() -> &'static Foundation {
    Foundation::initialize_default().unwrap()
}

fn default_world() -> World {
    let foundation = foundation();
    foundation.create_world(foundation.world_def()).unwrap()
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

fn is_invalid_value(error: &Error) -> bool {
    matches!(error, Error::InvalidValue { .. })
}

fn is_stale_shape(error: &Error) -> bool {
    matches!(
        error,
        Error::StaleHandle {
            kind: HandleKind::Shape
        }
    )
}

fn is_foreign_shape(error: &Error) -> bool {
    matches!(
        error,
        Error::ForeignHandle {
            kind: HandleKind::Shape
        }
    )
}

#[test]
fn voxel_data_validation_and_canonicalization_are_public_contracts() {
    let _guard = voxel_test_lock();
    let _foundation = foundation();
    let expected = vec![
        VoxelCell::new(-2, 1, 3),
        VoxelCell::new(0, 0, 0),
        VoxelCell::new(1, -1, 2),
    ];
    let inputs = [
        vec![[1, -1, 2], [0, 0, 0], [-2, 1, 3], [1, -1, 2]],
        vec![[-2, 1, 3], [1, -1, 2], [0, 0, 0], [-2, 1, 3]],
        vec![[0, 0, 0], [-2, 1, 3], [1, -1, 2], [0, 0, 0]],
    ];

    for input in inputs {
        let voxel = VoxelData::new_with_origin(input, 0.25, [0.125, -0.25, 0.5]).unwrap();
        assert_eq!(voxel.cell_count(), 3);
        assert_eq!(voxel.cells(), expected);
        assert_eq!(voxel.voxel_size(), 0.25);
        assert_eq!(voxel.origin(), Vec3::new(0.125, -0.25, 0.5));
        assert!(voxel.is_solid([-2, 1, 3]));
        assert!(!voxel.is_solid([-1, 1, 3]));
        assert_eq!(
            voxel.bounds(),
            Aabb {
                lower_bound: Vec3::new(-0.5, -0.625, 0.375),
                upper_bound: Vec3::new(0.5, 0.125, 1.375),
            }
        );
    }

    assert!(is_invalid_value(
        &VoxelData::new(Vec::<VoxelCell>::new(), 1.0).unwrap_err()
    ));
    for size in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(is_invalid_value(
            &VoxelData::new([[0, 0, 0]], size).unwrap_err()
        ));
    }
    for origin in [
        [f32::NAN, 0.0, 0.0],
        [0.0, f32::INFINITY, 0.0],
        [0.0, 0.0, f32::NEG_INFINITY],
    ] {
        assert!(is_invalid_value(
            &VoxelData::new_with_origin([[0, 0, 0]], 1.0, origin).unwrap_err()
        ));
    }
}

#[test]
fn voxel_shape_owns_data_and_rejects_stale_foreign_and_replaced_handles() {
    let _guard = voxel_test_lock();
    let mut world = default_world();
    let body = world.create_body(foundation().body_def()).unwrap();
    let shape = {
        let voxel = VoxelData::new_with_origin([[0, 0, 0], [2, 0, 0]], 0.5, [0.25; 3]).unwrap();
        world
            .create_voxel_shape(body, &foundation().shape_def(), voxel)
            .unwrap()
    };

    {
        let view = world.shape_voxel(shape).unwrap();
        assert_eq!(
            view.cells(),
            [VoxelCell::new(0, 0, 0), VoxelCell::new(2, 0, 0)]
        );
        assert_eq!(view.origin(), Vec3::new(0.25, 0.25, 0.25));
    }
    let hits = world
        .overlap_aabb(
            Aabb {
                lower_bound: Vec3::new(-0.1, -0.1, -0.1),
                upper_bound: Vec3::new(1.6, 0.6, 0.6),
            },
            QueryFilter::default(),
        )
        .unwrap();
    assert!(hits.iter().any(|hit| hit.shape_id == shape));

    world.destroy_shape(shape, false).unwrap();
    assert!(is_stale_shape(&world.shape_voxel(shape).unwrap_err()));

    for i in 0..32 {
        let shape = world
            .create_voxel_shape(
                body,
                &foundation().shape_def(),
                VoxelData::new([[i, 0, 0]], 0.125).unwrap(),
            )
            .unwrap();
        assert!(world.shape_voxel(shape).unwrap().is_solid([i, 0, 0]));
        world.destroy_shape(shape, false).unwrap();
        assert!(is_stale_shape(&world.shape_voxel(shape).unwrap_err()));
    }

    let doomed_body = world.create_body(foundation().body_def()).unwrap();
    let doomed_shape = world
        .create_voxel_shape(
            doomed_body,
            &foundation().shape_def(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();
    world.destroy_body(doomed_body).unwrap();
    assert!(is_stale_shape(
        &world.shape_voxel(doomed_shape).unwrap_err()
    ));

    let replacement = world
        .create_voxel_shape(
            body,
            &foundation().shape_def(),
            VoxelData::new([[-1, 0, 0], [1, 0, 0]], 0.5).unwrap(),
        )
        .unwrap();
    world
        .set_shape_sphere(replacement, &Sphere::new(Vec3::ZERO, 0.4))
        .unwrap();
    assert_eq!(world.shape_type(replacement).unwrap(), ShapeType::Sphere);
    assert!(is_invalid_value(
        &world.shape_voxel(replacement).unwrap_err()
    ));
    world
        .set_shape_capsule(
            replacement,
            &Capsule::new([-0.25, 0.0, 0.0], [0.25, 0.0, 0.0], 0.2),
        )
        .unwrap();
    assert_eq!(world.shape_type(replacement).unwrap(), ShapeType::Capsule);
    assert!(
        world
            .shape_cast_ray(replacement, [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0])
            .unwrap()
            .is_some()
    );
    world.step(DT, 1).unwrap();

    let mut other = default_world();
    let other_body = other.create_body(foundation().body_def()).unwrap();
    let foreign = other
        .create_voxel_shape(
            other_body,
            &foundation().shape_def(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();
    assert!(is_foreign_shape(&world.shape_voxel(foreign).unwrap_err()));
}

#[test]
fn voxel_shapes_participate_in_every_safe_query_and_filter_path() {
    let _guard = voxel_test_lock();
    let mut world = default_world();
    let body = world.create_body(foundation().body_def()).unwrap();
    let shape = world
        .create_voxel_shape(
            body,
            &foundation()
                .shape_def_builder()
                .filter(Filter {
                    category_bits: 0b10,
                    mask_bits: u64::MAX,
                    group_index: 0,
                })
                .build()
                .unwrap(),
            VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
        )
        .unwrap();
    let proxy = ShapeProxy::sphere(0.2).unwrap();
    let included = QueryFilter::default().mask_bits(0b10);
    let excluded = QueryFilter::default().mask_bits(0b100);

    assert!(
        world
            .overlap_shape(Vec3::ZERO, &proxy, included)
            .unwrap()
            .iter()
            .any(|hit| hit.shape_id == shape)
    );
    assert!(
        world
            .overlap_shape(Vec3::ZERO, &proxy, excluded)
            .unwrap()
            .is_empty()
    );
    assert!(
        world
            .cast_ray([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], included)
            .unwrap()
            .iter()
            .any(|hit| hit.shape_id == shape)
    );
    assert!(
        world
            .cast_ray([-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], excluded)
            .unwrap()
            .is_empty()
    );
    assert!(
        world
            .cast_shape(
                [-2.0, 0.0, 0.0],
                ShapeCastInput::new(proxy.clone(), [4.0, 0.0, 0.0]).unwrap(),
                included,
            )
            .unwrap()
            .iter()
            .any(|hit| hit.shape_id == shape)
    );

    assert_eq!(
        world
            .body_cast_ray(body, [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0], included)
            .unwrap()
            .unwrap()
            .shape_id,
        shape
    );
    assert_eq!(
        world
            .body_cast_shape(
                body,
                [-2.0, 0.0, 0.0],
                ShapeCastInput::new(proxy.clone(), [4.0, 0.0, 0.0]).unwrap(),
                included,
            )
            .unwrap()
            .unwrap()
            .shape_id,
        shape
    );
    assert!(
        world
            .body_overlap_shape(body, Vec3::ZERO, &proxy, included)
            .unwrap()
    );
    assert!(
        !world
            .body_overlap_shape(body, Vec3::ZERO, &proxy, excluded)
            .unwrap()
    );
    assert!(
        world
            .shape_cast_ray(shape, [-2.0, 0.0, 0.0], [4.0, 0.0, 0.0])
            .unwrap()
            .is_some()
    );

    let touching_mover = Capsule::new([-0.6, -0.2, 0.0], [-0.6, 0.2, 0.0], 0.2);
    assert!(
        world
            .collide_mover(Vec3::ZERO, &touching_mover, included)
            .unwrap()
            .iter()
            .any(|plane| plane.shape_id == shape)
    );
    assert!(
        world
            .collide_mover(Vec3::ZERO, &touching_mover, excluded)
            .unwrap()
            .is_empty()
    );

    let cast_mover = Capsule::new([-2.0, -0.2, 0.0], [-2.0, 0.2, 0.0], 0.2);
    assert!(
        world
            .cast_mover(Vec3::ZERO, &cast_mover, [4.0, 0.0, 0.0], included)
            .unwrap()
            < 1.0
    );
}

fn contact_normal_from_static_to_dynamic(reverse_creation: bool, voxel_voxel: bool) -> Vec3 {
    let mut world = world_with_gravity(Vec3::ZERO);

    let make_dynamic = |world: &mut World| {
        let body = world
            .create_body(
                foundation()
                    .body_def_builder()
                    .body_type(BodyType::Dynamic)
                    .position([0.7, 0.0, 0.0])
                    .build()
                    .unwrap(),
            )
            .unwrap();
        let shape = if voxel_voxel {
            world
                .create_voxel_shape(
                    body,
                    &foundation()
                        .shape_def_builder()
                        .density(1.0)
                        .build()
                        .unwrap(),
                    VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
                )
                .unwrap()
        } else {
            world
                .create_hull_shape(
                    body,
                    &foundation()
                        .shape_def_builder()
                        .density(1.0)
                        .build()
                        .unwrap(),
                    &BoxHull::cube(0.5).unwrap(),
                )
                .unwrap()
        };
        (body, shape)
    };
    let make_static = |world: &mut World| {
        let body = world.create_body(foundation().body_def()).unwrap();
        let shape = world
            .create_voxel_shape(
                body,
                &foundation().shape_def(),
                VoxelData::new([[0, 0, 0]], 1.0).unwrap(),
            )
            .unwrap();
        (body, shape)
    };

    let (static_body, static_shape, dynamic_body, dynamic_shape) = if reverse_creation {
        let (dynamic_body, dynamic_shape) = make_dynamic(&mut world);
        let (static_body, static_shape) = make_static(&mut world);
        (static_body, static_shape, dynamic_body, dynamic_shape)
    } else {
        let (static_body, static_shape) = make_static(&mut world);
        let (dynamic_body, dynamic_shape) = make_dynamic(&mut world);
        (static_body, static_shape, dynamic_body, dynamic_shape)
    };

    world.step(DT, 1).unwrap();
    let contacts = world.body_contacts(dynamic_body).unwrap();
    let contact = contacts
        .iter()
        .find(|contact| {
            (contact.shape_id_a == static_shape && contact.shape_id_b == dynamic_shape)
                || (contact.shape_id_a == dynamic_shape && contact.shape_id_b == static_shape)
        })
        .expect("expected the physical pair to produce a contact");
    assert_eq!(world.shape_body(static_shape).unwrap(), static_body);
    assert_eq!(world.shape_body(dynamic_shape).unwrap(), dynamic_body);
    assert_eq!(world.shape_type(static_shape).unwrap(), ShapeType::Voxel);
    assert_eq!(
        world.shape_type(dynamic_shape).unwrap(),
        if voxel_voxel {
            ShapeType::Voxel
        } else {
            ShapeType::Hull
        }
    );
    assert_eq!(contact.manifolds.len(), 1);
    assert!(contact.manifolds[0].point_count > 0);

    let normal = contact.manifolds[0].normal;
    if contact.shape_id_a == static_shape {
        normal
    } else {
        Vec3::new(-normal.x, -normal.y, -normal.z)
    }
}

#[test]
fn voxel_contact_normals_follow_shape_ownership_across_creation_order() {
    let _guard = voxel_test_lock();
    for voxel_voxel in [false, true] {
        let forward = contact_normal_from_static_to_dynamic(false, voxel_voxel);
        let reversed = contact_normal_from_static_to_dynamic(true, voxel_voxel);
        assert!(forward.x > 0.8, "unexpected physical normal {forward:?}");
        assert!(reversed.x > 0.8, "unexpected physical normal {reversed:?}");
        assert!((forward.x - reversed.x).abs() < 1.0e-5);
        assert!((forward.y - reversed.y).abs() < 1.0e-5);
        assert!((forward.z - reversed.z).abs() < 1.0e-5);
    }
}

#[test]
fn loaded_worlds_own_and_resume_voxel_geometry_without_rust_sidecars() {
    let _guard = voxel_test_lock();
    let (image, source_body, source_shape, expected_cells) = {
        let mut source = world_with_gravity([0.0, -10.0, 0.0]);
        let ground = source.create_body(foundation().body_def()).unwrap();
        source
            .create_hull_shape(
                ground,
                &foundation().shape_def(),
                &BoxHull::new(5.0, 0.1, 5.0).unwrap(),
            )
            .unwrap();
        let body = source
            .create_body(
                foundation()
                    .body_def_builder()
                    .body_type(BodyType::Dynamic)
                    .position([0.0, 0.3, 0.0])
                    .build()
                    .unwrap(),
            )
            .unwrap();
        let old_shape = source
            .create_voxel_shape(
                body,
                &foundation()
                    .shape_def_builder()
                    .density(1.0)
                    .build()
                    .unwrap(),
                VoxelData::new([[0, 0, 0]], 0.5).unwrap(),
            )
            .unwrap();
        source.destroy_shape(old_shape, true).unwrap();

        let expected_cells = vec![
            VoxelCell::new(-2, 0, 0),
            VoxelCell::new(0, 0, 0),
            VoxelCell::new(0, 1, 0),
            VoxelCell::new(2, 0, 0),
        ];
        let shape = source
            .create_voxel_shape(
                body,
                &foundation()
                    .shape_def_builder()
                    .density(1.0)
                    .friction(0.6)
                    .build()
                    .unwrap(),
                VoxelData::new_with_origin(expected_cells.clone(), 0.25, [0.125; 3]).unwrap(),
            )
            .unwrap();
        for _ in 0..3 {
            source.step(DT, 4).unwrap();
        }
        source
            .set_body_linear_velocity(body, [0.4, -0.2, 0.1])
            .unwrap();
        source
            .set_body_angular_velocity(body, [0.2, -0.1, 0.3])
            .unwrap();
        (
            source.save_state().unwrap(),
            source.body_snapshot_id(body).unwrap(),
            source.shape_snapshot_id(shape).unwrap(),
            expected_cells,
        )
    };

    let mut first = default_world();
    let mut second = default_world();
    first.load_state(&image).unwrap();
    second.load_state(&image).unwrap();
    drop(image);

    let first_body = first.resolve_body_snapshot_id(source_body).unwrap();
    let second_body = second.resolve_body_snapshot_id(source_body).unwrap();
    let first_shape = first.resolve_shape_snapshot_id(source_shape).unwrap();
    let second_shape = second.resolve_shape_snapshot_id(source_shape).unwrap();

    for (world, shape) in [(&first, first_shape), (&second, second_shape)] {
        let voxel = world.shape_voxel(shape).unwrap();
        assert_eq!(voxel.cells(), expected_cells);
        assert_eq!(voxel.voxel_size(), 0.25);
        assert_eq!(voxel.origin(), Vec3::new(0.125, 0.125, 0.125));
        assert!(voxel.is_solid([0, 1, 0]));
        assert!(!voxel.is_solid([1, 1, 0]));
        assert_eq!(
            voxel.bounds(),
            Aabb {
                lower_bound: Vec3::new(-0.5, 0.0, 0.0),
                upper_bound: Vec3::new(0.75, 0.5, 0.25),
            }
        );
    }

    assert_eq!(
        first.body_transform(first_body),
        second.body_transform(second_body)
    );
    assert_eq!(
        first.body_linear_velocity(first_body),
        second.body_linear_velocity(second_body)
    );
    assert_eq!(
        first.body_angular_velocity(first_body),
        second.body_angular_velocity(second_body)
    );
    let first_contacts = first.body_contacts(first_body).unwrap();
    let second_contacts = second.body_contacts(second_body).unwrap();
    assert_eq!(first_contacts.len(), second_contacts.len());
    for (a, b) in first_contacts.iter().zip(&second_contacts) {
        assert_eq!(a.manifolds, b.manifolds);
    }

    for _ in 0..10 {
        first.step(DT, 4).unwrap();
        second.step(DT, 4).unwrap();
        assert_eq!(
            first.body_transform(first_body),
            second.body_transform(second_body)
        );
        assert_eq!(
            first.body_linear_velocity(first_body),
            second.body_linear_velocity(second_body)
        );
        assert_eq!(
            first.body_angular_velocity(first_body),
            second.body_angular_velocity(second_body)
        );
        let contacts_a = first.body_contacts(first_body).unwrap();
        let contacts_b = second.body_contacts(second_body).unwrap();
        assert_eq!(contacts_a.len(), contacts_b.len());
        for (a, b) in contacts_a.iter().zip(&contacts_b) {
            assert_eq!(a.manifolds, b.manifolds);
        }
    }
}

#[test]
fn voxel_recording_replays_offset_occupancy_and_query_metadata() {
    let _guard = voxel_test_lock();
    let mut world = default_world();
    let mut recording = Recording::new().unwrap();
    {
        let mut session = world.record(&mut recording).unwrap();
        let world = session.world();
        let body = world.create_body(foundation().body_def()).unwrap();
        let shape = world
            .create_voxel_shape(
                body,
                &foundation().shape_def(),
                VoxelData::new_with_origin([[-2, 0, 0], [0, 0, 0], [3, 1, 0]], 0.25, [0.125; 3])
                    .unwrap(),
            )
            .unwrap();
        let aabb = world.shape_aabb(shape).unwrap();
        for _ in 0..3 {
            let hits = world
                .overlap_aabb(aabb, QueryFilter::new().mask_bits(0b1).id(42))
                .unwrap();
            assert!(hits.iter().any(|hit| hit.shape_id == shape));
            world.step(DT, 2).unwrap();
        }
        session.finish().unwrap();
    }
    let bytes = recording.bytes().unwrap();
    drop(world);
    assert!(foundation().validate_replay(bytes, 1).unwrap());
    let mut player = foundation().create_replay_player(bytes, 1).unwrap();
    let mut saw_query = false;
    for frame in 0..=player.frame_count().unwrap() {
        player.seek_frame(frame).unwrap();
        for query_index in 0..player.frame_query_count().unwrap() {
            let query = player.frame_query(query_index).unwrap();
            if query.id == 42 {
                saw_query = true;
                assert_eq!(query.filter.mask_bits, 0b1);
                assert!(query.hit_count > 0);
                for hit_index in 0..query.hit_count {
                    assert!(
                        player
                            .frame_query_hit(query_index, hit_index)
                            .unwrap()
                            .shape_id
                            .slot_index()
                            > 0
                    );
                }
            }
        }
    }
    assert!(saw_query);
    assert!(!player.has_diverged().unwrap());
}
