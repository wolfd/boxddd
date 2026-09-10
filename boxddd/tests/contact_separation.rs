use boxddd::{BodyId, BodyType, BoxHull, Foundation, TaskSystem, World};

fn foundation() -> &'static Foundation {
    Foundation::initialize_default().unwrap()
}

fn world(workers: i32) -> World {
    let mut world = foundation()
        .create_world(
            foundation()
                .world_def_builder()
                .gravity([0.0, -10.0, 0.0])
                .worker_count(workers.try_into().unwrap())
                .task_system(TaskSystem::blocking_threads().unwrap())
                .build()
                .unwrap(),
        )
        .unwrap();
    world.enable_sleeping(false).unwrap();
    world
}

fn support(world: &mut World, moving: bool) -> BodyId {
    let body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(if moving {
                    BodyType::Kinematic
                } else {
                    BodyType::Static
                })
                .linear_velocity(if moving { [0.02, 0.0, 0.0] } else { [0.0; 3] })
                .build()
                .unwrap(),
        )
        .unwrap();
    world
        .create_hull_shape(
            body,
            &foundation().shape_def(),
            &BoxHull::offset(30.0, 0.5, 10.0, [0.0, -0.5, 0.0]).unwrap(),
        )
        .unwrap();
    body
}

fn scene(workers: i32, tiled: bool, support_first: bool, moving: bool) -> (World, Vec<BodyId>) {
    let mut world = world(workers);
    let mut bodies = Vec::new();
    if support_first {
        bodies.push(support(&mut world, moving));
    }
    for index in 0..4 {
        let body = world
            .create_body(
                foundation()
                    .body_def_builder()
                    .body_type(BodyType::Dynamic)
                    .position([index as f32 * 12.0 - 18.0, 0.2, 0.0])
                    .build()
                    .unwrap(),
            )
            .unwrap();
        let shape = foundation()
            .shape_def_builder()
            .density(1.0)
            .friction(0.5)
            .build()
            .unwrap();
        if tiled {
            for x in 0..8 {
                for z in 0..8 {
                    world
                        .create_hull_shape(
                            body,
                            &shape,
                            &BoxHull::offset(
                                0.45,
                                0.2,
                                0.45,
                                [x as f32 - 3.5, 0.0, z as f32 - 3.5],
                            )
                            .unwrap(),
                        )
                        .unwrap();
                }
            }
        } else {
            world
                .create_hull_shape(body, &shape, &BoxHull::new(4.0, 0.2, 4.0).unwrap())
                .unwrap();
        }
        bodies.push(body);
    }
    let upper = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position([0.0, 0.8, 0.0])
                .angular_velocity([0.0, 0.02, 0.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    world
        .create_hull_shape(
            upper,
            &foundation()
                .shape_def_builder()
                .density(1.0)
                .build()
                .unwrap(),
            &BoxHull::new(24.0, 0.4, 4.0).unwrap(),
        )
        .unwrap();
    bodies.push(upper);
    if !support_first {
        bodies.push(support(&mut world, moving));
    }
    (world, bodies)
}

fn assert_same(a: &World, ids_a: &[BodyId], b: &World, ids_b: &[BodyId]) {
    for (&a_id, &b_id) in ids_a.iter().zip(ids_b) {
        assert_eq!(a.body_transform(a_id), b.body_transform(b_id));
        assert_eq!(a.body_linear_velocity(a_id), b.body_linear_velocity(b_id));
        assert_eq!(a.body_angular_velocity(a_id), b.body_angular_velocity(b_id));
        assert_eq!(a.body_mass_data(a_id), b.body_mass_data(b_id));
        assert_eq!(a.body_awake(a_id), b.body_awake(b_id));
        let a_contacts = a.body_contacts(a_id).unwrap();
        let b_contacts = b.body_contacts(b_id).unwrap();
        assert_eq!(a_contacts.len(), b_contacts.len());
        for (a, b) in a_contacts.iter().zip(&b_contacts) {
            assert_eq!(a.manifolds, b.manifolds);
        }
    }
}

#[test]
fn contact_separation_survives_substep_changes_and_body_order() {
    for tiled in [false, true] {
        for support_first in [false, true] {
            for workers in [1, 4, 8] {
                let (mut serial, ids) = scene(1, tiled, support_first, false);
                let (mut parallel, parallel_ids) = scene(workers, tiled, support_first, false);
                for round in 0..3 {
                    let dt = if round == 1 { 1.0 / 90.0 } else { 1.0 / 60.0 };
                    for substeps in [1, 2, 8, 4, 1, 8] {
                        serial.step(dt, substeps).unwrap();
                        parallel.step(dt, substeps).unwrap();
                        assert_same(&serial, &ids, &parallel, &parallel_ids);
                    }
                }
                assert!(
                    ids.iter()
                        .any(|&id| serial.body_contacts(id).unwrap().len() > 1)
                );
            }
        }
    }
}

#[test]
fn moving_support_contacts_continue_after_snapshot_restore() {
    for tiled in [false, true] {
        let (mut serial, ids) = scene(1, tiled, false, true);
        for _ in 0..12 {
            serial.step(1.0 / 60.0, 8).unwrap();
        }
        let snapshot_ids: Vec<_> = ids
            .iter()
            .map(|&id| serial.body_snapshot_id(id).unwrap())
            .collect();
        let image = serial.save_state().unwrap();
        let mut restored = world(8);
        restored.load_state(&image).unwrap();
        let restored_ids: Vec<_> = snapshot_ids
            .into_iter()
            .map(|id| restored.resolve_body_snapshot_id(id).unwrap())
            .collect();
        for substeps in [1, 8, 2, 4, 8, 1].into_iter().cycle().take(36) {
            serial.step(1.0 / 60.0, substeps).unwrap();
            restored.step(1.0 / 60.0, substeps).unwrap();
            assert_same(&serial, &ids, &restored, &restored_ids);
        }
    }
}

#[test]
fn tiled_contacts_survive_tree_recycling_and_body_type_changes() {
    for workers in [1, 4, 8] {
        let (mut serial, mut ids) = scene(1, true, true, false);
        let (mut parallel, mut parallel_ids) = scene(workers, true, true, false);
        for round in 0..6 {
            for (world, ids) in [(&mut serial, &mut ids), (&mut parallel, &mut parallel_ids)] {
                world
                    .set_body_type(
                        ids[2],
                        [BodyType::Kinematic, BodyType::Static, BodyType::Dynamic][round % 3],
                    )
                    .unwrap();
                let shapes = world.body_shapes(ids[1]).unwrap();
                assert!(shapes.len() > 1);
                for shape in shapes.into_iter().step_by(3) {
                    world.destroy_shape(shape, true).unwrap();
                }
                for x in 0..8 {
                    world
                        .create_hull_shape(
                            ids[1],
                            &foundation()
                                .shape_def_builder()
                                .density(1.0)
                                .build()
                                .unwrap(),
                            &BoxHull::offset(
                                0.45,
                                0.2,
                                0.45,
                                [x as f32 - 3.5, 0.0, round as f32 - 2.5],
                            )
                            .unwrap(),
                        )
                        .unwrap();
                }
                let last = ids.len() - 1;
                world.destroy_body(ids[last]).unwrap();
                ids[last] = world
                    .create_body(
                        foundation()
                            .body_def_builder()
                            .body_type(BodyType::Dynamic)
                            .position([0.0, 0.7 + round as f32 * 0.05, 0.0])
                            .build()
                            .unwrap(),
                    )
                    .unwrap();
                world
                    .create_hull_shape(
                        ids[last],
                        &foundation()
                            .shape_def_builder()
                            .density(1.0)
                            .build()
                            .unwrap(),
                        &BoxHull::new(24.0, 0.4, 4.0).unwrap(),
                    )
                    .unwrap();
            }
            for _ in 0..8 {
                serial.step(1.0 / 60.0, 4).unwrap();
                parallel.step(1.0 / 60.0, 4).unwrap();
                assert_same(&serial, &ids, &parallel, &parallel_ids);
            }
            if round == 2 {
                let image = serial.save_state().unwrap();
                let keys: Vec<_> = ids
                    .iter()
                    .map(|&id| serial.body_snapshot_id(id).unwrap())
                    .collect();
                parallel.load_state(&image).unwrap();
                parallel_ids = keys
                    .into_iter()
                    .map(|key| parallel.resolve_body_snapshot_id(key).unwrap())
                    .collect();
                assert_same(&serial, &ids, &parallel, &parallel_ids);
            }
        }
        assert!(
            ids.iter()
                .any(|&id| serial.body_contacts(id).unwrap().len() > 1)
        );
    }
}
