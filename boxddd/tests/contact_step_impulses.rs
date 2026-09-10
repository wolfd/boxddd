use boxddd::{
    BodyId, BodyType, BoxHull, ContactBuffer, Foundation, MeshData, Quat, ShapeDef, TaskSystem,
    Vec3, World,
};

type V = [f64; 3];
const DT: f32 = 1.0 / 60.0;

fn v(value: Vec3) -> V {
    [value.x.into(), value.y.into(), value.z.into()]
}

fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}

fn scale(a: V, s: f64) -> V {
    a.map(|x| x * s)
}

fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: V) -> f64 {
    a.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn foundation() -> &'static Foundation {
    Foundation::initialize_default().unwrap()
}

fn world(workers: u32) -> World {
    foundation()
        .create_world(
            foundation()
                .world_def_builder()
                .gravity([0.0, -10.0, 0.0])
                .worker_count(workers)
                .task_system(TaskSystem::blocking_threads().unwrap())
                .build()
                .unwrap(),
        )
        .unwrap()
}

fn support(world: &mut World, shape: &ShapeDef, mesh: bool, moving: bool) -> BodyId {
    let body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(if moving {
                    BodyType::Kinematic
                } else {
                    BodyType::Static
                })
                .linear_velocity(if moving { [0.05, 0.0, 0.0] } else { [0.0; 3] })
                .build()
                .unwrap(),
        )
        .unwrap();
    if mesh {
        world
            .create_mesh_shape(
                body,
                shape,
                MeshData::box_mesh([0.0, -0.5, 0.0], [10.0, 0.5, 10.0], true).unwrap(),
                [1.0; 3],
            )
            .unwrap();
    } else {
        world
            .create_hull_shape(
                body,
                shape,
                &BoxHull::offset(10.0, 0.5, 10.0, [0.0, -0.5, 0.0]).unwrap(),
            )
            .unwrap();
    }
    body
}

fn cube(world: &mut World, shape: &ShapeDef, tiled: bool, position: [f32; 3]) -> BodyId {
    let body = world
        .create_body(
            foundation()
                .body_def_builder()
                .body_type(BodyType::Dynamic)
                .position(position)
                .rotation(Quat::new(Vec3::new(0.0, 0.0, 0.12467473), 0.9921977))
                .linear_velocity([2.0, -3.0, 0.4])
                .angular_velocity([0.1, 1.0, 2.0])
                .build()
                .unwrap(),
        )
        .unwrap();
    let count = if tiled { 4 } else { 1 };
    let size = 1.0 / count as f32;
    for x in 0..count {
        for y in 0..count {
            for z in 0..count {
                let center = [x, y, z].map(|n| (n as f32 + 0.5) * size - 0.5);
                world
                    .create_hull_shape(
                        body,
                        shape,
                        &BoxHull::offset(size * 0.5, size * 0.5, size * 0.5, center).unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    body
}

fn momentum_step(world: &mut World, body: BodyId, substeps: i32) -> usize {
    let mass = world.body_mass_data(body).unwrap();
    let inertia = f64::from(mass.inertia.cx.x);
    assert!((inertia - f64::from(mass.inertia.cy.y)).abs() < 1.0e-6);
    assert!((inertia - f64::from(mass.inertia.cz.z)).abs() < 1.0e-6);
    let before = v(world.body_linear_velocity(body).unwrap());
    let spin_before = v(world.body_angular_velocity(body).unwrap());
    world.step(DT, substeps).unwrap();
    let after = v(world.body_linear_velocity(body).unwrap());
    let spin_after = v(world.body_angular_velocity(body).unwrap());
    let expected = scale(
        add(
            add(after, scale(before, -1.0)),
            [0.0, 10.0 * f64::from(DT), 0.0],
        ),
        f64::from(mass.mass),
    );
    let expected_angular = scale(add(spin_after, scale(spin_before, -1.0)), inertia);
    let mut contacts = ContactBuffer::new();
    world.body_contacts_buffered(body, &mut contacts).unwrap();
    let mut linear = [0.0; 3];
    let mut angular = [0.0; 3];
    let mut scale_linear = 0.0;
    let mut scale_angular = 0.0;
    for contact in contacts.iter() {
        let ours_a = world.shape_body(contact.shape_id_a).unwrap() == body;
        let sign = if ours_a { -1.0 } else { 1.0 };
        for manifold in contact.manifolds {
            let step = &manifold.step_impulses;
            assert!(step.is_current(world.step_index().unwrap()));
            for (i, point) in manifold.points().iter().enumerate() {
                let impulse = scale(
                    v(manifold.normal),
                    sign * f64::from(step.normal_impulses[i]),
                );
                let anchor = v(if ours_a {
                    point.anchor_a
                } else {
                    point.anchor_b
                });
                let torque = cross(anchor, impulse);
                linear = add(linear, impulse);
                angular = add(angular, torque);
                scale_linear += norm(impulse);
                scale_angular += norm(torque);
            }
            let friction = scale(v(step.friction_impulse), sign);
            let center = v(if ours_a {
                step.friction_anchor_a
            } else {
                step.friction_anchor_b
            });
            let torque = add(
                cross(center, friction),
                scale(v(step.angular_impulse), sign),
            );
            linear = add(linear, friction);
            angular = add(angular, torque);
            scale_linear += norm(friction);
            scale_angular += norm(torque);
        }
    }
    let linear_error = norm(add(linear, scale(expected, -1.0)));
    let angular_error = norm(add(angular, scale(expected_angular, -1.0)));
    assert!(
        linear_error < 2.0e-5 * (1.0 + scale_linear),
        "linear {linear_error}: {linear:?} != {expected:?}"
    );
    assert!(
        angular_error < 2.0e-5 * (1.0 + scale_angular),
        "angular {angular_error}: {angular:?} != {expected_angular:?}"
    );
    contacts.len()
}

#[test]
fn complete_step_impulses_balance_linear_and_angular_momentum() {
    for mesh in [false, true] {
        for moving in [false, true] {
            if mesh && moving {
                continue;
            }
            for support_first in [false, true] {
                for tiled in [false, true] {
                    for warm in [false, true] {
                        let mut world = world(4);
                        world.enable_sleeping(false).unwrap();
                        world.enable_warm_starting(warm).unwrap();
                        let mut shape = foundation()
                            .shape_def_builder()
                            .density(1.0)
                            .friction(0.5)
                            .restitution(0.6)
                            .build()
                            .unwrap();
                        shape.base_material.rolling_resistance = 0.03;
                        if support_first {
                            support(&mut world, &shape, mesh, moving);
                        }
                        let body = cube(&mut world, &shape, tiled, [0.0, 0.7, 0.0]);
                        if !support_first {
                            support(&mut world, &shape, mesh, moving);
                        }
                        let mut max_contacts = 0;
                        for substeps in [1, 8, 2, 4].into_iter().cycle().take(120) {
                            max_contacts =
                                max_contacts.max(momentum_step(&mut world, body, substeps));
                        }
                        assert!(max_contacts > 0);
                        if tiled && !mesh {
                            assert!(max_contacts > 12, "{max_contacts}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn complete_step_impulses_survive_snapshot_and_worker_changes() {
    let mut original = world(1);
    original.enable_sleeping(false).unwrap();
    let shape = foundation()
        .shape_def_builder()
        .density(1.0)
        .friction(0.5)
        .build()
        .unwrap();
    support(&mut original, &shape, true, false);
    let body = cube(&mut original, &shape, true, [0.0, 0.7, 0.0]);
    cube(&mut original, &shape, true, [0.0, 1.8, 0.0]);
    for _ in 0..20 {
        momentum_step(&mut original, body, 4);
    }
    let id = original.body_snapshot_id(body).unwrap();
    let image = original.save_state().unwrap();
    let mut restored = world(8);
    restored.load_state(&image).unwrap();
    let restored_body = restored.resolve_body_snapshot_id(id).unwrap();
    for substeps in [1, 4, 8, 2].into_iter().cycle().take(80) {
        momentum_step(&mut original, body, substeps);
        momentum_step(&mut restored, restored_body, substeps);
        assert_eq!(original.step_index(), restored.step_index());
        assert_eq!(
            original.body_transform(body),
            restored.body_transform(restored_body)
        );
        let a = original.body_contacts(body).unwrap();
        let b = restored.body_contacts(restored_body).unwrap();
        assert_eq!(a.len(), b.len());
        for (a, b) in a.iter().zip(&b) {
            assert_eq!(a.manifolds, b.manifolds);
        }
    }
}

#[test]
fn sleeping_contacts_do_not_report_stale_impulses_as_current() {
    let mut world = world(1);
    assert_eq!(world.step_index().unwrap(), 0);
    let shape = foundation()
        .shape_def_builder()
        .density(1.0)
        .build()
        .unwrap();
    support(&mut world, &shape, false, false);
    let body = cube(&mut world, &shape, false, [0.0, 0.5, 0.0]);
    for _ in 0..120 {
        world.step(DT, 4).unwrap();
    }
    world.set_body_awake(body, false).unwrap();
    let previous = world.step_index().unwrap();
    world.step(DT, 4).unwrap();
    assert_eq!(world.step_index().unwrap(), previous + 1);
    let contacts = world.body_contacts(body).unwrap();
    assert!(!contacts.is_empty());
    for contact in &contacts {
        for manifold in &contact.manifolds {
            assert!(
                !manifold
                    .step_impulses
                    .is_current(world.step_index().unwrap())
            );
        }
    }
    world.set_body_awake(body, true).unwrap();
    world.step(DT, 4).unwrap();
    for contact in world.body_contacts(body).unwrap() {
        for manifold in contact.manifolds {
            assert!(
                manifold
                    .step_impulses
                    .is_current(world.step_index().unwrap())
            );
        }
    }
}

#[test]
fn collision_only_step_invalidates_refreshed_impulses() {
    let mut world = world(1);
    world.enable_sleeping(false).unwrap();
    let shape = foundation()
        .shape_def_builder()
        .density(1.0)
        .build()
        .unwrap();
    support(&mut world, &shape, false, false);
    let body = cube(&mut world, &shape, false, [0.0, 0.5, 0.0]);
    world.step(DT, 4).unwrap();
    let step = world.step_index().unwrap();
    world.step(0.0, 4).unwrap();
    assert_eq!(world.step_index().unwrap(), step);
    let contacts = world.body_contacts(body).unwrap();
    assert!(!contacts.is_empty());
    for contact in contacts {
        for manifold in contact.manifolds {
            assert!(!manifold.step_impulses.is_current(step));
        }
    }
}

#[test]
fn material_changes_do_not_report_unapplied_rolling_impulses() {
    for tiled in [false, true] {
        let mut world = world(4);
        world.enable_sleeping(false).unwrap();
        world.set_contact_recycle_distance(0.0).unwrap();
        let mut shape = foundation()
            .shape_def_builder()
            .density(1.0)
            .friction(0.5)
            .build()
            .unwrap();
        shape.base_material.rolling_resistance = 0.2;
        let ground = support(&mut world, &shape, false, false);
        let body = cube(&mut world, &shape, tiled, [0.0, 0.5, 0.0]);
        for _ in 0..8 {
            momentum_step(&mut world, body, 4);
        }
        shape.base_material.rolling_resistance = 0.0;
        for owner in [ground, body] {
            for id in world.body_shapes(owner).unwrap() {
                world
                    .set_shape_surface_material(id, shape.base_material)
                    .unwrap();
            }
        }
        for substeps in [1, 8, 2, 4] {
            momentum_step(&mut world, body, substeps);
        }
    }
}
