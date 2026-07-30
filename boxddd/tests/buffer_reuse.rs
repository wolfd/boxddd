use boxddd::{
    Aabb, BodyDef, BodySleepBuffer, BodyType, BoxHull, ContactBuffer, ContactId,
    IslandCensusBuffer, Manifold, QueryFilter, ShapeDef, ShapeId, Sphere, World, WorldDef,
};
use std::collections::{HashMap, HashSet};

type ContactSnapshot = (ShapeId, ShapeId, Vec<Manifold>);

fn merged_contact_snapshots(buffers: &[&ContactBuffer]) -> HashMap<ContactId, ContactSnapshot> {
    let mut merged = HashMap::new();
    for contact in buffers.iter().flat_map(|buffer| buffer.iter()) {
        let snapshot = (
            contact.shape_id_a,
            contact.shape_id_b,
            contact.manifolds.to_vec(),
        );
        if let Some(previous) = merged.insert(contact.contact_id, snapshot.clone()) {
            assert_eq!(
                previous, snapshot,
                "both body queries must copy the same native contact"
            );
        }
    }
    merged
}

fn assert_world_contacts(
    world: &World,
    all: &mut ContactBuffer,
    expected: &HashMap<ContactId, ContactSnapshot>,
) {
    world.try_world_contacts_buffered(all).unwrap();
    assert_eq!(all.len(), expected.len(), "each native pair appears once");
    let mut seen = HashSet::new();
    for contact in all.iter() {
        assert!(seen.insert(contact.contact_id), "duplicate world contact");
        let snapshot = expected
            .get(&contact.contact_id)
            .expect("world contact must appear in a body query");
        assert_eq!(contact.shape_id_a, snapshot.0);
        assert_eq!(contact.shape_id_b, snapshot.1);
        assert_eq!(contact.manifolds, snapshot.2.as_slice());
    }
}

/// A dynamic box resting on a static ground slab after settling — the
/// minimal world with one live contact.
fn settled_box_on_ground() -> (World, boxddd::BodyId) {
    let mut world = World::new(WorldDef::default()).unwrap();
    let ground = world.create_body(BodyDef::builder().body_type(BodyType::Static).build());
    world.create_hull_shape(ground, &ShapeDef::default(), &BoxHull::new(10.0, 0.5, 10.0));
    let cube = world.create_body(
        BodyDef::builder()
            .body_type(BodyType::Dynamic)
            .position([0.0, 1.2, 0.0])
            .build(),
    );
    world.create_hull_shape(cube, &ShapeDef::default(), &BoxHull::new(0.5, 0.5, 0.5));
    for _ in 0..120 {
        world.step(1.0 / 60.0, 4);
    }
    (world, cube)
}

/// The buffered contacts query returns the same pairs and manifolds as the
/// allocating `try_body_contacts_into`, and refilling reuses its storage.
#[test]
fn buffered_contacts_match_the_allocating_query_and_reuse_storage() {
    let (world, cube) = settled_box_on_ground();

    let mut allocating = Vec::new();
    world.try_body_contacts_into(cube, &mut allocating).unwrap();
    assert!(!allocating.is_empty(), "the cube must rest in contact");

    let mut buf = ContactBuffer::new();
    world.try_body_contacts_buffered(cube, &mut buf).unwrap();
    assert_eq!(buf.len(), allocating.len());
    for (view, data) in buf.iter().zip(&allocating) {
        assert_eq!(view.contact_id, data.contact_id);
        assert_eq!(view.shape_id_a, data.shape_id_a);
        assert_eq!(view.shape_id_b, data.shape_id_b);
        assert_eq!(view.manifolds, data.manifolds.as_slice());
    }

    // Refill: same contents, no fresh growth beyond the warmed capacity.
    world.try_body_contacts_buffered(cube, &mut buf).unwrap();
    assert_eq!(buf.len(), allocating.len());
}

/// The sleep observers report a settled cube as sleep-qualified inside its
/// island, agree with each other, and refilling reuses the warm buffers.
#[test]
fn sleep_observers_report_a_settled_cube_and_reuse_storage() {
    let (world, cube) = settled_box_on_ground();

    let mut bodies = BodySleepBuffer::new();
    world.try_world_body_sleep_buffered(&mut bodies).unwrap();
    // The static ground and the dynamic cube.
    assert_eq!(bodies.len(), 2);
    let sample = *bodies
        .iter()
        .find(|s| s.body_id == cube)
        .expect("the cube must be sampled");
    assert!(
        sample.sleep_time >= 0.5,
        "120 settled steps must reach the engine time-to-sleep ({})",
        sample.sleep_time
    );
    assert!(sample.island_id >= 0, "a dynamic body is always in an island");
    let ground_sample = bodies.iter().find(|s| s.body_id != cube).unwrap();
    assert!(
        ground_sample.island_id < 0,
        "static bodies are never in islands"
    );

    let mut islands = IslandCensusBuffer::new();
    world.try_world_island_census_buffered(&mut islands).unwrap();
    assert_eq!(islands.len(), 1, "one dynamic body forms one island");
    let island = islands.entries()[0];
    assert_eq!(island.island_id, sample.island_id);
    assert_eq!(island.body_count, 1);
    assert_eq!(island.min_sleep_time, sample.sleep_time);
    assert_eq!(island.min_sleep_time_body, cube);
    assert_eq!(island.sleep_ready_count, 1);

    // Refill: same contents from the warm buffers.
    world.try_world_body_sleep_buffered(&mut bodies).unwrap();
    assert_eq!(bodies.len(), 2);
    world.try_world_island_census_buffered(&mut islands).unwrap();
    assert_eq!(islands.len(), 1);
    assert_eq!(islands.entries()[0], island);
}

#[test]
fn world_contact_buffer_visits_each_touching_pair_once() {
    let mut world = World::new(WorldDef::default()).unwrap();
    let ground = world.create_body(BodyDef::builder().body_type(BodyType::Static).build());
    world.create_hull_shape(ground, &ShapeDef::default(), &BoxHull::new(10.0, 0.5, 10.0));
    let lower = world.create_body(
        BodyDef::builder()
            .body_type(BodyType::Dynamic)
            .position([0.0, 1.2, 0.0])
            .build(),
    );
    world.create_hull_shape(lower, &ShapeDef::default(), &BoxHull::new(0.5, 0.5, 0.5));
    let upper = world.create_body(
        BodyDef::builder()
            .body_type(BodyType::Dynamic)
            .position([0.0, 2.2, 0.0])
            .build(),
    );
    world.create_hull_shape(upper, &ShapeDef::default(), &BoxHull::new(0.5, 0.5, 0.5));
    for _ in 0..120 {
        world.step(1.0 / 60.0, 4);
    }

    let mut lower_contacts = ContactBuffer::new();
    world
        .try_body_contacts_buffered(lower, &mut lower_contacts)
        .unwrap();
    let mut upper_contacts = ContactBuffer::new();
    world
        .try_body_contacts_buffered(upper, &mut upper_contacts)
        .unwrap();
    let body_count = lower_contacts.len() + upper_contacts.len();
    let expected = merged_contact_snapshots(&[&lower_contacts, &upper_contacts]);
    assert!(
        body_count > expected.len(),
        "the body queries must duplicate the shared dynamic pair"
    );

    let mut all = ContactBuffer::new();
    assert_world_contacts(&world, &mut all, &expected);

    // Destroying one participant removes the shared pair and leaves a hole
    // in the native contact-id slots. The world query must shrink cleanly
    // without revisiting the stale slot.
    world.destroy_body(upper);
    world
        .try_body_contacts_buffered(lower, &mut lower_contacts)
        .unwrap();
    let after_destroy = merged_contact_snapshots(&[&lower_contacts]);
    assert!(after_destroy.len() < expected.len());
    assert_world_contacts(&world, &mut all, &after_destroy);

    // Refill once more to cover warm-buffer reuse after the shrink.
    assert_world_contacts(&world, &mut all, &after_destroy);
}

/// Regression: refilling a WARM buffer whose previous fill was smaller must
/// grow it safely. `fill_from_ffi` used to under-reserve for reused vecs
/// (`reserve(capacity - out.capacity())` guarantees `len + additional`, not
/// a total), so the FFI wrote past the allocation — UB that a fresh-vec
/// caller could never hit.
#[test]
fn warm_buffer_grows_safely_when_contact_count_increases() {
    let mut world = World::new(WorldDef::default()).unwrap();
    let ground = world.create_body(BodyDef::builder().body_type(BodyType::Static).build());
    world.create_hull_shape(ground, &ShapeDef::default(), &BoxHull::new(20.0, 0.5, 20.0));
    let center = world.create_body(
        BodyDef::builder()
            .body_type(BodyType::Dynamic)
            .position([0.0, 1.0, 0.0])
            .build(),
    );
    world.create_hull_shape(center, &ShapeDef::default(), &BoxHull::new(0.5, 0.5, 0.5));
    for _ in 0..120 {
        world.step(1.0 / 60.0, 4);
    }

    // Warm the buffer on ONE contact (the ground).
    let mut buf = ContactBuffer::new();
    world.try_body_contacts_buffered(center, &mut buf).unwrap();
    let warm = buf.len();
    assert!(warm >= 1);

    // Surround the center cube so its contact list outgrows the warm fill.
    for (dx, dz) in [(1.0f32, 0.0f32), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
        let n = world.create_body(
            BodyDef::builder()
                .body_type(BodyType::Dynamic)
                .position([dx, 1.0, dz])
                .build(),
        );
        world.create_hull_shape(n, &ShapeDef::default(), &BoxHull::new(0.5, 0.5, 0.5));
    }
    for _ in 0..120 {
        world.step(1.0 / 60.0, 4);
    }

    world.try_body_contacts_buffered(center, &mut buf).unwrap();
    assert!(
        buf.len() > warm,
        "the grown contact list must exceed the warm fill ({} vs {warm})",
        buf.len()
    );
    let mut allocating = Vec::new();
    world
        .try_body_contacts_into(center, &mut allocating)
        .unwrap();
    assert_eq!(buf.len(), allocating.len());
}

/// `update_body_mass(false)` defers the per-attach mass recompute; one
/// `apply_mass_from_shapes` afterwards lands on the same mass data as the
/// default eager path.
#[test]
fn deferred_batch_mass_equals_eager_per_shape_mass() {
    let hulls = [
        BoxHull::offset(0.5, 0.5, 0.5, boxddd::Vec3::new(0.0, 0.0, 0.0)),
        BoxHull::offset(0.5, 0.5, 0.5, boxddd::Vec3::new(1.0, 0.0, 0.0)),
        BoxHull::offset(0.5, 1.0, 0.5, boxddd::Vec3::new(0.0, 1.5, 0.0)),
    ];

    let build = |defer: bool| -> boxddd::MassData {
        let mut world = World::new(WorldDef::default()).unwrap();
        let body = world.create_body(BodyDef::builder().body_type(BodyType::Dynamic).build());
        let def = if defer {
            ShapeDef::builder()
                .density(2.0)
                .update_body_mass(false)
                .build()
        } else {
            ShapeDef::builder().density(2.0).build()
        };
        for hull in &hulls {
            world.create_hull_shape(body, &def, hull);
        }
        if defer {
            world.try_apply_mass_from_shapes(body).unwrap();
        }
        world.try_body_mass_data(body).unwrap()
    };

    let eager = build(false);
    let deferred = build(true);
    assert_eq!(eager.mass, deferred.mass);
    assert_eq!(eager.center, deferred.center);
    assert_eq!(eager.inertia, deferred.inertia);
}

#[test]
fn query_buffers_are_cleared_and_capacity_is_reused() {
    let mut world = World::new(WorldDef::default()).unwrap();
    let body = world.create_body(BodyDef::builder().body_type(BodyType::Static).build());
    world.create_sphere_shape(
        body,
        &ShapeDef::default(),
        &Sphere::new([0.0, 0.0, 0.0], 0.5),
    );

    let mut hits = Vec::with_capacity(8);
    hits.push(boxddd::QueryHit {
        shape_id: Default::default(),
    });
    let before_capacity = hits.capacity();
    let hit_aabb = Aabb {
        lower_bound: [-1.0, -1.0, -1.0].into(),
        upper_bound: [1.0, 1.0, 1.0].into(),
    };
    world
        .overlap_aabb_into(hit_aabb, QueryFilter::default(), &mut hits)
        .unwrap();
    assert_eq!(hits.capacity(), before_capacity);
    assert_eq!(hits.len(), 1);

    let miss_aabb = Aabb {
        lower_bound: [10.0, 10.0, 10.0].into(),
        upper_bound: [11.0, 11.0, 11.0].into(),
    };
    world
        .overlap_aabb_into(miss_aabb, QueryFilter::default(), &mut hits)
        .unwrap();
    assert_eq!(hits.capacity(), before_capacity);
    assert!(hits.is_empty());
}
