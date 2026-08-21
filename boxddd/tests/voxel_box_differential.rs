use boxddd_sys::ffi;

fn vec3(x: f32, y: f32, z: f32) -> ffi::b3Vec3 {
    ffi::b3Vec3 { x, y, z }
}

fn transform(position: ffi::b3Vec3, rotation: ffi::b3Quat) -> ffi::b3Transform {
    ffi::b3Transform {
        p: position,
        q: rotation,
    }
}

fn random_unit_quat(state: &mut u64) -> ffi::b3Quat {
    let mut sample = || {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let unit = (*state >> 40) as f32 / (1u32 << 24) as f32;
        2.0 * unit - 1.0
    };
    let mut x = sample();
    let mut y = sample();
    let mut z = sample();
    let mut w = sample();
    let mut length_squared = x * x + y * y + z * z + w * w;
    if length_squared < 1.0e-6 {
        x = 0.0;
        y = 0.0;
        z = 0.0;
        w = 1.0;
        length_squared = 1.0;
    }
    let inverse_length = length_squared.sqrt().recip();
    ffi::b3Quat {
        v: vec3(x * inverse_length, y * inverse_length, z * inverse_length),
        s: w * inverse_length,
    }
}

fn collide_generic(
    cell: &ffi::b3BoxHull,
    convex: &ffi::b3BoxHull,
    transform_b_to_a: ffi::b3Transform,
) -> (ffi::b3LocalManifold, [ffi::b3LocalManifoldPoint; 8]) {
    let mut points: [ffi::b3LocalManifoldPoint; 8] = unsafe { std::mem::zeroed() };
    let mut manifold: ffi::b3LocalManifold = unsafe { std::mem::zeroed() };
    manifold.points = points.as_mut_ptr();
    let mut cache: ffi::b3SATCache = unsafe { std::mem::zeroed() };
    unsafe {
        ffi::b3CollideHulls(
            &mut manifold,
            points.len() as i32,
            &cell.base,
            &convex.base,
            transform_b_to_a,
            &mut cache,
        );
    }
    (manifold, points)
}

fn collide_voxel(
    voxel: *const ffi::b3VoxelData,
    convex: &ffi::b3BoxHull,
    transform_b_to_a: ffi::b3Transform,
) -> (ffi::b3LocalManifold, [ffi::b3LocalManifoldPoint; 8]) {
    let mut points: [ffi::b3LocalManifoldPoint; 8] = unsafe { std::mem::zeroed() };
    let mut manifold: ffi::b3LocalManifold = unsafe { std::mem::zeroed() };
    manifold.points = points.as_mut_ptr();
    unsafe {
        ffi::b3CollideVoxelAndHull(
            &mut manifold,
            points.len() as i32,
            voxel,
            &convex.base,
            transform_b_to_a,
        );
    }
    (manifold, points)
}

fn min_separation(manifold: &ffi::b3LocalManifold, points: &[ffi::b3LocalManifoldPoint]) -> f32 {
    points[..manifold.pointCount as usize]
        .iter()
        .map(|point| point.separation)
        .fold(f32::INFINITY, f32::min)
}

#[test]
fn specialized_voxel_box_path_matches_generic_sat_classification() {
    let cell_index = ffi::b3Vec3i { x: 0, y: 0, z: 0 };
    let voxel = unsafe { ffi::b3CreateOffsetVoxelData(&cell_index, 1, 1.0, vec3(0.0, 0.0, 0.0)) };
    assert!(!voxel.is_null());

    let cell = unsafe { ffi::b3MakeBoxHull(0.5, 0.5, 0.5) };
    let convex = unsafe { ffi::b3MakeBoxHull(0.37, 0.29, 0.43) };
    let mut random_state = 0x4d59_5df4_d0f3_3173u64;

    for sample in 0..2048 {
        let rotation = random_unit_quat(&mut random_state);
        // Project the rotated convex onto the voxel's X axis. These fixtures
        // cover unambiguous overlap, shallow face contact, and separation;
        // they avoid the differing speculative-distance policies of the two
        // public entry points.
        // Derive the OBB's projection radius from the quaternion's X matrix row.
        let x = rotation.v.x;
        let y = rotation.v.y;
        let z = rotation.v.z;
        let w = rotation.s;
        let row_x = [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ];
        let projected_half = row_x[0].abs() * 0.37 + row_x[1].abs() * 0.29 + row_x[2].abs() * 0.43;
        let cases = [
            (vec3(0.01, -0.02, 0.015), true, false),
            (vec3(0.5 + projected_half - 0.02, 0.0, 0.0), true, true),
            (vec3(0.5 + projected_half + 0.20, 0.0, 0.0), false, false),
        ];

        for (position, expect_contact, compare_depth) in cases {
            let relative = transform(position, rotation);
            let (generic, generic_points) = collide_generic(&cell, &convex, relative);
            let (specialized, specialized_points) = collide_voxel(voxel, &convex, relative);
            assert_eq!(
                generic.pointCount > 0,
                expect_contact,
                "generic classification differed at sample {sample} position {:?}",
                [position.x, position.y, position.z]
            );
            assert_eq!(
                specialized.pointCount > 0,
                expect_contact,
                "specialized classification differed at sample {sample} position {:?}",
                [position.x, position.y, position.z]
            );

            if expect_contact {
                let generic_depth = -min_separation(&generic, &generic_points);
                let specialized_depth = -min_separation(&specialized, &specialized_points);
                assert!(generic_depth.is_finite() && specialized_depth.is_finite());
                if compare_depth {
                    assert!(
                        (generic_depth - specialized_depth).abs() < 1.0e-2,
                        "depth mismatch at sample {sample}: generic={generic_depth} specialized={specialized_depth}"
                    );
                    let dot = generic.normal.x * specialized.normal.x
                        + generic.normal.y * specialized.normal.y
                        + generic.normal.z * specialized.normal.z;
                    assert!(dot > 0.98, "normal mismatch at sample {sample}: dot={dot}");
                }
            }
        }
    }

    unsafe { ffi::b3DestroyVoxelData(voxel) };
}
