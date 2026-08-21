//! Runtime capacity, profiling, counter, and version snapshots.

use super::*;

/// Native world capacity snapshot.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Capacity {
    /// Static shape capacity.
    pub static_shape_count: i32,
    /// Dynamic shape capacity.
    pub dynamic_shape_count: i32,
    /// Static body capacity.
    pub static_body_count: i32,
    /// Dynamic body capacity.
    pub dynamic_body_count: i32,
    /// Contact capacity.
    pub contact_count: i32,
}

impl Capacity {
    /// Converts a raw Box3D capacity snapshot into the Rust value type.
    #[inline]
    pub const fn from_raw(raw: ffi::b3Capacity) -> Self {
        Self {
            static_shape_count: raw.staticShapeCount,
            dynamic_shape_count: raw.dynamicShapeCount,
            static_body_count: raw.staticBodyCount,
            dynamic_body_count: raw.dynamicBodyCount,
            contact_count: raw.contactCount,
        }
    }

    /// Converts this value into the raw Box3D representation.
    #[inline]
    pub const fn into_raw(self) -> ffi::b3Capacity {
        ffi::b3Capacity {
            staticShapeCount: self.static_shape_count,
            dynamicShapeCount: self.dynamic_shape_count,
            staticBodyCount: self.static_body_count,
            dynamicBodyCount: self.dynamic_body_count,
            contactCount: self.contact_count,
        }
    }
}

/// Per-step timing profile reported by Box3D.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Profile {
    /// Total step time.
    pub step: f32,
    /// Broad-phase pair update time.
    pub pairs: f32,
    /// Collision detection time.
    pub collide: f32,
    /// Solver time.
    pub solve: f32,
    /// Solver setup time.
    pub solver_setup: f32,
    /// Constraint solve time.
    pub constraints: f32,
    /// Constraint preparation time.
    pub prepare_constraints: f32,
    /// Velocity integration time.
    pub integrate_velocities: f32,
    /// Warm-start time.
    pub warm_start: f32,
    /// Impulse solve time.
    pub solve_impulses: f32,
    /// Position integration time.
    pub integrate_positions: f32,
    /// Impulse relaxation time.
    pub relax_impulses: f32,
    /// Restitution application time.
    pub apply_restitution: f32,
    /// Impulse storage time.
    pub store_impulses: f32,
    /// Island splitting time.
    pub split_islands: f32,
    /// Transform update time.
    pub transforms: f32,
    /// Sensor hit processing time.
    pub sensor_hits: f32,
    /// Joint event processing time.
    pub joint_events: f32,
    /// Hit event processing time.
    pub hit_events: f32,
    /// Broad-phase refit time.
    pub refit: f32,
    /// Bullet/continuous collision time.
    pub bullets: f32,
    /// Sleep island processing time.
    pub sleep_islands: f32,
    /// Sensor processing time.
    pub sensors: f32,
}

impl Profile {
    /// Converts a raw Box3D profile into the Rust value type.
    #[inline]
    pub const fn from_raw(raw: ffi::b3Profile) -> Self {
        Self {
            step: raw.step,
            pairs: raw.pairs,
            collide: raw.collide,
            solve: raw.solve,
            solver_setup: raw.solverSetup,
            constraints: raw.constraints,
            prepare_constraints: raw.prepareConstraints,
            integrate_velocities: raw.integrateVelocities,
            warm_start: raw.warmStart,
            solve_impulses: raw.solveImpulses,
            integrate_positions: raw.integratePositions,
            relax_impulses: raw.relaxImpulses,
            apply_restitution: raw.applyRestitution,
            store_impulses: raw.storeImpulses,
            split_islands: raw.splitIslands,
            transforms: raw.transforms,
            sensor_hits: raw.sensorHits,
            joint_events: raw.jointEvents,
            hit_events: raw.hitEvents,
            refit: raw.refit,
            bullets: raw.bullets,
            sleep_islands: raw.sleepIslands,
            sensors: raw.sensors,
        }
    }
}

/// Per-step voxel collision counters reported by Box3D.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct VoxelCounters {
    /// Voxel-versus-voxel collision calls.
    pub voxel_voxel_calls: i32,
    /// Voxel-versus-convex collision calls.
    pub voxel_convex_calls: i32,
    /// Occupancy queries, including count and fill passes.
    pub query_calls: i32,
    /// Chunk coordinates visited by occupancy queries.
    pub chunks_visited: i32,
    /// Occupied-list entries tested by occupancy queries.
    pub occupied_entries_scanned: i32,
    /// Cells returned across all occupancy-query passes.
    pub cells_returned: i32,
    /// Fill passes that repeat a preceding count pass.
    pub count_fill_rescans: i32,
    /// Exact voxel-versus-voxel OBB tests.
    pub obb_tests: i32,
    /// Exact cell-versus-convex tests.
    pub convex_leaf_tests: i32,
    /// Generic hull SAT calls made inside voxel-versus-convex collision.
    pub hull_sat_calls: i32,
    /// Cache hits from those generic hull SAT calls.
    pub hull_sat_cache_hits: i32,
    /// Raw exact-collision points before surface filtering and reduction.
    pub raw_contact_points: i32,
    /// Points rejected because their directed voxel face is internal.
    pub surface_rejects: i32,
    /// Deep-overlap fallback contacts emitted.
    pub deep_overlap_fallbacks: i32,
    /// Continuous-collision encounters involving voxel shapes.
    pub ccd_encounters: i32,
    /// Broad-phase shape proxies visited for moving voxel CCD.
    pub ccd_broad_phase_visits: i32,
    /// Convex targets admitted to moving voxel CCD.
    pub ccd_convex_targets: i32,
    /// Aggregate targets admitted to moving voxel CCD.
    pub ccd_aggregate_targets: i32,
    /// Occupied cells visited by convex-target CCD queries.
    pub ccd_cells_visited: i32,
    /// Fully interior cells rejected before exact CCD.
    pub ccd_interior_rejects: i32,
    /// Surface cells rejected by the swept corridor.
    pub ccd_corridor_rejects: i32,
    /// Exact generic time-of-impact calls made for voxel cells.
    pub ccd_exact_toi_calls: i32,
    /// Raw points presented after the online reducer became full.
    pub reducer_overflow_insertions: i32,
    /// Normal clusters constructed for voxel contacts.
    pub normal_clusters: i32,
    /// Solver manifolds emitted for voxel contacts.
    pub emitted_manifolds: i32,
    /// Solver points emitted for voxel contacts.
    pub emitted_points: i32,
    /// Emitted points whose prior impulse was restored.
    pub persisted_points: i32,
    /// Voxel contact manifold-array reallocations.
    pub manifold_reallocations: i32,
    /// Touching voxel contacts routed through the scalar solver path.
    pub scalar_contacts: i32,
    /// Touching voxel contacts that emitted exactly one manifold.
    pub single_manifold_contacts: i32,
    /// Touching voxel contacts moved between wide and scalar solver storage.
    pub solver_class_changes: i32,
    /// Real cell-pair visits classified by canonical patch identity.
    pub patch_visits: i32,
    /// Unique exact canonical patch keys.
    pub patch_unique_keys: i32,
    /// Patch visits that reused an exact key.
    pub patch_duplicate_visits: i32,
    /// Visits covered by the symmetric canonical topology traversal.
    pub patch_eligible_visits: i32,
    /// Unique keys covered by that topology traversal.
    pub patch_eligible_unique_keys: i32,
    /// Largest visit multiplicity of one key in one contact update.
    pub patch_max_multiplicity: i32,
    /// Largest unique-key count in one contact update.
    pub patch_max_unique_keys: i32,
    /// Key multiplicities in buckets 1, 2–3, 4–7, 8–15, 16–31, and 32+.
    pub patch_multiplicity_counts: [i32; 6],
    /// Visit counts indexed by ordered `(topology_a * 4 + topology_b)`.
    pub patch_topology_pairs: [i32; 16],
    /// Exact leaf SAT results selecting a face axis.
    pub patch_sat_face_axes: i32,
    /// Exact leaf SAT results selecting an edge axis.
    pub patch_sat_edge_axes: i32,
    /// Exact leaf SAT separations.
    pub patch_sat_separations: i32,
    /// Real cell pairs excluded by topology pruning.
    pub topology_pruned_pairs: i32,
    /// Real leaf pairs represented by canonical candidates.
    pub represented_leaf_pairs: i32,
    /// Pseudo-cuboid SAT calls.
    pub pseudo_sat_calls: i32,
    /// Canonical keys rejected by pseudo-cuboid separation.
    pub pseudo_separated_keys: i32,
    /// Canonical keys retaining at least one real-alias-selected point.
    pub selected_patch_keys: i32,
    /// Positive pseudo keys whose points no real alias selected.
    pub empty_selected_patch_keys: i32,
    /// Pseudo witnesses rejected by real exposed-face validation.
    pub pseudo_witness_rejects: i32,
    /// Pseudo penetrating points rejected by real-cell support depth.
    pub pseudo_depth_rejects: i32,
    /// Keys first selected by an alias after their first represented pair.
    pub pseudo_late_selections: i32,
    /// Selected patches encountered beyond the solver-manifold budget.
    pub patch_budget_overflows: i32,
    /// Solver manifolds emitted from exact canonical patch keys.
    pub emitted_patch_manifolds: i32,
    /// Persistent workspace exact-key matches.
    pub workspace_key_hits: i32,
    /// Persistent workspace exact-key misses.
    pub workspace_key_misses: i32,
    /// Solver points persisted through an exact patch-key match.
    pub exact_patch_persisted_points: i32,
    /// Feature matches rejected by the anchor-distance guard.
    pub feature_remap_rejects: i32,
    /// Positive pseudo keys recovered by exact leaves after no alias selected a pseudo point.
    pub empty_patch_fallback_keys: i32,
    /// Exact leaf SAT calls made by empty-selection recovery.
    pub patch_leaf_fallback_tests: i32,
    /// Geometric growth operations in transient patch tables.
    pub patch_table_growths: i32,
    /// Largest transient patch scratch allocation in bytes.
    pub patch_scratch_peak_bytes: i32,
    /// Large aggregate pairs routed through one complete leaf traversal.
    pub adaptive_leaf_pairs: i32,
}

impl VoxelCounters {
    /// Converts raw Box3D voxel counters into the Rust value type.
    #[inline]
    pub const fn from_raw(raw: ffi::b3VoxelCounters) -> Self {
        Self {
            voxel_voxel_calls: raw.voxelVoxelCalls,
            voxel_convex_calls: raw.voxelConvexCalls,
            query_calls: raw.queryCalls,
            chunks_visited: raw.chunksVisited,
            occupied_entries_scanned: raw.occupiedEntriesScanned,
            cells_returned: raw.cellsReturned,
            count_fill_rescans: raw.countFillRescans,
            obb_tests: raw.obbTests,
            convex_leaf_tests: raw.convexLeafTests,
            hull_sat_calls: raw.hullSatCalls,
            hull_sat_cache_hits: raw.hullSatCacheHits,
            raw_contact_points: raw.rawContactPoints,
            surface_rejects: raw.surfaceRejects,
            deep_overlap_fallbacks: raw.deepOverlapFallbacks,
            ccd_encounters: raw.ccdEncounters,
            ccd_broad_phase_visits: raw.ccdBroadPhaseVisits,
            ccd_convex_targets: raw.ccdConvexTargets,
            ccd_aggregate_targets: raw.ccdAggregateTargets,
            ccd_cells_visited: raw.ccdCellsVisited,
            ccd_interior_rejects: raw.ccdInteriorRejects,
            ccd_corridor_rejects: raw.ccdCorridorRejects,
            ccd_exact_toi_calls: raw.ccdExactToiCalls,
            reducer_overflow_insertions: raw.reducerOverflowInsertions,
            normal_clusters: raw.normalClusters,
            emitted_manifolds: raw.emittedManifolds,
            emitted_points: raw.emittedPoints,
            persisted_points: raw.persistedPoints,
            manifold_reallocations: raw.manifoldReallocations,
            scalar_contacts: raw.scalarContacts,
            single_manifold_contacts: raw.singleManifoldContacts,
            solver_class_changes: raw.solverClassChanges,
            patch_visits: raw.patchVisits,
            patch_unique_keys: raw.patchUniqueKeys,
            patch_duplicate_visits: raw.patchDuplicateVisits,
            patch_eligible_visits: raw.patchEligibleVisits,
            patch_eligible_unique_keys: raw.patchEligibleUniqueKeys,
            patch_max_multiplicity: raw.patchMaxMultiplicity,
            patch_max_unique_keys: raw.patchMaxUniqueKeys,
            patch_multiplicity_counts: raw.patchMultiplicityCounts,
            patch_topology_pairs: raw.patchTopologyPairs,
            patch_sat_face_axes: raw.patchSatFaceAxes,
            patch_sat_edge_axes: raw.patchSatEdgeAxes,
            patch_sat_separations: raw.patchSatSeparations,
            topology_pruned_pairs: raw.topologyPrunedPairs,
            represented_leaf_pairs: raw.representedLeafPairs,
            pseudo_sat_calls: raw.pseudoSatCalls,
            pseudo_separated_keys: raw.pseudoSeparatedKeys,
            selected_patch_keys: raw.selectedPatchKeys,
            empty_selected_patch_keys: raw.emptySelectedPatchKeys,
            pseudo_witness_rejects: raw.pseudoWitnessRejects,
            pseudo_depth_rejects: raw.pseudoDepthRejects,
            pseudo_late_selections: raw.pseudoLateSelections,
            patch_budget_overflows: raw.patchBudgetOverflows,
            emitted_patch_manifolds: raw.emittedPatchManifolds,
            workspace_key_hits: raw.workspaceKeyHits,
            workspace_key_misses: raw.workspaceKeyMisses,
            exact_patch_persisted_points: raw.exactPatchPersistedPoints,
            feature_remap_rejects: raw.featureRemapRejects,
            empty_patch_fallback_keys: raw.emptyPatchFallbackKeys,
            patch_leaf_fallback_tests: raw.patchLeafFallbackTests,
            patch_table_growths: raw.patchTableGrowths,
            patch_scratch_peak_bytes: raw.patchScratchPeakBytes,
            adaptive_leaf_pairs: raw.adaptiveLeafPairs,
        }
    }
}

/// World counters reported by Box3D for diagnostics and tests.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Counters {
    /// Number of live bodies.
    pub body_count: i32,
    /// Number of live shapes.
    pub shape_count: i32,
    /// Number of live contacts.
    pub contact_count: i32,
    /// Number of live joints.
    pub joint_count: i32,
    /// Number of solver islands.
    pub island_count: i32,
    /// Stack memory used by Box3D.
    pub stack_used: i32,
    /// Arena memory capacity.
    pub arena_capacity: i32,
    /// Static broad-phase tree height.
    pub static_tree_height: i32,
    /// Dynamic broad-phase tree height.
    pub tree_height: i32,
    /// Number of separating-axis test calls.
    pub sat_call_count: i32,
    /// Number of separating-axis cache hits.
    pub sat_cache_hit_count: i32,
    /// Native byte count reported by Box3D.
    pub byte_count: i32,
    /// Number of native tasks scheduled by the last step.
    pub task_count: i32,
    /// Solver graph color distribution.
    pub color_counts: [i32; 24],
    /// Per-contact manifold-count distribution. Index zero counts contacts
    /// with one manifold; the final bucket includes all larger counts.
    pub manifold_counts: [i32; 8],
    /// Detailed per-step voxel collision work.
    pub voxel: VoxelCounters,
    /// Number of awake contacts.
    pub awake_contact_count: i32,
    /// Number of recycled contacts.
    pub recycled_contact_count: i32,
    /// Distance solver iteration count.
    pub distance_iterations: i32,
    /// Push-back solver iteration count.
    pub push_back_iterations: i32,
    /// Root solver iteration count.
    pub root_iterations: i32,
}

impl Counters {
    /// Converts raw Box3D counters into the Rust value type.
    #[inline]
    pub const fn from_raw(raw: ffi::b3Counters) -> Self {
        Self {
            body_count: raw.bodyCount,
            shape_count: raw.shapeCount,
            contact_count: raw.contactCount,
            joint_count: raw.jointCount,
            island_count: raw.islandCount,
            stack_used: raw.stackUsed,
            arena_capacity: raw.arenaCapacity,
            static_tree_height: raw.staticTreeHeight,
            tree_height: raw.treeHeight,
            sat_call_count: raw.satCallCount,
            sat_cache_hit_count: raw.satCacheHitCount,
            byte_count: raw.byteCount,
            task_count: raw.taskCount,
            color_counts: raw.colorCounts,
            manifold_counts: raw.manifoldCounts,
            voxel: VoxelCounters::from_raw(raw.voxel),
            awake_contact_count: raw.awakeContactCount,
            recycled_contact_count: raw.recycledContactCount,
            distance_iterations: raw.distanceIterations,
            push_back_iterations: raw.pushBackIterations,
            root_iterations: raw.rootIterations,
        }
    }
}

impl Default for Counters {
    fn default() -> Self {
        Self {
            body_count: 0,
            shape_count: 0,
            contact_count: 0,
            joint_count: 0,
            island_count: 0,
            stack_used: 0,
            arena_capacity: 0,
            static_tree_height: 0,
            tree_height: 0,
            sat_call_count: 0,
            sat_cache_hit_count: 0,
            byte_count: 0,
            task_count: 0,
            color_counts: [0; 24],
            manifold_counts: [0; 8],
            voxel: VoxelCounters::default(),
            awake_contact_count: 0,
            recycled_contact_count: 0,
            distance_iterations: 0,
            push_back_iterations: 0,
            root_iterations: 0,
        }
    }
}

/// Box3D version number.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Version {
    /// Major version.
    pub major: i32,
    /// Minor version.
    pub minor: i32,
    /// Revision or patch version.
    pub revision: i32,
}

impl Version {
    /// Converts a raw Box3D version into the Rust value type.
    #[inline]
    pub const fn from_raw(raw: ffi::b3Version) -> Self {
        Self {
            major: raw.major,
            minor: raw.minor,
            revision: raw.revision,
        }
    }
}
