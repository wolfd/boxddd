//! Read-only sleep observation data: per-body sleep state and per-island
//! sleep census snapshots.

use super::*;

/// Sleep state of one body, copied from the native world.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BodySleepData {
    /// The body this sample describes.
    pub body_id: BodyId,
    /// Seconds this body has continuously stayed below its sleep threshold.
    pub sleep_time: f32,
    /// The body's most recent sleep velocity measure (updated while awake).
    pub sleep_velocity: f32,
    /// Id of the island containing this body, or negative when the body is
    /// not in an island (static bodies are never in islands).
    pub island_id: i32,
}

impl BodySleepData {
    #[inline]
    pub(crate) fn from_raw(raw: ffi::b3BodySleepData, body_id: BodyId) -> Self {
        Self {
            body_id,
            sleep_time: raw.sleepTime,
            sleep_velocity: raw.sleepVelocity,
            island_id: raw.islandId,
        }
    }
}

/// Sleep census of one island, copied from the native world.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IslandCensus {
    /// The island id.
    pub island_id: i32,
    /// Number of bodies in the island.
    pub body_count: i32,
    /// Number of contacts in the island.
    pub contact_count: i32,
    /// Number of constraints removed from this island since it was created
    /// or last split. A non-zero count blocks sleep for multi-body islands.
    pub constraint_remove_count: i32,
    /// Smallest sleep time across the island's bodies, in seconds.
    pub min_sleep_time: f32,
    /// The body holding the smallest sleep time — the body currently keeping
    /// the island awake. The null id when the island has no bodies.
    pub min_sleep_time_body: BodyId,
    /// Number of bodies whose sleep time has reached the engine
    /// time-to-sleep duration (the per-body qualification for island sleep).
    pub sleep_ready_count: i32,
}

impl IslandCensus {
    #[inline]
    pub(crate) fn from_raw(raw: ffi::b3IslandCensus, min_sleep_time_body: BodyId) -> Self {
        Self {
            island_id: raw.islandId,
            body_count: raw.bodyCount,
            contact_count: raw.contactCount,
            constraint_remove_count: raw.constraintRemoveCount,
            min_sleep_time: raw.minSleepTime,
            min_sleep_time_body,
            sleep_ready_count: raw.sleepReadyCount,
        }
    }
}

/// Reusable buffer for [`World::world_body_sleep_buffered`], following
/// the allocation-free-once-warm pattern of [`ContactBuffer`].
///
/// [`World::world_body_sleep_buffered`]: crate::World::world_body_sleep_buffered
#[derive(Default)]
pub struct BodySleepBuffer {
    pub(crate) raw: Vec<ffi::b3BodySleepData>,
    entries: Vec<BodySleepData>,
}

impl BodySleepBuffer {
    /// Creates an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of body samples currently held.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer holds no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The buffered body sleep samples.
    #[inline]
    pub fn entries(&self) -> &[BodySleepData] {
        &self.entries
    }

    /// Iterates the buffered body sleep samples.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &BodySleepData> {
        self.entries.iter()
    }

    /// Converts the raw snapshots into owned entries. Clears any previous
    /// contents first. The raw structs are plain values with no pointers
    /// into world storage.
    pub(crate) fn convert_raw<F>(&mut self, mut resolve: F) -> Result<()>
    where
        F: FnMut(ffi::b3BodyId) -> Result<BodyId>,
    {
        self.entries.clear();
        self.entries.reserve(self.raw.len());
        for raw in &self.raw {
            let body_id = match resolve(raw.bodyId) {
                Ok(id) => id,
                Err(error) => {
                    self.entries.clear();
                    return Err(error);
                }
            };
            self.entries.push(BodySleepData::from_raw(*raw, body_id));
        }
        Ok(())
    }
}

/// Reusable buffer for [`World::world_island_census_buffered`],
/// following the allocation-free-once-warm pattern of [`ContactBuffer`].
///
/// [`World::world_island_census_buffered`]: crate::World::world_island_census_buffered
#[derive(Default)]
pub struct IslandCensusBuffer {
    pub(crate) raw: Vec<ffi::b3IslandCensus>,
    entries: Vec<IslandCensus>,
}

impl IslandCensusBuffer {
    /// Creates an empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of island census records currently held.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the buffer holds no records.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The buffered island census records.
    #[inline]
    pub fn entries(&self) -> &[IslandCensus] {
        &self.entries
    }

    /// Iterates the buffered island census records.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &IslandCensus> {
        self.entries.iter()
    }

    /// Converts the raw snapshots into owned entries. Clears any previous
    /// contents first. The raw structs are plain values with no pointers
    /// into world storage.
    pub(crate) fn convert_raw<F>(&mut self, mut resolve: F) -> Result<()>
    where
        F: FnMut(ffi::b3BodyId) -> Result<BodyId>,
    {
        self.entries.clear();
        self.entries.reserve(self.raw.len());
        for raw in &self.raw {
            let min_sleep_time_body = match resolve(raw.minSleepTimeBody) {
                Ok(id) => id,
                Err(error) => {
                    self.entries.clear();
                    return Err(error);
                }
            };
            self.entries
                .push(IslandCensus::from_raw(*raw, min_sleep_time_body));
        }
        Ok(())
    }
}
