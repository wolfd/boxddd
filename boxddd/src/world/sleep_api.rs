//! Read-only sleep observation queries: per-body sleep state and per-island
//! sleep census. Pure observers — no world mutation.

use super::*;
use crate::types::{BodySleepBuffer, IslandCensusBuffer};

impl World {
    /// Tries to refill `buf` with the current sleep state of every enabled
    /// body: sleep time, sleep velocity, and containing island id.
    ///
    /// Allocation-free once `buf`'s capacity has warmed up — see
    /// [`BodySleepBuffer`]. Read-only observation data; the world is not
    /// mutated.
    pub fn world_body_sleep_buffered(&self, buf: &mut BodySleepBuffer) -> Result<()> {
        let _call = self.enter_world_call()?;
        let capacity = unsafe { ffi::b3World_GetBodySleepCapacity(self.raw()) }.max(0) as usize;
        unsafe {
            crate::core::ffi_vec::fill_from_ffi(&mut buf.raw, capacity, |ptr, cap| {
                ffi::b3World_GetBodySleepData(self.raw(), ptr, cap)
            });
        }
        buf.convert_raw(|raw| self.state().ledger.resolve_body(raw))
    }

    /// Tries to refill `buf` with a sleep census of every island (awake or
    /// sleeping): body/contact counts, constraint remove count, minimum
    /// sleep time with the body holding it, and the count of bodies whose
    /// sleep time has reached the engine time-to-sleep duration.
    ///
    /// Allocation-free once `buf`'s capacity has warmed up — see
    /// [`IslandCensusBuffer`]. Read-only observation data; the world is not
    /// mutated.
    pub fn world_island_census_buffered(&self, buf: &mut IslandCensusBuffer) -> Result<()> {
        let _call = self.enter_world_call()?;
        let capacity = unsafe { ffi::b3World_GetIslandCensusCapacity(self.raw()) }.max(0) as usize;
        unsafe {
            crate::core::ffi_vec::fill_from_ffi(&mut buf.raw, capacity, |ptr, cap| {
                ffi::b3World_GetIslandCensusData(self.raw(), ptr, cap)
            });
        }
        buf.convert_raw(|raw| self.state().ledger.resolve_body(raw))
    }
}
