#![allow(dead_code)]

pub(crate) unsafe fn fill_from_ffi<T>(
    out: &mut Vec<T>,
    capacity: usize,
    fill: impl FnOnce(*mut T, i32) -> i32,
) {
    out.clear();
    if capacity == 0 {
        return;
    }
    // `reserve(additional)` guarantees room for `len + additional` elements
    // (len is 0 after the clear above), so pass the FULL target capacity.
    // The old `reserve(capacity - out.capacity())` under-reserved whenever
    // the vec was warm but smaller than `capacity` (e.g. 5 of 8: it asked
    // for room for 3), letting `fill` write past the allocation — heap
    // corruption that only a REUSED buffer could trigger; `read_from_ffi`'s
    // always-fresh vec made the subtraction accidentally correct.
    out.reserve(capacity);
    let cap = i32::try_from(capacity).expect("ffi capacity exceeds i32::MAX");
    let wrote = fill(out.as_mut_ptr(), cap).max(0) as usize;
    unsafe { out.set_len(wrote.min(capacity)) };
}

pub(crate) unsafe fn read_from_ffi<T>(
    capacity: usize,
    fill: impl FnOnce(*mut T, i32) -> i32,
) -> Vec<T> {
    let mut out = Vec::new();
    unsafe { fill_from_ffi(&mut out, capacity, fill) };
    out
}
