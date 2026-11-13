//! What the memory of a [`MemoryScope`] covers on a runtime's device, whose
//! streams each keep pools of their own.
//!
//! Not part of `testgen_all`: the check reads and cleans what the whole memory
//! holds, so it needs a process to itself. A runtime calls it from an
//! integration test of its own.

use crate::MemoryScope;
use cubecl_runtime::{client::Client, runtime::Runtime};

const SIZE: usize = 4 * 1024 * 1024;

/// The device's memory is every stream's: its report counts what another
/// thread's stream holds, and its cleanup gives back the pages another
/// thread's stream no longer uses.
pub fn the_device_covers_every_threads_stream<R: Runtime>() {
    let client = R::client(&Default::default());
    let (to_main, from_other) = std::sync::mpsc::channel::<u64>();
    let (to_other, from_main) = std::sync::mpsc::channel::<()>();
    let other = client.clone();
    let thread = std::thread::spawn(move || {
        let stream = |client: &Client| client.memory_report(MemoryScope::CurrentStream).usage();
        let allocation = other.empty(SIZE);
        to_main.send(stream(&other).bytes_in_use).unwrap();
        from_main.recv().unwrap();
        drop(allocation);
        let kept = stream(&other).bytes_reserved;
        to_main.send(kept).unwrap();
        from_main.recv().unwrap();
        (kept, stream(&other).bytes_reserved)
    });

    let held = from_other.recv().unwrap();
    assert!(held >= SIZE as u64, "the other thread holds its allocation");
    let device = client.memory_report(MemoryScope::Device).usage();
    assert!(
        device.bytes_in_use >= held,
        "the device's report counts the other thread's stream: {} bytes in use, {held} there",
        device.bytes_in_use
    );

    to_other.send(()).unwrap();
    let kept = from_other.recv().unwrap();
    assert!(kept >= SIZE as u64, "the freed allocation's page stays reserved");
    client
        .memory_cleanup()
        .expect("no stream records a graph");
    to_other.send(()).unwrap();
    let (kept, after) = thread.join().expect("the other thread panicked");
    assert!(
        after < kept,
        "the device's cleanup gives back the other thread's free page: {kept} bytes reserved before, {after} after"
    );
}
