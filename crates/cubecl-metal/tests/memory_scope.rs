//! The device's memory across streams, see [`cubecl_core::runtime_tests::memory_scope`].
//! One test per binary: it reads and cleans what the whole memory holds.
#![cfg(target_vendor = "apple")]

use cubecl_core::runtime_tests::memory_scope;
use cubecl_metal::MetalRuntime;

#[test]
fn the_device_covers_every_threads_stream() {
    memory_scope::the_device_covers_every_threads_stream::<MetalRuntime>();
}
