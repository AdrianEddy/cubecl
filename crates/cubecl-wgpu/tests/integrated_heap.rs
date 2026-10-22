//! An integrated GPU's storage comes from its largest device-local heap, not from whichever heap
//! holds the first host-visible device-local type.
//!
//! AMD's Windows driver exposes that type in a 256 MiB heap of its own, beside the large heap that
//! is the GPU's real share of system RAM, and every allocation past 250 MiB failed there with
//! `ERROR_OUT_OF_DEVICE_MEMORY` while the real heap stood empty. 768 MiB held at once is past any
//! such side heap and well inside an integrated GPU's share. Vulkan only: elsewhere wgpu allocates.
#![cfg(all(feature = "spirv", not(target_family = "wasm")))]

use cubecl_common::device::Device;
use cubecl_server::runtime::Runtime;
use cubecl_wgpu::{WgpuBackend, WgpuDevice, WgpuDeviceKind, WgpuRuntime};

#[test]
fn an_integrated_gpu_holds_more_than_a_side_heap() {
    let device = WgpuDevice::new(WgpuDeviceKind::IntegratedGpu(0)).on(WgpuBackend::Vulkan);
    if <WgpuRuntime>::find_device(device.to_id()).is_err() {
        eprintln!("no integrated GPU on Vulkan");
        return;
    }
    let client = <WgpuRuntime>::client(&device);

    const BUFFER: usize = 128 << 20;
    let held: Vec<_> = (0..6u8)
        .map(|i| client.create_from_slice(&vec![i + 1; BUFFER]))
        .collect();
    for (i, handle) in held.into_iter().enumerate() {
        let bytes = client.read_one(handle).expect("the buffer reads back");
        assert_eq!(bytes.len(), BUFFER);
        let want = i as u8 + 1;
        assert!(
            bytes[0] == want && bytes[BUFFER / 2] == want && bytes[BUFFER - 1] == want,
            "buffer {i} holds what was written"
        );
    }
}
