use core::hash::{Hash, Hasher};

use cubecl::prelude::*;
use cubecl_core as cubecl;

// A comptime enum whose variants hold only runtime values, like a cube mapping strategy.
#[derive(CubeType, CubeLaunch)]
pub enum Mapping {
    Flattened { x: u32, y: u32 },
    Spread { x: u32, y: u32, z: u32 },
    Exact,
}

fn hash_of(value: &impl Hash) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

// Regression: the compilation argument hashed only the fields, which runtime values make empty,
// so its variants hashed alike while comparing unequal. A kernel's persistent cache key is that
// hash, and a launch of one variant ran the kernel compiled for the other.
#[test]
fn variants_of_runtime_fields_hash_apart() {
    let variants = [
        MappingCompilationArg::Flattened { x: (), y: () },
        MappingCompilationArg::Spread { x: (), y: (), z: () },
        MappingCompilationArg::Exact,
    ];
    for (i, a) in variants.iter().enumerate() {
        for b in &variants[i + 1..] {
            assert_ne!(a, b);
            assert_ne!(hash_of(a), hash_of(b), "{a:?} and {b:?} hash alike");
            let (id_a, id_b) = (KernelId::new::<Mapping>().info(a.clone()), KernelId::new::<Mapping>().info(b.clone()));
            assert_ne!(id_a.stable_hash(), id_b.stable_hash(), "{a:?} and {b:?} share a cache key");
        }
    }
}
