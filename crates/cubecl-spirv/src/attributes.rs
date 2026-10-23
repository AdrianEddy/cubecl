use cubecl_ir::{
    attributes::{BoolAttr, IndexAttr, ZeroAttr},
    prelude::*,
    verify_attr_succ,
};

use pliron::{
    attribute::{AttrObj, attr_cast},
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{IntegerAttr, VecAttr},
        types::IntegerType,
    },
    utils::apint::{APInt, bw},
};
use pliron_spirv::{
    attrs::{CompositeAttr, FloatAttr, NullAttr},
    types::{FloatType, khr::CooperativeMatrixType},
};

use crate::types::ty_to_spirv_dialect;

pub fn attr_to_spirv_dialect(ctx: &Context, attr: &AttrObj) -> AttrObj {
    if let Some(to_spirv_dialect) = attr_cast::<dyn ToSpirvDialectAttr>(&**attr) {
        to_spirv_dialect.to_spirv_dialect(ctx)
    } else {
        attr.clone()
    }
}

#[attr_interface]
pub trait ToSpirvDialectAttr {
    verify_attr_succ!();
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj;
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for IndexAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let value = self.0;
        let width = bw(ctx.address_type().size_bits());
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be integer");
        IntegerAttr::new(ty, APInt::from_usize(value, width)).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for IntegerAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let ty = ty_to_spirv_dialect(ctx, self.get_type());
        IntegerAttr::new(TypedHandle::from_handle(ty, ctx).unwrap(), self.value()).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for cubecl_ir::attributes::FloatAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));

        // A minifloat the device cannot convert lowers to the integer holding its bits (see
        // `types.rs`), and a constant of that type has to follow it there. Folding reaches one
        // whenever a cast's operand is known — the saturating arm of a scale's round-up returns a
        // literal, and the cast to the storage type folds with it — and the polyfill rewrites
        // casts, so a folded constant never passes through it. The value *is* the bit pattern, so
        // the integer attribute carries it unchanged.
        if let Ok(int_ty) = TypedHandle::<IntegerType>::from_handle(ty, ctx) {
            let width = int_ty.deref(ctx).width();
            let bits = self.val.to_bits() as u64;
            return IntegerAttr::new(int_ty, APInt::from_u64(bits, bw(width as usize))).into();
        }

        let ty = TypedHandle::from_handle(ty, ctx).expect("Should be float");
        FloatAttr::new(self.val.to_bits() as u64, ty).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for BoolAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        let value = if self.0 { 1 } else { 0 };
        let value = APInt::from_u8(value, bw(1));
        IntegerAttr::new(TypedHandle::from_handle(ty, ctx).unwrap(), value).into()
    }
}

#[attr_interface_impl]
impl ToSpirvDialectAttr for ZeroAttr {
    fn to_spirv_dialect(&self, ctx: &Context) -> AttrObj {
        let ty = ty_to_spirv_dialect(ctx, self.get_type(ctx));
        // A cooperative matrix is zeroed as a composite of its component's zero, the same value
        // `OpConstantNull` names. AMD's Windows Vulkan compiler (driverVersion 2.0.353) crashes
        // compiling a kernel that holds a null cooperative matrix, as the zero-initialised
        // fragments of a double-buffered matmul did.
        if let Ok(matrix) = TypedHandle::<CooperativeMatrixType>::from_handle(ty, ctx) {
            let component = matrix.deref(ctx).component_type;
            return CompositeAttr::new(VecAttr(vec![scalar_zero(ctx, component)]), ty).into();
        }
        NullAttr::new(ty).into()
    }
}

/// The zero of the SPIR-V scalar type `ty`.
fn scalar_zero(ctx: &Context, ty: TypeHandle) -> AttrObj {
    if let Ok(float) = TypedHandle::<FloatType>::from_handle(ty, ctx) {
        return FloatAttr::new(0, float).into();
    }
    let int = TypedHandle::<IntegerType>::from_handle(ty, ctx)
        .expect("a cooperative matrix holds floats or integers");
    let width = int.deref(ctx).width();
    IntegerAttr::new(int, APInt::from_u64(0, bw(width as usize))).into()
}
