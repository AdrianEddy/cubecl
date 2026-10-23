use cubecl_core::{self as cubecl, frontend::polyfills::*, prelude::*};
use cubecl_ir::{dialect::math, interfaces::TypedExt, prelude::*};
use cubecl_opt::passes::uniformity::op_dyn_uniformity;
use pliron::{
    attribute::{AttrObj, boxed_attr_cast},
    builtin::{
        attributes::VecAttr,
        types::{IntegerType, Signedness},
    },
};
use pliron_spirv::{
    attrs::{CompositeAttr, FloatAttr as SpirvFloatAttr, PackedVectorFormatAttr},
    ext::gl,
    ops,
    spirv::PackedVectorFormat,
    types::{FloatType as SpirvFloatType, StructType, VectorType as SpirvVectorType},
};

use crate::{
    decorate_uniform,
    lower::{LowerOp, lower_binop, lower_unop},
    ops::{
        base::{binop_to_spirv_dialect, ternop_to_spirv_dialect, unop_to_spirv_dialect},
        to_spirv_dialect::ToSpirvDialectOp,
    },
    types::ty_to_spirv_dialect,
};

unop_to_spirv_dialect!(math::SNegOp => ops::SNegateOp);
unop_to_spirv_dialect!(math::FNegOp => ops::FNegateOp);

binop_to_spirv_dialect!(math::IAddOp => ops::IAddOp);
binop_to_spirv_dialect!(math::FAddOp => ops::FAddOp);
binop_to_spirv_dialect!(math::ISubOp => ops::ISubOp);
binop_to_spirv_dialect!(math::FSubOp => ops::FSubOp);
binop_to_spirv_dialect!(math::IMulOp => ops::IMulOp);
binop_to_spirv_dialect!(math::FMulOp => ops::FMulOp);
binop_to_spirv_dialect!(math::SDivOp => ops::SDivOp);
binop_to_spirv_dialect!(math::UDivOp => ops::UDivOp);
binop_to_spirv_dialect!(math::FDivOp => ops::FDivOp);
lower_binop!(math::SRemOp, signed_remainder);
binop_to_spirv_dialect!(math::URemOp => ops::UModOp);
binop_to_spirv_dialect!(math::FRemOp => ops::FRemOp);
lower_binop!(math::SModFloorOp, signed_mod_floor);
binop_to_spirv_dialect!(math::FModFloorOp => ops::FModOp);

// Vulkan only defines OpSRem/OpSMod for negative operands when maintenance8 is
// enabled. Reconstruct the truncating remainder using integer arithmetic so
// signed inputs work on devices without that feature too.
#[cube]
fn signed_remainder<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    // Division by zero and MIN / -1 are outside the portable contract.
    lhs - (lhs / rhs) * rhs
}

#[cube]
fn signed_mod_floor<I: Int, N: Size>(lhs: Vector<I, N>, rhs: Vector<I, N>) -> Vector<I, N> {
    let remainder = signed_remainder(lhs, rhs);
    let zero = Vector::new(I::from_int(0));
    let different_signs = (remainder ^ rhs).less_than(&zero);
    let adjust = remainder.not_equal(&zero).vec_and(different_signs);
    remainder + select_many(adjust, rhs, zero)
}

#[op_interface_impl]
impl LowerOp for math::Dp4aOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        !ctx.aux_ty::<cubecl_core::WgpuCompilationOptions>()
            .vulkan
            .supports_dp4a
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        let ctx = scope.ctx();
        vec![expand_dp4a_polyfill(
            scope,
            self.a(ctx),
            self.b(ctx),
            self.c(ctx),
        )]
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for math::Dp4aOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let packed = PackedVectorFormatAttr::new(PackedVectorFormat::PackedVectorFormat4x8Bit);
        let dot = ops::SDotOp::new(ctx, out_ty, self.a(ctx), self.b(ctx), Some(packed));
        rewriter.append_op(ctx, &dot);
        let add = ops::IAddOp::new(ctx, out_ty, dot.get_result(ctx), self.c(ctx));
        rewriter.append_op(ctx, &add);
        rewriter.replace_operation(ctx, op, add.get_operation());
        Ok(())
    }
}

binop_to_spirv_dialect!(SimplePowOp => gl::PowOp);

unop_to_spirv_dialect!(math::SAbsOp => gl::SAbsOp);
unop_to_spirv_dialect!(math::FAbsOp => gl::FAbsOp);
unop_to_spirv_dialect!(math::ExpOp => gl::ExpOp);
unop_to_spirv_dialect!(math::LogOp => gl::LogOp);
unop_to_spirv_dialect!(math::IsNanOp => ops::IsNanOp);
unop_to_spirv_dialect!(math::IsInfOp => ops::IsInfOp);

unop_to_spirv_dialect!(math::SinOp => gl::SinOp);
unop_to_spirv_dialect!(math::CosOp => gl::CosOp);
unop_to_spirv_dialect!(math::TanOp => gl::TanOp);
unop_to_spirv_dialect!(math::SinhOp => gl::SinhOp);
unop_to_spirv_dialect!(math::CoshOp => gl::CoshOp);

// GLSL.std.450 Tanh inherits its precision from sinh / cosh, and AMD's Windows driver evaluates it
// through an exponential that overflows where tanh is already ±1: NaN from |x| = 89 in f32 and from
// |x| = 12 in f16, which turned large GELU arguments into NaN. tanh rounds to ±1 from |x| = 10 in
// f32 (20 in f64), so the argument is clamped there, which changes no finite result; a NaN skips
// the clamp, whose result for one is undefined (a bound in practice, which would make it ±1). A
// half-precision tanh is evaluated in f32: a clamp low enough for an f16 exponential would sit
// where tanh is not yet ±1 in f16.
#[op_interface_impl]
impl ToSpirvDialectOp for math::TanhOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let uniformity = op_dyn_uniformity(ctx, op);
        let input = op.operand(ctx, 0);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let (lanes, width) = float_shape(ctx, out_ty);
        let (eval_width, limit) = if width == 64 { (64, 20.0) } else { (32, 10.0) };
        let eval_scalar: TypeHandle = SpirvFloatType::get(ctx, eval_width, None).to_handle();
        let bool_scalar: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).to_handle();
        let (eval_ty, bool_ty) = match lanes {
            Some(count) => (
                SpirvVectorType::get(ctx, count, eval_scalar).to_handle(),
                SpirvVectorType::get(ctx, count, bool_scalar).to_handle(),
            ),
            None => (eval_scalar, bool_scalar),
        };

        let mut value = input;
        if width != eval_width {
            let widen = ops::FConvertOp::new(ctx, eval_ty, value);
            value = rewriter.append_op_with_result(ctx, &widen);
        }
        let bound = |ctx: &mut Context, rewriter: &mut DialectConversionRewriter, v: f64| {
            let bits = if eval_width == 64 {
                v.to_bits()
            } else {
                u64::from((v as f32).to_bits())
            };
            let scalar: AttrObj =
                SpirvFloatAttr::new(bits, TypedHandle::from_handle(eval_scalar, ctx).unwrap())
                    .into();
            let attr: AttrObj = match lanes {
                Some(count) => {
                    CompositeAttr::new(VecAttr(vec![scalar; count as usize]), eval_ty).into()
                }
                None => scalar,
            };
            let constant =
                pliron::builtin::ops::ConstantOp::new(ctx, boxed_attr_cast(attr).unwrap());
            rewriter.append_op_with_result(ctx, &constant)
        };
        let (low, high) = (bound(ctx, rewriter, -limit), bound(ctx, rewriter, limit));
        let clamp = gl::FClampOp::new(ctx, eval_ty, value, low, high);
        let clamped = rewriter.append_op_with_result(ctx, &clamp);
        let is_nan = ops::IsNanOp::new(ctx, bool_ty, value);
        let is_nan = rewriter.append_op_with_result(ctx, &is_nan);
        let select = ops::SelectOp::new(ctx, eval_ty, is_nan, value, clamped);
        let argument = rewriter.append_op_with_result(ctx, &select);
        let tanh = gl::TanhOp::new(ctx, eval_ty, argument);
        let result = if width == eval_width {
            decorate_uniform(ctx, tanh.get_operation(), uniformity);
            rewriter.append_op(ctx, &tanh);
            tanh.get_operation()
        } else {
            let tanh = rewriter.append_op_with_result(ctx, &tanh);
            let narrow = ops::FConvertOp::new(ctx, out_ty, tanh);
            decorate_uniform(ctx, narrow.get_operation(), uniformity);
            rewriter.append_op(ctx, &narrow);
            narrow.get_operation()
        };
        rewriter.replace_operation(ctx, op, result);
        Ok(())
    }
}

/// The lane count (`None` for a scalar) and component width of the SPIR-V float type `ty`.
fn float_shape(ctx: &Context, ty: TypeHandle) -> (Option<u32>, u32) {
    let (lanes, scalar) = match TypedHandle::<SpirvVectorType>::from_handle(ty, ctx) {
        Ok(vector) => {
            let vector = vector.deref(ctx);
            (Some(vector.count), vector.element_type)
        }
        Err(_) => (None, ty),
    };
    let width = TypedHandle::<SpirvFloatType>::from_handle(scalar, ctx)
        .expect("tanh of a float")
        .deref(ctx)
        .width;
    (lanes, width)
}
unop_to_spirv_dialect!(math::ArcSinOp => gl::AsinOp);
unop_to_spirv_dialect!(math::ArcCosOp => gl::AcosOp);
unop_to_spirv_dialect!(math::ArcTanOp => gl::AtanOp);
unop_to_spirv_dialect!(math::ArcSinhOp => gl::AsinhOp);
unop_to_spirv_dialect!(math::ArcCoshOp => gl::AcoshOp);
unop_to_spirv_dialect!(math::ArcTanhOp => gl::AtanhOp);
binop_to_spirv_dialect!(math::ArcTan2Op => gl::Atan2Op);

unop_to_spirv_dialect!(math::DegreesOp => gl::DegreesOp);
unop_to_spirv_dialect!(math::RadiansOp => gl::RadiansOp);

unop_to_spirv_dialect!(math::SqrtOp => gl::SqrtOp);
unop_to_spirv_dialect!(math::RsqrtOp => gl::InverseSqrtOp);

unop_to_spirv_dialect!(math::RoundOp => gl::RoundOp);
unop_to_spirv_dialect!(math::FloorOp => gl::FloorOp);
unop_to_spirv_dialect!(math::CeilOp => gl::CeilOp);
unop_to_spirv_dialect!(math::TruncOp => gl::TruncOp);

#[op_interface_impl]
impl ToSpirvDialectOp for math::SMulHiOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let uniformity = op_dyn_uniformity(ctx, op);
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let out_st = StructType::get(ctx, vec![out_ty, out_ty], vec![], vec![], vec![]).into();
        let mul = ops::SMulExtendedOp::new(ctx, out_st, lhs, rhs);
        decorate_uniform(ctx, mul.get_operation(), uniformity);
        rewriter.append_op(ctx, &mul);
        let new_op = ops::CompositeExtractOp::new(ctx, out_ty, mul.get_result(ctx), vec![1.into()]);
        decorate_uniform(ctx, new_op.get_operation(), uniformity);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for math::UMulHiOp {
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op = self.get_operation();
        let uniformity = op_dyn_uniformity(ctx, op);
        let lhs = self.lhs(ctx);
        let rhs = self.rhs(ctx);
        let out_ty = ty_to_spirv_dialect(ctx, self.get_result(ctx).get_type(ctx));
        let out_st = StructType::get(ctx, vec![out_ty, out_ty], vec![], vec![], vec![]).into();
        let mul = ops::UMulExtendedOp::new(ctx, out_st, lhs, rhs);
        decorate_uniform(ctx, mul.get_operation(), uniformity);
        rewriter.append_op(ctx, &mul);
        let new_op = ops::CompositeExtractOp::new(ctx, out_ty, mul.get_result(ctx), vec![1.into()]);
        decorate_uniform(ctx, new_op.get_operation(), uniformity);
        rewriter.append_op(ctx, &new_op);
        rewriter.replace_operation(ctx, op, new_op.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl ToSpirvDialectOp for gl::PowOp {
    fn should_convert(&self, ctx: &Context) -> bool {
        ty_to_spirv_dialect(ctx, self.result_type(ctx)) != self.result_type(ctx)
    }
    fn to_spirv_dialect(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let ty = ty_to_spirv_dialect(ctx, self.result_type(ctx));
        rewriter.set_value_type(ctx, self.get_result(ctx), ty);
        Ok(())
    }
}

lower_unop!(math::RecipOp, recip);
lower_unop!(math::Log1pOp, log1p);
lower_unop!(math::Expm1Op, expm1);
lower_unop!(math::CbrtOp, cbrt);
lower_unop!(math::RcbrtOp, inverse_cbrt);
lower_unop!(math::ErfOp, erf);
lower_binop!(math::HypotOp, hypot);
lower_binop!(math::RhypotOp, rhypot);

lower_binop!(math::PowfOp, powf);

#[op_interface_impl]
impl LowerOp for math::PowiOp {
    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let lhs = self.lhs(scope.ctx());
        let rhs = self.rhs(scope.ctx());
        scope.register_value_type::<T, S>(lhs);

        let ty = lhs.scalar_ty(scope.ctx());
        let result = if ty.is_int(scope.ctx()) || ty.is_index(scope.ctx()) {
            powi_int::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)
        } else {
            powi::expand::<T, S>(scope, lhs.into(), rhs.into()).read_value(scope)
        };
        vec![result]
    }
}

ternop_to_spirv_dialect!(math::FmaOp => gl::FmaOp);

/// `gl::FmaOp` is defined for floats only, so an integer fma keeps its multiply and add.
#[op_interface_impl]
impl LowerOp for math::FmaOp {
    fn should_lower(&self, ctx: &Context) -> bool {
        self.get_result(ctx).scalar_ty(ctx).is_int(ctx)
    }

    fn lower(&self, scope: &Scope) -> Vec<Value> {
        define_scalar!(T);
        define_size!(S);
        let a = self.a(scope.ctx());
        let b = self.b(scope.ctx());
        let c = self.c(scope.ctx());
        scope.register_value_type::<T, S>(a);
        vec![int_fma::expand::<T, S>(scope, a.into(), b.into(), c.into()).read_value(scope)]
    }
}

#[cube]
fn int_fma<T: Int, N: Size>(a: Vector<T, N>, b: Vector<T, N>, c: Vector<T, N>) -> Vector<T, N> {
    a * b + c
}
