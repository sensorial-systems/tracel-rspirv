use tracel_rspirv::spirv::{TensorClampMode, Word};

use crate::{ToSpirvType, interfaces::VerCapExtTypeInterface, prelude::*, types::u32_ty};

#[pliron_type(
    name = "spirv.tensor_layout",
    format = "`<` $dim `d, ` $clamp_mode `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TensorLayoutType {
    pub dim: u32,
    pub clamp_mode: TensorClampMode,
}

#[type_interface_impl]
impl ToSpirvType for TensorLayoutType {
    fn to_spirv(&self, ctx: &Context, builder: &mut PlironBuilder) -> Result<Word> {
        let dim_id = builder.constant_bit32(ctx, u32_ty(ctx), self.dim)?;
        let clamp_mode_id = builder.constant_bit32(ctx, u32_ty(ctx), self.clamp_mode as u32)?;
        Ok(builder.type_tensor_layout_nv(dim_id, clamp_mode_id))
    }
}

#[type_interface_impl]
impl VerCapExtTypeInterface for TensorLayoutType {
    fn min_version(&self, _ctx: &Context) -> Option<(u8, u8)> {
        None
    }
    fn required_extensions(&self, _ctx: &Context) -> Vec<Vec<&'static str>> {
        Operand::from(self.clamp_mode).required_extensions()
    }
    fn required_capabilities(&self, _ctx: &Context) -> Vec<Vec<Capability>> {
        let mut out = vec![vec![Capability::TensorAddressingNV]];
        out.extend(Operand::from(self.clamp_mode).required_capabilities());
        out
    }
}

#[pliron_type(
    name = "spirv.tensor_view",
    format = "`<` $dim `d, has_dimensions: ` $has_dimensions `, permutation: [` vec($permutation, CharSpace(`,`)) `] `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct TensorViewType {
    pub dim: u32,
    pub has_dimensions: bool,
    pub permutation: Vec<u32>,
}

#[type_interface_impl]
impl ToSpirvType for TensorViewType {
    fn to_spirv(&self, ctx: &Context, builder: &mut PlironBuilder) -> Result<Word> {
        let bool = builder.type_bool();
        let dim_id = builder.constant_bit32(ctx, u32_ty(ctx), self.dim)?;
        let has_dims = match self.has_dimensions {
            true => builder.constant_true(bool),
            false => builder.constant_false(bool),
        };
        let permutation = self
            .permutation
            .iter()
            .map(|dim| builder.constant_bit32(ctx, u32_ty(ctx), *dim))
            .collect::<Result<Vec<_>>>()?;
        Ok(builder.type_tensor_view_nv(dim_id, has_dims, permutation))
    }
}

#[type_interface_impl]
impl VerCapExtTypeInterface for TensorViewType {
    fn min_version(&self, _ctx: &Context) -> Option<(u8, u8)> {
        None
    }
    fn required_extensions(&self, _ctx: &Context) -> Vec<Vec<&'static str>> {
        vec![]
    }
    fn required_capabilities(&self, _ctx: &Context) -> Vec<Vec<Capability>> {
        vec![vec![Capability::TensorAddressingNV]]
    }
}

/// `OpTypeCooperativeMatrixNV` (`SPV_NV_cooperative_matrix`), the matrix type before the KHR
/// extension, the only one some drivers expose (NVIDIA's on Turing). It has no use operand: a
/// matrix of one component type, scope and shape is the same type whether it is an A, a B or
/// an accumulator operand. `use_` only keeps the frontend's types apart, and is not emitted.
#[pliron_type(
    name = "spirv.cooperative_matrix_nv",
    format = "`<` $rows ` x ` $columns ` x ` $component_type `, ` $use_ `, ` $scope `>`",
    generate_get = true,
    verifier = "succ"
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CooperativeMatrixType {
    pub component_type: TypeHandle,
    pub scope: tracel_rspirv::spirv::Scope,
    pub rows: u32,
    pub columns: u32,
    pub use_: tracel_rspirv::spirv::CooperativeMatrixUse,
}

#[type_interface_impl]
impl ToSpirvType for CooperativeMatrixType {
    fn to_spirv(&self, ctx: &Context, builder: &mut PlironBuilder) -> Result<Word> {
        let component_id = spirv_type_id(ctx, builder, self.component_type)?;
        let scope_id = builder.constant_bit32(ctx, u32_ty(ctx), self.scope as u32)?;
        let rows_id = builder.constant_bit32(ctx, u32_ty(ctx), self.rows)?;
        let cols_id = builder.constant_bit32(ctx, u32_ty(ctx), self.columns)?;
        Ok(builder.type_cooperative_matrix_nv(component_id, scope_id, rows_id, cols_id))
    }
}

#[type_interface_impl]
impl VerCapExtTypeInterface for CooperativeMatrixType {
    fn min_version(&self, _ctx: &Context) -> Option<(u8, u8)> {
        None
    }
    fn required_extensions(&self, ctx: &Context) -> Vec<Vec<&'static str>> {
        let mut out = vec![vec!["SPV_NV_cooperative_matrix"]];
        let component_ty = self.component_type.deref(ctx);
        if let Some(ver_cap_ext) = pliron::r#type::type_cast::<dyn VerCapExtTypeInterface>(&*component_ty) {
            out.extend(ver_cap_ext.required_extensions(ctx))
        }
        out.extend(Operand::from(self.scope).required_extensions());
        out
    }
    fn required_capabilities(&self, ctx: &Context) -> Vec<Vec<Capability>> {
        let mut out = vec![vec![Capability::CooperativeMatrixNV]];
        let component_ty = self.component_type.deref(ctx);
        if let Some(ver_cap_ext) = pliron::r#type::type_cast::<dyn VerCapExtTypeInterface>(&*component_ty) {
            out.extend(ver_cap_ext.required_capabilities(ctx))
        }
        out.extend(Operand::from(self.scope).required_capabilities());
        out
    }
}
