use std::{ffi::c_void, mem};

use metal::MTLSize;

use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    tensor::{DType, Tensor},
};

const SHADER_SOURCE: &str = include_str!("../../kernels/gqa_matmul.metal");
const TILE_SIZE: u64 = 16;

/// Computes `Q × Kᵀ` without materializing KV heads to query-head count.
pub fn gqa_qk_matmul(context: &MetalContext, q: &Tensor, k: &Tensor) -> Result<Tensor> {
    let (batch, q_heads, q_len, kv_heads, kv_len, head_dim) = validate_qk(q, k)?;
    // A q_len=1 head permutation is layout-equivalent on device. Keep the
    // view to avoid a decode-only copy; prefill still materializes it.
    let q = if q_len == 1 {
        q.clone()
    } else {
        contiguous(context, q)?
    };
    let k = if q_len == 1 {
        k.clone()
    } else {
        contiguous(context, k)?
    };
    let output = Tensor::empty(context, &[batch, q_heads, q_len, kv_len], q.dtype())?;
    let kernel = match q.dtype() {
        DType::F32 => "gqa_qk_f32",
        DType::F16 => "gqa_qk_f16",
    };

    dispatch(
        context,
        &q,
        &k,
        &output,
        [q_heads, kv_heads, q_len, kv_len, head_dim],
        kernel,
        kv_len,
        q_len,
        batch,
        k.strides().values()[1] / head_dim,
    )?;
    Ok(output)
}

/// Computes `P × V` without materializing KV heads to query-head count.
pub fn gqa_pv_matmul(context: &MetalContext, probs: &Tensor, value: &Tensor) -> Result<Tensor> {
    let (batch, q_heads, q_len, kv_heads, kv_len, head_dim) = validate_pv(probs, value)?;
    let probs = contiguous(context, probs)?;
    let value = if q_len == 1 {
        value.clone()
    } else {
        contiguous(context, value)?
    };
    let output = Tensor::empty(context, &[batch, q_heads, q_len, head_dim], probs.dtype())?;
    let kernel = match probs.dtype() {
        DType::F32 => "gqa_pv_f32",
        DType::F16 => "gqa_pv_f16",
    };

    dispatch(
        context,
        &probs,
        &value,
        &output,
        [q_heads, kv_heads, q_len, kv_len, head_dim],
        kernel,
        head_dim,
        q_len,
        batch,
        value.strides().values()[1] / head_dim,
    )?;
    Ok(output)
}

fn validate_qk(q: &Tensor, k: &Tensor) -> Result<(usize, usize, usize, usize, usize, usize)> {
    validate_dtype_and_rank(q, k, "QK", "Q [B,QH,Q,D] and K [B,KVH,K,D]")?;
    let (batch, q_heads, q_len, head_dim) = (q.dim(0)?, q.dim(1)?, q.dim(2)?, q.dim(3)?);
    let (k_batch, kv_heads, kv_len, k_head_dim) = (k.dim(0)?, k.dim(1)?, k.dim(2)?, k.dim(3)?);
    if batch != k_batch || head_dim != k_head_dim {
        return Err(TinyError::ShapeMismatch {
            left: q.shape().dims().to_vec(),
            right: k.shape().dims().to_vec(),
        });
    }
    validate_head_groups(q_heads, kv_heads)?;
    Ok((batch, q_heads, q_len, kv_heads, kv_len, head_dim))
}

fn validate_pv(
    probs: &Tensor,
    value: &Tensor,
) -> Result<(usize, usize, usize, usize, usize, usize)> {
    validate_dtype_and_rank(probs, value, "PV", "P [B,QH,Q,K] and V [B,KVH,K,D]")?;
    let (batch, q_heads, q_len, kv_len) =
        (probs.dim(0)?, probs.dim(1)?, probs.dim(2)?, probs.dim(3)?);
    let (value_batch, kv_heads, value_len, head_dim) =
        (value.dim(0)?, value.dim(1)?, value.dim(2)?, value.dim(3)?);
    if batch != value_batch || kv_len != value_len {
        return Err(TinyError::ShapeMismatch {
            left: probs.shape().dims().to_vec(),
            right: value.shape().dims().to_vec(),
        });
    }
    validate_head_groups(q_heads, kv_heads)?;
    Ok((batch, q_heads, q_len, kv_heads, kv_len, head_dim))
}

fn validate_dtype_and_rank(
    left: &Tensor,
    right: &Tensor,
    operation: &str,
    expected_shapes: &str,
) -> Result<()> {
    if left.dtype() != right.dtype() {
        return Err(TinyError::UnsupportedDType(format!(
            "GQA {operation} dtype mismatch: {:?} vs {:?}",
            left.dtype(),
            right.dtype(),
        )));
    }
    if left.rank() != 4 || right.rank() != 4 {
        return Err(TinyError::ModelFormat(format!(
            "GQA {operation} expects {expected_shapes}, got {:?} and {:?}",
            left.shape().dims(),
            right.shape().dims(),
        )));
    }
    Ok(())
}

fn validate_head_groups(q_heads: usize, kv_heads: usize) -> Result<()> {
    if kv_heads == 0 || !q_heads.is_multiple_of(kv_heads) {
        return Err(TinyError::ModelFormat(format!(
            "Q heads {q_heads} must be divisible by non-zero KV heads {kv_heads}",
        )));
    }
    Ok(())
}

fn contiguous(context: &MetalContext, tensor: &Tensor) -> Result<Tensor> {
    if tensor.is_contiguous() {
        Ok(tensor.clone())
    } else {
        tensor.contiguous(context)
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    context: &MetalContext,
    left: &Tensor,
    right: &Tensor,
    output: &Tensor,
    dimensions: [usize; 5],
    kernel_name: &str,
    output_width: usize,
    query_len: usize,
    batch: usize,
    kv_capacity: usize,
) -> Result<()> {
    let [q_heads, kv_heads, q_len, kv_len, head_dim] = dimensions.map(|value| {
        u32::try_from(value)
            .map_err(|_| TinyError::InvalidShape("GQA dimension exceeds u32".to_string()))
    });
    let q_heads = q_heads?;
    let batch_heads = u64::try_from(batch.checked_mul(q_heads as usize).ok_or_else(|| {
        TinyError::InvalidShape("GQA batch × query-head count overflow".to_string())
    })?)
    .map_err(|_| TinyError::InvalidShape("GQA grid depth exceeds u64".to_string()))?;

    let pipeline = context.pipeline(SHADER_SOURCE, kernel_name);
    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(left.metal_buffer()?.raw()), 0);
    encoder.set_buffer(1, Some(right.metal_buffer()?.raw()), 0);
    encoder.set_buffer(2, Some(output.metal_buffer()?.raw()), 0);
    for (index, value) in [
        q_heads,
        kv_heads?,
        q_len?,
        kv_len?,
        head_dim?,
        u32::try_from(kv_capacity)
            .map_err(|_| TinyError::InvalidShape("GQA KV capacity exceeds u32".into()))?,
    ]
    .iter()
    .enumerate()
    {
        encoder.set_bytes(
            (index + 3) as u64,
            mem::size_of::<u32>() as u64,
            value as *const u32 as *const c_void,
        );
    }
    encoder.dispatch_thread_groups(
        MTLSize::new(
            (output_width as u64).div_ceil(TILE_SIZE),
            (query_len as u64).div_ceil(TILE_SIZE),
            batch_heads,
        ),
        MTLSize::new(TILE_SIZE, TILE_SIZE, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}
