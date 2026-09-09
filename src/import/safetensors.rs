use std::fs;
use std::path::Path;

use half::{
    bf16,
    f16,
};

use safetensors::{
    Dtype,
    SafeTensors,
};

use crate::error::{
    Result,
    TinyError,
};

use crate::model::ModelWeights;

#[derive(Debug, Clone)]
pub struct SafeTensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub numel: usize,
}

impl SafeTensorInfo {
    pub fn rank(
        &self,
    ) -> usize {
        self.shape.len()
    }
}

pub fn inspect_safetensors(
    path: impl AsRef<Path>,
) -> Result<Vec<SafeTensorInfo>> {
    let bytes =
        fs::read(
            path,
        )?;

    let tensors =
        SafeTensors::deserialize(
            &bytes,
        )
        .map_err(|error| {
            TinyError::ModelFormat(
                format!(
                    "failed to parse safetensors: {error}"
                ),
            )
        })?;

    let mut result =
        Vec::with_capacity(
            tensors.len(),
        );

    for (
        name,
        tensor,
    ) in tensors.iter()
    {
        let shape =
            tensor.shape()
                .to_vec();

        let numel =
            checked_numel(
                &shape,
                name,
            )?;

        result.push(
            SafeTensorInfo {
                name:
                    name.to_string(),

                dtype:
                    tensor.dtype(),

                shape,

                numel,
            },
        );
    }

    result.sort_by(
        |left, right| {
            left.name
                .cmp(
                    &right.name,
                )
        },
    );

    Ok(result)
}

pub fn import_safetensors(
    path: impl AsRef<Path>,
) -> Result<ModelWeights> {
    let bytes =
        fs::read(
            path,
        )?;

    let tensors =
        SafeTensors::deserialize(
            &bytes,
        )
        .map_err(|error| {
            TinyError::ModelFormat(
                format!(
                    "failed to parse safetensors: {error}"
                ),
            )
        })?;

    let mut weights =
        ModelWeights::new();

    for (
        name,
        tensor,
    ) in tensors.iter()
    {
        let shape =
            tensor.shape()
                .to_vec();

        if shape.is_empty() {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "scalar tensor is not supported yet: {name}"
                    ),
                ),
            );
        }

        if shape.contains(&0) {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "zero-sized tensor is not supported yet: {name} {shape:?}"
                    ),
                ),
            );
        }

        let data =
            decode_to_f32(
                name,
                tensor.dtype(),
                tensor.data(),
            )?;

        let expected =
            checked_numel(
                &shape,
                name,
            )?;

        if data.len()
            != expected
        {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "tensor {name} decoded to {} values, but shape {shape:?} requires {expected}",
                        data.len(),
                    ),
                ),
            );
        }

        weights.insert_f32(
            name.to_string(),
            &shape,
            data,
        )?;
    }

    Ok(weights)
}

fn decode_to_f32(
    name: &str,
    dtype: Dtype,
    bytes: &[u8],
) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            decode_f32(
                name,
                bytes,
            )
        }

        Dtype::F16 => {
            decode_f16(
                name,
                bytes,
            )
        }

        Dtype::BF16 => {
            decode_bf16(
                name,
                bytes,
            )
        }

        _ => {
            Err(
                TinyError::UnsupportedDType(
                    format!(
                        "safetensors tensor {name} uses {dtype:?}; importer currently supports only F32, F16 and BF16"
                    ),
                ),
            )
        }
    }
}

fn decode_f32(
    name: &str,
    bytes: &[u8],
) -> Result<Vec<f32>> {
    if bytes.len() % 4
        != 0
    {
        return Err(
            TinyError::ModelFormat(
                format!(
                    "F32 tensor {name} has invalid byte length {}",
                    bytes.len(),
                ),
            ),
        );
    }

    let mut values =
        Vec::with_capacity(
            bytes.len() / 4,
        );

    for chunk
        in bytes.chunks_exact(4)
    {
        values.push(
            f32::from_le_bytes(
                [
                    chunk[0],
                    chunk[1],
                    chunk[2],
                    chunk[3],
                ],
            ),
        );
    }

    Ok(values)
}

fn decode_f16(
    name: &str,
    bytes: &[u8],
) -> Result<Vec<f32>> {
    if bytes.len() % 2
        != 0
    {
        return Err(
            TinyError::ModelFormat(
                format!(
                    "F16 tensor {name} has invalid byte length {}",
                    bytes.len(),
                ),
            ),
        );
    }

    let mut values =
        Vec::with_capacity(
            bytes.len() / 2,
        );

    for chunk
        in bytes.chunks_exact(2)
    {
        let bits =
            u16::from_le_bytes(
                [
                    chunk[0],
                    chunk[1],
                ],
            );

        let value =
            f16::from_bits(
                bits,
            );

        values.push(
            value.to_f32(),
        );
    }

    Ok(values)
}

fn decode_bf16(
    name: &str,
    bytes: &[u8],
) -> Result<Vec<f32>> {
    if bytes.len() % 2
        != 0
    {
        return Err(
            TinyError::ModelFormat(
                format!(
                    "BF16 tensor {name} has invalid byte length {}",
                    bytes.len(),
                ),
            ),
        );
    }

    let mut values =
        Vec::with_capacity(
            bytes.len() / 2,
        );

    for chunk
        in bytes.chunks_exact(2)
    {
        let bits =
            u16::from_le_bytes(
                [
                    chunk[0],
                    chunk[1],
                ],
            );

        let value =
            bf16::from_bits(
                bits,
            );

        values.push(
            value.to_f32(),
        );
    }

    Ok(values)
}

fn checked_numel(
    shape: &[usize],
    name: &str,
) -> Result<usize> {
    let mut numel =
        1usize;

    for &dimension
        in shape
    {
        numel =
            numel
                .checked_mul(
                    dimension,
                )
                .ok_or_else(|| {
                    TinyError::ModelFormat(
                        format!(
                            "tensor element count overflow for {name}: {shape:?}"
                        ),
                    )
                })?;
    }

    Ok(numel)
}
