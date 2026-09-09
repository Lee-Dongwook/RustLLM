use std::collections::HashMap;

use std::fs::File;

use std::io::{
    BufReader,
    BufWriter,
    Read,
    Write,
};

use std::path::Path;

use crate::error::{
    Result,
    TinyError,
};

const MAGIC: &[u8; 8] =
    b"TMLLWGHT";

const VERSION: u32 =
    1;

const DTYPE_F32: u8 =
    1;

const MAX_RANK: u32 =
    16;

const MAX_NAME_LEN: u32 =
    1024;

#[derive(Debug)]
pub struct WeightTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl WeightTensor {
    pub fn shape(
        &self,
    ) -> &[usize] {
        &self.shape
    }

    pub fn data(
        &self,
    ) -> &[f32] {
        &self.data
    }

    pub fn into_parts(
        self,
    ) -> (
        Vec<usize>,
        Vec<f32>,
    ) {
        (
            self.shape,
            self.data,
        )
    }
}

#[derive(Debug, Default)]
pub struct ModelWeights {
    tensors:
        HashMap<
            String,
            WeightTensor,
        >,
}

impl ModelWeights {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(
        &self,
    ) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(
        &self,
    ) -> bool {
        self.tensors.is_empty()
    }

    pub fn insert_f32(
        &mut self,
        name: impl Into<String>,
        shape: &[usize],
        data: Vec<f32>,
    ) -> Result<()> {
        let name =
            name.into();

        if name.is_empty() {
            return Err(
                TinyError::ModelFormat(
                    "weight name cannot be empty"
                        .to_string(),
                ),
            );
        }

        if shape.is_empty() {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "weight {name} has an empty shape"
                    ),
                ),
            );
        }

        if shape.contains(&0) {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "weight {name} has a zero-sized dimension"
                    ),
                ),
            );
        }

        let numel =
            checked_numel(
                shape,
            )?;

        if data.len()
            != numel
        {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "weight {name} has {} values, but shape {:?} requires {numel}",
                        data.len(),
                        shape,
                    ),
                ),
            );
        }

        if self.tensors
            .contains_key(
                &name,
            )
        {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "duplicate weight: {name}"
                    ),
                ),
            );
        }

        self.tensors.insert(
            name,
            WeightTensor {
                shape:
                    shape.to_vec(),
                data,
            },
        );

        Ok(())
    }

    pub fn take(
        &mut self,
        name: &str,
    ) -> Result<WeightTensor> {
        self.tensors
            .remove(name)
            .ok_or_else(|| {
                TinyError::MissingWeight(
                    name.to_string(),
                )
            })
    }

    pub fn save(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<()> {
        let file =
            File::create(
                path,
            )?;

        let mut writer =
            BufWriter::new(
                file,
            );

        writer.write_all(
            MAGIC,
        )?;

        write_u32(
            &mut writer,
            VERSION,
        )?;

        let tensor_count =
            u32::try_from(
                self.tensors.len(),
            )
            .map_err(|_| {
                TinyError::ModelFormat(
                    "too many tensors"
                        .to_string(),
                )
            })?;

        write_u32(
            &mut writer,
            tensor_count,
        )?;

        // HashMap 순서는 랜덤이므로
        // 파일 결과를 deterministic하게 만든다.
        let mut names:
            Vec<&String> =
            self.tensors
                .keys()
                .collect();

        names.sort();

        for name in names {
            let tensor =
                &self.tensors[name];

            let name_bytes =
                name.as_bytes();

            let name_len =
                u32::try_from(
                    name_bytes.len(),
                )
                .map_err(|_| {
                    TinyError::ModelFormat(
                        format!(
                            "weight name too long: {name}"
                        ),
                    )
                })?;

            write_u32(
                &mut writer,
                name_len,
            )?;

            writer.write_all(
                name_bytes,
            )?;

            // dtype
            writer.write_all(
                &[DTYPE_F32],
            )?;

            let rank =
                u32::try_from(
                    tensor.shape.len(),
                )
                .map_err(|_| {
                    TinyError::ModelFormat(
                        format!(
                            "weight rank too large: {name}"
                        ),
                    )
                })?;

            write_u32(
                &mut writer,
                rank,
            )?;

            for &dim
                in &tensor.shape
            {
                write_u64(
                    &mut writer,
                    dim as u64,
                )?;
            }

            write_u64(
                &mut writer,
                tensor.data.len()
                    as u64,
            )?;

            for &value
                in &tensor.data
            {
                writer.write_all(
                    &value
                        .to_le_bytes(),
                )?;
            }
        }

        writer.flush()?;

        Ok(())
    }

    pub fn load(
        path: impl AsRef<Path>,
    ) -> Result<Self> {
        let file =
            File::open(
                path,
            )?;

        let mut reader =
            BufReader::new(
                file,
            );

        let mut magic =
            [0u8; 8];

        reader.read_exact(
            &mut magic,
        )?;

        if &magic != MAGIC {
            return Err(
                TinyError::ModelFormat(
                    "invalid model weight magic"
                        .to_string(),
                ),
            );
        }

        let version =
            read_u32(
                &mut reader,
            )?;

        if version != VERSION {
            return Err(
                TinyError::ModelFormat(
                    format!(
                        "unsupported weight format version {version}"
                    ),
                ),
            );
        }

        let tensor_count =
            read_u32(
                &mut reader,
            )?;

        let mut weights =
            Self::new();

        for _ in 0..tensor_count {
            let name_len =
                read_u32(
                    &mut reader,
                )?;

            if name_len == 0
                || name_len
                    > MAX_NAME_LEN
            {
                return Err(
                    TinyError::ModelFormat(
                        format!(
                            "invalid weight name length {name_len}"
                        ),
                    ),
                );
            }

            let mut name_bytes =
                vec![
                    0u8;
                    name_len as usize
                ];

            reader.read_exact(
                &mut name_bytes,
            )?;

            let name =
                String::from_utf8(
                    name_bytes,
                )
                .map_err(|error| {
                    TinyError::ModelFormat(
                        format!(
                            "invalid UTF-8 weight name: {error}"
                        ),
                    )
                })?;

            let mut dtype =
                [0u8; 1];

            reader.read_exact(
                &mut dtype,
            )?;

            if dtype[0]
                != DTYPE_F32
            {
                return Err(
                    TinyError::ModelFormat(
                        format!(
                            "unsupported dtype {} for weight {name}",
                            dtype[0],
                        ),
                    ),
                );
            }

            let rank =
                read_u32(
                    &mut reader,
                )?;

            if rank == 0
                || rank > MAX_RANK
            {
                return Err(
                    TinyError::ModelFormat(
                        format!(
                            "invalid rank {rank} for weight {name}"
                        ),
                    ),
                );
            }

            let mut shape =
                Vec::with_capacity(
                    rank as usize,
                );

            for _ in 0..rank {
                let dim =
                    read_u64(
                        &mut reader,
                    )?;

                let dim =
                    usize::try_from(
                        dim,
                    )
                    .map_err(|_| {
                        TinyError::ModelFormat(
                            format!(
                                "dimension too large for weight {name}"
                            ),
                        )
                    })?;

                if dim == 0 {
                    return Err(
                        TinyError::ModelFormat(
                            format!(
                                "weight {name} has a zero-sized dimension"
                            ),
                        ),
                    );
                }

                shape.push(
                    dim,
                );
            }

            let element_count =
                read_u64(
                    &mut reader,
                )?;

            let element_count =
                usize::try_from(
                    element_count,
                )
                .map_err(|_| {
                    TinyError::ModelFormat(
                        format!(
                            "weight {name} is too large"
                        ),
                    )
                })?;

            let expected =
                checked_numel(
                    &shape,
                )?;

            if element_count
                != expected
            {
                return Err(
                    TinyError::ModelFormat(
                        format!(
                            "weight {name} declares {element_count} elements, shape {:?} requires {expected}",
                            shape,
                        ),
                    ),
                );
            }

            let byte_len =
                element_count
                    .checked_mul(
                        std::mem::size_of::<f32>(),
                    )
                    .ok_or_else(|| {
                        TinyError::ModelFormat(
                            format!(
                                "weight byte size overflow: {name}"
                            ),
                        )
                    })?;

            let mut bytes =
                vec![
                    0u8;
                    byte_len
                ];

            reader.read_exact(
                &mut bytes,
            )?;

            let mut data =
                Vec::with_capacity(
                    element_count,
                );

            for chunk
                in bytes
                    .chunks_exact(4)
            {
                data.push(
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

            weights.insert_f32(
                name,
                &shape,
                data,
            )?;
        }

        Ok(weights)
    }
}

fn checked_numel(
    shape: &[usize],
) -> Result<usize> {
    let mut numel =
        1usize;

    for &dim in shape {
        numel =
            numel
                .checked_mul(
                    dim,
                )
                .ok_or_else(|| {
                    TinyError::ModelFormat(
                        format!(
                            "shape element count overflow: {shape:?}"
                        ),
                    )
                })?;
    }

    Ok(numel)
}

fn write_u32(
    writer: &mut impl Write,
    value: u32,
) -> Result<()> {
    writer.write_all(
        &value.to_le_bytes(),
    )?;

    Ok(())
}

fn write_u64(
    writer: &mut impl Write,
    value: u64,
) -> Result<()> {
    writer.write_all(
        &value.to_le_bytes(),
    )?;

    Ok(())
}

fn read_u32(
    reader: &mut impl Read,
) -> Result<u32> {
    let mut bytes =
        [0u8; 4];

    reader.read_exact(
        &mut bytes,
    )?;

    Ok(
        u32::from_le_bytes(
            bytes,
        ),
    )
}

fn read_u64(
    reader: &mut impl Read,
) -> Result<u64> {
    let mut bytes =
        [0u8; 8];

    reader.read_exact(
        &mut bytes,
    )?;

    Ok(
        u64::from_le_bytes(
            bytes,
        ),
    )
}
