use super::{
    weights::{checked_numel, DTYPE_F32, MAGIC, MAX_NAME_LEN, MAX_RANK, VERSION},
    ModelWeights,
};
use crate::error::{Result, TinyError};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

pub(super) fn save(weights: &ModelWeights, path: impl AsRef<Path>) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    out.write_all(MAGIC)?;
    u32w(&mut out, VERSION)?;
    u32w(
        &mut out,
        u32::try_from(weights.tensors.len())
            .map_err(|_| TinyError::ModelFormat("too many tensors".into()))?,
    )?;
    let mut names: Vec<_> = weights.tensors.keys().collect();
    names.sort();
    for name in names {
        let t = &weights.tensors[name];
        let b = name.as_bytes();
        u32w(
            &mut out,
            u32::try_from(b.len())
                .map_err(|_| TinyError::ModelFormat(format!("weight name too long: {name}")))?,
        )?;
        out.write_all(b)?;
        out.write_all(&[DTYPE_F32])?;
        u32w(
            &mut out,
            u32::try_from(t.shape.len())
                .map_err(|_| TinyError::ModelFormat(format!("weight rank too large: {name}")))?,
        )?;
        for &d in &t.shape {
            u64w(&mut out, d as u64)?;
        }
        u64w(&mut out, t.data.len() as u64)?;
        for &v in &t.data {
            out.write_all(&v.to_le_bytes())?;
        }
    }
    out.flush()?;
    Ok(())
}
pub(super) fn load(path: impl AsRef<Path>) -> Result<ModelWeights> {
    let mut input = BufReader::new(File::open(path)?);
    let mut magic = [0; 8];
    input.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(TinyError::ModelFormat("invalid model weight magic".into()));
    }
    if u32r(&mut input)? != VERSION {
        return Err(TinyError::ModelFormat(
            "unsupported weight format version".into(),
        ));
    }
    let mut weights = ModelWeights::new();
    for _ in 0..u32r(&mut input)? {
        let n = u32r(&mut input)?;
        if n == 0 || n > MAX_NAME_LEN {
            return Err(TinyError::ModelFormat(format!(
                "invalid weight name length {n}"
            )));
        }
        let mut nb = vec![0; n as usize];
        input.read_exact(&mut nb)?;
        let name = String::from_utf8(nb)
            .map_err(|e| TinyError::ModelFormat(format!("invalid UTF-8 weight name: {e}")))?;
        let mut dtype = [0];
        input.read_exact(&mut dtype)?;
        if dtype[0] != DTYPE_F32 {
            return Err(TinyError::ModelFormat(format!(
                "unsupported dtype {} for weight {name}",
                dtype[0]
            )));
        }
        let rank = u32r(&mut input)?;
        if rank == 0 || rank > MAX_RANK {
            return Err(TinyError::ModelFormat(format!(
                "invalid rank {rank} for weight {name}"
            )));
        }
        let mut shape = Vec::with_capacity(rank as usize);
        for _ in 0..rank {
            let d = usize::try_from(u64r(&mut input)?).map_err(|_| {
                TinyError::ModelFormat(format!("dimension too large for weight {name}"))
            })?;
            if d == 0 {
                return Err(TinyError::ModelFormat(format!(
                    "weight {name} has a zero-sized dimension"
                )));
            }
            shape.push(d);
        }
        let count = usize::try_from(u64r(&mut input)?)
            .map_err(|_| TinyError::ModelFormat(format!("weight {name} is too large")))?;
        if count != checked_numel(&shape)? {
            return Err(TinyError::ModelFormat(format!(
                "weight {name} declares an invalid element count"
            )));
        }
        let mut bytes = vec![
            0;
            count
                .checked_mul(4)
                .ok_or_else(|| TinyError::ModelFormat(format!(
                    "weight byte size overflow: {name}"
                )))?
        ];
        input.read_exact(&mut bytes)?;
        let data = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        weights.insert_f32(name, &shape, data)?;
    }
    Ok(weights)
}
fn u32w(w: &mut impl Write, n: u32) -> Result<()> {
    w.write_all(&n.to_le_bytes())?;
    Ok(())
}
fn u64w(w: &mut impl Write, n: u64) -> Result<()> {
    w.write_all(&n.to_le_bytes())?;
    Ok(())
}
fn u32r(r: &mut impl Read) -> Result<u32> {
    let mut b = [0; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn u64r(r: &mut impl Read) -> Result<u64> {
    let mut b = [0; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
