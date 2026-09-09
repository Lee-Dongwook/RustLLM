use crate::error::Result;

use crate::metal::MetalBuffer;

use super::Device;

pub enum Storage {
    Metal(MetalBuffer),
}

impl Storage {
    pub fn device(&self) -> Device {
        match self {
            Storage::Metal(_) => Device::Metal,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Metal(buffer) => buffer.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        match self {
            Storage::Metal(buffer) => Ok(buffer.as_slice()),
        }
    }

    pub fn as_f16_slice(&self) -> Result<&[half::f16]> {
        match self {
            Storage::Metal(buffer) => Ok(buffer.as_f16_slice()),
        }
    }

    pub fn byte_len(&self) -> usize {
        match self {
            Storage::Metal(buffer) => buffer.byte_len(),
        }
    }

    pub fn metal_buffer(&self) -> Result<&MetalBuffer> {
        match self {
            Storage::Metal(buffer) => Ok(buffer),
        }
    }
}
