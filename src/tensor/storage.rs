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

    pub fn metal_buffer(&self) -> Result<&MetalBuffer> {
        match self {
            Storage::Metal(buffer) => Ok(buffer),
        }
    }
}
