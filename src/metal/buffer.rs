use std::ffi::c_void;
use std::mem;

use ::metal::{
    Buffer,
    MTLResourceOptions,
};

use super::MetalContext;

pub struct MetalBuffer {
    raw: Buffer,
    len: usize,
}

impl MetalBuffer {
    pub fn from_slice(
        context: &MetalContext,
        data: &[f32],
    ) -> Self {
        assert!(
            !data.is_empty(),
            "빈 데이터로 MetalBuffer를 만들 수 없습니다."
        );

        let byte_len = 
            (data.len() * mem::size_of::<f32>()) as u64;
        
        let raw = context.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            raw,
            len: data.len(),
        }
    }

      pub fn empty(
        context: &MetalContext,
        len: usize,
    ) -> Self {
        assert!(
            len > 0,
            "길이가 0인 MetalBuffer는 만들 수 없습니다."
        );

        let byte_len =
            (len * mem::size_of::<f32>()) as u64;

        let raw = context.device.new_buffer(
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            raw,
            len,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn raw(&self) -> &Buffer {
        &self.raw
    }

    pub fn as_slice(&self) -> &[f32] {
        let ptr = self.raw.contents() as *const f32;

        unsafe {
            std::slice::from_raw_parts(
                ptr,
                self.len,
            )
        }
    }
}
