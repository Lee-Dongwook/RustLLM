use std::ffi::c_void;
use std::mem;

use metal::{Buffer, MTLResourceOptions};

use super::MetalContext;

pub struct MetalBuffer {
    raw: Buffer,
    len: usize,
    byte_len: usize,
}

impl MetalBuffer {
    pub fn from_slice(context: &MetalContext, data: &[f32]) -> Self {
        assert!(
            !data.is_empty(),
            "빈 데이터로 MetalBuffer를 만들 수 없습니다."
        );

        let byte_len = std::mem::size_of_val(data) as u64;

        let raw = context.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            raw,
            len: data.len(),
            byte_len: std::mem::size_of_val(data),
        }
    }

    pub fn from_u32_slice(context: &MetalContext, data: &[u32]) -> Self {
        assert!(
            !data.is_empty(),
            "빈 데이터로 MetalBuffer를 만들 수 없습니다."
        );

        let byte_len = std::mem::size_of_val(data) as u64;

        let raw = context.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            raw,
            len: data.len(),
            byte_len: std::mem::size_of_val(data),
        }
    }

    pub fn from_f16_slice(context: &MetalContext, data: &[half::f16]) -> Self {
        assert!(
            !data.is_empty(),
            "빈 데이터로 MetalBuffer를 만들 수 없습니다."
        );
        let byte_len = std::mem::size_of_val(data);
        let raw = context.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            raw,
            len: data.len(),
            byte_len,
        }
    }

    pub fn from_i8_slice(context: &MetalContext, data: &[i8]) -> Self {
        assert!(
            !data.is_empty(),
            "빈 데이터로 MetalBuffer를 만들 수 없습니다."
        );
        let byte_len = std::mem::size_of_val(data);
        let raw = context.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            raw,
            len: data.len(),
            byte_len,
        }
    }

    pub fn from_u8_slice(context: &MetalContext, data: &[u8]) -> Self {
        assert!(
            !data.is_empty(),
            "빈 데이터로 MetalBuffer를 만들 수 없습니다."
        );
        let byte_len = std::mem::size_of_val(data);
        let raw = context.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_len as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Self {
            raw,
            len: data.len(),
            byte_len,
        }
    }

    pub fn empty(context: &MetalContext, len: usize) -> Self {
        assert!(len > 0, "길이가 0인 MetalBuffer는 만들 수 없습니다.");

        let byte_len = (len * mem::size_of::<f32>()) as u64;

        let raw = context
            .device
            .new_buffer(byte_len, MTLResourceOptions::StorageModeShared);
        context.record_buffer_allocation(byte_len as usize);

        Self {
            raw,
            len,
            byte_len: byte_len as usize,
        }
    }

    pub fn empty_with_element_size(
        context: &MetalContext,
        len: usize,
        element_size: usize,
    ) -> Self {
        assert!(len > 0 && element_size > 0, "invalid MetalBuffer size");
        let byte_len = len
            .checked_mul(element_size)
            .expect("MetalBuffer byte size overflow");
        let raw = context
            .device
            .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
        context.record_buffer_allocation(byte_len);
        Self { raw, len, byte_len }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn raw(&self) -> &Buffer {
        &self.raw
    }

    pub fn as_slice(&self) -> &[f32] {
        let ptr = self.raw.contents() as *const f32;

        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    pub fn as_f16_slice(&self) -> &[half::f16] {
        let ptr = self.raw.contents() as *const half::f16;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }
}
