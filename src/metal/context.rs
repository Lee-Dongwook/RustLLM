use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::Arc;

use metal::{CommandQueue, CompileOptions, ComputePipelineState, Device};

use crate::error::{Result, TinyError};
use crate::profile::MetalDecodeProfile;

pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,

    pipelines: RefCell<HashMap<PipelineKey, Arc<ComputePipelineState>>>,
    // None in ordinary inference: no allocation, locking, or counter updates.
    submission_profile: RefCell<Option<MetalDecodeProfile>>,
    active_command_buffers: Cell<usize>,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct PipelineKey {
    source_address: usize,
    function_name: String,
}

impl MetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| {
            TinyError::Metal(
                "no Metal GPU is available; run inference on macOS with an Apple Metal-capable GPU"
                    .to_string(),
            )
        })?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue,
            pipelines: RefCell::new(HashMap::new()),
            submission_profile: RefCell::new(None),
            active_command_buffers: Cell::new(0),
        })
    }

    /// Starts decode-only submission profiling. This is deliberately explicit
    /// so prefill and normal inference never update per-kernel counters.
    pub fn begin_decode_submission_profile(&self) {
        *self.submission_profile.borrow_mut() = Some(MetalDecodeProfile::default());
    }

    pub fn take_decode_submission_profile(&self) -> Option<MetalDecodeProfile> {
        self.submission_profile.borrow_mut().take()
    }

    pub(crate) fn record_command_buffer(&self) {
        if let Some(profile) = self.submission_profile.borrow_mut().as_mut() {
            profile.record_command_buffer();
        }
    }

    pub(crate) fn record_commit(&self) {
        if let Some(profile) = self.submission_profile.borrow_mut().as_mut() {
            profile.record_commit();
        }
    }

    pub(crate) fn record_wait(&self) {
        if let Some(profile) = self.submission_profile.borrow_mut().as_mut() {
            profile.record_wait();
        }
    }

    pub(crate) fn record_buffer_allocation(&self, bytes: usize) {
        if let Some(profile) = self.submission_profile.borrow_mut().as_mut() {
            profile.record_buffer_allocation(bytes);
        }
    }

    pub(crate) fn begin_execution(&self) {
        self.active_command_buffers
            .set(self.active_command_buffers.get() + 1);
    }

    pub(crate) fn end_execution(&self) {
        self.active_command_buffers
            .set(self.active_command_buffers.get().saturating_sub(1));
    }

    pub fn pipeline(&self, shader_source: &str, function_name: &str) -> Arc<ComputePipelineState> {
        // Every current call site creates exactly one command buffer and compute
        // encoder, dispatches exactly one kernel, then commits and waits. Keep
        // this branch here rather than in every op so disabled profiling does
        // not allocate or synchronize per kernel.
        if let Some(profile) = self.submission_profile.borrow_mut().as_mut() {
            profile.record_kernel_dispatch(function_name);
            profile.record_encoder();
            if self.active_command_buffers.get() == 0 {
                profile.record_command_buffer();
                profile.record_commit();
                profile.record_wait();
            }
        }
        let key = PipelineKey {
            source_address: shader_source.as_ptr() as usize,
            function_name: function_name.to_owned(),
        };
        if let Some(pipeline) = self.pipelines.borrow().get(&key).cloned() {
            return pipeline;
        }

        if std::env::var_os("TINY_METAL_LLM_DEBUG").is_some() {
            eprintln!("[metal] compiling pipeline: {}", function_name);
        }

        let compile_options = CompileOptions::new();

        let library = self
            .device
            .new_library_with_source(shader_source, &compile_options)
            .expect("Metal shader 컴파일에 실패했습니다.");

        let function = library
            .get_function(function_name, None)
            .expect("Metal kernel function을 찾을 수 없습니다.");

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .expect("Compute pipeline 생성에 실패했습니다.");

        let pipeline = Arc::new(pipeline);

        self.pipelines
            .borrow_mut()
            .insert(key, Arc::clone(&pipeline));
        pipeline
    }
}
