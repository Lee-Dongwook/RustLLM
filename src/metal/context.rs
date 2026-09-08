use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use ::metal:: {
    CommandQueue,
    CompileOptions,
    ComputePipelineState,
    Device,
};

pub struct MetalContext {
    pub device: Device,
    pub command_queue: CommandQueue,

    pipelines:
        RefCell<HashMap<String, Arc<ComputePipelineState>>>,
}

impl MetalContext {
    pub fn new() -> Self {
        let device =
            Device::system_default().expect("Metal GPU를 찾을 수 없습니다.");
        
        let command_queue = device.new_command_queue();

        Self {
            device,
            command_queue,
            pipelines: RefCell::new(
                HashMap::new(),
            )
        }
    }

    pub fn pipeline(
        &self,
        shader_source: &str,
        function_name: &str,
    ) -> Arc<ComputePipelineState> {
        if let Some(pipeline) = 
            self.pipelines
                .borrow()
                .get(function_name)
                .cloned()
        {
            return pipeline;
        }

        println!(
            "[metal] compiling pipeline: {}",
            function_name
        );

        let compile_options = CompileOptions::new();

        let library = self
            .device
            .new_library_with_source(shader_source, &compile_options)
            .expect("Metal shader 컴파일에 실패했습니다.");

        let function = library
            .get_function(function_name, None)
            .expect("Metal kernel function을 찾을 수 없습니다.");

        let pipeline = self.device.new_compute_pipeline_state_with_function(&function)
            .expect("Compute pipeline 생성에 실패했습니다.");

        let pipeline = Arc::new(pipeline);

        self.pipelines
             .borrow_mut()
             .insert(
                function_name.to_string(),
                Arc::clone(&pipeline),
             );
        pipeline
    }
}
