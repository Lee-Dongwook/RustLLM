use metal::{CommandBuffer, CommandBufferRef};

use super::MetalContext;

pub struct MetalExecution<'a> {
    context: &'a MetalContext,
    command_buffer: CommandBuffer,
}

impl<'a> MetalExecution<'a> {
    pub fn new(context: &'a MetalContext) -> Self {
        let command_buffer = context.command_queue.new_command_buffer().to_owned();

        context.record_command_buffer();
        context.begin_execution();
        Self {
            context,
            command_buffer,
        }
    }

    pub fn context(&self) -> &'a MetalContext {
        self.context
    }

    pub fn command_buffer(&self) -> &CommandBufferRef {
        self.command_buffer.as_ref()
    }

    pub fn finish(self) {
        self.context.end_execution();
        self.context.record_commit();
        self.command_buffer.commit();
        self.context.record_wait();
        self.command_buffer.wait_until_completed();
    }
}
