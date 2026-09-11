use metal::CommandBufferRef;
use ::metal::MTLSize;

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

pub fn add_f32(context: &MetalContext, a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = add_f32_encode(
        context,
        execution.command_buffer(),
        a,
        b,
    )?;

    execution.finish();

    Ok(output)
}

pub(crate) fn add_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    a: &MetalBuffer,
    b: &MetalBuffer,
) -> Result<MetalBuffer> {
     if a.len() != b.len() {
        return Err(TinyError::InvalidShape(format!(
            "add requires equal buffer lengths, got {} and {}",
            a.len(),
            b.len(),
        )));
    }

    let output = MetalBuffer::empty(
        context,
        a.len(),
    );

    let shader_source = include_str!("../../kernels/vector_add.metal");
    let pipeline = context.pipeline(shader_source, "vector_add");

    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(a.raw()), 0);
    encoder.set_buffer(1, Some(b.raw()), 0);
    encoder.set_buffer(2, Some(output.raw()), 0);

    let grid = MTLSize::new(a.len() as u64, 1, 1);
    let threads_per_group = MTLSize::new(a.len().min(256) as u64, 1, 1);
    encoder.dispatch_threads(grid, threads_per_group);
    encoder.end_encoding();

    Ok(output)
}   

pub fn add_f16(
    context: &MetalContext,
    a: &MetalBuffer,
    b: &MetalBuffer,
) -> Result<MetalBuffer> {
    let execution =
        MetalExecution::new(context);

    let output =
        add_f16_encode(
            context,
            execution.command_buffer(),
            a,
            b,
        )?;

    execution.finish();

    Ok(output)
}

pub(crate) fn add_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    a: &MetalBuffer,
    b: &MetalBuffer,
) -> Result<MetalBuffer> {
    if a.len() != b.len()
        || a.byte_len() != a.len() * 2
        || b.byte_len() != b.len() * 2
    {
        return Err(
            TinyError::InvalidShape(
                format!(
                    "F16 add requires equal F16 buffers, got {} elements/{} bytes and {} elements/{} bytes",
                    a.len(),
                    a.byte_len(),
                    b.len(),
                    b.byte_len(),
                ),
            ),
        );
    }

    let output =
        MetalBuffer::empty_with_element_size(
            context,
            a.len(),
            2,
        );

    let pipeline =
        context.pipeline(
            include_str!(
                "../../kernels/vector_add.metal"
            ),
            "vector_add_f16",
        );

    let encoder =
        command_buffer
            .new_compute_command_encoder();

    encoder.set_compute_pipeline_state(
        pipeline.as_ref(),
    );

    encoder.set_buffer(
        0,
        Some(a.raw()),
        0,
    );

    encoder.set_buffer(
        1,
        Some(b.raw()),
        0,
    );

    encoder.set_buffer(
        2,
        Some(output.raw()),
        0,
    );

    encoder.dispatch_threads(
        MTLSize::new(
            a.len() as u64,
            1,
            1,
        ),
        MTLSize::new(
            a.len().min(256) as u64,
            1,
            1,
        ),
    );

    encoder.end_encoding();
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::add_f16_encode;

    use crate:: {
        metal::{MetalContext, MetalExecution},
        tensor::{DType, Tensor},
    };

    #[test]
    fn batches_two_f16_adds_in_one_command_buffer() {
        let Ok(context) = MetalContext::new() else {
            return;
        };

        let a = Tensor::from_f16_slice(
            &context,
            &[
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
            ],
            &[2],
        )
        .unwrap();

        let b = Tensor::from_f16_slice(
            &context,
            &[
                half::f16::from_f32(10.0),
                half::f16::from_f32(20.0),
            ],
            &[2],
        )
        .unwrap();

        let c = Tensor::from_f16_slice(
            &context,
            &[
                half::f16::from_f32(100.0),
                half::f16::from_f32(200.0),
            ],
            &[2],
        )
        .unwrap();

        let execution =
            MetalExecution::new(&context);

        let first =
            add_f16_encode(
                &context,
                execution.command_buffer(),
                a.metal_buffer().unwrap(),
                b.metal_buffer().unwrap(),
            )
            .unwrap();

        let second =
            add_f16_encode(
                &context,
                execution.command_buffer(),
                &first,
                c.metal_buffer().unwrap(),
            )
            .unwrap();

        // 여기까지 GPU 완료를 기다린 적이 없다.

        execution.finish();

        let result =
            Tensor::from_metal_buffer(
                second,
                &[2],
                DType::F16,
            )
            .unwrap();

        let values =
            result.to_f32_vec().unwrap();

        assert!(
            (values[0] - 111.0).abs()
                < 0.01
        );

        assert!(
            (values[1] - 222.0).abs()
                < 0.01
        );
    }
}
