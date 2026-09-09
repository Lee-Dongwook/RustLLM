#include <metal_stdlib>

using namespace metal;

kernel void concat_sequence_f32(
    device const float* left
        [[buffer(0)]],

    device const float* right
        [[buffer(1)]],

    device float* output
        [[buffer(2)]],

    constant uint& left_seq
        [[buffer(3)]],

    constant uint& right_seq
        [[buffer(4)]],

    constant uint& num_heads
        [[buffer(5)]],

    constant uint& head_dim
        [[buffer(6)]],

    uint index
        [[thread_position_in_grid]]
) {
    uint output_seq = left_seq + right_seq;
    uint elements_per_batch = num_heads * output_seq * head_dim;

    uint batch = index / elements_per_batch;

    uint batch_offset = index % elements_per_batch;

    uint head =
        batch_offset
        / (output_seq * head_dim);

    uint head_offset =
        batch_offset
        % (output_seq * head_dim);

    uint seq =
        head_offset
        / head_dim;

    uint dim =
        head_offset
        % head_dim;
    
    if (seq < left_seq) {
        uint source_index =
            (
                (
                    batch
                    * num_heads
                    + head
                )
                * left_seq
                + seq
            )
            * head_dim
            + dim;

        output[index] =
            left[source_index];
    } else {
        uint right_position =
            seq - left_seq;

        uint source_index =
            (
                (
                    batch
                    * num_heads
                    + head
                )
                * right_seq
                + right_position
            )
            * head_dim
            + dim;

        output[index] =
            right[source_index];
    }
}
