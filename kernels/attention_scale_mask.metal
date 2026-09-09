#include <metal_stdlib>

using namespace metal;

kernel void attention_scale_mask_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],

    constant float& scale [[buffer(2)]],
    constant uint& query_len [[buffer(3)]],
    constant uint& key_len [[buffer(4)]],
    constant uint& query_start_pos [[buffer(5)]],
    constant uint& numel [[buffer(6)]],

    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) {
        return;
    }

    uint key_index =
        id % key_len;

    uint row =
        id / key_len;

    uint query_index =
        row % query_len;

    uint absolute_query_position =
        query_start_pos
        + query_index;

    if (key_index > absolute_query_position) {
        output[id] =
            -INFINITY;
    } else {
        output[id] =
            input[id]
            * scale;
    }
}
