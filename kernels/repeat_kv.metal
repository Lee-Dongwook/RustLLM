#include <metal_stdlib>

using namespace metal;

kernel void repeat_kv_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],

    constant uint& kv_heads [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& repeats [[buffer(5)]],

    uint index [[thread_position_in_grid]]
) {
    uint query_heads = kv_heads * repeats;
    uint d = index % head_dim;
    uint temp = index / head_dim;

    uint seq = temp % seq_len;
    temp /= seq_len;

    uint query_head = temp % query_heads;
    uint batch = temp / query_heads;

    uint kv_head = query_head / repeats;

    uint input_index = ((batch * kv_heads + kv_head) * seq_len + seq) * head_dim + d;

    output[index] = input[input_index];
}


kernel void repeat_kv_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],

    constant uint& kv_heads [[buffer(2)]],
    constant uint& seq_len [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant uint& repeats [[buffer(5)]],

    uint index [[thread_position_in_grid]]
) {
    uint query_heads =
        kv_heads * repeats;

    uint d =
        index % head_dim;

    uint temp =
        index / head_dim;

    uint seq =
        temp % seq_len;

    temp /=
        seq_len;

    uint query_head =
        temp % query_heads;

    uint batch =
        temp / query_heads;

    uint kv_head =
        query_head / repeats;

    uint input_index = ((batch * kv_heads + kv_head) * seq_len + seq) * head_dim + d;

    output[index] = input[input_index];
}
