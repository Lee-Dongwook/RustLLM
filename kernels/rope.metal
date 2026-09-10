#include <metal_stdlib>

using namespace metal;

kernel void rope_f32(
    device const float* input [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    device float* output [[buffer(3)]],

    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& half_dim [[buffer(6)]],
    constant uint& start_pos [[buffer(7)]],
    constant uint& total_pairs [[buffer(8)]],

    uint id [[thread_position_in_grid]]
) {
    if (id >= total_pairs) {
        return;
    }

    uint pair_index =
        id % half_dim;

    uint row =
        id / half_dim;

    uint position =
        row % seq_len;

    uint base =
        row * head_dim;

    uint first_index =
        base + pair_index;

    uint second_index =
        base + half_dim + pair_index;

    uint table_index =
        (start_pos + position) * half_dim
        + pair_index;

    float cosine =
        cos_table[table_index];

    float sine =
        sin_table[table_index];

    float first =
        input[first_index];

    float second =
        input[second_index];

    output[first_index] =
        first * cosine
        - second * sine;

    output[second_index] =
        second * cosine
        + first * sine;
}

kernel void rope_f16(
    device const half* input [[buffer(0)]],
    device const float* cos_table [[buffer(1)]],
    device const float* sin_table [[buffer(2)]],
    device half* output [[buffer(3)]],
    device half* output [[buffer(3)]],
    constant uint& seq_len [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& half_dim [[buffer(6)]],
    constant uint& start_pos [[buffer(7)]],
    constant uint& total_pairs [[buffer(8)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_pairs) {
        return;
    }

    uint pair_index = id % half_dim;
    uint row = id / half_dim;
    uint position = row % seq_len;
    uint base = row * head_dim;
    uint first_index = base + pair_index;
    uint second_index = base + half_dim + pair_index;
    uint table_index = (start_pos + position) * half_dim + pair_index;

    float cosine = cos_table[table_index];
    float sine = sin_table[table_index];
    float first = float(input[first_index]);
    float second = float(input[second_index]);

    output[first_index] = half(first * cosine - second * sine);
    output[second_index] = half(second * cosine + first * sine);
}
