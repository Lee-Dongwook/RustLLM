#include <metal_stdlib>

using namespace metal;

kernel void cast_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = half(input[id]);
}

kernel void cast_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    output[id] = float(input[id]);
}
