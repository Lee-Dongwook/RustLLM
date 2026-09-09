#include <metal_stdlib>

using namespace metal;

kernel void silu_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],

    constant uint& numel [[buffer(2)]],

    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) {
        return;
    }

    float x = input[id];

    float sigmoid =
        1.0f / (1.0f + exp(-x));

    output[id] =
        x * sigmoid;
}
