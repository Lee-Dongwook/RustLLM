#include <metal_stdlib>

using namespace metal;

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],

    constant uint& numel [[buffer(3)]],

    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) {
        return;
    }

    output[id] =
        a[id] * b[id];
}
