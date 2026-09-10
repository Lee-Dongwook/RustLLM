#include <metal_stdlib>

using namespace metal;

kernel void vector_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = a[id] + b[id];
}

kernel void vector_add_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* result [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    result[id] = half(float(a[id]) + float(b[id]));
}
