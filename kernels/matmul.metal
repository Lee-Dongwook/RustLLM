#include <metal_stdlib>

using namespace metal;

kernel void matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],

    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],

    uint2 gid [[thread_position_in_grid]]
) {
    uint col = gid.x;
    uint row = gid.y;

    if(row >= m || col >= n) {
        return;
    }

    float sum = 0.0f;

    for (uint i = 0; i < k; ++i) {
        float a_value = a[row * k + i];
        float b_value = b[i * n + col];

        sum += a_value * b_value;
    }

    result[row * n + col] = sum;
}  
