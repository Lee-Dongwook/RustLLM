#include <metal_stdlib>

using namespace metal;

kernel void quantized_linear_i8_f16(
    device const half* input [[buffer(0)]],
    device const char* weight [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= n || gid.y >= m) {
        return;
    }

    float sum = 0.0f;
    const float scale = scales[gid.x];
    for (uint index = 0; index < k; ++index) {
        sum += float(input[gid.y * k + index])
            * float(weight[index * n + gid.x])
            * scale;
    }
    output[gid.y * n + gid.x] = half(sum);
}
