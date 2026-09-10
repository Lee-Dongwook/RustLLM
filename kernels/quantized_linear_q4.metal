#include <metal_stdlib>
using namespace metal;

kernel void quantized_linear_q4_f16(
    device const half* input [[buffer(0)]],
    device const uchar* packed_weight [[buffer(1)]],
    device const float* scales [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant uint& m [[buffer(4)]], constant uint& k [[buffer(5)],
    constant uint& n [[buffer(6)]], constant uint& block_size [[buffer(7)],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= n || gid.y >= m) return;
    float sum = 0.0f;
    for (uint index = 0; index < k; ++index) {
        const uchar packed = packed_weight[(index / 2) * n + gid.x];
        const int code = (index & 1) == 0 ? int(packed & 0x0f) : int(packed >> 4);
        const float value = float(code - 8) * scales[(index / block_size) * n + gid.x];
        sum += float(input[gid.y * k + index]) * value;
    }
    output[gid.y * n + gid.x] = half(sum);
}
