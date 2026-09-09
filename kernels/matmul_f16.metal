#include <metal_stdlib>
using namespace metal;
kernel void matmul_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]], device half* result [[buffer(2)]], constant uint& m [[buffer(3)]], constant uint& k [[buffer(4)]], constant uint& n [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= n || gid.y >= m) return;
    float sum = 0.0f;
    for (uint i = 0; i < k; ++i) sum += float(a[gid.y * k + i]) * float(b[i * n + gid.x]);
    result[gid.y * n + gid.x] = half(sum);
}
