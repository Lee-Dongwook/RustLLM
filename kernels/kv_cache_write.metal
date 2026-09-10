#include <metal_stdlib>
using namespace metal;

kernel void kv_cache_write_f32(
    device float* cache [[buffer(0)]],
    device const float* source [[buffer(1)]],
    constant uint& start [[buffer(2)]],
    constant uint& source_seq [[buffer(3)]],
    constant uint& max_seq [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& numel [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint dim = id % head_dim;
    uint row = id / head_dim;
    uint seq = row % source_seq;
    uint head = row / source_seq;
    cache[(head * max_seq + start + seq) * head_dim + dim] = source[id];
}

kernel void kv_cache_write_f16(
    device half* cache [[buffer(0)]],
    device const half* source [[buffer(1)]],
    constant uint& start [[buffer(2)]],
    constant uint& source_seq [[buffer(3)]],
    constant uint& max_seq [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant uint& numel [[buffer(6)]],
    uint id [[thread_position_in_grid]]) {
    if (id >= numel) return;
    uint dim = id % head_dim;
    uint row = id / head_dim;
    uint seq = row % source_seq;
    uint head = row / source_seq;
    cache[(head * max_seq + start + seq) * head_dim + dim] = source[id];
}
