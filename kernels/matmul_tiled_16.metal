#include <metal_stdlib>

using namespace metal;

#define TILE_SIZE 16

kernel void matmul_tiled_16(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],

    constant uint& m [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    constant uint& n [[buffer(5)]],

    uint2 local_id [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]]
) {
    threadgroup float tile_a[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_b[TILE_SIZE][TILE_SIZE];

    uint local_col = local_id.x;
    uint local_row = local_id.y;

    uint row =
        group_id.y * TILE_SIZE + local_row;

    uint col =
        group_id.x * TILE_SIZE + local_col;

    float sum = 0.0f;

    uint tile_count =
        (k + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < tile_count; ++tile) {
        uint a_col =
            tile * TILE_SIZE + local_col;

        uint b_row =
            tile * TILE_SIZE + local_row;

        if (row < m && a_col < k) {
            tile_a[local_row][local_col] =
                a[row * k + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0f;
        }

        if (b_row < k && col < n) {
            tile_b[local_row][local_col] =
                b[b_row * n + col];
        } else {
            tile_b[local_row][local_col] = 0.0f;
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (uint i = 0; i < TILE_SIZE; ++i) {
            sum +=
                tile_a[local_row][i] *
                tile_b[i][local_col];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    if (row < m && col < n) {
        result[row * n + col] = sum;
    }
}
