#include <metal_stdlib>

using namespace metal;

#define TILE_SIZE 16

kernel void batched_matmul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],

    constant uint& batch_count [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],

    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    uint batch =
        group_id.z;

    if (batch >= batch_count) {
        return;
    }

    uint local_col =
        local_id.x;

    uint local_row =
        local_id.y;

    uint row =
        group_id.y * TILE_SIZE
        + local_row;

    uint col =
        group_id.x * TILE_SIZE
        + local_col;

    threadgroup float tile_a[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_b[TILE_SIZE][TILE_SIZE];

    uint a_batch_offset =
        batch * m * k;

    uint b_batch_offset =
        batch * k * n;

    uint output_batch_offset =
        batch * m * n;

    float sum =
        0.0f;

    uint tile_count =
        (k + TILE_SIZE - 1)
        / TILE_SIZE;

    for (
        uint tile = 0;
        tile < tile_count;
        ++tile
    ) {
        uint a_col =
            tile * TILE_SIZE
            + local_col;

        uint b_row =
            tile * TILE_SIZE
            + local_row;

        if (
            row < m
            && a_col < k
        ) {
            tile_a[local_row][local_col] =
                a[
                    a_batch_offset
                    + row * k
                    + a_col
                ];
        } else {
            tile_a[local_row][local_col] =
                0.0f;
        }

        if (
            b_row < k
            && col < n
        ) {
            tile_b[local_row][local_col] =
                b[
                    b_batch_offset
                    + b_row * n
                    + col
                ];
        } else {
            tile_b[local_row][local_col] =
                0.0f;
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (
            uint i = 0;
            i < TILE_SIZE;
            ++i
        ) {
            sum +=
                tile_a[local_row][i]
                *
                tile_b[i][local_col];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    if (
        row < m
        && col < n
    ) {
        output[
            output_batch_offset
            + row * n
            + col
        ] = sum;
    }
}

// Keep FP32 accumulation so the F16 path has the same numerical policy as
// matmul_f16: storage bandwidth is reduced without needlessly accumulating
// products in half precision.
kernel void batched_matmul_f16(
    device const half* a [[buffer(0)]],
    device const half* b [[buffer(1)]],
    device half* output [[buffer(2)]],

    constant uint& batch_count [[buffer(3)]],
    constant uint& m [[buffer(4)]],
    constant uint& k [[buffer(5)]],
    constant uint& n [[buffer(6)]],

    uint3 local_id [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]
) {
    uint batch = group_id.z;

    if (batch >= batch_count) {
        return;
    }

    uint local_col = local_id.x;
    uint local_row = local_id.y;
    uint row = group_id.y * TILE_SIZE + local_row;
    uint col = group_id.x * TILE_SIZE + local_col;

    threadgroup half tile_a[TILE_SIZE][TILE_SIZE];
    threadgroup half tile_b[TILE_SIZE][TILE_SIZE];

    uint a_batch_offset = batch * m * k;
    uint b_batch_offset = batch * k * n;
    uint output_batch_offset = batch * m * n;
    float sum = 0.0f;
    uint tile_count = (k + TILE_SIZE - 1) / TILE_SIZE;

    for (uint tile = 0; tile < tile_count; ++tile) {
        uint a_col = tile * TILE_SIZE + local_col;
        uint b_row = tile * TILE_SIZE + local_row;

        tile_a[local_row][local_col] =
            row < m && a_col < k
                ? a[a_batch_offset + row * k + a_col]
                : half(0.0h);

        tile_b[local_row][local_col] =
            b_row < k && col < n
                ? b[b_batch_offset + b_row * n + col]
                : half(0.0h);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_SIZE; ++i) {
            sum += float(tile_a[local_row][i]) * float(tile_b[i][local_col]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < m && col < n) {
        output[output_batch_offset + row * n + col] = half(sum);
    }
}
