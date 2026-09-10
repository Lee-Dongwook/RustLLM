#include <metal_stdlib>

using namespace metal;

constant uint TILE_SIZE = 16;

kernel void gqa_qk_f32(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device float* output [[buffer(2)]],

    constant uint& q_heads [[buffer(3)]],
    constant uint& kv_heads [[buffer(4)]],
    constant uint& q_len [[buffer(5)]],
    constant uint& kv_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],

    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float tile_q[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_k[TILE_SIZE][TILE_SIZE];

    uint flat_head =
        group_id.z;

    uint batch =
        flat_head / q_heads;

    uint q_head =
        flat_head % q_heads;

    uint groups =
        q_heads / kv_heads;

    uint kv_head =
        q_head / groups;

    uint query_pos =
        group_id.y * TILE_SIZE
        + tid.y;

    uint key_pos =
        group_id.x * TILE_SIZE
        + tid.x;

    float sum =
        0.0f;

    uint q_base =
        (
            batch * q_heads
            + q_head
        )
        * q_len
        * head_dim;

    uint k_base =
        (
            batch * kv_heads
            + kv_head
        )
        * kv_len
        * head_dim;

    uint tile_count =
        (
            head_dim
            + TILE_SIZE
            - 1
        )
        / TILE_SIZE;

    for (
        uint tile = 0;
        tile < tile_count;
        ++tile
    ) {
        uint q_dim =
            tile * TILE_SIZE
            + tid.x;

        if (
            query_pos < q_len
            && q_dim < head_dim
        ) {
            tile_q[tid.y][tid.x] =
                q[
                    q_base
                    + query_pos * head_dim
                    + q_dim
                ];
        } else {
            tile_q[tid.y][tid.x] =
                0.0f;
        }

        uint k_pos =
            group_id.x * TILE_SIZE
            + tid.y;

        uint k_dim =
            tile * TILE_SIZE
            + tid.x;

        if (
            k_pos < kv_len
            && k_dim < head_dim
        ) {
            tile_k[tid.y][tid.x] =
                k[
                    k_base
                    + k_pos * head_dim
                    + k_dim
                ];
        } else {
            tile_k[tid.y][tid.x] =
                0.0f;
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (
            uint inner = 0;
            inner < TILE_SIZE;
            ++inner
        ) {
            sum +=
                tile_q[tid.y][inner]
                *
                tile_k[tid.x][inner];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    if (
        query_pos < q_len
        && key_pos < kv_len
    ) {
        uint output_base =
            (
                batch * q_heads
                + q_head
            )
            * q_len
            * kv_len;

        output[
            output_base
            + query_pos * kv_len
            + key_pos
        ] =
            sum;
    }
}

kernel void gqa_qk_f16(
    device const half* q [[buffer(0)]],
    device const half* k [[buffer(1)]],
    device half* output [[buffer(2)]],

    constant uint& q_heads [[buffer(3)]],
    constant uint& kv_heads [[buffer(4)]],
    constant uint& q_len [[buffer(5)]],
    constant uint& kv_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],

    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    threadgroup half tile_q[TILE_SIZE][TILE_SIZE];
    threadgroup half tile_k[TILE_SIZE][TILE_SIZE];

    uint flat_head =
        group_id.z;

    uint batch =
        flat_head / q_heads;

    uint q_head =
        flat_head % q_heads;

    uint groups =
        q_heads / kv_heads;

    uint kv_head =
        q_head / groups;

    uint query_pos =
        group_id.y * TILE_SIZE
        + tid.y;

    uint key_pos =
        group_id.x * TILE_SIZE
        + tid.x;

    float sum =
        0.0f;

    uint q_base =
        (
            batch * q_heads
            + q_head
        )
        * q_len
        * head_dim;

    uint k_base =
        (
            batch * kv_heads
            + kv_head
        )
        * kv_len
        * head_dim;

    uint tile_count =
        (
            head_dim
            + TILE_SIZE
            - 1
        )
        / TILE_SIZE;

    for (
        uint tile = 0;
        tile < tile_count;
        ++tile
    ) {
        uint q_dim =
            tile * TILE_SIZE
            + tid.x;

        if (
            query_pos < q_len
            && q_dim < head_dim
        ) {
            tile_q[tid.y][tid.x] =
                q[
                    q_base
                    + query_pos * head_dim
                    + q_dim
                ];
        } else {
            tile_q[tid.y][tid.x] =
                half(0.0h);
        }

        uint k_pos =
            group_id.x * TILE_SIZE
            + tid.y;

        uint k_dim =
            tile * TILE_SIZE
            + tid.x;

        if (
            k_pos < kv_len
            && k_dim < head_dim
        ) {
            tile_k[tid.y][tid.x] =
                k[
                    k_base
                    + k_pos * head_dim
                    + k_dim
                ];
        } else {
            tile_k[tid.y][tid.x] =
                half(0.0h);
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (
            uint inner = 0;
            inner < TILE_SIZE;
            ++inner
        ) {
            sum +=
                float(
                    tile_q[tid.y][inner]
                )
                *
                float(
                    tile_k[tid.x][inner]
                );
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    if (
        query_pos < q_len
        && key_pos < kv_len
    ) {
        uint output_base =
            (
                batch * q_heads
                + q_head
            )
            * q_len
            * kv_len;

        output[
            output_base
            + query_pos * kv_len
            + key_pos
        ] =
            half(sum);
    }
}

kernel void gqa_pv_f32(
    device const float* probs [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* output [[buffer(2)]],

    constant uint& q_heads [[buffer(3)]],
    constant uint& kv_heads [[buffer(4)]],
    constant uint& q_len [[buffer(5)]],
    constant uint& kv_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],

    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float tile_p[TILE_SIZE][TILE_SIZE];
    threadgroup float tile_v[TILE_SIZE][TILE_SIZE];

    uint flat_head =
        group_id.z;

    uint batch =
        flat_head / q_heads;

    uint q_head =
        flat_head % q_heads;

    uint groups =
        q_heads / kv_heads;

    uint kv_head =
        q_head / groups;

    uint query_pos =
        group_id.y * TILE_SIZE
        + tid.y;

    uint dim =
        group_id.x * TILE_SIZE
        + tid.x;

    uint p_base =
        (
            batch * q_heads
            + q_head
        )
        * q_len
        * kv_len;

    uint v_base =
        (
            batch * kv_heads
            + kv_head
        )
        * kv_len
        * head_dim;

    float sum =
        0.0f;

    uint tile_count =
        (
            kv_len
            + TILE_SIZE
            - 1
        )
        / TILE_SIZE;

    for (
        uint tile = 0;
        tile < tile_count;
        ++tile
    ) {
        uint key =
            tile * TILE_SIZE
            + tid.x;

        if (
            query_pos < q_len
            && key < kv_len
        ) {
            tile_p[tid.y][tid.x] =
                probs[
                    p_base
                    + query_pos * kv_len
                    + key
                ];
        } else {
            tile_p[tid.y][tid.x] =
                0.0f;
        }

        uint v_key =
            tile * TILE_SIZE
            + tid.y;

        if (
            v_key < kv_len
            && dim < head_dim
        ) {
            tile_v[tid.y][tid.x] =
                v[
                    v_base
                    + v_key * head_dim
                    + dim
                ];
        } else {
            tile_v[tid.y][tid.x] =
                0.0f;
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (
            uint inner = 0;
            inner < TILE_SIZE;
            ++inner
        ) {
            sum +=
                tile_p[tid.y][inner]
                *
                tile_v[inner][tid.x];
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    if (
        query_pos < q_len
        && dim < head_dim
    ) {
        uint output_base =
            (
                batch * q_heads
                + q_head
            )
            * q_len
            * head_dim;

        output[
            output_base
            + query_pos * head_dim
            + dim
        ] =
            sum;
    }
}

kernel void gqa_pv_f16(
    device const half* probs [[buffer(0)]],
    device const half* v [[buffer(1)]],
    device half* output [[buffer(2)]],

    constant uint& q_heads [[buffer(3)]],
    constant uint& kv_heads [[buffer(4)]],
    constant uint& q_len [[buffer(5)]],
    constant uint& kv_len [[buffer(6)]],
    constant uint& head_dim [[buffer(7)]],

    uint3 group_id [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
) {
    threadgroup half tile_p[TILE_SIZE][TILE_SIZE];
    threadgroup half tile_v[TILE_SIZE][TILE_SIZE];

    uint flat_head =
        group_id.z;

    uint batch =
        flat_head / q_heads;

    uint q_head =
        flat_head % q_heads;

    uint groups =
        q_heads / kv_heads;

    uint kv_head =
        q_head / groups;

    uint query_pos =
        group_id.y * TILE_SIZE
        + tid.y;

    uint dim =
        group_id.x * TILE_SIZE
        + tid.x;

    uint p_base =
        (
            batch * q_heads
            + q_head
        )
        * q_len
        * kv_len;

    uint v_base =
        (
            batch * kv_heads
            + kv_head
        )
        * kv_len
        * head_dim;

    float sum =
        0.0f;

    uint tile_count =
        (
            kv_len
            + TILE_SIZE
            - 1
        )
        / TILE_SIZE;

    for (
        uint tile = 0;
        tile < tile_count;
        ++tile
    ) {
        uint key =
            tile * TILE_SIZE
            + tid.x;

        if (
            query_pos < q_len
            && key < kv_len
        ) {
            tile_p[tid.y][tid.x] =
                probs[
                    p_base
                    + query_pos * kv_len
                    + key
                ];
        } else {
            tile_p[tid.y][tid.x] =
                half(0.0h);
        }

        uint v_key =
            tile * TILE_SIZE
            + tid.y;

        if (
            v_key < kv_len
            && dim < head_dim
        ) {
            tile_v[tid.y][tid.x] =
                v[
                    v_base
                    + v_key * head_dim
                    + dim
                ];
        } else {
            tile_v[tid.y][tid.x] =
                half(0.0h);
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );

        for (
            uint inner = 0;
            inner < TILE_SIZE;
            ++inner
        ) {
            sum +=
                float(
                    tile_p[tid.y][inner]
                )
                *
                float(
                    tile_v[inner][tid.x]
                );
        }

        threadgroup_barrier(
            mem_flags::mem_threadgroup
        );
    }

    if (
        query_pos < q_len
        && dim < head_dim
    ) {
        uint output_base =
            (
                batch * q_heads
                + q_head
            )
            * q_len
            * head_dim;

        output[
            output_base
            + query_pos * head_dim
            + dim
        ] =
            half(sum);
    }
}
