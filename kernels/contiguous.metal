#include <metal_stdlib>

using namespace metal;

kernel void contiguous_f32(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],

    constant uint* dims [[buffer(2)]],
    constant uint* strides [[buffer(3)]],

    constant uint& rank [[buffer(4)]],
    constant uint& numel [[buffer(5)]],

    uint id [[thread_position_in_grid]]
) {
    if (id >= numel) {
        return;
    }

    uint remaining = id;
    uint source_index = 0;

    for (int dim = int(rank) -1; dim >= 0; --dim) {
        uint dimension_size = dims[dim];

        uint coordinate = remaining % dimension_size;

        remaining /= dimension_size;

        source_index += coordinate * strides[dim];
    }

    destination[id] = source[source_index];
}
