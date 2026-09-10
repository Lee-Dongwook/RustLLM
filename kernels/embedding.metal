#include <metal_stdlib>

using namespace metal;

kernel void embedding_f32(
    device const float* weight [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device float* output [[buffer(2)]],

    constant uint& hidden_size [[buffer(3)]],
    constant uint& output_elements [[buffer(4)]],

    uint id [[thread_position_in_grid]]
) {
    if (id >= output_elements) {
        return;
    }

    uint token_index = 
        id / hidden_size;
    
    uint hidden_index = 
        id % hidden_size;
    
    uint token_id = 
        token_ids[token_index];
    
    uint weight_index =
        token_id * hidden_size
            + hidden_index;
    
    output[id] = 
        weight[weight_index];
}

kernel void embedding_f16(
    device const half* weight [[buffer(0)]],
    device const uint* token_ids [[buffer(1)]],
    device half* output [[buffer(2)]],

    constant uint& hidden_size [[buffer(3)]],
    constant uint& output_elements [[buffer(4)]],

    uint id [[thread_position_in_grid]]
) {
    if (id >= output_elements) {
        return;
    }

    uint token_index = id / hidden_size;
    uint hidden_index = id % hidden_size;
    uint token_id = token_ids[token_index];
    output[id] = weight[token_id * hidden_size + hidden_index];
}
