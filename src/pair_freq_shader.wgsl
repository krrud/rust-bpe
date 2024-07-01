@group(0) @binding(0) var<storage, read> token_indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> pair_frequencies: array<array<atomic<u32>, 128>, 128>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i < arrayLength(&token_indices) - 1u {
        let a = token_indices[i];
        let b = token_indices[i + 1];
        if (a != 0u && b != 0u) { // Skipping pairs where either token is '0'
            atomicAdd(&pair_frequencies[a][b], 1u);
        }
    }
}
