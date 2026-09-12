# RustLLM Test Knowledge Base

Rust uses ownership to manage memory safely.

Ownership means that each value has an owner.
When the owner goes out of scope, the value is dropped.

Cargo is Rust's package manager and build tool.

Apple Metal is a low-level graphics and compute API.
RustLLM uses Metal to execute tensor operations on Apple GPUs.

RustLLM supports F16 model inference.

RustLLM also supports INT8 weight-only quantization with F16 activations.

The project uses a KV cache during autoregressive generation.
The KV cache avoids recomputing keys and values for previously processed tokens.
