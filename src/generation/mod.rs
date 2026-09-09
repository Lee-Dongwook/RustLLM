mod sampler;
mod generate;

pub use generate::{
    generate_greedy,
    generate_greedy_stream,
};
pub use sampler::greedy_next_token;
