use crate::error::{Result, TinyError};

#[derive(Debug, Clone, Copy)]
pub struct AgentConfig {
    max_steps: usize,
}

impl AgentConfig {
    pub fn new(max_steps: usize) -> Result<Self> {
        if max_steps == 0 {
            return Err(TinyError::InvalidArgument(
                "agent max_steps must be greater than zero".to_string(),
            ));
        }

        Ok(Self { max_steps })
    }

    pub fn max_steps(&self) -> usize {
        self.max_steps
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self { max_steps: 5 }
    }
}

#[cfg(test)]
mod tests {
    use super::AgentConfig;

    #[test]
    fn creates_agent_config() {
        let config = AgentConfig::new(3).unwrap();

        assert_eq!(config.max_steps(), 3);
    }

    #[test]
    fn rejects_zero_steps() {
        assert!(AgentConfig::new(0).is_err());
    }
}
