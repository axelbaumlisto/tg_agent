use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("provider error: {0}")]
    Provider(String),

    #[error(transparent)]
    ProviderTyped(#[from] crate::provider::error::ProviderError),

    #[error("tool error: {tool}: {message}")]
    Tool { tool: String, message: String },

    #[error("session error: {0}")]
    Session(String),

    #[error("session not found: {0}")]
    SessionNotFound(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("provider not configured: {0}")]
    ProviderNotConfigured(String),

    #[error("config parse error: {0}")]
    ConfigParse(String),

    #[error("cancelled")]
    Cancelled,

    #[error("max iterations ({0}) exceeded")]
    MaxIterations(usize),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, AgentError>;
