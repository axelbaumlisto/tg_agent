pub mod classifier;
pub mod daily;
pub mod digest;
pub mod dreams;
pub mod service;
pub mod store;
pub mod types;

#[cfg(test)]
pub(crate) static MEMORY_INJECTION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) fn record_injection_failed(stage: &'static str, error: impl std::fmt::Display) {
    use std::sync::atomic::Ordering;

    crate::types::MEMORY_INJECTION_FAILED_COUNT.fetch_add(1, Ordering::Relaxed);
    tracing::warn!(
        stage,
        error = %error,
        "memory injection failed; continuing without memory context"
    );
}
