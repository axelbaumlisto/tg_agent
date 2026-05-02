//! Thread-local image collector for tool results.
//!
//! Tools that produce images (read_file on PNG, screenshot tools) push
//! base64 image data here. The agent loop picks it up after tool execution
//! and injects ContentBlock::Image into the tool result message.
//!
//! This avoids changing ToolResult (181 call sites) while still allowing
//! vision models to see tool-produced images.

use std::cell::RefCell;

thread_local! {
    static PENDING_IMAGES: RefCell<Vec<(String, String)>> = const { RefCell::new(Vec::new()) };
}

/// Push an image from a tool execution. Called by tools like read_file.
pub fn push_image(mime: &str, base64_data: &str) {
    PENDING_IMAGES.with(|cell| {
        cell.borrow_mut()
            .push((mime.to_string(), base64_data.to_string()));
    });
}

/// Drain all pending images. Called by the agent loop after tool execution.
pub fn drain_images() -> Vec<(String, String)> {
    PENDING_IMAGES.with(|cell| cell.borrow_mut().drain(..).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_drain() {
        push_image("image/png", "iVBOR...");
        push_image("image/jpeg", "/9j/4AAQ...");

        let images = drain_images();
        assert_eq!(images.len(), 2);
        assert_eq!(images[0].0, "image/png");
        assert_eq!(images[1].0, "image/jpeg");

        // Drain again should be empty
        assert!(drain_images().is_empty());
    }
}
