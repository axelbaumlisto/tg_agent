use super::*;

pub(super) async fn handle_error_callback(
    bot: &Bot,
    q: &CallbackQuery,
    action: &str,
) -> Result<(), teloxide::RequestError> {
    match action {
        "retry" => {
            bot.answer_callback_query(q.id.clone())
                .text("🔄 Отправь сообщение ещё раз — переотправлю")
                .await?;
        }
        "switch" => {
            bot.answer_callback_query(q.id.clone())
                .text("🔀 Используй /model для смены")
                .await?;
        }
        _ => {
            bot.answer_callback_query(q.id.clone()).await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::callbacks::CallbackAction;

    #[test]
    fn error_callbacks_route_through_typed_action() {
        assert_eq!(
            CallbackAction::parse("err:retry"),
            CallbackAction::Error { action: "retry" }
        );
        assert_eq!(
            CallbackAction::parse("err:switch"),
            CallbackAction::Error { action: "switch" }
        );
    }

    // ── BUG_REGISTRY D-CB-ERR-ACTION ───────────────────────────────
    //
    // `handle_error_callback` answers the callback query with
    // action-specific text: "retry" → the re-send hint, "switch" → the
    // /model hint, and any other action → a bare acknowledgement with NO
    // text. The only existing coverage
    // (`error_callbacks_route_through_typed_action`) asserts *parsing*
    // (`CallbackAction::parse("err:retry")`) — nothing exercises the
    // handler's answer text. Mutation probe confirmed: swapping the
    // "retry" and "switch" texts (or giving the fallback arm a text)
    // keeps all 25 callback tests green. This behavioral test closes that
    // gap by driving the real `handle_error_callback` against a wiremock
    // Telegram server and asserting the `answerCallbackQuery` JSON body's
    // `text` field EQUALS exactly the expected text per action, and is
    // ABSENT entirely for the fallback (bare acknowledgement).
    use teloxide::Bot;
    use teloxide::types::CallbackQuery;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const RETRY_TEXT: &str = "🔄 Отправь сообщение ещё раз — переотправлю";
    const SWITCH_TEXT: &str = "🔀 Используй /model для смены";

    /// Minimal `CallbackQuery` fixture — `handle_error_callback` only reads
    /// `q.id`, so the remaining fields are just enough to deserialize.
    fn err_callback_query(id: &str) -> CallbackQuery {
        serde_json::from_value(serde_json::json!({
            "id": id,
            "from": {"id": 42, "is_bot": false, "first_name": "tester"},
            "chat_instance": "chat-instance",
        }))
        .expect("valid callback query")
    }

    /// Drive `handle_error_callback` once for `action` against a fresh mock
    /// Telegram server and return the parsed `answerCallbackQuery` JSON
    /// body. teloxide serializes requests as `application/json`, so we
    /// deserialize with `serde_json` and inspect the `text` field directly
    /// (equality when present, absence for the bare acknowledgement).
    async fn answer_body_for(action: &str) -> serde_json::Value {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "ok": true,
                "result": true
            })))
            .mount(&server)
            .await;

        let url = reqwest::Url::parse(&server.uri()).unwrap();
        let bot = Bot::new("0:TEST_TOKEN").set_api_url(url);
        let q = err_callback_query(&format!("err-cb-{action}"));

        super::handle_error_callback(&bot, &q, action)
            .await
            .expect("handle_error_callback");

        let received = server.received_requests().await.unwrap();
        assert_eq!(
            received.len(),
            1,
            "expected exactly one answerCallbackQuery for action {action:?}, got {}",
            received.len()
        );
        assert!(
            received[0]
                .url
                .path()
                .to_ascii_lowercase()
                .ends_with("answercallbackquery"),
            "expected an answerCallbackQuery call, got path {}",
            received[0].url.path()
        );
        serde_json::from_slice(&received[0].body).unwrap_or_else(|e| {
            panic!(
                "answerCallbackQuery body must be valid JSON for action {action:?}: {e}; body: {}",
                String::from_utf8_lossy(&received[0].body)
            )
        })
    }

    /// Extract the `text` field of an `answerCallbackQuery` body, if present.
    fn answer_text(body: &serde_json::Value) -> Option<&str> {
        body.get("text").map(|v| {
            v.as_str()
                .expect("answerCallbackQuery `text` field must be a string")
        })
    }

    #[tokio::test]
    async fn handle_error_callback_answers_with_action_specific_text() {
        // retry → the `text` field EQUALS exactly the re-send hint.
        let retry = answer_body_for("retry").await;
        assert_eq!(
            answer_text(&retry),
            Some(RETRY_TEXT),
            "retry action must answer with exactly the retry text; got body: {retry}"
        );

        // switch → the `text` field EQUALS exactly the /model hint.
        let switch = answer_body_for("switch").await;
        assert_eq!(
            answer_text(&switch),
            Some(SWITCH_TEXT),
            "switch action must answer with exactly the switch text; got body: {switch}"
        );

        // fallback (unknown action) → bare acknowledgement: the `text` field
        // must be ABSENT entirely (not merely different from the known
        // texts). This guards against a regression like `.text("unknown
        // error")` on the fallback arm.
        let fallback = answer_body_for("bogus").await;
        assert_eq!(
            answer_text(&fallback),
            None,
            "fallback action must answer with NO text field; got body: {fallback}"
        );
    }
}
