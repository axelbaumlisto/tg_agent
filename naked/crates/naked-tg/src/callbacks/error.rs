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
}
