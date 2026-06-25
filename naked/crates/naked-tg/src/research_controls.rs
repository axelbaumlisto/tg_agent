//! Pure inline-keyboard builders for B86 research schedule controls.
//!
//! This module intentionally contains no callback handling or streaming wiring:
//! it only centralises the short Telegram callback payloads and keyboard shapes
//! used by later B86 steps.

use naked_core::research::ResearchSpec;
use teloxide::types::{InlineKeyboardButton, InlineKeyboardMarkup};

const CB_SCHED: &str = "r:sch:";
const CB_UNSCHED: &str = "r:uns:";
const CB_DELETE: &str = "r:rm:";
const CB_DELETE_CONFIRM: &str = "r:rmc:";

/// Keyboard shown after a one-shot run completes: explicit opt-in to recurring.
pub fn keyboard_schedule_opt_in(spec_id: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![InlineKeyboardButton::callback(
        "⏰ Повторять по расписанию",
        format!("{CB_SCHED}{spec_id}"),
    )]])
}

/// Keyboard shown after each scheduled run: delete or keep but unschedule.
pub fn keyboard_scheduled(spec_id: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("🗑 Удалить", format!("{CB_DELETE}{spec_id}")),
        InlineKeyboardButton::callback("⏹ Отменить расписание", format!("{CB_UNSCHED}{spec_id}")),
    ]])
}

/// Confirmation keyboard for deleting a research spec.
pub fn keyboard_delete_confirm(spec_id: &str) -> InlineKeyboardMarkup {
    InlineKeyboardMarkup::new(vec![vec![
        InlineKeyboardButton::callback("✅ Да, удалить", format!("{CB_DELETE_CONFIRM}{spec_id}")),
        InlineKeyboardButton::callback("↩ Отмена", format!("{CB_UNSCHED}{spec_id}")),
    ]])
}

/// Return scheduled-run controls only for recurring specs.
pub fn scheduled_keyboard_for_spec(spec: &ResearchSpec) -> Option<InlineKeyboardMarkup> {
    (spec.interval_seconds.is_some() || spec.cron.is_some()).then(|| keyboard_scheduled(&spec.id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use teloxide::types::InlineKeyboardButtonKind;

    const LONG_SPEC_ID: &str = "n-ng-2026-04-23-30-2026-03-24-30-77-000-000-vnd--39d27efd";

    fn callbacks(markup: &InlineKeyboardMarkup) -> Vec<&str> {
        markup
            .inline_keyboard
            .iter()
            .flatten()
            .map(|button| match &button.kind {
                InlineKeyboardButtonKind::CallbackData(data) => data.as_str(),
                other => panic!("expected callback button, got {other:?}"),
            })
            .collect()
    }

    fn buttons(markup: &InlineKeyboardMarkup) -> Vec<(&str, &str)> {
        markup
            .inline_keyboard
            .iter()
            .flatten()
            .map(|button| {
                let data = match &button.kind {
                    InlineKeyboardButtonKind::CallbackData(data) => data.as_str(),
                    other => panic!("expected callback button, got {other:?}"),
                };
                (button.text.as_str(), data)
            })
            .collect()
    }

    fn spec_with(id: &str, interval_seconds: Option<u64>, cron: Option<&str>) -> ResearchSpec {
        ResearchSpec {
            id: id.to_string(),
            interval_seconds,
            cron: cron.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn callback_short_codes_fit_57_byte_spec_id() {
        assert_eq!(LONG_SPEC_ID.len(), 57);
        assert_eq!(CB_SCHED, "r:sch:");
        assert_eq!(CB_UNSCHED, "r:uns:");
        assert_eq!(CB_DELETE, "r:rm:");
        assert_eq!(CB_DELETE_CONFIRM, "r:rmc:");

        let recurring_spec = spec_with(LONG_SPEC_ID, Some(3600), None);
        let markups = [
            keyboard_schedule_opt_in(LONG_SPEC_ID),
            keyboard_scheduled(LONG_SPEC_ID),
            keyboard_delete_confirm(LONG_SPEC_ID),
            scheduled_keyboard_for_spec(&recurring_spec).expect("recurring"),
        ];

        for data in markups.iter().flat_map(callbacks) {
            assert!(data.len() <= 64, "callback_data too long: {data}");
            assert!(
                !data.starts_with("r:sched:"),
                "banned long schedule prefix emitted: {data}"
            );
            assert!(
                !data.starts_with("r:unsched:"),
                "banned long unschedule prefix emitted: {data}"
            );
        }

        assert_eq!(format!("{CB_SCHED}{LONG_SPEC_ID}").len(), 63);
        assert_eq!(format!("{CB_UNSCHED}{LONG_SPEC_ID}").len(), 63);
        assert_eq!(format!("{CB_DELETE}{LONG_SPEC_ID}").len(), 62);
        assert_eq!(format!("{CB_DELETE_CONFIRM}{LONG_SPEC_ID}").len(), 63);
    }

    #[test]
    fn scheduled_keyboard_has_delete_and_unschedule() {
        let keyboard = keyboard_scheduled("spec-1");
        let items = buttons(&keyboard);

        assert_eq!(items.len(), 2);
        assert!(items[0].0.contains("Удалить"));
        assert_eq!(items[0].1, "r:rm:spec-1");
        assert!(items[1].0.contains("Отменить"));
        assert_eq!(items[1].1, "r:uns:spec-1");
    }

    #[test]
    fn schedule_opt_in_single_button() {
        let keyboard = keyboard_schedule_opt_in("spec-1");
        let items = buttons(&keyboard);

        assert_eq!(items.len(), 1);
        assert!(items[0].0.contains("Повторять"));
        assert_eq!(items[0].1, "r:sch:spec-1");
    }

    #[test]
    fn delete_confirm_keyboard_has_rmc_and_cancel() {
        let keyboard = keyboard_delete_confirm(LONG_SPEC_ID);
        let items = buttons(&keyboard);

        assert_eq!(items.len(), 2);
        assert!(items[0].0.contains("Да"));
        assert_eq!(items[0].1, format!("{CB_DELETE_CONFIRM}{LONG_SPEC_ID}"));
        assert!(items[1].0.contains("Отмена"));
        assert_eq!(items[1].1, format!("{CB_UNSCHED}{LONG_SPEC_ID}"));
        assert!(items.iter().all(|(_, data)| data.len() <= 64));
    }

    #[test]
    fn scheduled_keyboard_for_spec_some_only_when_recurring() {
        let interval_only = spec_with("interval", Some(3600), None);
        let cron_only = spec_with("cron", None, Some("0 */6 * * *"));
        let one_shot = spec_with("one-shot", None, None);

        let interval_keyboard = scheduled_keyboard_for_spec(&interval_only).expect("interval");
        assert_eq!(
            callbacks(&interval_keyboard),
            vec!["r:rm:interval", "r:uns:interval"]
        );

        let cron_keyboard = scheduled_keyboard_for_spec(&cron_only).expect("cron");
        assert_eq!(callbacks(&cron_keyboard), vec!["r:rm:cron", "r:uns:cron"]);

        assert!(scheduled_keyboard_for_spec(&one_shot).is_none());
    }
}
