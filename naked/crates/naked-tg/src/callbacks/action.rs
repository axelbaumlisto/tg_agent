#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CallbackAction<'a> {
    Permission { call_id: &'a str, action: &'a str },
    Provider { name: String },
    Model { name: String },
    Reasoning { level: &'a str },
    Research { action: &'a str, spec_id: &'a str },
    Command { name: &'a str },
    ModelPage { page: &'a str },
    Stream { action: &'a str },
    Error { action: &'a str },
    Unknown,
}

impl<'a> CallbackAction<'a> {
    pub(crate) fn parse(data: &'a str) -> Self {
        let parts: Vec<&str> = data.splitn(3, ':').collect();
        match parts.first().copied().unwrap_or("") {
            "p" if parts.len() == 3 => Self::Permission {
                call_id: parts[1],
                action: parts[2],
            },
            "sp" if parts.len() >= 2 => Self::Provider {
                name: parts[1..].join(":"),
            },
            "sm" if parts.len() >= 2 => Self::Model {
                name: parts[1..].join(":"),
            },
            "sr" if parts.len() >= 2 => Self::Reasoning { level: parts[1] },
            "r" if parts.len() >= 3 => Self::Research {
                action: parts[1],
                spec_id: parts[2],
            },
            "cmd" if parts.len() >= 2 => Self::Command { name: parts[1] },
            "mp" if parts.len() >= 2 => Self::ModelPage { page: parts[1] },
            "stream" if parts.len() >= 2 => Self::Stream { action: parts[1] },
            "err" if parts.len() >= 2 => Self::Error { action: parts[1] },
            _ => Self::Unknown,
        }
    }

    pub(crate) fn prefix(data: &str) -> &str {
        data.split(':').next().unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::CallbackAction;

    #[test]
    fn parses_research_action_with_spec_id() {
        assert_eq!(
            CallbackAction::parse("r:stop:da-nang-food"),
            CallbackAction::Research {
                action: "stop",
                spec_id: "da-nang-food",
            }
        );
    }

    #[test]
    fn parses_stream_action() {
        assert_eq!(
            CallbackAction::parse("stream:sendnow"),
            CallbackAction::Stream { action: "sendnow" }
        );
    }

    #[test]
    fn keeps_colons_inside_provider_and_model_names() {
        assert_eq!(
            CallbackAction::parse("sp:openrouter:free"),
            CallbackAction::Provider {
                name: "openrouter:free".to_string(),
            }
        );
        assert_eq!(
            CallbackAction::parse("sm:vendor:model:v1"),
            CallbackAction::Model {
                name: "vendor:model:v1".to_string(),
            }
        );
    }

    #[test]
    fn malformed_callbacks_are_unknown() {
        assert_eq!(CallbackAction::parse(""), CallbackAction::Unknown);
        assert_eq!(CallbackAction::parse("r:stop"), CallbackAction::Unknown);
        assert_eq!(CallbackAction::parse("stream"), CallbackAction::Unknown);
    }
}
