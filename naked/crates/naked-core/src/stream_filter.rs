//! Streaming text filter — strips fake tool-call wrappers from model output.

const START_MARKERS: [&str; 5] = [
    "[TOOL_CALL]",
    "<deepseek:tool_call",
    "<tool_call",
    "<invoke ",
    "<function_calls>",
];

const END_MARKERS: [&str; 5] = [
    "[/TOOL_CALL]",
    "</deepseek:tool_call>",
    "</tool_call>",
    "</invoke>",
    "</function_calls>",
];

pub fn contains_fake_wrapper(text: &str) -> bool {
    START_MARKERS.iter().any(|m| text.contains(m))
}

pub fn filter_fake_tool_delta(delta: &str, in_fake: &mut bool) -> String {
    if delta.is_empty() {
        return String::new();
    }
    let mut output = String::new();
    let mut rest = delta;
    loop {
        if *in_fake {
            match find_first_marker(rest, &END_MARKERS) {
                Some((idx, len)) => {
                    rest = &rest[idx + len..];
                    *in_fake = false;
                }
                None => break,
            }
        } else {
            match find_first_marker(rest, &START_MARKERS) {
                Some((idx, _)) => {
                    output.push_str(&rest[..idx]);
                    let mlen = START_MARKERS
                        .iter()
                        .find(|m| rest[idx..].starts_with(**m))
                        .map(|m| m.len())
                        .unwrap_or(1);
                    rest = &rest[idx + mlen..];
                    *in_fake = true;
                }
                None => {
                    output.push_str(rest);
                    break;
                }
            }
        }
    }
    output
}

fn find_first_marker(text: &str, markers: &[&str]) -> Option<(usize, usize)> {
    markers
        .iter()
        .filter_map(|m| text.find(m).map(|idx| (idx, m.len())))
        .min_by_key(|(idx, _)| *idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_text_passes_through() {
        let mut f = false;
        assert_eq!(filter_fake_tool_delta("hello world", &mut f), "hello world");
    }

    #[test]
    fn complete_wrapper_stripped() {
        let mut f = false;
        assert_eq!(
            filter_fake_tool_delta("before[TOOL_CALL]json[/TOOL_CALL]after", &mut f),
            "beforeafter"
        );
    }

    #[test]
    fn xml_style_wrapper_stripped() {
        let mut f = false;
        assert_eq!(
            filter_fake_tool_delta("text<function_calls>json</function_calls>more", &mut f),
            "textmore"
        );
    }

    #[test]
    fn split_across_deltas() {
        let mut f = false;
        assert_eq!(
            filter_fake_tool_delta("hello[TOOL_CALL]partial", &mut f),
            "hello"
        );
        assert!(f);
        assert_eq!(
            filter_fake_tool_delta("rest[/TOOL_CALL]world", &mut f),
            "world"
        );
        assert!(!f);
    }

    #[test]
    fn empty_delta_noop() {
        let mut f = false;
        assert_eq!(filter_fake_tool_delta("", &mut f), "");
    }

    #[test]
    fn contains_check() {
        assert!(contains_fake_wrapper("foo[TOOL_CALL]bar"));
        assert!(!contains_fake_wrapper("normal text"));
    }

    #[test]
    fn nested_markers_outer_only() {
        let mut f = false;
        assert_eq!(
            filter_fake_tool_delta("a[TOOL_CALL]b<function_calls>c[/TOOL_CALL]d", &mut f),
            "ad"
        );
    }

    #[test]
    fn invoke_style() {
        let mut f = false;
        assert_eq!(
            filter_fake_tool_delta("pre<invoke tool=\"x\">cmd</invoke>post", &mut f),
            "prepost"
        );
    }
}
