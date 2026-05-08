//! Three-state patch field + `ResearchPatch` for partial spec updates.

use super::ResearchSpec;

/// Three-state patch field: leave unchanged, clear to `None`, or set to a value.
///
/// Replaces `Option<Option<T>>` for readability. `Default` = `Unchanged`.
#[derive(Debug, Clone, Default)]
pub enum PatchField<T> {
    /// Leave the target field as-is.
    #[default]
    Unchanged,
    /// Clear the target field to `None`.
    Clear,
    /// Set the target field to this value.
    Set(T),
}

impl<T> PatchField<T> {
    /// Apply this patch to an `Option<T>` target.
    pub fn apply(self, target: &mut Option<T>) {
        match self {
            Self::Unchanged => {}
            Self::Clear => *target = None,
            Self::Set(v) => *target = Some(v),
        }
    }
}

/// Backward-compat conversion from `Option<Option<T>>`.
impl<T> From<Option<Option<T>>> for PatchField<T> {
    fn from(opt: Option<Option<T>>) -> Self {
        match opt {
            None => Self::Unchanged,
            Some(None) => Self::Clear,
            Some(Some(v)) => Self::Set(v),
        }
    }
}

/// Partial mutation applied to a [`ResearchSpec`] by [`AgentCore::update_research`].
#[derive(Debug, Default, Clone)]
pub struct ResearchPatch {
    pub topic: Option<String>,
    /// Replace the entire sources list (after dedup, empty entries dropped).
    pub sources_replace: Option<Vec<String>>,
    /// Append to the sources list (skipping duplicates).
    pub sources_add: Option<Vec<String>>,
    pub interval_seconds: PatchField<u64>,
    /// One-shot at-time trigger.
    pub run_at: PatchField<chrono::DateTime<chrono::Utc>>,
    /// Recurring cron expression.
    pub cron: PatchField<String>,
    /// Per-spec scheduler-task timeout override (seconds).
    pub task_timeout_seconds: PatchField<u64>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub max_iterations: PatchField<u32>,
    pub max_wall_seconds: PatchField<u64>,
}

/// Apply a [`ResearchPatch`] to an in-memory [`ResearchSpec`] following the
/// same semantics used by [`AgentCore::update_research`]. Exposed so tests can
/// assert patch behavior without spinning up an [`AgentCore`].
pub fn apply_research_patch(spec: &mut ResearchSpec, patch: ResearchPatch) {
    if let Some(topic) = patch.topic {
        let trimmed = topic.trim();
        if !trimmed.is_empty() {
            spec.topic = trimmed.to_string();
        }
    }
    if let Some(sources) = patch.sources_replace {
        let mut seen = std::collections::HashSet::new();
        spec.sources = sources
            .into_iter()
            .filter(|s| !s.trim().is_empty())
            .filter(|s| seen.insert(s.clone()))
            .collect();
    }
    if let Some(extra) = patch.sources_add {
        for s in extra {
            let s = s.trim().to_string();
            if !s.is_empty() && !spec.sources.contains(&s) {
                spec.sources.push(s);
            }
        }
    }
    patch.interval_seconds.apply(&mut spec.interval_seconds);
    patch.run_at.apply(&mut spec.run_at);
    patch.cron.apply(&mut spec.cron);
    patch
        .task_timeout_seconds
        .apply(&mut spec.task_timeout_seconds);
    if let Some(provider) = patch.provider {
        spec.provider = if provider.trim().is_empty() {
            None
        } else {
            Some(provider)
        };
    }
    if let Some(model) = patch.model {
        spec.model = if model.trim().is_empty() {
            None
        } else {
            Some(model)
        };
    }
    patch.max_iterations.apply(&mut spec.max_iterations);
    patch.max_wall_seconds.apply(&mut spec.max_wall_seconds);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spec() -> ResearchSpec {
        ResearchSpec {
            id: "test-1".into(),
            topic: "Original topic".into(),
            sources: vec!["src1".into(), "src2".into()],
            interval_seconds: Some(3600),
            max_iterations: Some(10),
            max_wall_seconds: Some(300),
            ..Default::default()
        }
    }

    #[test]
    fn patch_unchanged_leaves_spec_intact() {
        let mut spec = make_spec();
        let original = spec.clone();
        apply_research_patch(&mut spec, ResearchPatch::default());
        assert_eq!(spec.topic, original.topic);
        assert_eq!(spec.interval_seconds, original.interval_seconds);
        assert_eq!(spec.max_iterations, original.max_iterations);
    }

    #[test]
    fn patch_set_overrides_field() {
        let mut spec = make_spec();
        apply_research_patch(
            &mut spec,
            ResearchPatch {
                interval_seconds: PatchField::Set(7200),
                max_iterations: PatchField::Set(20),
                ..Default::default()
            },
        );
        assert_eq!(spec.interval_seconds, Some(7200));
        assert_eq!(spec.max_iterations, Some(20));
    }

    #[test]
    fn patch_clear_removes_field() {
        let mut spec = make_spec();
        apply_research_patch(
            &mut spec,
            ResearchPatch {
                interval_seconds: PatchField::Clear,
                max_iterations: PatchField::Clear,
                ..Default::default()
            },
        );
        assert_eq!(spec.interval_seconds, None);
        assert_eq!(spec.max_iterations, None);
    }

    #[test]
    fn patch_topic_empty_string_is_ignored() {
        let mut spec = make_spec();
        apply_research_patch(
            &mut spec,
            ResearchPatch {
                topic: Some("   ".into()),
                ..Default::default()
            },
        );
        assert_eq!(spec.topic, "Original topic");
    }

    #[test]
    fn patch_sources_replace_deduplicates() {
        let mut spec = make_spec();
        apply_research_patch(
            &mut spec,
            ResearchPatch {
                sources_replace: Some(vec!["a".into(), "b".into(), "a".into(), "".into()]),
                ..Default::default()
            },
        );
        assert_eq!(spec.sources, vec!["a", "b"]);
    }

    #[test]
    fn patch_sources_add_skips_existing() {
        let mut spec = make_spec();
        apply_research_patch(
            &mut spec,
            ResearchPatch {
                sources_add: Some(vec!["src1".into(), "new".into()]),
                ..Default::default()
            },
        );
        assert_eq!(spec.sources, vec!["src1", "src2", "new"]);
    }

    #[test]
    fn patch_provider_empty_clears() {
        let mut spec = make_spec();
        spec.provider = Some("kimi".into());
        apply_research_patch(
            &mut spec,
            ResearchPatch {
                provider: Some("".into()),
                ..Default::default()
            },
        );
        assert_eq!(spec.provider, None);
    }

    #[test]
    fn patch_field_from_option_option() {
        let f: PatchField<u64> = None.into();
        assert!(matches!(f, PatchField::Unchanged));

        let f: PatchField<u64> = Some(None).into();
        assert!(matches!(f, PatchField::Clear));

        let f: PatchField<u64> = Some(Some(42)).into();
        assert!(matches!(f, PatchField::Set(42)));
    }

    #[test]
    fn patch_field_apply_unchanged() {
        let mut val = Some(100u64);
        PatchField::Unchanged.apply(&mut val);
        assert_eq!(val, Some(100));
    }

    #[test]
    fn patch_field_apply_clear() {
        let mut val = Some(100u64);
        PatchField::Clear.apply(&mut val);
        assert_eq!(val, None);
    }

    #[test]
    fn patch_field_apply_set() {
        let mut val: Option<u64> = None;
        PatchField::Set(42).apply(&mut val);
        assert_eq!(val, Some(42));

        mod prop {
            use super::super::*;
            use proptest::prelude::*;

            proptest! {
                #[test]
                fn apply_set_always_stores(value in 0u64..1000) {
                    let mut target: Option<u64> = None;
                    PatchField::Set(value).apply(&mut target);
                    prop_assert_eq!(target, Some(value));
                }

                #[test]
                fn apply_clear_always_clears(initial in proptest::option::of(0u64..1000)) {
                    let mut target = initial;
                    PatchField::Clear.apply(&mut target);
                    prop_assert_eq!(target, None);
                }

                #[test]
                fn apply_unchanged_preserves(initial in proptest::option::of(0u64..1000)) {
                    let mut target = initial;
                    PatchField::<u64>::Unchanged.apply(&mut target);
                    prop_assert_eq!(target, initial);
                }
            }
        }
    }
}
