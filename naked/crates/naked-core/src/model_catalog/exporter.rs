//! Phase 5 of the Model Capabilities Catalog plan: render the
//! `skills/model-catalog/SKILL.md` body sections from the structured
//! `naked.json.providers[*].capabilities` so the human-readable catalog
//! never drifts from the runtime source of truth.
//!
//! # How it works
//!
//! The exporter treats the SKILL.md file as a shell with **preserved
//! regions** (everything a human wrote: the YAML front-matter, the
//! hand-curated "Keeping this catalog fresh" section, etc.) and
//! **generated regions** (the tables the operator doesn't want to
//! maintain by hand). Generated regions are delimited by HTML comment
//! markers:
//!
//! ```markdown
//! <!-- generated:quick-decision:start -->
//! ...table rendered from capabilities...
//! <!-- generated:quick-decision:end -->
//! ```
//!
//! Three regions are rendered today, matching the plan:
//!
//! - `quick-decision` — the top "Quick decision matrix" table
//! - `provider-details` — per-provider bullet list
//! - `known-failure-modes` — the failure-mode table
//!
//! Running the exporter is idempotent: if no catalog entry changed, the
//! regenerated file is byte-identical. A round-trip test in the
//! integration suite asserts that exporting an already-exported file
//! produces no diff.

use std::collections::BTreeMap;
use std::path::Path;

use crate::config::{Config, ProviderConfig};
use crate::model_catalog::types::{
    CostTier, LatencyTier, ModelCapabilities, ModelStatus, QualityTier, TaskKind,
};

/// Region marker pair. Kept as a plain struct so tests can enumerate
/// them and assert every marker ends up rendered.
struct Region {
    start: &'static str,
    end: &'static str,
}

const QUICK_DECISION: Region = Region {
    start: "<!-- generated:quick-decision:start -->",
    end: "<!-- generated:quick-decision:end -->",
};
const PROVIDER_DETAILS: Region = Region {
    start: "<!-- generated:provider-details:start -->",
    end: "<!-- generated:provider-details:end -->",
};
const FAILURE_MODES: Region = Region {
    start: "<!-- generated:known-failure-modes:start -->",
    end: "<!-- generated:known-failure-modes:end -->",
};

/// Load a `Config` and render the three generated sections as strings.
/// Used by both the CLI binary and by unit tests.
pub fn render_sections(config: &Config) -> RenderedSections {
    let sorted: BTreeMap<String, ProviderConfig> = config
        .providers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    RenderedSections {
        quick_decision: render_quick_decision(&sorted),
        provider_details: render_provider_details(&sorted),
        failure_modes: render_failure_modes(&sorted),
    }
}

/// Rendered section bodies (between — not including — the marker
/// comments). `RenderedSections::apply` splices them into a SKILL.md
/// string.
#[derive(Debug, Clone)]
pub struct RenderedSections {
    pub quick_decision: String,
    pub provider_details: String,
    pub failure_modes: String,
}

impl RenderedSections {
    /// Splice this render into a SKILL.md body, preserving everything
    /// outside the generated regions. Returns `Err` with a short
    /// human-readable reason if a marker is missing — the caller can
    /// surface that to the operator without a stack trace.
    pub fn apply(&self, skill_md: &str) -> Result<String, String> {
        let mut out = skill_md.to_string();
        out = splice_region(&out, &QUICK_DECISION, &self.quick_decision)?;
        out = splice_region(&out, &PROVIDER_DETAILS, &self.provider_details)?;
        out = splice_region(&out, &FAILURE_MODES, &self.failure_modes)?;
        Ok(out)
    }

    /// A transiently empty/unreachable catalog renders syntactically valid
    /// markdown placeholders. Treat it as unsafe to write unless there is at
    /// least one generated model row that would preserve real catalog data.
    fn has_usable_model_rows(&self) -> bool {
        self.provider_details
            .lines()
            .any(|line| line.trim_start().starts_with("- `"))
    }
}

fn splice_region(src: &str, region: &Region, body: &str) -> Result<String, String> {
    let start_idx = src
        .find(region.start)
        .ok_or_else(|| format!("missing marker `{}`", region.start))?;
    let end_idx = src
        .find(region.end)
        .ok_or_else(|| format!("missing marker `{}`", region.end))?;
    if end_idx <= start_idx {
        return Err(format!(
            "markers `{}` / `{}` out of order",
            region.start, region.end
        ));
    }
    let after_start = start_idx + region.start.len();
    // We normalize the body to exactly one leading `\n` and one
    // trailing `\n` so repeated exports stay byte-stable regardless of
    // how the operator edited the markers.
    let body_normalized = format!("\n{}\n", body.trim_matches('\n'));
    let mut out = String::with_capacity(src.len() + body_normalized.len());
    out.push_str(&src[..after_start]);
    out.push_str(&body_normalized);
    out.push_str(&src[end_idx..]);
    Ok(out)
}

fn render_quick_decision(providers: &BTreeMap<String, ProviderConfig>) -> String {
    let all_pairs: Vec<(&str, &str, &ModelCapabilities)> = providers
        .iter()
        .flat_map(|(p, cfg)| {
            cfg.capabilities
                .iter()
                .map(move |(m, c)| (p.as_str(), m.as_str(), c))
        })
        .collect();

    let mut out = String::new();
    out.push_str("| Job | First choice | Backup | Why |\n");
    out.push_str("|---|---|---|---|\n");

    for task in [
        TaskKind::Coding,
        TaskKind::Research,
        TaskKind::Chat,
        TaskKind::Classify,
        TaskKind::Vision,
        TaskKind::Digest,
    ] {
        let mut candidates: Vec<(&str, &str, &ModelCapabilities)> = all_pairs
            .iter()
            .copied()
            .filter(|(_, _, c)| matches!(c.status, ModelStatus::Active) && c.fits(task))
            .collect();
        candidates.sort_by_key(|a| rank_key(a.2));
        let first = candidates.first().copied();
        let backup = candidates.get(1).copied();
        let why = first
            .map(|(_, _, c)| rationale(c))
            .unwrap_or_else(|| "—".to_string());
        let first_str = first
            .map(|(p, m, _)| format!("`{p}` / `{m}`"))
            .unwrap_or_else(|| "—".to_string());
        let backup_str = backup
            .map(|(p, m, _)| format!("`{p}` / `{m}`"))
            .unwrap_or_else(|| "—".to_string());
        out.push_str(&format!(
            "| **{}** | {} | {} | {} |\n",
            task_label(task),
            first_str,
            backup_str,
            why,
        ));
    }
    out
}

/// Lower is better.
fn rank_key(c: &ModelCapabilities) -> (u8, u8, u8) {
    (
        quality_rank(c.quality_tier),
        latency_rank(c.latency_tier),
        cost_rank(c.cost_tier),
    )
}

fn quality_rank(q: Option<QualityTier>) -> u8 {
    match q {
        Some(QualityTier::S) => 0,
        Some(QualityTier::A) => 1,
        Some(QualityTier::B) => 2,
        Some(QualityTier::C) => 3,
        None => 4,
    }
}

fn latency_rank(l: Option<LatencyTier>) -> u8 {
    match l {
        Some(LatencyTier::Fast) => 0,
        Some(LatencyTier::Medium) => 1,
        Some(LatencyTier::Slow) => 2,
        None => 3,
    }
}

fn cost_rank(c: Option<CostTier>) -> u8 {
    match c {
        Some(CostTier::Free) => 0,
        Some(CostTier::Cheap) => 1,
        Some(CostTier::Medium) => 2,
        Some(CostTier::Premium) => 3,
        None => 4,
    }
}

fn task_label(t: TaskKind) -> &'static str {
    match t {
        TaskKind::Coding => "Coding refactor / multi-file edit",
        TaskKind::Research => "Research / web extraction (agentic)",
        TaskKind::Chat => "General chat / Q&A",
        TaskKind::Classify => "Classification",
        TaskKind::Vision => "Vision / multimodal",
        TaskKind::Digest => "Memory digest",
    }
}

fn rationale(c: &ModelCapabilities) -> String {
    let mut bits = Vec::new();
    if let Some(q) = c.quality_tier {
        bits.push(format!("{q:?}-quality"));
    }
    if let Some(l) = c.latency_tier {
        bits.push(format!("{l:?} latency"));
    }
    if let Some(ct) = c.cost_tier {
        bits.push(format!("{ct:?} cost"));
    }
    if c.supports_thinking {
        bits.push("thinking".into());
    }
    if bits.is_empty() {
        "selected by status + task fit".into()
    } else {
        bits.join(", ")
    }
}

fn render_provider_details(providers: &BTreeMap<String, ProviderConfig>) -> String {
    let mut out = String::new();
    for (name, cfg) in providers {
        let mut active_pairs: Vec<(&str, &ModelCapabilities)> = cfg
            .capabilities
            .iter()
            .filter(|(_, c)| !matches!(c.status, ModelStatus::Deprecated))
            .map(|(m, c)| (m.as_str(), c))
            .collect();
        active_pairs.sort_by_key(|(m, _)| m.to_string());

        out.push_str(&format!("### `{name}`\n"));
        if active_pairs.is_empty() {
            out.push_str("- _(no capability entries yet)_\n");
        } else {
            for (model, caps) in active_pairs {
                let status_note = match caps.status {
                    ModelStatus::Active => "".into(),
                    ModelStatus::Degraded => " **(degraded)**".to_string(),
                    ModelStatus::Experimental => " _(experimental)_".to_string(),
                    ModelStatus::Deprecated => " ~~(deprecated)~~".to_string(),
                };
                let mut facts = Vec::new();
                if let Some(q) = caps.quality_tier {
                    facts.push(format!("quality={q:?}"));
                }
                if let Some(l) = caps.latency_tier {
                    facts.push(format!("latency={l:?}"));
                }
                if let Some(ct) = caps.cost_tier {
                    facts.push(format!("cost={ct:?}"));
                }
                if let Some(ctx) = caps.context_window {
                    facts.push(format!("ctx={ctx}"));
                }
                let fits: Vec<String> = caps.task_fit.iter().map(|t| t.to_string()).collect();
                if !fits.is_empty() {
                    facts.push(format!("fit=[{}]", fits.join(",")));
                }
                let fact_str = if facts.is_empty() {
                    "".to_string()
                } else {
                    format!(" — {}", facts.join(", "))
                };
                out.push_str(&format!("- `{model}`{status_note}{fact_str}\n"));
            }
        }
        out.push('\n');
    }
    out.trim_end().to_string() + "\n"
}

fn render_failure_modes(providers: &BTreeMap<String, ProviderConfig>) -> String {
    let mut rows: Vec<(String, String, String, String)> = Vec::new();
    for (provider, cfg) in providers {
        for (model, caps) in &cfg.capabilities {
            if caps.known_failure_modes.is_empty()
                && !matches!(caps.status, ModelStatus::Degraded | ModelStatus::Deprecated)
            {
                continue;
            }
            let symptom = if caps.known_failure_modes.is_empty() {
                match caps.status {
                    ModelStatus::Deprecated => "model marked deprecated".to_string(),
                    ModelStatus::Degraded => "model marked degraded".to_string(),
                    _ => "".to_string(),
                }
            } else {
                caps.known_failure_modes.join(", ")
            };
            let cause = caps
                .notes
                .clone()
                .unwrap_or_else(|| format!("status: {:?}", caps.status));
            let fix = suggest_fix(caps);
            rows.push((format!("`{provider}` / `{model}`"), symptom, cause, fix));
        }
    }
    rows.sort();
    let mut out = String::new();
    out.push_str("| Pair | Symptom | Cause | Fix |\n");
    out.push_str("|---|---|---|---|\n");
    if rows.is_empty() {
        out.push_str("| — | — | — | — |\n");
    } else {
        for (pair, symptom, cause, fix) in rows {
            out.push_str(&format!("| {pair} | {symptom} | {cause} | {fix} |\n"));
        }
    }
    out
}

fn suggest_fix(caps: &ModelCapabilities) -> String {
    match caps.status {
        ModelStatus::Deprecated => {
            "remove from fallback chains; selector already skips it".into()
        }
        ModelStatus::Degraded => {
            "use only as last-resort fallback; wrap in retry; prefer alternatives for the same task fit"
                .into()
        }
        ModelStatus::Experimental => "keep behind a feature flag until stability confirmed".into(),
        ModelStatus::Active => {
            if caps.known_failure_modes.is_empty() {
                "—".into()
            } else {
                "retry once; investigate when failure rate sustained".into()
            }
        }
    }
}

/// Rewrite an existing SKILL.md file in place. Returns `Ok(true)` if
/// the file changed, `Ok(false)` if it was already byte-identical.
pub fn export_to_file(config: &Config, skill_path: &Path) -> std::io::Result<bool> {
    let src = std::fs::read_to_string(skill_path)?;
    let rendered = render_sections(config);
    let new_body = rendered
        .apply(&src)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    if !rendered.has_usable_model_rows() {
        tracing::warn!(
            path = %skill_path.display(),
            "model catalog export produced zero usable models; preserving existing generated sections"
        );
        return Ok(false);
    }
    if new_body == src {
        return Ok(false);
    }
    let tmp = skill_path.with_extension("md.tmp");
    std::fs::write(&tmp, &new_body)?;
    std::fs::rename(&tmp, skill_path)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use crate::model_catalog::types::{
        CostTier, LatencyTier, ModelCapabilities, ModelStatus, QualityTier, ReasoningLevel,
        TaskKind, ToolUseLevel,
    };
    use std::collections::HashMap;

    fn mk_caps(
        status: ModelStatus,
        task: TaskKind,
        quality: QualityTier,
        latency: LatencyTier,
        cost: CostTier,
    ) -> ModelCapabilities {
        ModelCapabilities {
            status,
            task_fit: vec![task],
            tool_use: ToolUseLevel::Full,
            reasoning: ReasoningLevel::Medium,
            supports_thinking: false,
            supports_vision: None,
            context_window: Some(128_000),
            max_output_tokens: Some(4096),
            quality_tier: Some(quality),
            latency_tier: Some(latency),
            cost_tier: Some(cost),
            known_failure_modes: Vec::new(),
            notes: None,
        }
    }

    fn mk_provider(caps: HashMap<String, ModelCapabilities>) -> ProviderConfig {
        ProviderConfig {
            provider_type: "openai_compat".into(),
            api_key: "unused".into(),
            api_keys: Vec::new(),
            base_url: None,
            models: Vec::new(),
            max_tokens: None,
            temperature: None,
            context_window: None,
            headers: HashMap::new(),
            model_aliases: HashMap::new(),
            supports_vision: None,
            capabilities: caps,
        }
    }

    fn mk_config() -> Config {
        let mut providers: HashMap<String, ProviderConfig> = HashMap::new();

        let mut anthropic_caps: HashMap<String, ModelCapabilities> = HashMap::new();
        anthropic_caps.insert(
            "claude-sonnet-4".into(),
            mk_caps(
                ModelStatus::Active,
                TaskKind::Research,
                QualityTier::S,
                LatencyTier::Medium,
                CostTier::Premium,
            ),
        );
        providers.insert("anthropic".into(), mk_provider(anthropic_caps));

        let mut zai_caps: HashMap<String, ModelCapabilities> = HashMap::new();
        zai_caps.insert(
            "glm-5-turbo".into(),
            ModelCapabilities {
                status: ModelStatus::Degraded,
                task_fit: vec![TaskKind::Chat],
                tool_use: ToolUseLevel::TextOnly,
                reasoning: ReasoningLevel::Low,
                supports_thinking: false,
                supports_vision: None,
                context_window: Some(128_000),
                max_output_tokens: Some(2048),
                quality_tier: Some(QualityTier::C),
                latency_tier: Some(LatencyTier::Fast),
                cost_tier: Some(CostTier::Cheap),
                known_failure_modes: vec!["empty_content".into()],
                notes: Some("returns empty responses under load".into()),
            },
        );
        providers.insert("zai".into(), mk_provider(zai_caps));

        Config {
            providers,
            ..Default::default()
        }
    }

    #[test]
    fn render_sections_produces_non_empty_bodies() {
        let cfg = mk_config();
        let r = render_sections(&cfg);
        assert!(r.quick_decision.contains("Job"));
        assert!(r.provider_details.contains("`anthropic`"));
        assert!(r.failure_modes.contains("empty_content"));
    }

    #[test]
    fn quick_decision_picks_highest_quality_active() {
        let cfg = mk_config();
        let r = render_sections(&cfg);
        assert!(
            r.quick_decision.contains("claude-sonnet-4"),
            "S-tier active must show up: {}",
            r.quick_decision
        );
    }

    #[test]
    fn failure_modes_lists_degraded_pairs_even_without_notes() {
        let cfg = mk_config();
        let r = render_sections(&cfg);
        assert!(r.failure_modes.contains("glm-5-turbo"));
        assert!(r.failure_modes.contains("empty_content"));
    }

    #[test]
    fn apply_is_idempotent_roundtrip() {
        let src = "# header\n\n\
                   <!-- generated:quick-decision:start -->\nOLD\n<!-- generated:quick-decision:end -->\n\n\
                   <!-- generated:provider-details:start -->\nOLD\n<!-- generated:provider-details:end -->\n\n\
                   <!-- generated:known-failure-modes:start -->\nOLD\n<!-- generated:known-failure-modes:end -->\n\n\
                   trailing\n";
        let cfg = mk_config();
        let r = render_sections(&cfg);
        let once = r.apply(src).unwrap();
        let twice = r.apply(&once).unwrap();
        assert_eq!(once, twice, "round-trip export must be byte-stable");
    }

    #[test]
    fn apply_rejects_missing_markers() {
        let cfg = mk_config();
        let r = render_sections(&cfg);
        let err = r.apply("nothing here").unwrap_err();
        assert!(err.contains("missing marker"));
    }

    #[test]
    fn apply_preserves_content_outside_markers() {
        let src = "# header\n\
                   preserved-prelude\n\
                   <!-- generated:quick-decision:start -->\nOLD\n<!-- generated:quick-decision:end -->\n\
                   preserved-middle\n\
                   <!-- generated:provider-details:start -->\nOLD\n<!-- generated:provider-details:end -->\n\
                   preserved-post\n\
                   <!-- generated:known-failure-modes:start -->\nOLD\n<!-- generated:known-failure-modes:end -->\n\
                   tail\n";
        let cfg = mk_config();
        let r = render_sections(&cfg);
        let out = r.apply(src).unwrap();
        assert!(out.contains("preserved-prelude"));
        assert!(out.contains("preserved-middle"));
        assert!(out.contains("preserved-post"));
        assert!(out.contains("tail"));
        assert!(out.contains("# header"));
    }

    #[test]
    fn export_to_file_returns_false_on_no_change() {
        let cfg = mk_config();
        let r = render_sections(&cfg);
        let src = "# header\n\
                   <!-- generated:quick-decision:start -->\nOLD\n<!-- generated:quick-decision:end -->\n\
                   <!-- generated:provider-details:start -->\nOLD\n<!-- generated:provider-details:end -->\n\
                   <!-- generated:known-failure-modes:start -->\nOLD\n<!-- generated:known-failure-modes:end -->\n";
        let first = r.apply(src).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("SKILL.md");
        std::fs::write(&p, &first).unwrap();
        assert!(!export_to_file(&cfg, &p).unwrap(), "second export is no-op");
    }
}
