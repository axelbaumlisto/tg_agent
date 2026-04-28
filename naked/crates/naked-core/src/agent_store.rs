//! Disk-backed loader for [`crate::agent_role::AgentRole`] definitions.
//!
//! Layout, one entry per agent role:
//!
//! ```text
//! <root>/
//!   web_researcher/
//!     role.json        # serialised AgentRole (no inline `system` —
//!                      # uses prompt_file instead)
//!     prompt.md        # system prompt body
//!     criteria.txt     # default acceptance-criteria template (optional)
//!   browser_extractor/
//!     role.json
//!     prompt.md
//! ```
//!
//! The store is built once at startup from `Config::agent_dirs` (each
//! root scanned in declaration order — earlier wins on name conflict)
//! and shared via `Arc<AgentStore>`. Both file paths in `role.json`
//! (`prompt_file`, `criteria_file`) are resolved relative to the role
//! directory, so role JSONs stay portable.
//!
//! What the store is NOT responsible for:
//!
//! * Applying `naked.json` `agent_roles` per-name overrides — that's
//!   the caller's job (see [`resolve_role`]) so the store stays a
//!   pure read-from-disk thing testable without a `Config`.
//! * Hot reload — roles are read once. A dev workflow that wants to
//!   tweak prompts can `naked agent reload-roles` (future) or just
//!   restart the process. The whole point of the data-driven layout is
//!   that prompt edits don't require a Rust rebuild — operating-system
//!   restart is cheap.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::agent_role::{AgentRole, AgentRoleOverride, apply_role_override};
use crate::error::{AgentError, Result};

/// In-memory catalog of agent roles loaded from disk.
///
/// Cheap to clone (`Arc<HashMap<...>>` under the hood). Cloning the
/// store does NOT re-read from disk; mutating one of the loaded roles
/// affects every Arc holder, which is fine — we never mutate roles
/// after load.
#[derive(Debug, Clone, Default)]
pub struct AgentStore {
    roles: Arc<HashMap<String, AgentRole>>,
    /// The roots scanned at load time. Surfaced for diagnostics
    /// (`naked agent list-roles --verbose`) and for tests.
    roots: Vec<PathBuf>,
}

impl AgentStore {
    /// Empty store. Useful in tests / programmatic construction where
    /// roles are built inline via `AgentRole::new`.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Build a store from a fixed set of fully-hydrated roles. Used by
    /// tests that don't want to touch the filesystem.
    pub fn from_roles<I: IntoIterator<Item = AgentRole>>(it: I) -> Self {
        let mut roles = HashMap::new();
        for r in it {
            roles.insert(r.name.clone(), r);
        }
        Self {
            roles: Arc::new(roles),
            roots: Vec::new(),
        }
    }

    /// Scan `roots` in declaration order, load every `<root>/<name>/role.json`
    /// directory, hydrate prompt + criteria from disk. The first root
    /// to define a given role name wins; later definitions are
    /// silently ignored. Missing roots are skipped (a fresh checkout
    /// without `~/.naked/agents` is normal).
    pub fn load_dirs(roots: &[PathBuf]) -> Result<Self> {
        let mut roles: HashMap<String, AgentRole> = HashMap::new();
        for root in roots {
            if !root.exists() {
                continue;
            }
            let entries = std::fs::read_dir(root).map_err(|e| {
                AgentError::Config(format!("agent_dirs: read_dir {}: {e}", root.display()))
            })?;
            for entry in entries.flatten() {
                let dir = entry.path();
                if !dir.is_dir() {
                    continue;
                }
                let role_json = dir.join("role.json");
                if !role_json.exists() {
                    continue;
                }
                let role = load_role(&dir, &role_json)?;
                roles.entry(role.name.clone()).or_insert(role);
            }
        }
        Ok(Self {
            roles: Arc::new(roles),
            roots: roots.to_vec(),
        })
    }

    /// Lookup by name. Returns a clone (roles are small + ownership
    /// keeps callers honest about per-task overrides).
    pub fn get(&self, name: &str) -> Option<AgentRole> {
        self.roles.get(name).cloned()
    }

    /// Iterate every loaded role (for `naked agent list-roles`).
    pub fn list(&self) -> Vec<AgentRole> {
        self.roles.values().cloned().collect()
    }

    /// Number of roles loaded.
    pub fn len(&self) -> usize {
        self.roles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roles.is_empty()
    }

    /// Roots the store was built from. Empty for an in-memory store.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }
}

/// Read one `<dir>/role.json`, hydrate `system` from `prompt_file` and
/// `default_criteria` from `criteria_file`. All file paths in the JSON
/// are resolved relative to `dir`.
fn load_role(dir: &Path, role_json: &Path) -> Result<AgentRole> {
    let text = std::fs::read_to_string(role_json)
        .map_err(|e| AgentError::Config(format!("read {}: {e}", role_json.display())))?;
    let mut role: AgentRole = serde_json::from_str(&text)
        .map_err(|e| AgentError::Config(format!("parse {}: {e}", role_json.display())))?;

    if let Some(p) = &role.prompt_file {
        let prompt_path = resolve_relative(dir, p);
        let prompt = std::fs::read_to_string(&prompt_path).map_err(|e| {
            AgentError::Config(format!(
                "agent `{}`: prompt_file {}: {e}",
                role.name,
                prompt_path.display()
            ))
        })?;
        role.system = prompt;
    } else if role.system.is_empty() {
        return Err(AgentError::Config(format!(
            "agent `{}` ({}): neither `prompt_file` nor inline `system` is set",
            role.name,
            role_json.display(),
        )));
    }

    if let Some(p) = &role.criteria_file {
        let crit_path = resolve_relative(dir, p);
        let crit = std::fs::read_to_string(&crit_path).map_err(|e| {
            AgentError::Config(format!(
                "agent `{}`: criteria_file {}: {e}",
                role.name,
                crit_path.display()
            ))
        })?;
        role.default_criteria = Some(crit);
    }

    Ok(role)
}

fn resolve_relative(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base.join(p)
    }
}

/// One-stop role lookup: consult the store, apply `naked.json` per-name
/// override on top. Returns `None` for unknown names.
///
/// This is the function the CLI / coordinator calls. Keeping it here
/// (rather than as an [`AgentStore`] method) keeps the override layer
/// — which is `Config`-shaped — out of the store's pure read-from-disk
/// surface.
pub fn resolve_role(
    name: &str,
    store: &AgentStore,
    overrides: &HashMap<String, AgentRoleOverride>,
) -> Option<AgentRole> {
    let role = store.get(name)?;
    Some(match overrides.get(name) {
        Some(ov) => apply_role_override(role, ov),
        None => role,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_role::ToolFilter;

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn empty_store_returns_none() {
        let s = AgentStore::empty();
        assert!(s.get("anything").is_none());
        assert!(s.is_empty());
    }

    #[test]
    fn from_roles_round_trips() {
        let role = AgentRole::new("inline", "you are inline");
        let s = AgentStore::from_roles([role]);
        let r = s.get("inline").expect("inline must be present");
        assert_eq!(r.name, "inline");
        assert_eq!(r.system, "you are inline");
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn load_dirs_hydrates_prompt_and_criteria() {
        let tmp = tempfile::tempdir().unwrap();
        let role_dir = tmp.path().join("test_role");
        write(
            &role_dir.join("role.json"),
            r#"{
                "name": "test_role",
                "description": "fixture",
                "prompt_file": "prompt.md",
                "criteria_file": "criteria.txt",
                "max_iters": 12,
                "tool_filter": {"kind": "allow", "tools": ["web_fetch"]}
            }"#,
        );
        write(&role_dir.join("prompt.md"), "system body for {topic}");
        write(&role_dir.join("criteria.txt"), "criteria for {topic}");

        let store = AgentStore::load_dirs(&[tmp.path().to_path_buf()]).unwrap();
        assert_eq!(store.len(), 1);
        let r = store.get("test_role").unwrap();
        assert_eq!(r.system, "system body for {topic}");
        assert_eq!(r.default_criteria.as_deref(), Some("criteria for {topic}"));
        assert_eq!(r.max_iters, 12);
        match r.tool_filter {
            ToolFilter::Allow { tools } => assert_eq!(tools, vec!["web_fetch".to_string()]),
            other => panic!("unexpected filter: {other:?}"),
        }
    }

    #[test]
    fn load_dirs_missing_root_is_silent() {
        let store = AgentStore::load_dirs(&[PathBuf::from("/nonexistent")]).unwrap();
        assert!(store.is_empty());
    }

    #[test]
    fn load_dirs_first_root_wins_on_conflict() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        for (root, body) in [(tmp1.path(), "from-1"), (tmp2.path(), "from-2")] {
            write(
                &root.join("dup").join("role.json"),
                r#"{"name": "dup", "prompt_file": "prompt.md"}"#,
            );
            write(&root.join("dup").join("prompt.md"), body);
        }
        let store =
            AgentStore::load_dirs(&[tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]).unwrap();
        assert_eq!(store.get("dup").unwrap().system, "from-1");
    }

    #[test]
    fn load_dirs_role_without_prompt_or_inline_system_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        write(
            &tmp.path().join("bad").join("role.json"),
            r#"{"name": "bad"}"#,
        );
        let err = AgentStore::load_dirs(&[tmp.path().to_path_buf()]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("bad") && msg.contains("prompt_file"),
            "expected diagnostic about missing prompt, got: {msg}"
        );
    }

    #[test]
    fn load_dirs_missing_prompt_file_reports_path() {
        let tmp = tempfile::tempdir().unwrap();
        write(
            &tmp.path().join("dangling").join("role.json"),
            r#"{"name": "dangling", "prompt_file": "missing.md"}"#,
        );
        let err = AgentStore::load_dirs(&[tmp.path().to_path_buf()]).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("dangling") && msg.contains("missing.md"),
            "expected diagnostic about missing prompt file, got: {msg}"
        );
    }

    #[test]
    fn resolve_role_applies_override() {
        let store = AgentStore::from_roles([AgentRole::new("rt", "sys")
            .with_max_iters(20)
            .with_model("base-m")]);
        let mut overrides = HashMap::new();
        overrides.insert(
            "rt".to_string(),
            AgentRoleOverride {
                model: Some("override-m".into()),
                ..Default::default()
            },
        );
        let resolved = resolve_role("rt", &store, &overrides).unwrap();
        assert_eq!(resolved.model.as_deref(), Some("override-m"));
        assert_eq!(resolved.max_iters, 20);
    }

    #[test]
    fn resolve_role_unknown_name_returns_none() {
        let store = AgentStore::empty();
        let overrides = HashMap::new();
        assert!(resolve_role("nope", &store, &overrides).is_none());
    }

    #[test]
    fn load_dirs_skips_dirs_without_role_json() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("not-a-role")).unwrap();
        write(
            &tmp.path().join("real").join("role.json"),
            r#"{"name": "real", "prompt_file": "prompt.md"}"#,
        );
        write(&tmp.path().join("real").join("prompt.md"), "body");
        let store = AgentStore::load_dirs(&[tmp.path().to_path_buf()]).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.get("real").is_some());
    }

    /// Locate the workspace-shipped `naked/agents/` directory by
    /// walking up from `CARGO_MANIFEST_DIR`. Returns `None` if the
    /// fixture isn't present (e.g. when the crate is consumed as a
    /// dependency from a different workspace) — callers should skip
    /// the test gracefully in that case.
    fn shipped_agents_dir() -> Option<PathBuf> {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // .../naked/crates/naked-core
        let candidate = manifest
            .parent() // .../naked/crates
            .and_then(|p| p.parent()) // .../naked
            .map(|p| p.join("agents"))?;
        if candidate.is_dir() {
            Some(candidate)
        } else {
            None
        }
    }

    #[test]
    fn shipped_browser_extractor_has_required_skills_and_filter() {
        let Some(dir) = shipped_agents_dir() else {
            eprintln!("skipping: naked/agents not found");
            return;
        };
        let store = AgentStore::load_dirs(&[dir]).unwrap();
        let r = store
            .get("browser_extractor")
            .expect("agents/browser_extractor must ship in-tree");
        assert!(r.default_skills.iter().any(|s| s == "web-browser-playbook"));
        assert!(
            r.default_validators.iter().any(|v| v == "gatekeeper"),
            "browser_extractor must declare `gatekeeper` as a default validator \
             (validation lives on the role, not the CLI). Got: {:?}",
            r.default_validators
        );
        match r.tool_filter {
            ToolFilter::Allow { tools } => {
                assert!(tools.iter().any(|t| t == "web_fetch"));
                assert!(tools.iter().any(|t| t == "research_save"));
                assert!(!tools.iter().any(|t| t == "bash"));
            }
            other => panic!("expected Allow filter, got {other:?}"),
        }
    }

    #[test]
    fn shipped_web_researcher_declares_gatekeeper_default_validator() {
        let Some(dir) = shipped_agents_dir() else {
            eprintln!("skipping: naked/agents not found");
            return;
        };
        let store = AgentStore::load_dirs(&[dir]).unwrap();
        let r = store
            .get("web_researcher")
            .expect("agents/web_researcher must ship in-tree");
        assert!(
            r.default_validators.iter().any(|v| v == "gatekeeper"),
            "web_researcher must declare `gatekeeper` as a default validator \
             so `naked agent batch-from-spec` works without repeating \
             --validators on every invocation. Got: {:?}",
            r.default_validators
        );
    }

    /// `web_researcher` must stay universal: no country / language /
    /// currency / phone-format hints sneaking into the system prompt.
    /// The whole point is one role for VN real-estate, TH condos,
    /// Bangkok mopeds, used Jaguars in Vietnam, etc.
    #[test]
    fn shipped_web_researcher_is_universal_no_country_assumptions() {
        let Some(dir) = shipped_agents_dir() else {
            eprintln!("skipping: naked/agents not found");
            return;
        };
        let store = AgentStore::load_dirs(&[dir]).unwrap();
        let r = store
            .get("web_researcher")
            .expect("agents/web_researcher must ship in-tree");
        let sys = r.system.to_lowercase();
        for forbidden in [
            "vietnam",
            "thailand",
            "indonesia",
            "tiếng việt",
            "việt",
            " vnd ",
            " thb ",
            " usd ",
            "0905",
            "+66",
            "+84",
        ] {
            assert!(
                !sys.contains(forbidden),
                "web_researcher.system must NOT contain `{forbidden}` — \
                 push country/lang/format knowledge into the spec or \
                 the gatekeeper criteria, not the role prompt."
            );
        }
        for required in [
            "universal",
            "{url}",
            "{topic}",
            "{max_listings}",
            "research_save",
            "Contacts hidden behind site captcha — visit URL",
            "Wayback",
        ] {
            assert!(
                r.system.contains(required),
                "web_researcher.system must contain `{required}`"
            );
        }
        assert!(r.default_skills.iter().any(|s| s == "web-browser-playbook"));
        match r.tool_filter {
            ToolFilter::Allow { tools } => {
                assert!(!tools.iter().any(|t| t == "bash"));
                assert!(!tools.iter().any(|t| t == "write_file"));
                assert!(tools.iter().any(|t| t == "research_save"));
            }
            other => panic!("web_researcher must use Allow filter, got {other:?}"),
        }
        assert!(
            r.max_iters >= 50,
            "web_researcher needs a real traversal budget — got {}",
            r.max_iters
        );
        assert!(
            r.default_criteria.is_some(),
            "web_researcher must ship a default acceptance criteria template"
        );
    }

    #[test]
    fn load_dirs_absolute_prompt_path_supported() {
        let tmp = tempfile::tempdir().unwrap();
        let prompt_abs = tmp.path().join("shared-prompt.md");
        std::fs::write(&prompt_abs, "shared body").unwrap();
        let role_dir = tmp.path().join("abs");
        write(
            &role_dir.join("role.json"),
            &format!(
                r#"{{"name": "abs", "prompt_file": "{}"}}"#,
                prompt_abs.display()
            ),
        );
        let store = AgentStore::load_dirs(&[tmp.path().to_path_buf()]).unwrap();
        assert_eq!(store.get("abs").unwrap().system, "shared body");
    }
}
