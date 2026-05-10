use std::path::PathBuf;

/// Disk form of a resolved skill.
///
/// The resolver looks at file extensions only — it does NOT parse
/// the body (that's [`super::tool::SkillTool`]'s job). Keeping the
/// distinction here lets the tool branch on form before reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillFile {
    /// Legacy markdown form. Body returned verbatim to the LLM.
    Markdown,
    /// JSON form, see [`super::spec::SkillSpec`]. May be advisory,
    /// executable, or hybrid — the mode lives inside the JSON.
    Json,
    /// Legacy `SKILL.toml` from another runtime. Rendered to an
    /// advisory manual on read — see [`super::toml_legacy`].
    Toml,
}

/// Resolved skill location + form. Cheap to clone.
#[derive(Debug, Clone)]
pub struct ResolvedSkill {
    pub path: PathBuf,
    pub kind: SkillFile,
}

pub struct SkillResolver {
    roots: Vec<PathBuf>,
}

impl SkillResolver {
    pub fn new(roots: Vec<PathBuf>) -> Self {
        Self { roots }
    }

    /// T9 of PLAN_QUALITY_v1 — build a resolver that augments the
    /// user-configured roots with ecosystem-shared paths (Claude
    /// Code, Agents SDK). Walks parent dirs from `cwd` up to root
    /// looking for `.claude/skills/` and `.agents/skills/`, then
    /// appends global `~/.claude/skills/` and `~/.agents/skills/`.
    /// User roots take precedence (declared first); a duplicate
    /// skill name resolves to the user-owned copy.
    #[must_use]
    pub fn with_ecosystem_paths(mut user_roots: Vec<PathBuf>, cwd: &std::path::Path) -> Self {
        let mut walker = Some(cwd.to_path_buf());
        while let Some(d) = walker {
            for sub in [".claude/skills", ".agents/skills"] {
                let candidate = d.join(sub);
                if candidate.is_dir() {
                    user_roots.push(candidate);
                }
            }
            walker = d.parent().map(|p| p.to_path_buf());
        }
        if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
            for sub in [".claude/skills", ".agents/skills"] {
                let candidate = home.join(sub);
                if candidate.is_dir() {
                    user_roots.push(candidate);
                }
            }
        }
        Self { roots: user_roots }
    }

    /// Resolve a skill name to its disk path. Searches roots in
    /// declaration order; within each root tries exact then
    /// case-insensitive directory match. Inside the matched directory
    /// `SKILL.json` wins over `SKILL.md` so authors who add a JSON
    /// form alongside a legacy MD file get the new behaviour
    /// automatically.
    pub fn resolve(&self, skill: &str) -> Option<ResolvedSkill> {
        let requested = skill.trim().trim_start_matches('/').trim_start_matches('$');
        if requested.is_empty() || requested.contains("..") {
            return None;
        }

        for root in &self.roots {
            let candidate = root.join(requested);
            if let (Ok(real), Ok(base)) = (candidate.canonicalize(), root.canonicalize())
                && !real.starts_with(&base)
            {
                continue;
            }
            if let Some(hit) = pick_form(&candidate) {
                return Some(hit);
            }

            if let Ok(entries) = std::fs::read_dir(root) {
                for entry in entries.flatten() {
                    if !entry
                        .file_name()
                        .to_string_lossy()
                        .eq_ignore_ascii_case(requested)
                    {
                        continue;
                    }
                    if let Some(hit) = pick_form(&entry.path()) {
                        return Some(hit);
                    }
                }
            }
        }

        None
    }

    /// List all available skills across all roots. JSON form wins
    /// over MD inside each directory (same precedence as `resolve`).
    pub fn list(&self) -> Vec<(String, ResolvedSkill)> {
        let mut skills = Vec::new();
        for root in &self.roots {
            if let Ok(entries) = std::fs::read_dir(root) {
                for entry in entries.flatten() {
                    if let Some(hit) = pick_form(&entry.path()) {
                        let name = entry.file_name().to_string_lossy().to_string();
                        skills.push((name, hit));
                    }
                }
            }
        }
        skills
    }

    /// Subdirectories that *look* like skill slots (they live directly
    /// under a configured root) but have no `SKILL.{json,md,toml}`
    /// manifest. These are invisible to the `Skill` tool — logging
    /// them on startup makes "why can't the agent see my skill"
    /// obvious at a glance.
    ///
    /// Returns `(root, orphan_dir_path)` tuples. The `root` is the
    /// configured path as stored in `self.roots` (so the operator can
    /// map it back to `naked.json::skill_roots`), and the
    /// `orphan_dir_path` is the absolute path of the offending
    /// directory.
    pub fn find_orphans(&self) -> Vec<(PathBuf, PathBuf)> {
        let mut orphans = Vec::new();
        for root in &self.roots {
            let Ok(entries) = std::fs::read_dir(root) else {
                continue;
            };
            for entry in entries.flatten() {
                let p = entry.path();
                if !p.is_dir() {
                    continue;
                }
                if is_service_dir_name(&p) {
                    continue;
                }
                if pick_form(&p).is_none() {
                    orphans.push((root.clone(), p));
                }
            }
        }
        orphans
    }
}

/// Decide whether a directory under a skill_root should be exempt from
/// the orphan check.
///
/// Two classes of exemption:
///
/// 1. **Hidden / service directories.** Names starting with `.` (e.g.
///    `.pytest_cache`, `.venv`, `.session`) and a small list of
///    well-known tool names (`scripts`, `tests`, `__pycache__`,
///    `node_modules`, `data`, `state`, `venv`) that often sit next to
///    skill directories but aren't skills themselves. The
///    `deep-research` root in particular keeps shared helpers under
///    `scripts/`.
///
/// 2. **Intentionally parked backups.** Names containing `.bak-` /
///    `.disabled-` / `.orphan-` are operator-marked stash folders —
///    they're orphan *by design*, shouldn't trigger the startup WARN.
fn is_service_dir_name(p: &std::path::Path) -> bool {
    let name = match p.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return false,
    };

    if name.starts_with('.') {
        return true;
    }

    const RESERVED: &[&str] = &[
        "scripts",
        "tests",
        "test",
        "__pycache__",
        "node_modules",
        "data",
        "state",
        "venv",
        "env",
        "legacy",
    ];
    if RESERVED.contains(&name) {
        return true;
    }

    if name.contains(".bak-") || name.contains(".disabled-") || name.contains(".orphan-") {
        return true;
    }

    false
}

/// Probe a candidate skill directory for SKILL.{json,md,toml}.
/// Precedence: `SKILL.json` (canonical) → `SKILL.md` (legacy) →
/// `SKILL.toml` (legacy from another runtime, rendered as advisory).
/// Returns the first one that exists, with the kind tagged for
/// downstream dispatch.
fn pick_form(dir: &std::path::Path) -> Option<ResolvedSkill> {
    let json = dir.join("SKILL.json");
    if json.exists() {
        return Some(ResolvedSkill {
            path: json,
            kind: SkillFile::Json,
        });
    }
    let md = dir.join("SKILL.md");
    if md.exists() {
        return Some(ResolvedSkill {
            path: md,
            kind: SkillFile::Markdown,
        });
    }
    let toml = dir.join("SKILL.toml");
    if toml.exists() {
        return Some(ResolvedSkill {
            path: toml,
            kind: SkillFile::Toml,
        });
    }
    None
}

pub fn parse_skill_description(contents: &str) -> Option<String> {
    for line in contents.lines() {
        if let Some(value) = line.strip_prefix("description:") {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Read a skill's short description, dispatching by form. Single
/// source of truth for both the LLM-facing capabilities catalog
/// (`AgentCore::capabilities_section`) and the `Skill` tool's own
/// JSON-schema description (`SkillTool::new`). MD skills look at
/// the YAML front-matter `description:` line; JSON skills read the
/// `description` field of [`super::spec::SkillSpec`]. Returns
/// `None` if the file is unreadable, malformed, or the description
/// is empty — callers should fall back to the bare skill name.
pub fn read_skill_description(hit: &ResolvedSkill) -> Option<String> {
    let raw = std::fs::read_to_string(&hit.path).ok()?;
    match hit.kind {
        SkillFile::Markdown => parse_skill_description(&raw),
        SkillFile::Json => serde_json::from_str::<super::spec::SkillSpec>(&raw)
            .ok()
            .map(|s| s.description)
            .filter(|d| !d.is_empty()),
        SkillFile::Toml => super::toml_legacy::TomlSkillManifest::parse(&raw)
            .map(|m| m.short_description())
            .filter(|d| !d.is_empty()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skill_description_found() {
        let content = "# My Skill\ndescription: Does awesome things\n\nBody here";
        assert_eq!(
            parse_skill_description(content),
            Some("Does awesome things".into())
        );
    }

    #[test]
    fn parse_skill_description_not_found() {
        let content = "# No desc line\nSome body";
        assert_eq!(parse_skill_description(content), None);
    }

    #[test]
    fn parse_skill_description_empty_value() {
        let content = "description:   \nother stuff";
        assert_eq!(parse_skill_description(content), None);
    }

    #[test]
    fn resolve_empty_name_returns_none() {
        let resolver = SkillResolver::new(vec![]);
        assert!(resolver.resolve("").is_none());
        assert!(resolver.resolve("  ").is_none());
    }

    #[test]
    fn resolve_strips_prefixes() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("my-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "# Skill").unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        assert!(resolver.resolve("/my-skill").is_some());
        assert!(resolver.resolve("$my-skill").is_some());
    }

    #[test]
    fn resolve_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("MySkill");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "# Skill").unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        assert!(resolver.resolve("myskill").is_some());
        assert!(resolver.resolve("MYSKILL").is_some());
    }

    #[test]
    fn resolve_direct_match_md() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("debug");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "content").unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("debug").unwrap();
        assert!(hit.path.ends_with("SKILL.md"));
        assert_eq!(hit.kind, SkillFile::Markdown);
    }

    #[test]
    fn resolve_prefers_json_over_md() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("dual");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "legacy md").unwrap();
        std::fs::write(
            skill_dir.join("SKILL.json"),
            r#"{"name":"dual","mode":"advisory","advisory_body":"x"}"#,
        )
        .unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("dual").unwrap();
        assert!(hit.path.ends_with("SKILL.json"));
        assert_eq!(hit.kind, SkillFile::Json);
    }

    #[test]
    fn resolve_falls_back_to_toml_if_no_json_or_md() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("legacy");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.toml"),
            "[skill]\nname = \"legacy\"\ndescription = \"legacy desc\"\n",
        )
        .unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("legacy").unwrap();
        assert!(hit.path.ends_with("SKILL.toml"));
        assert_eq!(hit.kind, SkillFile::Toml);
        assert_eq!(read_skill_description(&hit).as_deref(), Some("legacy desc"));
    }

    #[test]
    fn find_orphans_reports_dirs_without_manifest() {
        let dir = tempfile::tempdir().unwrap();
        // valid skill
        let good = dir.path().join("good");
        std::fs::create_dir(&good).unwrap();
        std::fs::write(good.join("SKILL.md"), "description: ok\n").unwrap();
        // orphan: directory without any manifest
        let orphan = dir.path().join("orphan");
        std::fs::create_dir(&orphan).unwrap();
        std::fs::write(orphan.join("scripts.py"), "print(1)\n").unwrap();
        // file at root level should be ignored (not a directory)
        std::fs::write(dir.path().join("stray.txt"), "noise").unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let orphans = resolver.find_orphans();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].0, dir.path());
        assert!(orphans[0].1.ends_with("orphan"));
    }

    #[test]
    fn find_orphans_skips_service_and_backup_dirs() {
        let dir = tempfile::tempdir().unwrap();
        // Directories that MUST be ignored:
        for name in [
            "scripts",
            "tests",
            "__pycache__",
            ".pytest_cache",
            ".venv",
            ".session",
            "yt-transcribe.bak-20260420",
            "telegram-mcp.disabled-20260101",
            "something.orphan-xyz",
            "legacy",
        ] {
            std::fs::create_dir(dir.path().join(name)).unwrap();
        }
        // And one real orphan that SHOULD be reported:
        let real_orphan = dir.path().join("no-manifest-here");
        std::fs::create_dir(&real_orphan).unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let orphans = resolver.find_orphans();

        assert_eq!(
            orphans.len(),
            1,
            "exactly one real orphan expected, got: {:?}",
            orphans.iter().map(|(_, p)| p.clone()).collect::<Vec<_>>()
        );
        assert!(orphans[0].1.ends_with("no-manifest-here"));
    }

    #[test]
    fn find_orphans_empty_when_all_have_manifest() {
        let dir = tempfile::tempdir().unwrap();
        for (name, fname, body) in [
            ("a", "SKILL.md", "description: a\n"),
            ("b", "SKILL.json", r#"{"name":"b","mode":"advisory"}"#),
            ("c", "SKILL.toml", "[skill]\nname=\"c\""),
        ] {
            let d = dir.path().join(name);
            std::fs::create_dir(&d).unwrap();
            std::fs::write(d.join(fname), body).unwrap();
        }
        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        assert!(resolver.find_orphans().is_empty());
    }

    #[test]
    fn resolve_prefers_md_over_toml() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("mixed");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.toml"), "[skill]\nname=\"mixed\"").unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "description: md\n").unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("mixed").unwrap();
        assert_eq!(hit.kind, SkillFile::Markdown);
    }

    #[test]
    fn resolve_nonexistent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        assert!(resolver.resolve("nonexistent").is_none());
    }

    #[test]
    fn list_returns_all_skills() {
        let dir = tempfile::tempdir().unwrap();
        for name in &["skill-a", "skill-b"] {
            let d = dir.path().join(name);
            std::fs::create_dir(&d).unwrap();
            std::fs::write(d.join("SKILL.md"), "# Skill").unwrap();
        }
        // JSON-only skill should also be listed.
        let json_only = dir.path().join("skill-json");
        std::fs::create_dir(&json_only).unwrap();
        std::fs::write(
            json_only.join("SKILL.json"),
            r#"{"name":"skill-json","mode":"advisory"}"#,
        )
        .unwrap();
        // A dir without either file should be excluded.
        std::fs::create_dir(dir.path().join("not-a-skill")).unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let skills = resolver.list();
        assert_eq!(skills.len(), 3);
        let names: Vec<_> = skills.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"skill-json"));
        let json_kind = skills
            .iter()
            .find(|(n, _)| n == "skill-json")
            .map(|(_, k)| k.kind)
            .unwrap();
        assert_eq!(json_kind, SkillFile::Json);
    }

    #[test]
    fn list_empty_root() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        assert!(resolver.list().is_empty());
    }

    #[test]
    fn list_nonexistent_root() {
        let resolver = SkillResolver::new(vec![PathBuf::from("/nonexistent/path")]);
        assert!(resolver.list().is_empty());
    }

    #[test]
    fn resolve_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("legit");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.md"), "# Skill").unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        assert!(resolver.resolve("../../../etc").is_none());
        assert!(resolver.resolve("foo/../../../etc").is_none());
    }

    #[test]
    fn read_skill_description_md_front_matter() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("md-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: md-skill\ndescription: Reads MD front-matter\n---\n# body",
        )
        .unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("md-skill").unwrap();
        assert_eq!(
            read_skill_description(&hit).as_deref(),
            Some("Reads MD front-matter")
        );
    }

    #[test]
    fn read_skill_description_json_field() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("json-skill");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.json"),
            r#"{"name":"json-skill","description":"Reads JSON description field","mode":"advisory","advisory_body":"x"}"#,
        )
        .unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("json-skill").unwrap();
        assert_eq!(
            read_skill_description(&hit).as_deref(),
            Some("Reads JSON description field")
        );
    }

    #[test]
    fn read_skill_description_json_empty_description_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("empty-desc");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.json"),
            r#"{"name":"empty-desc","description":"","mode":"advisory","advisory_body":"x"}"#,
        )
        .unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("empty-desc").unwrap();
        assert!(read_skill_description(&hit).is_none());
    }

    #[test]
    fn read_skill_description_json_malformed_is_none() {
        let dir = tempfile::tempdir().unwrap();
        let skill_dir = dir.path().join("broken");
        std::fs::create_dir(&skill_dir).unwrap();
        std::fs::write(skill_dir.join("SKILL.json"), "{not json").unwrap();

        let resolver = SkillResolver::new(vec![dir.path().to_path_buf()]);
        let hit = resolver.resolve("broken").unwrap();
        assert!(read_skill_description(&hit).is_none());
    }

    #[test]
    fn multiple_roots_searched_in_order() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();

        let s1 = dir1.path().join("shared");
        std::fs::create_dir(&s1).unwrap();
        std::fs::write(s1.join("SKILL.md"), "from root1").unwrap();

        let s2 = dir2.path().join("shared");
        std::fs::create_dir(&s2).unwrap();
        std::fs::write(s2.join("SKILL.md"), "from root2").unwrap();

        let resolver =
            SkillResolver::new(vec![dir1.path().to_path_buf(), dir2.path().to_path_buf()]);
        let hit = resolver.resolve("shared").unwrap();
        let content = std::fs::read_to_string(hit.path).unwrap();
        assert_eq!(content, "from root1");
    }
}
