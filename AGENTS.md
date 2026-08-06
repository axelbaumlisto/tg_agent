# AGENTS.md — `tg_agent`

Cross-tool instructions for any AI coding assistant working on this
repository. Sibling files: [`README.md`](README.md) (what this is),
[`OPERATIONS.md`](OPERATIONS.md) (how it runs in production on this
host), `naked/README.md` (the Rust agent itself).

## Architecture

| Component | Language | Code | Binary |
|-----------|----------|------|--------|
| Autonomous coding/research agent + Telegram bot | Rust 2024 edition | `naked/crates/{naked-core,naked-cli,naked-tg}` | `naked/target/release/{naked,naked-tg}` |

The Rust `naked-tg` is the **live production bot** (managed by
systemd — see `OPERATIONS.md`). It talks to LLM providers directly
(no intermediate proxy).

## Repo layout: code vs state

```
tg_agent/
├── naked/                 # ── CODE (pushed to both GitHub and Forgejo) ──
│   ├── Cargo.toml         # workspace root
│   ├── crates/
│   │   ├── naked-core/    # tool loop, providers, channels, memory, MCP
│   │   ├── naked-cli/     # `naked` CLI
│   │   └── naked-tg/      # `naked-tg` Telegram binary (systemd-managed)
│   ├── scripts/           # ops scripts (token health, research tick, …)
│   ├── skills/            # agent skills (JSON/MD)
│   ├── tests/             # bash e2e + fixtures
│   ├── ops/{systemd,cron} # unit files + cron snippets
│   ├── docs/              # architecture + planning docs
│   ├── data -> ../state/data        # symlink (gitignored)
│   ├── results -> ../state/results  # symlink (gitignored)
│   └── naked.json -> ../state/naked.json  # symlink (gitignored)
│
├── state/                 # ── DAEMON STATE (Forgejo only, not on GitHub) ──
│   ├── naked.json         # live config (providers, chat_ids, scheduler)
│   ├── data/              # research briefs, fx_rates
│   └── results/           # HTML reports, findings JSONL
│
├── .env                   # API keys + tokens (never committed anywhere)
├── .env.shared.example    # env template
├── AGENTS.md              # ← you are here
├── README.md
└── OPERATIONS.md
```

**Why the split?**
- `naked/` = pure code → safe to push to public GitHub
- `state/` = runtime config + research data → Forgejo only (private)
- Symlinks in `naked/` keep scripts working without path changes
- `NAKED_CONFIG` env var points systemd at `state/naked.json`

## Commands

```bash
cd naked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p naked-core --lib                # ~40s, must be green
cargo test -p naked-tg --lib                  # ~1s, must be green
cargo test -p naked-tg --bins                 # ~7s, must be green
                                              # `streaming_mod` (CompositeView,
                                              # render_final, delta/flush) is
                                              # wired into main.rs via #[path],
                                              # NOT lib.rs — `--lib` silently
                                              # skips ~361 tests incl. all
                                              # final-answer rendering.
cargo test -p naked-core --test loop_golden   # 8 golden tests, must be green
cargo build --release           # produces target/release/{naked,naked-tg}
```

Pre-commit must pass `cargo fmt + clippy --workspace -D warnings` and the
three targeted test commands above.

**Avoid `cargo test --workspace` in pre-commit.** It includes ~30 minutes
of live-LLM/live-DDG e2e tests (`t89_deep_research_full_cycle`,
`live_todo_tool` in `plan_v9_live_e2e`, etc.) that are stochastic and
depend on external services. Run them deliberately when needed:

```bash
cargo test -p naked-core --test e2e_provider   # live LLM, ~15min
cargo test -p naked-core --test plan_v9_live_e2e -- --test-threads=1
cargo test -p naked-core --test e2e_core       # includes DDG fallback
```

## Push targets

```bash
# Forgejo (full: code + state)
git push forgejo main

# GitHub (sanitized: code only, no state/)
naked/scripts/export_github_sanitized.sh
```

## Auto-detection registry (read FIRST on any new task)

Before drafting any plan or making non-trivial edits, consult
[`naked/docs/BUG_REGISTRY.md`](naked/docs/BUG_REGISTRY.md) —
specifically `§ 5 Agent self-check` for the M1-M7 / P1-P4 /
PL1-PL5 / C1-C4 rows. The registry catalogues 34 recurring bug
classes and 27 decomposed detection tasks so that the SAME bug
never needs the user to nudge me a second time.

New bug class discovered? Add a `B-NN` entry there BEFORE writing
the fix, then a matching `D-*` task in `§ 4` if no existing
detection layer would catch a re-occurrence.

**Mechanical gates already installed** (2026-05-13):
- **L1** lints: `naked/scripts/registry_lint.sh` (7 checks:
  WORKSPACE-TEST, NO-TARGET, ALLOW-SCOPE, SILENT-LET-ELSE,
  TAUTOLOGY, DEPS, PLAN-PUSH) — runs first in `pre-commit-check.sh`.
- **L3** integration: `naked-core/tests/config_invariants.rs` sweeps
  production `state/naked.json` for INV-1 / INV-2 / INV-3 consistency.
- **L4** boot: `wiring::check_multimodal_describer_health` +
  `boot_caps_invariant_sweep` + `D-BOOT-VISION-PROBE` (1×1 PNG
  content-shape probe). All fire on every restart, log INFO/WARN.
- **L5** metrics: `/metrics` exposes 22+ counters including
  `naked_core_provider_vision_capability_mismatch_total` (B06),
  `naked_tg_media_transcription_total{outcome}` (B01).
- **L6** audit: `naked/scripts/audit_all.sh` orchestrates 3
  sub-detectors (dead-key cycle, metric-vs-log, vision-cap); cron-
  ready; exit 1 on regression for `tg_alert.sh` integration.
- **L7** process: this file + `naked/docs/postmortems/_template.md`
  for structured retrospectives + `recall_load_rules` at session start.

**Before any new audit, run** `naked/scripts/audit_all.sh` to get
the current baseline. Compare deltas on follow-up runs.

## Calling scripts from bash (top error class)

`json.loads("")` → `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
is the #1 recurring failure. Empty stdout is normal when a command fails,
times out, or writes only to stderr.

```bash
# ❌ never
cmd | python3 -c 'import sys,json; json.load(sys.stdin)'
# ✅ always
out=$(timeout 120 cmd 2>/tmp/x.err); rc=$?
[ $rc -eq 0 ] && [ -n "$out" ] && printf '%s\n' "$out" | python3 -m json.tool \
  || echo "FAILED rc=$rc $(head -c 300 /tmp/x.err)"
```

Triage order: `echo $?` → stderr → `--help` → `pwd` → only then stdout.
Other recurring classes: guessed CLI flags (run `--help` first); cwd/path
(`SKILL.toml` paths resolve from `naked/`, `.env` is at repo root via the
`naked/.env` symlink); cron must export `NAKED_CONFIG` (B96); shell foot-guns
(`tail -n 2`, `grep -- "$negative"`, `pkill -f "[p]attern"`).
Full taxonomy with fixes: [`naked/skills/README.md`](naked/skills/README.md)
§ "Error taxonomy".

## Conventions

- **Editing scripts in `naked/scripts/`** — some deployments use
  absolute paths in `naked/ops/{systemd,cron}/`, the installed user
  service, and the active crontab. Keep those in sync with your local
  install path.
- **Never run `naked-tg` binary while the systemd service is active.**
  The pid-lock will kill the running daemon. Always
  `systemctl --user stop naked-tg.service` first if you need to invoke
  the binary directly (including `--help`).
- **Don't hand-edit between `# BEGIN naked/research_tick` markers** in
  the user crontab — regenerate via
  `make -C naked/ops/cron install-cron`.
- **Token health alerts** must include the deployment's current browser
  login URL; keep any private URL in local config rather than committed
  docs.
- **Restart after changing `naked/crates/`**:
  `cargo build --release && systemctl --user restart naked-tg.service`
  — the unit's `Type=notify` + `WatchdogSec=30s` make rolling
  restarts safe.
- **Logs**:
  - `journalctl --user -u naked-tg.service -f` (live bot)
  - `/tmp/{muaban,chotot}_token_health.log` (cron token watchers)
  - `/tmp/research_tick_full.log` (daily research sweep)
  - `/tmp/research_index_incremental.log` (hourly index)

## What lives outside this repo (not your concern)

`OPERATIONS.md` lists the optional sibling services used by a typical
deployment. They are referenced via URLs / docs / comments only; do not
commit host-local filesystem dependencies, secrets, or chat IDs.

## Live TG e2e verification pattern (proven 2026-05-13)

После больших refactor'ов (визибилити, провайдеры, tool factory) unit tests
проверяют компиляцию, но не dispatch chain. Final gate — live
request через реальный Telegram bot:

```bash
# Build + restart bot
cd naked && cargo build --release --bin naked-tg && \
  systemctl --user restart naked-tg.service

# Snapshot metrics + read recent history
export $(grep -E "^TELEGRAM_API" .env | xargs)
NAKED_TG_SESSION=research_session \
  python3 naked/skills/telegram-reader/scripts/tg_send.py last zGsR_bot --limit 5

# Progressive depth tests
NAKED_TG_SESSION=research_session python3 .../tg_send.py ask zGsR_bot "/help" --timeout 30
NAKED_TG_SESSION=research_session python3 .../tg_send.py ask zGsR_bot "research_list_specs" --timeout 60
NAKED_TG_SESSION=research_session python3 .../tg_send.py ask zGsR_bot "echo: ping pong" --timeout 30

# Counter delta + journal check
curl -s http://127.0.0.1:9898/metrics > /tmp/metrics_after.txt
diff /tmp/metrics_before.txt /tmp/metrics_after.txt | grep -E "turn_completed|provider_perm"
journalctl --user -u naked-tg.service --since "5 min ago" -p warning
```

Research account (`@Axelis_taurus_chats`, id `6196099449`) reserved
именно для этого — не спамить в zverozabr primary. Session file:
`naked/skills/telegram-reader/.session/research_session.session`.

Documented in `naked/docs/postmortems/2026-05-13-phase4-tg-verification.md`.

## Multi-agent workflow (proven 2026-05-08–2026-05-09)

Large architectural batches (T1-T26 across two waves, score 8.7 → 9.6)
were executed by parallel pi-subagents on `claude-sonnet-4-6`. Patterns
that worked, encoded for future batches:

- **Compact prompts (1-2 KB).** Reference the plan file from the
  worktree (`naked/docs/PLAN_*.md`) for templates and anti-footgun
  lists rather than re-pasting them. Long prompts (3-5 KB) cause
  workers to die after 1 commit due to context-budget exhaustion.
- **Worktrees + named-paths only.** Each task gets its own
  `git worktree add /tmp/agents/<slug>` from current `main`. **Never**
  `git add -A` from worktree root — it pulls in `target/` (one v1
  worker accidentally committed 2456 build artifacts).
- **Targeted test commands.** Workers use `cargo test -p <crate> --test
  <name>` and `-p <crate> --lib`, never `--workspace` (see Commands
  section).
- **Pre-flight scout-surveys.** Read-only `scout` agents with
  `subagent({async: true})` produce inventories (line numbers, risk
  registers) before workers touch code. Saved hours on T2/T3/T4/T11/T12.
- **`#[allow(clippy::...)]` only at module scope** of test-utility
  files (rationale required) or `#[cfg_attr(test, allow(...))]` on
  test functions. Function-level prod allows are instant reject.
- **Continuation pattern when worker dies mid-task.** Compact
  follow-up prompt referencing the killed worker's last commit:
  "Step 1 already in commit `<sha>`. Your job is just X." Saved T2,
  T3, T4 from manual recovery.
- **`needsAttentionAfterMs` override.** Set 600s (or 900s for heavy
  cargo cycles) on `subagent({control: {needsAttentionAfterMs: ...}})`
  to suppress 60-second false alarms during compilation.

Full lessons in [`naked/docs/postmortems/`](naked/docs/postmortems/);
plan templates in [`naked/docs/done/`](naked/docs/done/) for shape
reference. The plan-as-contract structure (`§ 0` pre-flight, `§ 3`
shared SPLIT TEMPLATE, `§ 4` per-task with acceptance one-liners,
`§ 6` instant-reject anti-patterns) is the recommended template for
future batches.
