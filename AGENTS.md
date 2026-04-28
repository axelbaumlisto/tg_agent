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
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo build --release           # produces target/release/{naked,naked-tg}
```

Pre-commit must pass `cargo fmt + clippy -D warnings + cargo test`
for the Rust workspace.

## Push targets

```bash
# Forgejo (full: code + state)
git push forgejo main

# GitHub (sanitized: code only, no state/)
naked/scripts/export_github_sanitized.sh
```

## Conventions

- **Editing scripts in `naked/scripts/`** — some deployments use
  absolute paths in `naked/ops/{systemd,cron}/`, the installed user
  service, and the active crontab. Keep those in sync with your local
  install path.
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
