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

## Code map

```
tg_agent/
├── README.md              # what this project is
├── OPERATIONS.md          # how it runs on this host (systemd, cron)
├── AGENTS.md              # ← you are here
├── .env                   # runtime secrets (gitignored)
├── .env.shared.example    # full env template (Rust agent + scripts)
└── naked/                 # ── Rust autonomous agent ──
    ├── Cargo.toml         # workspace root
    ├── naked.json         # agent config (models, MCPs, scheduler)
    ├── crates/
    │   ├── naked-core/    # tool loop, providers, channels, memory, MCP client
    │   ├── naked-cli/     # `naked` CLI
    │   └── naked-tg/      # `naked-tg` Telegram binary (systemd-managed)
    ├── scripts/           # ops scripts (token health, research tick, …)
    ├── ops/{systemd,cron} # unit files + cron snippets installed on the host
    ├── docs/              # architecture + planning docs
    ├── data/research_runs/# persistent research state (do not nuke)
    └── results/           # rendered HTML reports (persistent)
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
