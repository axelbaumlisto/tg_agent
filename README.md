# tg_agent — Autonomous Telegram Research Agent

A Rust-based autonomous coding and research agent with a Telegram
interface. Talks to LLM providers directly (Anthropic, OpenAI-compat,
Copilot, and others) — no intermediate proxy.

## Architecture

```
Telegram (teloxide)
     │
 naked-tg          ← Telegram bot binary (systemd-managed)
     │
 naked-core        ← Agent loop, tools, providers, memory, MCP, research
     │
 LLM APIs          ← Direct calls (Anthropic, OpenAI-compat, Copilot, …)
```

| Crate | Role | Binary |
|---|---|---|
| `naked-core` | Agent loop, tool system, providers, memory, research, MCP | (lib) |
| `naked-cli`  | Local CLI for chat / research / agent management | `naked` |
| `naked-tg`   | Telegram bot daemon | `naked-tg` |

## Quick start

```bash
cp .env.shared.example .env       # fill in API keys + TELEGRAM_BOT_TOKEN
cd naked
cargo build --release
./target/release/naked-tg          # or install as systemd service
```

## Configure

The agent loads `.env` by walking up from the current directory, then
loads JSON config in this order:

1. `NAKED_CONFIG` env var
2. `./.naked/config.json`
3. `./naked.json`
4. `~/.naked/config.json`

Minimal `.env`:

```dotenv
# Either TELEGRAM_BOT_TOKEN or NAKED_TELEGRAM_TOKEN works.
TELEGRAM_BOT_TOKEN=<telegram-bot-token>

# Provider key referenced from naked.json below.
OPENAI_API_KEY=<openai-api-key>

# Optional overrides.
# NAKED_CONFIG=/absolute/path/to/naked.json
# NAKED_PROVIDER=openai
# NAKED_MODEL=gpt-4o-mini
# NAKED_WORKSPACE=/absolute/path/to/workspace
# NAKED_ALLOWED_CHAT_IDS=123456789,987654321
```

Minimal `naked/naked.json`:

```json
{
  "default_provider": "openai",
  "default_model": "gpt-4o-mini",
  "workspace": "/absolute/path/to/workspace",
  "session_dir": "~/.naked/sessions",
  "telegram_bot_token": "$TELEGRAM_BOT_TOKEN",
  "allowed_chat_ids": [123456789],
  "providers": {
    "openai": {
      "type": "openai_compat",
      "api_key": "$OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1",
      "models": ["gpt-4o-mini", "gpt-4o"]
    }
  }
}
```

Notes:

- `allowed_chat_ids` is required for the Telegram bot. If empty, the
  bot boots but rejects all messages.
- Provider `api_key` values may be literal strings or `$ENV_VAR`
  references.
- For OpenAI-compatible providers, set `type: "openai_compat"` and point
  `base_url` at the provider's `/v1` API root.
- Anthropic-style providers can use `type: "anthropic"` with an
  Anthropic API key and model list.

See [`naked/.env.example`](naked/.env.example) and
[`naked/config.example.json`](naked/config.example.json) for templates.

## Layout

```text
.
├── README.md
├── AGENTS.md           ← cross-tool instructions for AI assistants
├── OPERATIONS.md       ← production deployment notes
├── .env.shared.example
└── naked/
    ├── Cargo.toml
    ├── Cargo.lock
    ├── crates/
    │   ├── naked-core/   (lib: agent loop, tools, providers, memory, research, MCP)
    │   ├── naked-cli/    (bin: `naked` — local CLI)
    │   └── naked-tg/     (bin: `naked-tg` — Telegram daemon)
    ├── scripts/          ops scripts (token health, research tick, …)
    ├── skills/           agent skills (JSON / MD)
    ├── tests/            bash e2e + fixtures
    ├── ops/              systemd units + cron snippets
    └── docs/             architecture, plans, postmortems
```

## Commands

```bash
cd naked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release
```

Pre-commit must pass `cargo fmt + clippy -D warnings + cargo test`.

## Status

| Metric | Value |
|---|---|
| Tests | ~1992 (passing on every push) |
| Clippy | `--all-targets -- -D warnings` clean |
| `clippy::unwrap_used` lint | enabled in all 3 crates |
| Top single-file LOC | ≤ 470 (largest: `naked-tg/runtime.rs`) |
| `AgentLoop::run()` | 297 LOC (down from 540) |
| Score (SOLID/DRY/KISS/TDD geomean) | 9.6 / 10 |

Architectural milestones documented in
[`naked/docs/postmortems/`](naked/docs/postmortems/) — see
`2026-05-08-multiagent.md` (v1: 12 tasks) and
`2026-05-09-multiagent-v2.md` (v2: 14 tasks). Each batch was driven by
parallel pi-subagents on `claude-sonnet-4-6` with shared SPLIT TEMPLATE
+ scout pre-flight surveys + acceptance one-liners.

## Documentation index

- [`AGENTS.md`](AGENTS.md) — repo conventions, push targets, anti-footguns
- [`OPERATIONS.md`](OPERATIONS.md) — systemd, cron, monitoring, sibling services
- [`naked/docs/`](naked/docs/) — architecture plans, completed plans (`done/`), postmortems
- [`naked/scripts/`](naked/scripts/) — ops scripts (token health, research tick, supervisor checks)

## Pushing

Two remotes:

```bash
# Forgejo (private, full repo with state/)
git push forgejo main

# GitHub (sanitized — code only, scans for leaked secrets)
naked/scripts/export_github_sanitized.sh
```

The sanitize script strips `state/`, `.env*`, `naked.json`, `target/`,
session files, and aborts if it finds any pattern matching API keys
(`sk-…`, `gho_…`, `xox[baprs]-…`), private SSH keys, or
`TELEGRAM_BOT_TOKEN=…`.

## License

MIT (see `LICENSE` once added). Configuration files and runtime state
(`state/`, `.env`, etc.) are private and never published to GitHub.
