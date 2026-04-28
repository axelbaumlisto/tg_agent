# tg_agent — Autonomous Telegram Research Agent

A Rust-based autonomous coding and research agent with a Telegram
interface. Talks to LLM providers directly (Anthropic, OpenAI-compat,
Copilot, and others) — no intermediate proxy.

## Quick start

```bash
cp .env.shared.example .env   # fill in API keys + TELEGRAM_BOT_TOKEN
cd naked
cargo build --release
./target/release/naked-tg      # or install as systemd service
```

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

See [`AGENTS.md`](AGENTS.md) for developer instructions,
[`OPERATIONS.md`](OPERATIONS.md) for production deployment.

## Commands

```bash
cd naked
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --workspace
cargo build --release
```

## Structure

- `naked/crates/naked-core/` — agent loop, tool system, providers, memory, MCP client
- `naked/crates/naked-cli/` — CLI binary
- `naked/crates/naked-tg/` — Telegram bot binary
- `naked/scripts/` — Python helper scripts (fetch, translate, index, etc.)
- `naked/ops/` — systemd units + cron snippets
