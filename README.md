# tg_agent — Autonomous Telegram Research Agent

A Rust-based autonomous coding and research agent with a Telegram
interface. Talks to LLM providers directly (Anthropic, OpenAI-compat,
Copilot, and others) — no intermediate proxy.

<<<<<<< Updated upstream
## Quick start
=======
For the complete feature documentation, see
[`docs/FULL_MANUAL.md`](docs/FULL_MANUAL.md). It covers configuration,
Telegram commands, tools and permissions, skills, MCP, sessions, memory,
media handling, research jobs, personas, testing, and troubleshooting.

## Layout

```text
.
├── README.md
├── naked/
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── crates/
│       ├── naked-core/
│       ├── naked-cli/
│       └── naked-tg/
└── python_bridge/
    ├── pyproject.toml
    ├── src/opencode_tg/
    ├── scripts/
    └── tests/
```

## Python Bridge

### 1. Configure

Create `.env` in the repository root, or export the same variables in
your shell:

```dotenv
TELEGRAM_BOT_TOKEN=<telegram-bot-token>
OC_BASE_URL=http://127.0.0.1:14096
OC_DIRECTORY=/absolute/path/to/workspace

# Optional: comma-separated chat ids. Empty means anyone can use the bridge.
ALLOWED_CHAT_IDS=

# Optional runtime files.
OC_TG_SESSIONS_FILE=./sessions.json
OC_TG_TELEGRAPH_TOKEN_FILE=./.telegraph_token
```

`OC_BASE_URL` must point to a running OpenCode-compatible HTTP API.

### 2. Run

```bash
cd python_bridge
uv sync --extra test
uv run python -m opencode_tg.bot
```

The bridge is a pluggable reference implementation with Telegram and
OpenCode backends:

```text
Messenger (Protocol)        AgentBackend (Protocol)
  └─ TelegramMessenger        └─ OpenCodeBackend
         │                           │
         └────── SessionManager ─────┘
                   │
              SessionRunner (per chat)
```

Add your own messenger or agent by implementing the corresponding protocol
in `python_bridge/src/opencode_tg/protocols.py`.

## Rust Agent

### 1. Configure

The Rust agent loads `.env` by walking up from the current directory,
then loads JSON config in this order:

1. `NAKED_CONFIG`
2. `.naked/config.json`
3. `naked.json`
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

- `allowed_chat_ids` is required for the Rust Telegram bot. If it is
  empty, the bot boots but rejects all messages.
- Provider `api_key` values may be literal strings or `$ENV_VAR`
  references.
- For OpenAI-compatible providers, set `type` to `openai_compat` and
  point `base_url` at the provider's `/v1` API root.
- Anthropic-style providers can use `type: "anthropic"` with an
  Anthropic API key and model list.

### 2. Build
>>>>>>> Stashed changes

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
