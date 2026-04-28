# tg_agent Source

Source-only export of the Telegram/OpenCode bridge and Rust `naked`
agent daemon. Runtime data, private operations docs, skills, local
configs, sessions, results, and deployment files are intentionally not
included.

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

```bash
cd naked
cargo build --release
```

The Telegram daemon source lives in `naked/crates/naked-tg`; the shared
agent core lives in `naked/crates/naked-core`.

### 3. Run

```bash
cd naked
./target/release/naked-tg
```

For CLI usage:

```bash
cd naked
cargo run -p naked-cli -- --help
```

## Runtime Manual

### Skills

Skills are reusable instructions or recipes that the Rust agent can load
during a run. They are discovered from `skill_roots`.

Default roots when `skill_roots` is omitted:

```text
~/.naked/skills
~/.agents/skills
```

You can override them in `naked.json`:

```json
{
  "skill_roots": [
    "./skills",
    "~/.naked/skills"
  ]
}
```

A skill lives in its own directory:

```text
skills/
└── my-skill/
    ├── SKILL.md
    └── SKILL.json
```

`SKILL.md` is the simple advisory format. The agent reads it as
instructions:

```markdown
# My Skill

Use this when the user asks for ...

## Steps

1. Inspect the input.
2. Call the relevant tools.
3. Return a concise result.
```

`SKILL.json` is the structured format. It supports three modes:

- `advisory`: return `advisory_body` to the model.
- `executable`: host runs `steps` and returns a bundle.
- `hybrid`: host runs `steps`, then gives `after_executable` to the
  model for follow-up judgement.

Example:

```json
{
  "name": "fetch-and-summarize",
  "description": "Fetch a URL and summarize it",
  "mode": "hybrid",
  "steps": [
    {
      "name": "Fetch page",
      "tool": "web_fetch",
      "args": { "url": "{url}" },
      "save_as": "page",
      "on_error": "abort"
    }
  ],
  "after_executable": "Summarize the fetched page for the user."
}
```

Placeholders like `{url}` are filled from skill arguments or previous
`save_as` outputs.

### MCP Servers

MCP servers add external tools to the agent. Configure them under
`mcpServers` in `naked.json`.

Stdio server:

```json
{
  "mcpServers": {
    "filesystem": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/workspace"],
      "env": {}
    }
  }
}
```

HTTP or SSE server:

```json
{
  "mcpServers": {
    "tools-api": {
      "transport": "http",
      "url": "http://127.0.0.1:3001/mcp",
      "headers": {
        "Authorization": "Bearer ${MCP_TOKEN}"
      },
      "tool_timeout_secs": 120
    }
  }
}
```

Config fields:

- `transport`: `stdio`, `http`, or `sse`.
- `command` and `args`: used by `stdio`.
- `url`: used by `http` / `sse`.
- `env`: extra environment for the MCP process.
- `headers`: request headers for HTTP/SSE transports.
- `tool_timeout_secs`: per-server timeout override.

Run with MCP:

```bash
cd naked
NAKED_CONFIG=/absolute/path/to/naked.json ./target/release/naked-tg
```

### Sessions

Rust sessions are persisted under `session_dir`.

Default:

```text
<workspace>/.naked/sessions
```

Configure it:

```json
{
  "workspace": "/absolute/path/to/workspace",
  "session_dir": ".naked/sessions"
}
```

Each session has its own directory containing:

```text
<session-id>/
├── session.jsonl
├── artifacts/
├── prompt.md
└── skills/
```

What is stored:

- `session.jsonl`: conversation events and tool-visible history.
- `artifacts/`: files created or downloaded for that session.
- `prompt.md`: prompt snapshot/material used for the session.
- `skills/`: session-local skill material when present.

Telegram chats map to sessions internally. The bot restores session
state from disk on restart when the session files still exist.

Python bridge sessions are separate and use:

```dotenv
OC_TG_SESSIONS_FILE=./sessions.json
```

### Memory

Memory is separate from session history. Session history is the current
conversation; memory is durable knowledge that can be recalled in later
sessions.

Memory scopes:

- `project`: tied to the workspace.
- `global`: shared across projects.
- `user:<id>`: tied to one user.

Memory files:

```text
~/.naked/memory/MEMORY.md
~/.naked/projects/<workspace-slug>/memory/MEMORY.md
~/.naked/users/<user-id>/memory/MEMORY.md
```

Daily draft/digest files live beside `MEMORY.md`:

```text
memory/YYYY-MM-DD.md
DREAMS.md
```

Memory entry types:

- `preference`
- `correction`
- `project_knowledge`
- `failure`

Minimal memory config:

```json
{
  "memory": {
    "daily_enabled": true,
    "daily_cron": "0 4 * * *",
    "mode": "summarize_and_promote"
  }
}
```

Useful Telegram commands:

```text
/memory help
/memory ls
/memory dreams
/memory drafts
/memory stats
```

### Per-Chat Personas

You can isolate a Telegram chat into its own workspace and memory
namespace without running another bot process:

```json
{
  "chat_personas": {
    "123456789": {
      "name": "client-a",
      "workspace": "/absolute/path/to/client-a-workspace",
      "allow_slash_commands": true
    }
  }
}
```

Changing `name` or `workspace` changes where that persona's memory and
project-local prompt material are found.

## Commands

| Command | Description |
|---------|-------------|
| `/models` | Show available models grouped by provider; reply with a number to switch |
| `/model provider/id` | Switch model directly |
| `/reset` / `/new` | Reset the current session |
| `/id` | Show chat/session debug info |
| `/memory` | Inspect project memory, drafts, dreams, and stats |

## Configuration

Configuration is provided by environment variables. Key variables:

- `TELEGRAM_BOT_TOKEN` — Telegram bot token (required)
- `OC_BASE_URL` — OpenCode server URL (default: `http://127.0.0.1:14096`)
- `OC_DIRECTORY` — default workspace directory
- `ALLOWED_CHAT_IDS` — comma-separated allowlist; empty = open access

## Tests

```bash
# Unit tests (no external services)
cd python_bridge
uv sync --extra test
uv run python -m pytest tests/test_oc_client.py tests/test_protocols.py -v

# Rust compile check
cd ../naked
cargo check --workspace

# E2E tests (requires running bot + OpenCode + Telethon session)
cd ../python_bridge
uv run python -m pytest tests/test_e2e.py -v -s
```
