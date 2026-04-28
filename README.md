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

```bash
cd naked
cargo test --workspace
cargo build --release
```

The Telegram daemon source lives in `naked/crates/naked-tg`; the shared
agent core lives in `naked/crates/naked-core`.

## Commands

| Command | Description |
|---------|-------------|
| `/models` | Show available models grouped by provider; reply with a number to switch |
| `/model provider/id` | Switch model directly |
| `/reset` / `/new` | Reset the current session |
| `/id` | Show chat/session debug info |

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

# E2E tests (requires running bot + OpenCode + Telethon session)
uv run python -m pytest tests/test_e2e.py -v -s
```
