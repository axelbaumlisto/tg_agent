# Agent Bridge — Telegram ↔ OpenCode

A pluggable bridge between messaging platforms and coding agents. Ships with **Telegram** (Bot API) and **OpenCode** backends.

## Quick start

```bash
cp .env.example ../.env   # config lives one level up (zeroclaws/.env)
# edit ../.env — set TELEGRAM_BOT_TOKEN, OC_BASE_URL, etc.

uv sync                   # install dependencies
uv run python -m opencode_tg.bot
```

## Architecture

```
Messenger (Protocol)        AgentBackend (Protocol)
  └─ TelegramMessenger        └─ OpenCodeBackend
         │                           │
         └────── SessionManager ─────┘
                   │
              SessionRunner (per chat)
```

Add your own messenger or agent by implementing the corresponding protocol in `protocols.py`.

## Commands

| Command | Description |
|---------|-------------|
| `/models` | Show available models grouped by provider; reply with a number to switch |
| `/model provider/id` | Switch model directly |
| `/reset` / `/new` | Reset the current session |
| `/id` | Show chat/session debug info |

## Configuration

All settings are read from environment variables (or `../.env`). See `.env.example` for the full list.

Key variables:

- `TELEGRAM_BOT_TOKEN` — Telegram bot token (required)
- `OC_BASE_URL` — OpenCode server URL (default: `http://127.0.0.1:14096`)
- `ALLOWED_CHAT_IDS` — comma-separated allowlist; empty = open access

## Tests

```bash
# Unit tests (no external services)
uv run python -m pytest tests/test_oc_client.py tests/test_protocols.py -v

# E2E tests (requires running bot + OpenCode + Telethon session)
uv run python -m pytest tests/test_e2e.py -v -s
```
