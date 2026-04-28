# naked-tg

Telegram channel for the [`naked`](../../README.md) agent runtime. Wraps the
core agent loop in a Telegram bot that handles text messages, photo / sticker
media, voice notes, documents, and operator commands (`/sessions`, `/abort`,
`/provider`, `/model`, `/metrics`, `/new`, `/clear`, …).

## Quick start

1. Get a bot token from [`@BotFather`](https://t.me/BotFather).
2. Configure `naked.json` (see [`config.example.json`](../../config.example.json)).
3. Set `TELEGRAM_BOT_TOKEN` and your provider API keys in the environment.
4. Run the bot:
   ```bash
   cargo run -p naked-tg --release
   ```
5. (Optional) Install as a user systemd service — see
   [`naked/docs/`](../../docs/) for the unit file.

## Telegram-specific features

### Multimodal (photos & stickers)

`naked-tg` ingests Telegram photos and **static** stickers and routes them
through the agent's multimodal pipeline:

- **Native path** (default): raw bytes are attached to the chat request as
  inline `image` content. Works with Anthropic Claude 3+, OpenAI gpt-4o
  family, Groq llama-4-scout, xAI grok-vision, Gemini 1.5/2, Qwen-VL.
- **Describer fallback**: text-only models route the image through a
  cheap vision provider (configured at `tg_media.vision`) which produces
  a text caption that is inlined into the prompt.

Vision capability is detected via:

1. `tg_media.model_vision_overrides[<id substring>] = true|false`
2. `providers.<x>.supports_vision = true|false` (provider-wide)
3. Built-in needles (`claude-3+`, `gpt-4o`, `llama-4-*`, `gemini-1.5+`, …)
4. `tg_media.vision_model_extras` (your own substring rules)

See [`naked/docs/vision-models.md`](../../docs/vision-models.md) for the
full playbook.

#### Per-provider image caps

Vision APIs reject oversize uploads with opaque 4xx errors. We pre-floor
the global `tg_media.native_image_max_bytes` against each provider's
documented ceiling: Anthropic 5 MB, Groq 4 MB, Gemini 7 MB, xAI 10 MB,
OpenAI / OpenRouter 20 MB. Photos that exceed the per-provider cap are
downgraded to the describer path with a `[⚠ vision describer failed: …]`
fallback if the describer is unavailable.

#### Image quality knob

`tg_media.image_detail = "low" | "high" | "auto"` controls token spend on
OpenAI-compatible vision endpoints (`image_url.detail`). Anthropic /
Gemini / xAI ignore the field. Default: provider's choice (`auto`).

#### Magic-byte sniffing

Before attaching `image/*` payloads to a vision API we check for known
JPEG / PNG / GIF / WebP magic headers. A truncated download or a misnamed
`.bin` file is downgraded to the describer path so the user gets a
useful error instead of an opaque API 400.

### Media groups (albums)

Telegram delivers each photo in a media group as a separate update tagged
with the same `media_group_id`. `naked-tg` debounces the batch (1.2 s
default) and dispatches **one** agent turn per album, with all images
attached to the same user message in the original order. The album's
caption (Telegram only stores it on one item) is preserved.

### Voice & audio transcription

Configure `tg_media.audio` with an OpenAI-compatible Whisper endpoint
(Groq `whisper-large-v3` is recommended). Transcripts are inlined into
the prompt with a `[🎤 voice 0:42]` header; the original file lands in
the workspace `artifacts/` directory and is referenced by path.

### Documents & files

Files smaller than `tg_media.docs_inline_max_bytes` (default 256 KiB) are
inlined as fenced code blocks; bigger files are saved to disk with a
`[📄 document …]` header and exposed to the agent via `read_file`.

### Sender attribution (groups)

In group chats, messages are prefixed with `@<username>:` so the agent
can address replies correctly. Disable per chat with
`/attribution off` (no restart required).

### Rate limiting

Per-chat rate limiter (default: 10s minimum interval, 60/min cap). Tunable
in `naked-tg/src/main.rs::TgRateLimiter` — currently not exposed in
config.

### Permission model

Only chat IDs in `allowed_chat_ids` are accepted; everything else is
silently dropped. Tools that need `bash` or `write_file` access prompt
the user via inline keyboard the first time they're invoked in a session;
subsequent calls are auto-approved unless the user toggled YOLO mode off
with `/new`.

## Operator commands

| Command            | Effect                                                         |
| ------------------ | -------------------------------------------------------------- |
| `/start`           | Sanity-check the bot is online.                                |
| `/new`             | Open a new session, disable YOLO.                              |
| `/sessions`        | List recent sessions for this chat.                            |
| `/abort`           | Cancel the in-flight agent turn.                               |
| `/provider`        | Show / switch the active provider (inline keyboard).           |
| `/model`           | Show / switch the active model (inline keyboard).              |
| `/clear`           | Reset the conversation history (keeps the session).            |
| `/attribution`     | Toggle `@username:` prefixing in groups (`on`/`off`/`status`). |
| `/metrics`         | Snapshot of media-routing counters (process lifetime).         |

## Health & ops

- `HEALTH_PORT=<port>` env var spins up a TCP health endpoint.
- Sessions live under `<workspace>/.naked/sessions/<id>/`. Image
  artifacts are externalized to `<id>/artifacts/img_*.<ext>` and
  rehydrated on load. Use `naked vacuum-sessions` (CLI) to migrate
  legacy sessions and garbage-collect orphan artifacts.
- Logs live under `~/.naked/logs/` with daily rotation.

## Architecture pointers

- [`src/main.rs`](src/main.rs) — manual `getUpdates` loop, message dispatch,
  command routing.
- [`src/album.rs`](src/album.rs) — media-group debounce buffer.
- [`src/media.rs`](src/media.rs) — Telegram media download + describer
  + transcriber clients.
- [`src/metrics.rs`](src/metrics.rs) — in-process counters surfaced by
  `/metrics`.
- [`src/channel_map.rs`](src/channel_map.rs) — per-(chat, thread) ↔ session
  mapping with persistence.

## See also

- [`naked/docs/vision-models.md`](../../docs/vision-models.md) — vision-model
  configuration playbook.
- [`naked/docs/multimodal-followups.md`](../../docs/multimodal-followups.md) —
  remaining multimodal roadmap items.
- [`naked/CHANGELOG.md`](../../CHANGELOG.md) — release notes.
