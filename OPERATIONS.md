# OPERATIONS

Public operations notes for running `tg_agent`. Keep host-specific
paths, tokens, chat IDs, and private URLs out of this file.

## Layout

| Path | Purpose |
|------|---------|
| `<repo>/.env` | local runtime secrets, never committed |
| `<repo>/.env.shared.example` | shared environment template |
| `<repo>/naked/` | Rust workspace and `naked-tg` Telegram binary |

## Build

```bash
cd naked
cargo build --release
```

## systemd

The production Telegram bot runs as a user service. Install the
unit template from `naked/ops/systemd/naked-tg.service`, then replace
absolute paths with your local repository path.

Useful commands:

```bash
systemctl --user status naked-tg.service
systemctl --user restart naked-tg.service
journalctl --user -u naked-tg.service -f
```

The unit should point to:

```ini
WorkingDirectory=<repo>/naked
EnvironmentFile=<repo>/.env
ExecStart=<repo>/naked/target/release/naked-tg
```

## cron

Cron entries are generated from `naked/ops/cron/research_tick.cron`.
Do not hand-edit generated blocks; reinstall them from the cron ops
directory.

```bash
make -C naked/ops/cron install-cron
```

## External Services

The repo may be configured to talk to services that are intentionally
not managed here, such as:

- Telegram Bot API.
- Optional browser/VNC infrastructure for manual login flows.
- Optional LLM/search provider APIs.

Configure those services via `.env`, not by committing host-specific
paths or credentials.

## Caddy reverse-proxy

Production TLS edge for `clipshot.cc` is a `spex-caddy` Docker container.
Source-controlled Caddyfile lives in [`naked/ops/caddy/Caddyfile`](naked/ops/caddy/Caddyfile);
production install point is `/home/spex/app/caddy/Caddyfile`.

Recommended symlink so repo ⇒ prod is one-direction:

```bash
sudo ln -sf /home/spex/work/tg_agent/naked/ops/caddy/Caddyfile \
            /home/spex/app/caddy/Caddyfile
```

Reload after edits (zero-downtime):

```bash
docker exec spex-caddy caddy reload --config /etc/caddy/Caddyfile
```

See [`naked/ops/caddy/README.md`](naked/ops/caddy/README.md) for routes,
anti-footguns, and smoke tests.

## Disk Hygiene

`naked/target/` is build cache and can be regenerated with Cargo.
`naked/results/`, `naked/data/`, and `.naked/` are runtime state;
back them up before destructive operations and keep them out of public
commits unless intentionally publishing sanitized fixtures.
