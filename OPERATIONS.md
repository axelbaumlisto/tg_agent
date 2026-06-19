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

## Pruning dead provider keys (offline)

Dead keys recorded in `_dead_api_keys*` / `_low_balance_keys*` buckets can
be removed from provider `api_keys[]` with the live service stopped. Follow
the AB-11 state-mutation protocol:

```bash
systemctl --user stop naked-tg.service
cd naked
scripts/prune_dead_keys.sh --config ../state/naked.json          # dry-run masked diff
scripts/prune_dead_keys.sh --config ../state/naked.json --apply  # after review
git -C .. status --short
# verify no raw key was printed, then commit state/naked.json to HEAD immediately
cargo build --release
systemctl --user restart naked-tg.service
```

The tool leaves primary `api_key` fields untouched; rotate those manually
if a primary key is dead. Do not run this against live `state/naked.json`
while `naked-tg.service` is active unless you intentionally use `--force`.

## Pre-turn workspace snapshots (the "stash gremlin")

`naked` takes a **pre-turn snapshot of its workspace before every turn**
that has uncommitted changes. Two mechanisms run fire-and-forget from
`dispatch_turn` (`naked-core/src/session_ops/turn.rs:159-188`):

1. **Legacy git stash** (`snapshot/legacy_stash.rs:14-28`):
   `git stash push -m "naked:pre-turn:<seq>" --include-untracked` on the
   workspace if it is a git repo and not clean. `git stash` internally
   performs a `reset: moving to HEAD`.
2. **Side-git SnapshotRepo** (`snapshot/repo.rs`) under
   `~/.naked/snapshots/<hash>/.git` — this is what `revert_turn` /
   `/restore` read; it does NOT touch the workspace tree.

**Consequence (important for development on this host).** The bot's
default `workspace` is `current_dir` = the systemd `WorkingDirectory`
(`<repo>/naked`, see config `default_workspace`). So when the live bot
processes a turn while you have **uncommitted edits in this repo**, the
legacy stash path will `git stash` them away (and `state/naked.json`,
reached via the tracked `naked/naked.json` symlink, is reverted to
HEAD). This produced 70+ `naked:pre-turn:N` stashes and silently ate
uncommitted plan files and a runtime-config edit during development.

**Operating rules:**
- **Commit (and push) each change immediately.** Never leave runtime
  config or docs uncommitted in the working tree while the service is
  live — a turn can stash it.
- **Runtime config changes (`state/naked.json`) must be committed** to
  the Forgejo branch, not left working-tree-only. A committed value
  survives the stash (the post-stash tree matches HEAD).
- Recover an eaten change with `git stash list` + `git stash show -p
  stash@{N}`; the side-git snapshots are an independent backup.
- `git stash clear` only after inventorying/exporting valuable stashes;
  it will re-populate as long as the bot runs turns against a dirty repo.
- If this snapshot-of-the-repo behaviour is unwanted in dev, point the
  bot at a dedicated scratch `workspace` (config/per-chat workspace) so
  it never stashes the source checkout.

## Disk Hygiene

`naked/target/` is build cache and can be regenerated with Cargo.
`naked/results/`, `naked/data/`, and `.naked/` are runtime state;
back them up before destructive operations and keep them out of public
commits unless intentionally publishing sanitized fixtures.
