"""Read configuration from zeroclaws .env and environment."""
from __future__ import annotations

import os
import pathlib

_ENV_FILE = pathlib.Path(__file__).parent.parent / ".env"


def _load_env(path: pathlib.Path) -> dict:
    result = {}
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            result[k.strip()] = v
    except OSError:
        pass
    return result


_env = _load_env(_ENV_FILE)


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key) or _env.get(key) or default


BOT_TOKEN: str = _get("TELEGRAM_BOT_TOKEN")
OC_BASE_URL: str = _get("OC_BASE_URL", "http://127.0.0.1:14096")
OC_DIRECTORY: str = _get("OC_DIRECTORY", str(pathlib.Path(__file__).parent.parent))

# Access control — comma-separated chat IDs; empty = allow all
_raw_allowed = _get("ALLOWED_CHAT_IDS")
ALLOWED_CHAT_IDS: set[str] = {s.strip() for s in _raw_allowed.split(",") if s.strip()} if _raw_allowed else set()

# Telegram API for tests (Telethon)
def _int(key: str, default: int = 0) -> int:
    raw = _get(key, str(default))
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


TG_API_ID: int = _int("TELEGRAM_API_ID")
TG_API_HASH: str = _get("TELEGRAM_API_HASH")
OPERATOR_CHAT_ID: int = _int("TELEGRAM_OPERATOR_CHAT_ID")

# Sessions file — lives next to this package
SESSIONS_FILE: pathlib.Path = pathlib.Path(__file__).parent / "sessions.json"

# Telegraph
TELEGRAPH_TOKEN_FILE: pathlib.Path = pathlib.Path(__file__).parent / ".telegraph_token"

# Runner config
IDLE_TIMEOUT_SECONDS: int = int(_get("OC_TG_IDLE_TIMEOUT", str(60 * 60)))  # 1 hour
EDIT_INTERVAL_SECONDS: float = float(_get("OC_TG_EDIT_INTERVAL", "2.0"))
MAX_MESSAGE_CHUNKS: int = 3
TYPING_INTERVAL_SECONDS: float = 4.0
RECONNECT_BACKOFF_MAX: float = 60.0
WATCHDOG_INTERVAL_SECONDS: float = float(_get("OC_TG_WATCHDOG_INTERVAL", "60"))
STALL_TIMEOUT_SECONDS: float = float(_get("OC_TG_STALL_TIMEOUT", "90"))
GENERATING_STALL_SECONDS: float = float(_get("OC_TG_GENERATING_STALL", "90"))
REASONING_STALL_SECONDS: float = float(_get("OC_TG_REASONING_STALL", "120"))

# Provider fallback — ordered list of "provider/model" to try on balance/auth errors
_raw_fallback = _get("OC_TG_FALLBACK_MODELS",
                      "minimax/MiniMax-M2.7-highspeed,"
                      "minimax-coding-plan/MiniMax-M2.7-highspeed,"
                      "kimi-for-coding/k2p5,"
                      "glm/glm-5.1")
FALLBACK_MODELS: list[tuple[str, str]] = []
for _entry in _raw_fallback.split(","):
    _entry = _entry.strip()
    if "/" in _entry:
        _p, _m = _entry.split("/", 1)
        FALLBACK_MODELS.append((_p, _m))

_BALANCE_ERROR_PATTERNS: list[str] = [
    "insufficient balance",
    "insufficient_quota",
    "rate_limit",
    "invalid api key",
    "authentication_error",
    "billing",
    "quota exceeded",
]


def is_provider_error(text: str) -> bool:
    """Return True if the error text looks like a provider balance/auth failure."""
    low = text.lower()
    return any(p in low for p in _BALANCE_ERROR_PATTERNS)
