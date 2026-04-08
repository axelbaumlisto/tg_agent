"""Read configuration from zeroclaws .env and environment."""
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
            result[k.strip()] = v.strip()
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
TG_API_ID: int = int(_get("TELEGRAM_API_ID", "0"))
TG_API_HASH: str = _get("TELEGRAM_API_HASH")
OPERATOR_CHAT_ID: int = int(_get("TELEGRAM_OPERATOR_CHAT_ID", "0"))

# Sessions file — lives next to this package
SESSIONS_FILE: pathlib.Path = pathlib.Path(__file__).parent / "sessions.json"

# Telegraph
TELEGRAPH_TOKEN_FILE: pathlib.Path = pathlib.Path(__file__).parent / ".telegraph_token"

# Runner config
IDLE_TIMEOUT_SECONDS: int = int(_get("OC_TG_IDLE_TIMEOUT", str(60 * 60)))  # 1 hour
EDIT_INTERVAL_SECONDS: float = float(_get("OC_TG_EDIT_INTERVAL", "2.0"))
MAX_MESSAGE_CHUNKS: int = 3
TYPING_INTERVAL_SECONDS: float = 5.0
RECONNECT_BACKOFF_MAX: float = 60.0
WATCHDOG_INTERVAL_SECONDS: float = float(_get("OC_TG_WATCHDOG_INTERVAL", "60"))
