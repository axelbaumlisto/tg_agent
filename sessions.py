"""Persistent mapping: chat_id:thread_id → agent session_id + model prefs.

All conversation state lives in the agent backend. We only store the
session ID and model preference so we can restore them after a bot restart.
"""
import json
import os
import pathlib
import tempfile
import threading
from typing import Optional

from . import config
from .protocols import ChatKey, ModelRef


class SessionStore:
    """Thread-safe JSON-backed store with atomic writes."""

    def __init__(self, path: pathlib.Path = config.SESSIONS_FILE):
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        try:
            raw = json.loads(self._path.read_text())
            if isinstance(raw, dict):
                migrated: dict[str, dict] = {}
                for k, v in raw.items():
                    if isinstance(v, str):
                        migrated[k] = {"session_id": v}
                    elif isinstance(v, dict):
                        migrated[k] = v
                self._data = migrated
            else:
                self._data = {}
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def key(chat_id: str, thread_id: Optional[str]) -> str:
        return str(ChatKey(chat_id, thread_id))

    def get(self, chat_id: str, thread_id: Optional[str] = None) -> Optional[str]:
        with self._lock:
            entry = self._data.get(self.key(chat_id, thread_id))
            if entry is None:
                return None
            return entry.get("session_id")

    def set(self, chat_id: str, thread_id: Optional[str], session_id: str):
        with self._lock:
            k = self.key(chat_id, thread_id)
            entry = self._data.get(k, {})
            entry["session_id"] = session_id
            self._data[k] = entry
            self._save()

    def delete(self, chat_id: str, thread_id: Optional[str] = None):
        with self._lock:
            k = self.key(chat_id, thread_id)
            if k in self._data:
                del self._data[k]
                self._save()

    def clear_session(self, chat_id: str, thread_id: Optional[str] = None):
        """Remove session_id and model but preserve directory and auto_approve."""
        with self._lock:
            k = self.key(chat_id, thread_id)
            entry = self._data.get(k)
            if entry is None:
                return
            changed = False
            if "session_id" in entry:
                del entry["session_id"]
                changed = True
            if "model" in entry:
                del entry["model"]
                changed = True
            if not entry:
                del self._data[k]
                changed = True
            if changed:
                self._save()

    def all_entries(self) -> dict[str, str]:
        """Return {key: session_id} for all stored sessions."""
        with self._lock:
            return {k: v["session_id"] for k, v in self._data.items() if "session_id" in v}

    def find_by_session_id(self, session_id: str) -> Optional[ChatKey]:
        """Return ChatKey for a given session_id."""
        with self._lock:
            for k, v in self._data.items():
                if v.get("session_id") == session_id:
                    return ChatKey.parse(k)
        return None

    # -- model preference ---------------------------------------------------

    def get_model(self, chat_id: str, thread_id: Optional[str] = None) -> Optional[ModelRef]:
        with self._lock:
            entry = self._data.get(self.key(chat_id, thread_id))
            if entry is None:
                return None
            mid = entry.get("model")
            if not mid or "/" not in mid:
                return None
            provider_id, model_id = mid.split("/", 1)
            return ModelRef(provider_id=provider_id, model_id=model_id)

    def set_model(self, chat_id: str, thread_id: Optional[str], provider_id: str, model_id: str):
        with self._lock:
            k = self.key(chat_id, thread_id)
            entry = self._data.get(k, {})
            entry["model"] = f"{provider_id}/{model_id}"
            self._data[k] = entry
            self._save()

    def delete_model(self, chat_id: str, thread_id: Optional[str] = None):
        with self._lock:
            k = self.key(chat_id, thread_id)
            entry = self._data.get(k)
            if entry and "model" in entry:
                del entry["model"]
                self._save()

    # -- per-chat directory ---------------------------------------------------

    def get_directory(self, chat_id: str, thread_id: Optional[str] = None) -> Optional[str]:
        with self._lock:
            entry = self._data.get(self.key(chat_id, thread_id))
            if entry is None:
                return None
            return entry.get("directory")

    def set_directory(self, chat_id: str, thread_id: Optional[str], directory: str):
        with self._lock:
            k = self.key(chat_id, thread_id)
            entry = self._data.get(k, {})
            entry["directory"] = directory
            self._data[k] = entry
            self._save()

    def delete_directory(self, chat_id: str, thread_id: Optional[str] = None):
        with self._lock:
            k = self.key(chat_id, thread_id)
            entry = self._data.get(k)
            if entry and "directory" in entry:
                del entry["directory"]
                self._save()

    # -- auto-approve toggle --------------------------------------------------

    def get_auto_approve(self, chat_id: str, thread_id: Optional[str] = None) -> bool:
        with self._lock:
            entry = self._data.get(self.key(chat_id, thread_id))
            if entry is None:
                return False
            return bool(entry.get("auto_approve"))

    def set_auto_approve(self, chat_id: str, thread_id: Optional[str], value: bool):
        with self._lock:
            k = self.key(chat_id, thread_id)
            entry = self._data.get(k, {})
            entry["auto_approve"] = value
            self._data[k] = entry
            self._save()
