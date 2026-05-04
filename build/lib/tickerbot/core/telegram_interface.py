from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import requests


log = logging.getLogger(__name__)


class TelegramInterface:
    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = str(chat_id)
        self.base_url = f"https://api.telegram.org/bot{token}"
        self._offset = 0
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def send_message(self, text: str) -> None:
        payload = {
            "chat_id": self.chat_id,
            "text": text[:4000],
        }
        try:
            requests.post(f"{self.base_url}/sendMessage", json=payload, timeout=15)
        except Exception as exc:
            log.warning("send_message failed: %s", exc)

    def send_document(self, file_path: str, caption: str = "") -> None:
        path = Path(file_path)
        if not path.exists():
            return
        try:
            with path.open("rb") as handle:
                requests.post(
                    f"{self.base_url}/sendDocument",
                    data={"chat_id": self.chat_id, "caption": caption[:1024]},
                    files={"document": (path.name, handle)},
                    timeout=30,
                )
        except Exception as exc:
            log.warning("send_document failed: %s", exc)

    def poll_commands(self, on_command: Callable[[str], str], interval_seconds: int = 5) -> None:
        while not self._stop.is_set():
            try:
                response = requests.get(
                    f"{self.base_url}/getUpdates",
                    params={"timeout": interval_seconds, "offset": self._offset + 1},
                    timeout=interval_seconds + 10,
                )
                payload = response.json()
                updates = payload.get("result", [])

                for upd in updates:
                    self._offset = max(self._offset, upd.get("update_id", 0))
                    message = upd.get("message", {})
                    text = message.get("text", "").strip()
                    chat = str(message.get("chat", {}).get("id", ""))

                    if not text or chat != self.chat_id:
                        continue

                    reply = on_command(text)
                    if reply:
                        self.send_message(reply)
            except Exception as exc:
                log.warning("poll_commands error: %s", exc)
