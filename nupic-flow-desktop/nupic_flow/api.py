from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests


class NupicFlowApi:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "NupicAI-Flow/0.1"

    def set_session_token(self, token: str) -> None:
        self.session.cookies.set("nupicai_session", token, path="/")

    def session_token(self) -> str:
        return str(self.session.cookies.get("nupicai_session") or "")

    def login(self, email: str, password: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"email": email, "password": password},
            timeout=20,
        )
        self._raise(response)
        return dict(response.json()["user"])

    def current_user(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/auth/me", timeout=15)
        self._raise(response)
        return dict(response.json()["user"])

    def transcribe(
        self,
        wav_path: Path,
        progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        with wav_path.open("rb") as audio:
            response = self.session.post(
                f"{self.base_url}/transcribe",
                files={"file": (wav_path.name, audio, "audio/wav")},
                timeout=30,
            )
        self._raise(response)
        job_id = str(response.json()["job_id"])
        return self._wait_for_job(job_id, progress)

    def polish(self, text: str, language: str = "auto") -> str:
        response = self.session.post(
            f"{self.base_url}/dictation/polish",
            json={"text": text, "language": language},
            timeout=self.timeout,
        )
        self._raise(response)
        return str(response.json().get("text") or text).strip()

    def _wait_for_job(
        self,
        job_id: str,
        progress: Callable[[str, float], None] | None,
    ) -> dict[str, Any]:
        with self.session.get(
            f"{self.base_url}/jobs/{job_id}/stream",
            stream=True,
            timeout=(15, self.timeout),
            headers={"Accept": "text/event-stream"},
        ) as response:
            self._raise(response)
            for raw_line in response.iter_lines(decode_unicode=True):
                line = str(raw_line or "")
                if not line.startswith("data:"):
                    continue
                event = json.loads(line[5:].strip())
                event_type = str(event.get("type") or "")
                if progress and event_type == "progress":
                    progress(str(event.get("message") or ""), float(event.get("progress") or 0.0))
                if event_type == "done":
                    return dict(event.get("result") or {})
                if event_type == "error":
                    raise RuntimeError(str(event.get("error") or "Transcription failed"))
        raise RuntimeError("The transcription stream ended without a result")

    @staticmethod
    def _raise(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            message = str(response.json().get("detail") or response.reason)
        except Exception:
            message = response.text.strip() or response.reason
        raise RuntimeError(f"NupicAI HTTP {response.status_code}: {message}")

