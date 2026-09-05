from __future__ import annotations

import tempfile
import threading
import wave
from pathlib import Path
import shutil
import subprocess


class MicrophoneRecorder:
    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self._process: subprocess.Popen[bytes] | None = None
        self._raw_path: Path | None = None
        self._raw_file: object | None = None
        self._lock = threading.Lock()

    @property
    def recording(self) -> bool:
        return self._process is not None

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                return
            executable = shutil.which("parec")
            if not executable:
                raise RuntimeError("Brak programu parec (pakiet pulseaudio-utils)")
            handle = tempfile.NamedTemporaryFile(prefix="nupic-flow-", suffix=".pcm", delete=False)
            self._raw_path = Path(handle.name)
            self._raw_file = handle
            self._process = subprocess.Popen(
                [
                    executable,
                    "--raw",
                    "--format=s16le",
                    f"--rate={self.sample_rate}",
                    "--channels=1",
                    "--latency-msec=40",
                ],
                stdout=handle,
                stderr=subprocess.PIPE,
            )

    def stop(self) -> Path | None:
        with self._lock:
            process, self._process = self._process, None
            raw_path, self._raw_path = self._raw_path, None
            raw_file, self._raw_file = self._raw_file, None
        if process is None or raw_path is None:
            return None
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
        if raw_file is not None:
            raw_file.close()  # type: ignore[attr-defined]
        byte_count = raw_path.stat().st_size if raw_path.exists() else 0
        if byte_count < int(self.sample_rate * 0.25) * 2:
            error = (process.stderr.read().decode("utf-8", errors="replace").strip() if process.stderr else "")
            raw_path.unlink(missing_ok=True)
            if process.returncode not in {0, -15} and error:
                raise RuntimeError(f"Nie udało się nagrać mikrofonu: {error}")
            return None
        handle = tempfile.NamedTemporaryFile(prefix="nupic-flow-", suffix=".wav", delete=False)
        handle.close()
        path = Path(handle.name)
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(raw_path.read_bytes())
        raw_path.unlink(missing_ok=True)
        return path
