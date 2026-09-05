from __future__ import annotations

import json
import os
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw
from pynput import keyboard
import pystray

from .api import NupicFlowApi
from .audio import MicrophoneRecorder
from .hotkey import PushToTalkHotkey


CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "nupicai-flow"
CONFIG_PATH = CONFIG_DIR / "config.json"


class FlowApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("NupicAI Flow")
        self.root.geometry("440x430")
        self.root.minsize(400, 390)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.config = self._load_config()
        self.api = NupicFlowApi(self.config.get("server_url", "http://127.0.0.1:8765"))
        token = str(self.config.get("session_token") or "")
        if token:
            self.api.set_session_token(token)
        self.recorder = MicrophoneRecorder()
        self.busy = False
        self.status_var = tk.StringVar(value="Zaloguj się do NupicAI")
        self.server_var = tk.StringVar(value=self.api.base_url)
        self.email_var = tk.StringVar(value=str(self.config.get("email") or ""))
        self.password_var = tk.StringVar()
        self.polish_var = tk.BooleanVar(value=bool(self.config.get("polish", False)))
        self.auto_paste_var = tk.BooleanVar(value=bool(self.config.get("auto_paste", True)))
        self.polish_enabled = bool(self.polish_var.get())
        self._build_ui()
        self.hotkey = PushToTalkHotkey(self.start_recording, self.stop_recording)
        self.hotkey.start()
        self.tray = pystray.Icon("nupicai-flow", self._tray_image(), "NupicAI Flow", menu=pystray.Menu(
            pystray.MenuItem("Otwórz", lambda: self.root.after(0, self.show), default=True),
            pystray.MenuItem("Zakończ", lambda: self.root.after(0, self.quit)),
        ))
        threading.Thread(target=self.tray.run, daemon=True).start()
        self.root.after(150, self._restore_session)

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=22)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="NupicAI Flow", font=("Sans", 22, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Przytrzymaj Ctrl+Alt+Space, mów i puść, aby wkleić tekst.", wraplength=390).pack(anchor="w", pady=(3, 18))

        form = ttk.Frame(frame)
        form.pack(fill="x")
        self._field(form, "Serwer", self.server_var)
        self._field(form, "E-mail", self.email_var)
        self._field(form, "Hasło", self.password_var, show="•")
        ttk.Button(form, text="Zaloguj", command=self.login).pack(fill="x", pady=(10, 14))

        ttk.Separator(frame).pack(fill="x", pady=4)
        ttk.Checkbutton(frame, text="Poprawiaj tekst przez AI (wolniej)", variable=self.polish_var, command=self._settings_changed).pack(anchor="w", pady=(12, 4))
        ttk.Checkbutton(frame, text="Automatycznie wklejaj do aktywnego okna", variable=self.auto_paste_var, command=self.save_config).pack(anchor="w")
        ttk.Label(frame, textvariable=self.status_var, wraplength=390).pack(anchor="w", pady=(18, 8))
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        ttk.Button(frame, text="Ukryj do zasobnika", command=self.hide).pack(fill="x", pady=(18, 0))

    @staticmethod
    def _field(parent: ttk.Frame, label: str, variable: tk.StringVar, show: str = "") -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(7, 2))
        ttk.Entry(parent, textvariable=variable, show=show).pack(fill="x")

    def login(self) -> None:
        if self.busy:
            return
        self.busy = True
        server_url = self.server_var.get().strip()
        email = self.email_var.get().strip()
        password = self.password_var.get()
        self._set_status("Logowanie…", 10)

        def worker() -> None:
            try:
                self.api = NupicFlowApi(server_url)
                user = self.api.login(email, password)
                self.config.update({
                    "server_url": self.api.base_url,
                    "email": email,
                    "session_token": self.api.session_token(),
                })
                self.root.after(0, lambda: self.password_var.set(""))
                self.root.after(0, self.save_config)
                self.root.after(0, lambda: self._set_status(f"Gotowy: {user.get('display_name') or user.get('email')}", 0))
            except Exception as exc:
                self.root.after(0, lambda message=str(exc): self._show_error(message))
            finally:
                self.busy = False

        threading.Thread(target=worker, daemon=True).start()

    def start_recording(self) -> None:
        if self.busy or self.recorder.recording:
            return
        try:
            self.recorder.start()
            self.root.after(0, lambda: self._set_status("Nagrywam…", 0))
        except Exception as exc:
            self.root.after(0, lambda message=str(exc): self._show_error(message))

    def stop_recording(self) -> None:
        try:
            path = self.recorder.stop()
        except Exception as exc:
            self.root.after(0, lambda message=str(exc): self._show_error(message))
            return
        if path is None or self.busy:
            return
        self.busy = True
        self.root.after(0, lambda: self._set_status("Wysyłam nagranie…", 5))
        threading.Thread(target=self._process_audio, args=(path,), daemon=True).start()

    def _process_audio(self, path: Path) -> None:
        try:
            result = self.api.transcribe(path, self._progress_from_worker)
            text = str(result.get("transcript") or "").strip()
            if not text:
                raise RuntimeError("Nie rozpoznano żadnego tekstu")
            if self.polish_enabled:
                self._progress_from_worker("Poprawiam tekst…", 0.92)
                try:
                    text = self.api.polish(text, str(result.get("detected_language") or "auto"))
                except Exception:
                    # Dictation must still be delivered when the optional language model is unavailable.
                    self._progress_from_worker("Korekta niedostępna, używam transkrypcji…", 0.96)
            self.root.after(0, lambda: self._deliver_text(text))
        except Exception as exc:
            self.root.after(0, lambda message=str(exc): self._show_error(message))
        finally:
            path.unlink(missing_ok=True)
            self.busy = False

    def _deliver_text(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update_idletasks()
        if self.auto_paste_var.get() and os.environ.get("XDG_SESSION_TYPE", "x11").lower() == "x11":
            def paste() -> None:
                time.sleep(0.12)
                controller = keyboard.Controller()
                with controller.pressed(keyboard.Key.ctrl):
                    controller.press("v")
                    controller.release("v")
            threading.Thread(target=paste, daemon=True).start()
            self._set_status("Wklejono transkrypcję", 100)
        else:
            self._set_status("Skopiowano transkrypcję do schowka", 100)
        self.root.after(1800, lambda: self.progress.configure(value=0))

    def _progress_from_worker(self, message: str, progress: float) -> None:
        self.root.after(0, lambda: self._set_status(message or "Transkrybuję…", progress * 100))

    def _restore_session(self) -> None:
        if not self.api.session_token():
            return
        def worker() -> None:
            try:
                user = self.api.current_user()
                self.root.after(0, lambda: self._set_status(f"Gotowy: {user.get('display_name') or user.get('email')}", 0))
            except Exception:
                self.config.pop("session_token", None)
                self.root.after(0, self.save_config)
        threading.Thread(target=worker, daemon=True).start()

    def _set_status(self, message: str, progress: float) -> None:
        self.status_var.set(message)
        self.progress.configure(value=max(0, min(100, progress)))

    def _show_error(self, message: str) -> None:
        self._set_status(f"Błąd: {message}", 0)
        self.show()
        messagebox.showerror("NupicAI Flow", message)

    def save_config(self) -> None:
        self.config.update({
            "server_url": self.server_var.get().strip(),
            "email": self.email_var.get().strip(),
            "polish": bool(self.polish_var.get()),
            "auto_paste": bool(self.auto_paste_var.get()),
        })
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(self.config, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(CONFIG_PATH, 0o600)

    def _settings_changed(self) -> None:
        self.polish_enabled = bool(self.polish_var.get())
        self.save_config()

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()

    def hide(self) -> None:
        self.save_config()
        self.root.withdraw()

    def quit(self) -> None:
        self.hotkey.stop()
        self.tray.stop()
        self.root.destroy()

    @staticmethod
    def _load_config() -> dict[str, object]:
        try:
            return dict(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return {}

    @staticmethod
    def _tray_image() -> Image.Image:
        image = Image.new("RGBA", (64, 64), (17, 13, 9, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((10, 10, 54, 54), fill=(224, 166, 91, 255))
        draw.rectangle((29, 18, 35, 46), fill=(17, 13, 9, 255))
        draw.rectangle((20, 29, 44, 35), fill=(17, 13, 9, 255))
        return image

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    FlowApp().run()


if __name__ == "__main__":
    main()
