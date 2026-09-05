from __future__ import annotations

import threading
from collections.abc import Callable

from pynput import keyboard


def _canonical_key(key: keyboard.Key | keyboard.KeyCode) -> keyboard.Key | keyboard.KeyCode:
    if key in {keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}:
        return keyboard.Key.ctrl
    if key in {keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr}:
        return keyboard.Key.alt
    if key in {keyboard.Key.shift_l, keyboard.Key.shift_r}:
        return keyboard.Key.shift
    return key


class PushToTalkHotkey:
    """Hold Ctrl+Alt+Space to record, release any combo key to stop."""

    def __init__(self, on_start: Callable[[], None], on_stop: Callable[[], None]) -> None:
        self.on_start = on_start
        self.on_stop = on_stop
        self.required = {keyboard.Key.ctrl, keyboard.Key.alt, keyboard.Key.space}
        self.pressed: set[keyboard.Key | keyboard.KeyCode] = set()
        self.active = False
        self.enabled = True
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._press, on_release=self._release)

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    def _press(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        start = False
        with self._lock:
            self.pressed.add(_canonical_key(key))
            if self.enabled and not self.active and self.required.issubset(self.pressed):
                self.active = True
                start = True
        if start:
            self.on_start()

    def _release(self, key: keyboard.Key | keyboard.KeyCode) -> None:
        stop = False
        with self._lock:
            self.pressed.discard(_canonical_key(key))
            if self.active and not self.required.issubset(self.pressed):
                self.active = False
                stop = True
        if stop:
            self.on_stop()

