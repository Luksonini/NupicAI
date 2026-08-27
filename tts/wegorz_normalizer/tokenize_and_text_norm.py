"""Compatibility bridge to the packaged Węgorz normalizer implementation."""

from .wegorz_normalizer import tokenize_and_text_norm as _impl

for _name in dir(_impl):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl, _name)

__all__ = [name for name in globals() if not name.startswith("__")]
