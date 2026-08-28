from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PASSWORD_ITERATIONS = 600_000


@dataclass(frozen=True)
class User:
    id: str
    email: str
    display_name: str
    created_at: float


class AuthStore:
    """Small SQLite account store with PBKDF2 passwords and opaque sessions."""

    def __init__(self, path: Path, *, session_days: int = 30) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_seconds = max(1, int(session_days)) * 24 * 3600
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user_id ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS sessions_expires_at ON sessions(expires_at);
                """
            )
        os.chmod(self.path, 0o600)

    @staticmethod
    def normalize_email(email: str) -> str:
        value = str(email or "").strip().lower()
        if len(value) > 254 or not EMAIL_RE.fullmatch(value):
            raise ValueError("Podaj prawidłowy adres e-mail")
        return value

    @staticmethod
    def normalize_name(name: str, email: str) -> str:
        value = " ".join(str(name or "").strip().split())
        if not value:
            value = email.split("@", 1)[0]
        if len(value) < 2 or len(value) > 80:
            raise ValueError("Nazwa użytkownika musi mieć od 2 do 80 znaków")
        return value

    @staticmethod
    def validate_password(password: str) -> str:
        value = str(password or "")
        if len(value) < 10:
            raise ValueError("Hasło musi mieć co najmniej 10 znaków")
        if len(value) > 256:
            raise ValueError("Hasło jest zbyt długie")
        return value

    @staticmethod
    def _hash_password(password: str, salt: bytes | None = None) -> str:
        salt = salt or secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
        )
        return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"

    @staticmethod
    def _verify_password(password: str, encoded: str) -> bool:
        try:
            algorithm, raw_iterations, raw_salt, expected = encoded.split("$", 3)
            if algorithm != "pbkdf2_sha256":
                return False
            digest = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                bytes.fromhex(raw_salt),
                int(raw_iterations),
            )
            return hmac.compare_digest(digest.hex(), expected)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            id=str(row["id"]),
            email=str(row["email"]),
            display_name=str(row["display_name"]),
            created_at=float(row["created_at"]),
        )

    def register(self, email: str, display_name: str, password: str) -> User:
        email = self.normalize_email(email)
        display_name = self.normalize_name(display_name, email)
        password = self.validate_password(password)
        user_id = uuid.uuid4().hex
        now = time.time()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO users(id, email, display_name, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                    (user_id, email, display_name, self._hash_password(password), now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Konto z tym adresem e-mail już istnieje") from exc
        return User(user_id, email, display_name, now)

    def authenticate(self, email: str, password: str) -> User | None:
        try:
            email = self.normalize_email(email)
        except ValueError:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, display_name, password_hash, created_at FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None or not self._verify_password(str(password or ""), str(row["password_hash"])):
            return None
        return self._row_to_user(row)

    def create_session(self, user_id: str) -> tuple[str, float]:
        token = secrets.token_urlsafe(48)
        now = time.time()
        expires_at = now + self.session_seconds
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (self._token_hash(token), user_id, now, expires_at),
            )
        return token, expires_at

    def user_for_session(self, token: str) -> User | None:
        if not token:
            return None
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.email, u.display_name, u.created_at
                FROM sessions s JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ? AND s.expires_at > ?
                """,
                (self._token_hash(token), now),
            ).fetchone()
        return self._row_to_user(row) if row is not None else None

    def delete_session(self, token: str) -> None:
        if not token:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (self._token_hash(token),))

    def cleanup_sessions(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
            return max(0, int(cursor.rowcount))

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            users = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            sessions = int(
                conn.execute("SELECT COUNT(*) FROM sessions WHERE expires_at > ?", (time.time(),)).fetchone()[0]
            )
        return {"users": users, "active_sessions": sessions}
