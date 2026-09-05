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
    is_admin: bool = False
    unlimited_usage: bool = False


class QuotaExceeded(ValueError):
    def __init__(self, requested_seconds: int, available_seconds: int) -> None:
        self.requested_seconds = requested_seconds
        self.available_seconds = available_seconds
        super().__init__(
            f"Brak wystarczającego limitu: potrzeba około {requested_seconds} s, "
            f"pozostało {available_seconds} s"
        )


class AuthStore:
    """Small SQLite account store with PBKDF2 passwords and opaque sessions."""

    def __init__(
        self,
        path: Path,
        *,
        session_days: int = 30,
        free_seconds: int = 300,
        admin_emails: tuple[str, ...] = (),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_seconds = max(1, int(session_days)) * 24 * 3600
        self.free_seconds = max(0, int(free_seconds))
        self.admin_emails = {
            str(email).strip().lower() for email in admin_emails if str(email).strip()
        }
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
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS password_reset_tokens_user_id
                    ON password_reset_tokens(user_id);
                CREATE INDEX IF NOT EXISTS password_reset_tokens_expires_at
                    ON password_reset_tokens(expires_at);
                CREATE TABLE IF NOT EXISTS usage_ledger (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    job_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    reserved_seconds INTEGER NOT NULL,
                    charged_seconds INTEGER,
                    status TEXT NOT NULL CHECK(status IN ('reserved', 'settled', 'released')),
                    created_at REAL NOT NULL,
                    settled_at REAL
                );
                CREATE INDEX IF NOT EXISTS usage_ledger_user_status
                    ON usage_ledger(user_id, status);
                """
            )
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(users)")}
            if "credit_seconds" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN credit_seconds INTEGER NOT NULL DEFAULT 0")
                conn.execute("UPDATE users SET credit_seconds = ?", (self.free_seconds,))
            if "used_seconds" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN used_seconds INTEGER NOT NULL DEFAULT 0")
            if "is_admin" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0")
            if "unlimited_usage" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN unlimited_usage INTEGER NOT NULL DEFAULT 0")
            if "terms_version" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN terms_version TEXT NOT NULL DEFAULT ''")
            if "privacy_version" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN privacy_version TEXT NOT NULL DEFAULT ''")
            if "terms_accepted_at" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN terms_accepted_at REAL")
            for email in self.admin_emails:
                conn.execute(
                    "UPDATE users SET is_admin = 1, unlimited_usage = 1 WHERE email = ? COLLATE NOCASE",
                    (email,),
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
            is_admin=bool(row["is_admin"]),
            unlimited_usage=bool(row["unlimited_usage"]),
        )

    def register(
        self,
        email: str,
        display_name: str,
        password: str,
        *,
        terms_version: str = "",
        privacy_version: str = "",
    ) -> User:
        email = self.normalize_email(email)
        display_name = self.normalize_name(display_name, email)
        password = self.validate_password(password)
        user_id = uuid.uuid4().hex
        now = time.time()
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO users(
                        id, email, display_name, password_hash, created_at, credit_seconds,
                        used_seconds, is_admin, unlimited_usage, terms_version,
                        privacy_version, terms_accepted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
                    (
                        user_id, email, display_name, self._hash_password(password), now,
                        self.free_seconds, int(email in self.admin_emails), int(email in self.admin_emails),
                        str(terms_version), str(privacy_version), now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("Konto z tym adresem e-mail już istnieje") from exc
        is_admin = email in self.admin_emails
        return User(user_id, email, display_name, now, is_admin, is_admin)

    def authenticate(self, email: str, password: str) -> User | None:
        try:
            email = self.normalize_email(email)
        except ValueError:
            return None
        with self._connect() as conn:
            row = conn.execute(
                """SELECT id, email, display_name, password_hash, created_at,
                          is_admin, unlimited_usage FROM users WHERE email = ?""",
                (email,),
            ).fetchone()
        if row is None or not self._verify_password(str(password or ""), str(row["password_hash"])):
            return None
        return self._row_to_user(row)

    def create_password_reset(self, email: str, *, ttl_seconds: int = 3600) -> str | None:
        """Return a one-time plaintext token only when the account exists."""
        try:
            email = self.normalize_email(email)
        except ValueError:
            return None
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM password_reset_tokens WHERE expires_at <= ?", (now,))
            row = conn.execute(
                "SELECT id FROM users WHERE email = ? COLLATE NOCASE", (email,)
            ).fetchone()
            if row is None:
                return None
            user_id = str(row["id"])
            token = secrets.token_urlsafe(48)
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute(
                """INSERT INTO password_reset_tokens(token_hash, user_id, created_at, expires_at)
                   VALUES (?, ?, ?, ?)""",
                (self._token_hash(token), user_id, now, now + max(300, int(ttl_seconds))),
            )
        return token

    def reset_password(self, token: str, new_password: str) -> bool:
        """Consume a reset token atomically and revoke every existing session."""
        if not token:
            return False
        password = self.validate_password(new_password)
        now = time.time()
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT user_id FROM password_reset_tokens
                   WHERE token_hash = ? AND expires_at > ?""",
                (self._token_hash(token), now),
            ).fetchone()
            if row is None:
                return False
            user_id = str(row["user_id"])
            cursor = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (self._hash_password(password), user_id),
            )
            if cursor.rowcount != 1:
                return False
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
        return True

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
                SELECT u.id, u.email, u.display_name, u.created_at,
                       u.is_admin, u.unlimited_usage
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

    def delete_user(self, user_id: str, password: str) -> bool:
        """Delete an account after password confirmation; cascades sessions and ledger."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE id = ?", (str(user_id),)
            ).fetchone()
            if row is None or not self._verify_password(
                str(password or ""), str(row["password_hash"])
            ):
                return False
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (str(user_id),))
        return cursor.rowcount == 1

    def cleanup_sessions(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (time.time(),))
            return max(0, int(cursor.rowcount))

    def usage(self, user_id: str) -> dict[str, int | str | bool]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT credit_seconds, used_seconds, unlimited_usage FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if row is None:
                raise ValueError("User not found")
            reserved = int(conn.execute(
                """SELECT COALESCE(SUM(reserved_seconds), 0)
                   FROM usage_ledger WHERE user_id = ? AND status = 'reserved'""",
                (user_id,),
            ).fetchone()[0])
        balance = max(0, int(row["credit_seconds"]))
        used = max(0, int(row["used_seconds"]))
        unlimited = bool(row["unlimited_usage"])
        return {
            "plan": "admin" if unlimited else "pilot",
            "total_seconds": balance + used,
            "used_seconds": used,
            "reserved_seconds": reserved,
            "available_seconds": max(0, balance - reserved),
            "unlimited": unlimited,
        }

    def reserve_usage(self, user_id: str, job_id: str, kind: str, seconds: float) -> dict[str, int | str | bool]:
        requested = max(1, int(round(float(seconds))))
        already_exists = False
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT user_id FROM usage_ledger WHERE job_id = ?", (job_id,)
            ).fetchone()
            if existing is not None:
                if str(existing["user_id"]) != user_id:
                    raise ValueError("Usage reservation belongs to another user")
                already_exists = True
            if not already_exists:
                row = conn.execute(
                    "SELECT credit_seconds, unlimited_usage FROM users WHERE id = ?", (user_id,)
                ).fetchone()
                if row is None:
                    raise ValueError("User not found")
                if bool(row["unlimited_usage"]):
                    return self.usage(user_id)
                reserved = int(conn.execute(
                    """SELECT COALESCE(SUM(reserved_seconds), 0)
                       FROM usage_ledger WHERE user_id = ? AND status = 'reserved'""",
                    (user_id,),
                ).fetchone()[0])
                available = max(0, int(row["credit_seconds"]) - reserved)
                if requested > available:
                    raise QuotaExceeded(requested, available)
                conn.execute(
                    """INSERT INTO usage_ledger(
                        user_id, job_id, kind, reserved_seconds, status, created_at
                    ) VALUES (?, ?, ?, ?, 'reserved', ?)""",
                    (user_id, job_id, str(kind), requested, time.time()),
                )
        return self.usage(user_id)

    def settle_usage(self, job_id: str, actual_seconds: float) -> dict[str, int | str | bool] | None:
        charged = max(0, int(float(actual_seconds) + 0.999999))
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT user_id, status, reserved_seconds FROM usage_ledger WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            user_id = str(row["user_id"])
            if str(row["status"]) == "reserved":
                balance_row = conn.execute(
                    "SELECT credit_seconds FROM users WHERE id = ?", (user_id,)
                ).fetchone()
                if balance_row is None:
                    return None
                other_reserved = int(conn.execute(
                    """SELECT COALESCE(SUM(reserved_seconds), 0) FROM usage_ledger
                       WHERE user_id = ? AND status = 'reserved' AND job_id != ?""",
                    (user_id, job_id),
                ).fetchone()[0])
                debit = min(
                    charged,
                    max(0, int(balance_row["credit_seconds"]) - other_reserved),
                )
                conn.execute(
                    """UPDATE users SET credit_seconds = credit_seconds - ?,
                       used_seconds = used_seconds + ? WHERE id = ?""",
                    (debit, debit, user_id),
                )
                conn.execute(
                    """UPDATE usage_ledger SET charged_seconds = ?, status = 'settled', settled_at = ?
                       WHERE job_id = ?""",
                    (debit, time.time(), job_id),
                )
        return self.usage(user_id)

    def release_usage(self, job_id: str) -> dict[str, int | str | bool] | None:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT user_id, status FROM usage_ledger WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                return None
            user_id = str(row["user_id"])
            if str(row["status"]) == "reserved":
                conn.execute(
                    """UPDATE usage_ledger SET status = 'released', settled_at = ?
                       WHERE job_id = ?""",
                    (time.time(), job_id),
                )
        return self.usage(user_id)

    def release_orphaned_reservations(self) -> int:
        """Release reservations whose in-memory jobs cannot survive a server restart."""
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE usage_ledger SET status = 'released', settled_at = ?
                   WHERE status = 'reserved'""",
                (time.time(),),
            )
        return max(0, int(cursor.rowcount))

    def add_credits(self, user_id: str, seconds: int) -> dict[str, int | str | bool]:
        amount = int(seconds)
        if amount <= 0:
            raise ValueError("Credit amount must be positive")
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET credit_seconds = credit_seconds + ? WHERE id = ?",
                (amount, user_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("User not found")
        return self.usage(user_id)

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            users = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
            sessions = int(
                conn.execute("SELECT COUNT(*) FROM sessions WHERE expires_at > ?", (time.time(),)).fetchone()[0]
            )
        return {"users": users, "active_sessions": sessions}
