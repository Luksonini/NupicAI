from __future__ import annotations

import unittest
import json
import os
import sqlite3
import tempfile
from unittest import mock
from pathlib import Path

import numpy as np
import soundfile as sf

import server
from auth_store import AuthStore, QuotaExceeded, User
from translate import parakeet_translation_core as translation_core


class PipelineIntegrityTests(unittest.TestCase):
    def test_auto_translation_prompt_handles_romanized_speech_without_global_language_state(self) -> None:
        response = json.dumps({"segments": [{"id": 1, "translation": "Przetłumaczony tekst."}]})
        with mock.patch.object(translation_core, "_call_chat_json_api", return_value=response) as api_call:
            translated, _ = translation_core.translate_segments_to_pl(
                segments=[{"index": 0, "text": "ye automatically pani ko apne andar samana shuru kar dete"}],
                source_lang="auto",
                target_lang="pl",
                api_key="test-key",
                endpoint="https://example.invalid/v1/chat/completions",
                model="test-model",
                mode="api_json_overlap",
                batch_segments=8,
            )
        self.assertEqual(translated[0]["text"], "Przetłumaczony tekst.")
        messages = api_call.call_args.kwargs["messages"]
        self.assertIn("romanized", messages[0]["content"])
        payload = json.loads(messages[-1]["content"].removeprefix("/no_think\n"))
        self.assertEqual(payload["source_language"], "the source language detected from the transcript")
        self.assertEqual(payload["target_language"], "Polish")

    def test_maskgit_continuity_profile_is_packaged_and_scoped(self) -> None:
        profile, checkpoint = server._resolve_tts_profile("maskgit_continuity")
        self.assertEqual(profile, "maskgit_continuity")
        self.assertTrue(checkpoint.is_file())
        self.assertTrue(server._tts_continuity_enabled(profile))
        self.assertFalse(server._tts_continuity_enabled("mini_dualpath"))
        self.assertFalse(server._tts_continuity_enabled("styleenc128_lstm"))

    def test_configured_admin_account_is_visible_and_has_unlimited_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(
                Path(tmp) / "accounts.sqlite3",
                free_seconds=1,
                admin_emails=("admin@example.com",),
            )
            user = store.register("admin@example.com", "Admin User", "bezpieczne-haslo-123")
            self.assertTrue(user.is_admin)
            self.assertTrue(user.unlimited_usage)
            store.reserve_usage(user.id, "unlimited-job", "tts_text", 86_400)
            usage = store.usage(user.id)
            self.assertTrue(usage["unlimited"])
            self.assertEqual(usage["reserved_seconds"], 0)

    def test_existing_account_is_promoted_during_admin_schema_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.sqlite3"
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """CREATE TABLE users (
                        id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                        display_name TEXT NOT NULL, password_hash TEXT NOT NULL,
                        created_at REAL NOT NULL, credit_seconds INTEGER NOT NULL DEFAULT 0,
                        used_seconds INTEGER NOT NULL DEFAULT 0
                    )"""
                )
                conn.execute(
                    "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("legacy", "admin@example.com", "Legacy Admin", "unused", 1.0, 1, 0),
                )
            store = AuthStore(db_path, admin_emails=("ADMIN@example.com",))
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT is_admin, unlimited_usage FROM users WHERE id = 'legacy'"
                ).fetchone()
            self.assertEqual((row["is_admin"], row["unlimited_usage"]), (1, 1))

    def test_usage_reservations_block_overbooking_and_settle_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "accounts.sqlite3", free_seconds=300)
            user = store.register("quota@example.com", "Quota User", "bezpieczne-haslo-123")
            store.reserve_usage(user.id, "job-a", "dub", 200)
            usage = store.usage(user.id)
            self.assertEqual(usage["available_seconds"], 100)
            self.assertEqual(usage["reserved_seconds"], 200)
            with self.assertRaises(QuotaExceeded):
                store.reserve_usage(user.id, "job-b", "tts_text", 101)

            settled = store.settle_usage("job-a", 125.2)
            self.assertIsNotNone(settled)
            self.assertEqual(settled["used_seconds"], 126)
            self.assertEqual(settled["available_seconds"], 174)
            self.assertEqual(store.settle_usage("job-a", 190)["used_seconds"], 126)

    def test_failed_usage_reservation_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "accounts.sqlite3", free_seconds=60)
            user = store.register("release@example.com", "Release User", "bezpieczne-haslo-123")
            store.reserve_usage(user.id, "job-a", "tts_text", 50)
            released = store.release_usage("job-a")
            self.assertIsNotNone(released)
            self.assertEqual(released["available_seconds"], 60)
            self.assertEqual(released["reserved_seconds"], 0)

    def test_restart_releases_orphaned_usage_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "accounts.sqlite3", free_seconds=60)
            user = store.register("restart@example.com", "Restart User", "bezpieczne-haslo-123")
            store.reserve_usage(user.id, "job-a", "dub", 50)
            self.assertEqual(store.release_orphaned_reservations(), 1)
            self.assertEqual(store.usage(user.id)["available_seconds"], 60)

    def test_settlement_can_exceed_its_estimate_without_spending_other_reservations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "accounts.sqlite3", free_seconds=300)
            user = store.register("concurrent@example.com", "Concurrent User", "bezpieczne-haslo-123")
            store.reserve_usage(user.id, "job-a", "tts_text", 100)
            store.reserve_usage(user.id, "job-b", "dub", 150)
            usage = store.settle_usage("job-a", 130)
            self.assertEqual(usage["used_seconds"], 130)
            self.assertEqual(usage["reserved_seconds"], 150)
            self.assertEqual(usage["available_seconds"], 20)

    def test_account_passwords_and_sessions_are_not_stored_in_plaintext(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.sqlite3"
            store = AuthStore(db_path, session_days=2)
            user = store.register("Test@Example.com", "Test User", "bezpieczne-haslo-123")
            self.assertEqual(user.email, "test@example.com")
            self.assertIsNone(store.authenticate(user.email, "zle-haslo"))
            self.assertEqual(store.authenticate(user.email, "bezpieczne-haslo-123"), user)
            token, expires_at = store.create_session(user.id)
            self.assertGreater(expires_at, user.created_at)
            self.assertEqual(store.user_for_session(token), user)
            raw_db = db_path.read_bytes()
            self.assertNotIn(b"bezpieczne-haslo-123", raw_db)
            self.assertNotIn(token.encode("utf-8"), raw_db)
            store.delete_session(token)
            self.assertIsNone(store.user_for_session(token))

    def test_password_reset_is_one_time_and_revokes_existing_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "accounts.sqlite3"
            store = AuthStore(db_path)
            user = store.register("reset@example.com", "Reset User", "stare-bezpieczne-haslo")
            session, _ = store.create_session(user.id)
            token = store.create_password_reset(user.email, ttl_seconds=600)
            self.assertIsNotNone(token)
            self.assertNotIn(str(token).encode("utf-8"), db_path.read_bytes())
            self.assertTrue(store.reset_password(str(token), "nowe-bezpieczne-haslo"))
            self.assertFalse(store.reset_password(str(token), "kolejne-bezpieczne-haslo"))
            self.assertIsNone(store.user_for_session(session))
            self.assertIsNone(store.authenticate(user.email, "stare-bezpieczne-haslo"))
            self.assertEqual(store.authenticate(user.email, "nowe-bezpieczne-haslo"), user)

    def test_password_reset_does_not_disclose_unknown_accounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "accounts.sqlite3")
            self.assertIsNone(store.create_password_reset("missing@example.com"))

    def test_youtube_errors_are_classified_for_actionable_feedback(self) -> None:
        message, retryable = server._yt_dlp_error_message(
            "HTTP Error 429: Too Many Requests; Sign in to confirm you're not a bot"
        )
        self.assertFalse(retryable)
        self.assertIn("429", message)
        self.assertIn("prześlij plik", message)
        message, retryable = server._yt_dlp_error_message("HTTP Error 403: Forbidden")
        self.assertTrue(retryable)
        self.assertIn("403", message)
        message, retryable = server._yt_dlp_error_message("This is a private video")
        self.assertFalse(retryable)
        self.assertIn("prywatny", message)

    def test_dub_segment_cache_key_tracks_voice_text_and_regeneration_nonce(self) -> None:
        req = server.DubRequest(
            segments=[], speaker_label="voice-a", target_lang="pl", tts_model_profile="model-a"
        )
        segment = {
            "translation": "Przykładowe zdanie.", "speaker_label": "voice-b",
            "seed": 1234, "render_nonce": 0,
        }
        base = server._dub_segment_render_key(segment, req, target_budget=2.0, position=0)
        same = server._dub_segment_render_key(dict(segment), req, target_budget=2.0, position=0)
        changed_nonce = server._dub_segment_render_key(
            {**segment, "render_nonce": 1}, req, target_budget=2.0, position=0
        )
        changed_voice = server._dub_segment_render_key(
            {**segment, "speaker_label": "voice-c"}, req, target_budget=2.0, position=0
        )
        self.assertEqual(base, same)
        self.assertNotEqual(base, changed_nonce)
        self.assertNotEqual(base, changed_voice)

    def test_dub_rerender_reserves_only_changed_segment_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "segment.wav"
            audio_path.write_bytes(b"cached")
            segment = {
                "segment_id": "scene-a", "start": 0.0, "end": 2.0,
                "translation": "Przykładowe zdanie.", "seed": 1234, "render_nonce": 0,
            }
            req = server.DubRequest(
                segments=[segment], speaker_label="voice-a", target_lang="pl",
                tts_model_profile="model-a", reuse_dub_job_id="prior-dub",
            )
            render_key = server._dub_segment_render_key(segment, req, target_budget=2.0, position=0)
            prior = server.Job("prior-dub", "dub", owner_id="owner", status="done")
            prior.result = {"segments": [{
                "segment_id": "scene-a", "target_budget": 2.0,
                "render_key": render_key, "segment_audio_path": str(audio_path),
            }]}
            server._jobs[prior.id] = prior
            try:
                self.assertEqual(server._estimate_dub_generation_seconds(req, 2.0), 1.0)
                req.segments[0]["render_nonce"] = 1
                self.assertEqual(server._estimate_dub_generation_seconds(req, 2.0), 3.0)
            finally:
                server._jobs.pop(prior.id, None)

    def test_registration_records_accepted_legal_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "accounts.sqlite3")
            user = store.register(
                "legal@example.com", "Legal User", "bezpieczne-haslo-123",
                terms_version="2026-08-29", privacy_version="2026-08-29",
            )
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT terms_version, privacy_version, terms_accepted_at FROM users WHERE id = ?",
                    (user.id,),
                ).fetchone()
            self.assertEqual(row["terms_version"], "2026-08-29")
            self.assertEqual(row["privacy_version"], "2026-08-29")
            self.assertGreater(float(row["terms_accepted_at"]), 0.0)

    def test_account_deletion_requires_password_and_cascades_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = AuthStore(Path(tmp) / "accounts.sqlite3")
            user = store.register("delete@example.com", "Delete User", "bezpieczne-haslo-123")
            token, _ = store.create_session(user.id)
            self.assertFalse(store.delete_user(user.id, "zle-haslo"))
            self.assertIsNotNone(store.user_for_session(token))
            self.assertTrue(store.delete_user(user.id, "bezpieczne-haslo-123"))
            self.assertIsNone(store.user_for_session(token))

    def test_job_owner_cannot_read_another_users_job(self) -> None:
        owner = User("a" * 32, "a@example.com", "A", 1.0)
        stranger = User("b" * 32, "b@example.com", "B", 1.0)
        job = server.Job(id="job-owner-test", kind="test", owner_id=owner.id)
        server._jobs[job.id] = job
        try:
            self.assertIs(server._require_job(job.id, owner), job)
            with self.assertRaises(server.HTTPException) as caught:
                server._require_job(job.id, stranger)
            self.assertEqual(caught.exception.status_code, 404)
        finally:
            server._jobs.pop(job.id, None)

    def test_retention_removes_expired_finished_job_but_not_active_job(self) -> None:
        old_work = server._WORK
        old_hours = server.DATA_RETENTION_HOURS
        with tempfile.TemporaryDirectory() as tmp:
            try:
                server._WORK = Path(tmp)
                server.DATA_RETENTION_HOURS = 1
                user = User("c" * 32, "c@example.com", "C", 1.0)
                old_dir = server._user_work_dir(user.id) / "jobs" / "old"
                active_dir = server._user_work_dir(user.id) / "jobs" / "active"
                old_dir.mkdir(parents=True)
                active_dir.mkdir(parents=True)
                old_file = old_dir / "source.wav"
                active_file = active_dir / "source.wav"
                old_file.write_bytes(b"old")
                active_file.write_bytes(b"active")
                old_time = 1_000.0
                os.utime(old_file, (old_time, old_time))
                os.utime(active_file, (old_time, old_time))
                finished = server.Job("old", "test", owner_id=user.id, work_dir=old_dir, status="done", created_at=old_time)
                active = server.Job("active", "test", owner_id=user.id, work_dir=active_dir, status="running", created_at=old_time)
                server._jobs[finished.id] = finished
                server._jobs[active.id] = active
                result = server._cleanup_expired_user_files(now=10_000.0)
                self.assertEqual(result["removed_directories"], 1)
                self.assertFalse(old_dir.exists())
                self.assertTrue(active_dir.exists())
                self.assertNotIn(finished.id, server._jobs)
                self.assertIn(active.id, server._jobs)
            finally:
                server._jobs.pop("old", None)
                server._jobs.pop("active", None)
                server._WORK = old_work
                server.DATA_RETENTION_HOURS = old_hours

    def test_local_env_loads_values_without_overriding_process_env(self) -> None:
        new_key = "WEGORZ_TEST_DOTENV_NEW"
        existing_key = "WEGORZ_TEST_DOTENV_EXISTING"
        os.environ.pop(new_key, None)
        os.environ[existing_key] = "from-process"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                env_file = Path(tmp) / ".env"
                env_file.write_text(
                    f"{new_key}='from-file'\n{existing_key}=must-not-win\n",
                    encoding="utf-8",
                )
                server._load_local_env(env_file)
            self.assertEqual(os.environ[new_key], "from-file")
            self.assertEqual(os.environ[existing_key], "from-process")
        finally:
            os.environ.pop(new_key, None)
            os.environ.pop(existing_key, None)

    def test_asr_slice_plan_covers_boundaries_without_ownership_gaps(self) -> None:
        for duration in (0.5, 179.9, 180.0, 180.05, 359.0, 541.37):
            plans = server._plan_asr_slices(duration, window=180.0, overlap=2.0)
            self.assertTrue(plans)
            self.assertAlmostEqual(float(plans[0]["ownership_start"]), 0.0, places=5)
            self.assertAlmostEqual(float(plans[-1]["ownership_end"]), duration, places=5)
            for left, right in zip(plans, plans[1:]):
                self.assertAlmostEqual(
                    float(left["ownership_end"]),
                    float(right["ownership_start"]),
                    places=5,
                )
                self.assertLess(float(right["start"]), float(left["start"]) + float(left["duration"]))

    def test_overlap_assigns_each_word_once(self) -> None:
        plans = server._plan_asr_slices(181.0, window=180.0, overlap=2.0)
        words = [
            {"word": "przed", "start": 178.7, "end": 179.0},
            {"word": "granica", "start": 179.0, "end": 179.4},
            {"word": "po", "start": 179.5, "end": 179.8},
        ]
        counts = [0] * len(words)
        for plan in plans:
            owned = server._words_owned_by_slice(words, plan)
            for word in owned:
                counts[words.index(word)] += 1
        self.assertEqual(counts, [1, 1, 1])

    def test_segment_builder_preserves_all_words_and_order(self) -> None:
        words = [
            {
                "word": f"slowo{i}{'.' if i % 17 == 16 else ''}",
                "start": i * 0.31,
                "end": i * 0.31 + 0.25,
            }
            for i in range(137)
        ]
        segments = server.build_segments(words)
        rebuilt = [word["word"] for segment in segments for word in segment["words"]]
        self.assertEqual(rebuilt, [word["word"] for word in words])
        server._validate_segment_timeline(segments, stage="test")

    def test_tts_split_preserves_normalized_text(self) -> None:
        text = (
            "To jest pierwsze bardzo dlugie zdanie, ktore sprawdza zachowanie podzialu tekstu oraz jego kolejnosc. "
            "Drugie zdanie zawiera jeszcze kilka slow, aby wymusic utworzenie nastepnego fragmentu bez utraty tresci."
        )
        chunks = server._split_text_for_tts(text, max_chars=70, hard_sentence_chars=90)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(" ".join(chunks).split()), " ".join(text.split()))

    def test_tts_split_isolates_short_opening_utterance(self) -> None:
        text = "Tak. Dlaczego nie chciałeś o tym wcześniej powiedzieć?"
        chunks = server._split_text_for_tts(text)
        self.assertEqual(chunks[0], "Tak.")
        self.assertEqual(" ".join(chunks), text)

        abbreviation = "Dr. Kowalski rozpoczął spotkanie punktualnie."
        self.assertEqual(server._split_text_for_tts(abbreviation), [abbreviation])

    def test_translation_validation_rejects_missing_and_empty_items(self) -> None:
        check = server.core._check_numbered_indices({1: "jeden", 2: ""}, [1, 2, 3])
        self.assertEqual(check["missing"], [3])
        self.assertEqual(check["empty"], [2])

    def test_production_voice_ids_map_to_learned_table(self) -> None:
        speaker_map = json.loads(server.LEARNED_VOICE_SPEAKER_MAP.read_text(encoding="utf-8"))
        for raw_id in server._SPEAKER_ID_BY_LABEL.values():
            self.assertGreaterEqual(raw_id, 0)
            self.assertLess(raw_id, 1535)
        self.assertEqual(int(speaker_map["1079801"]), 1529)

    def test_audio_mix_renders_source_and_dubbing(self) -> None:
        sr = 24000
        t = np.arange(sr, dtype=np.float32) / sr
        background = 0.08 * np.sin(2 * np.pi * 220 * t)
        voice = np.zeros(sr, dtype=np.float32)
        voice[sr // 4:3 * sr // 4] = 0.2 * np.sin(2 * np.pi * 440 * t[:sr // 2])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.wav"
            dubbed = root / "dubbed.wav"
            mixed = root / "mixed.wav"
            sf.write(source, background, sr)
            sf.write(dubbed, voice, sr)
            server._render_audio_mix(
                source, dubbed, mixed,
                original_gain=0.25, dubbing_gain=1.0, ducking_strength=0.65,
            )
            audio, mixed_sr = sf.read(mixed)
            self.assertEqual(mixed_sr, sr)
            self.assertGreater(len(audio), int(0.95 * sr))
            self.assertGreater(float(np.max(np.abs(audio))), 0.05)


if __name__ == "__main__":
    unittest.main()
