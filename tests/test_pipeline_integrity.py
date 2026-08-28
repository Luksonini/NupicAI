from __future__ import annotations

import unittest
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

import server
from auth_store import AuthStore, User


class PipelineIntegrityTests(unittest.TestCase):
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
