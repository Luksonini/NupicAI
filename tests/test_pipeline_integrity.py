from __future__ import annotations

import unittest
import json

import server


class PipelineIntegrityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
