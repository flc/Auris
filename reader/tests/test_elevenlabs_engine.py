import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core.elevenlabs_engine import (
    ElevenLabsAPIError,
    ElevenLabsTTSEngine,
    _decode_audio,
    _sends_language_code,
    _strip_inline_tags,
)
from core.tts_router import TTSEngineRouter


def _settings(**overrides):
    base = {
        "elevenlabs_api_key": "sk_test",
        "elevenlabs_voice_id": "voice-1",
        "elevenlabs_model_id": "eleven_multilingual_v2",
        "elevenlabs_output_format": "pcm_24000",
        "elevenlabs_stability": 0.5,
        "elevenlabs_similarity_boost": 0.75,
        "elevenlabs_style": 0.0,
        "elevenlabs_speaker_boost": True,
    }
    base.update(overrides)
    return lambda key, default: base.get(key, default)


class TextPreparationTests(unittest.TestCase):
    def test_omnivoice_nonverbal_tags_are_removed_not_spoken(self):
        self.assertEqual(
            _strip_inline_tags("Wait. [laughter] Really? [question-oh]").split(),
            ["Wait.", "Really?"],
        )

    def test_bracketed_prose_survives(self):
        text = "He read the note [it was smudged] twice."
        self.assertEqual(_strip_inline_tags(text), text)

    def test_language_code_only_for_models_that_accept_it(self):
        self.assertFalse(_sends_language_code("eleven_multilingual_v2"))
        self.assertTrue(_sends_language_code("eleven_turbo_v2_5"))
        self.assertTrue(_sends_language_code("eleven_flash_v2_5"))


class DecodeTests(unittest.TestCase):
    def test_pcm_payload_is_scaled_to_float_at_the_declared_rate(self):
        pcm = np.array([0, 16384, -16384], dtype="<i2")
        audio, rate = _decode_audio(pcm.tobytes(), "pcm_24000")
        self.assertEqual(rate, 24_000)
        np.testing.assert_allclose(audio, [0.0, 0.5, -0.5], atol=1e-6)

    def test_container_payload_is_downmixed_to_mono(self):
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros((100, 2), dtype=np.float32), 44_100, format="WAV")
        audio, rate = _decode_audio(buffer.getvalue(), "mp3_44100_128")
        self.assertEqual(rate, 44_100)
        self.assertEqual(audio.ndim, 1)


class CacheKeyTests(unittest.TestCase):
    def test_voice_and_model_change_the_key(self):
        with patch("core.elevenlabs_engine._setting", side_effect=_settings()):
            first = ElevenLabsTTSEngine.cache_key("Hello", None, None, 1.0)
        with patch(
            "core.elevenlabs_engine._setting",
            side_effect=_settings(elevenlabs_voice_id="voice-2"),
        ):
            second = ElevenLabsTTSEngine.cache_key("Hello", None, None, 1.0)
        self.assertNotEqual(first, second)

    def test_instruct_alone_does_not_split_the_cache(self):
        """One voice reads everything, so per-character keys would just re-bill."""
        with patch("core.elevenlabs_engine._setting", side_effect=_settings()):
            narrator = ElevenLabsTTSEngine.cache_key("Hello", "male, elderly", None, 1.0)
            character = ElevenLabsTTSEngine.cache_key("Hello", "female, child", None, 1.0)
        self.assertEqual(narrator, character)


class RequestPayloadTests(unittest.TestCase):
    def _engine(self):
        engine = ElevenLabsTTSEngine()
        engine._ready = True
        return engine

    def test_speed_is_clamped_into_the_supported_range(self):
        engine = self._engine()
        captured = {}

        def fake_post(voice_id, payload, output_format):
            captured.update(payload=payload, voice_id=voice_id, fmt=output_format)
            return np.zeros(10, dtype="<i2").tobytes()

        with patch("core.elevenlabs_engine._setting", side_effect=_settings()), patch.object(
            engine, "_post_speech", side_effect=fake_post
        ):
            engine._synthesize("Hello", None, None, 3.0, "en", False)

        self.assertEqual(captured["voice_id"], "voice-1")
        self.assertEqual(captured["payload"]["voice_settings"]["speed"], 1.2)
        # Multilingual v2 does not take an explicit language code.
        self.assertNotIn("language_code", captured["payload"])

    def test_default_speed_is_not_sent_at_all(self):
        engine = self._engine()
        captured = {}

        with patch("core.elevenlabs_engine._setting", side_effect=_settings()), patch.object(
            engine,
            "_post_speech",
            side_effect=lambda v, p, f: captured.update(p) or np.zeros(10, dtype="<i2").tobytes(),
        ):
            engine._synthesize("Hello", None, None, 1.0, "en", False)

        self.assertNotIn("speed", captured["voice_settings"])

    def test_turbo_model_receives_the_book_language(self):
        engine = self._engine()
        captured = {}

        with patch(
            "core.elevenlabs_engine._setting",
            side_effect=_settings(elevenlabs_model_id="eleven_turbo_v2_5"),
        ), patch.object(
            engine,
            "_post_speech",
            side_effect=lambda v, p, f: captured.update(p) or np.zeros(10, dtype="<i2").tobytes(),
        ):
            engine._synthesize("Szia", None, None, 1.0, "hu", False)

        self.assertEqual(captured["language_code"], "hu")


class RetryTests(unittest.TestCase):
    def test_rejected_optional_field_is_dropped_and_the_call_retried(self):
        engine = ElevenLabsTTSEngine()
        engine._ready = True
        seen: list[dict] = []

        def fake_request(method, path, *, body=None, accept="application/json"):
            seen.append(json.loads(json.dumps(body)))
            if "speed" in body.get("voice_settings", {}):
                raise ElevenLabsAPIError("HTTP 422: speed not supported", 422)
            return b"ok"

        with patch("core.elevenlabs_engine._setting", side_effect=_settings()), patch.object(
            engine, "_request", side_effect=fake_request
        ):
            result = engine._post_speech(
                "voice-1",
                {"text": "Hi", "voice_settings": {"stability": 0.5, "speed": 1.2}},
                "pcm_24000",
            )

        self.assertEqual(result, b"ok")
        self.assertEqual(len(seen), 2)
        self.assertNotIn("speed", seen[1]["voice_settings"])

    def test_client_errors_that_are_not_optional_fields_propagate(self):
        engine = ElevenLabsTTSEngine()

        def fake_request(method, path, *, body=None, accept="application/json"):
            raise ElevenLabsAPIError("HTTP 401: invalid key", 401)

        with patch.object(engine, "_request", side_effect=fake_request):
            with self.assertRaises(ElevenLabsAPIError) as ctx:
                engine._post_speech("voice-1", {"text": "Hi"}, "pcm_24000")
        self.assertEqual(ctx.exception.status, 401)

    def test_rate_limit_is_retried_then_reported(self):
        engine = ElevenLabsTTSEngine()
        calls = {"n": 0}

        def fake_request(method, path, *, body=None, accept="application/json"):
            calls["n"] += 1
            raise ElevenLabsAPIError("HTTP 429: too many requests", 429)

        with patch.object(engine, "_request", side_effect=fake_request), patch(
            "core.elevenlabs_engine._BACKOFF_BASE_SEC", 0.0
        ):
            with self.assertRaises(ElevenLabsAPIError):
                engine._post_speech("voice-1", {"text": "Hi"}, "pcm_24000")
        self.assertEqual(calls["n"], 4)


class StatusTests(unittest.TestCase):
    def test_missing_api_key_is_an_actionable_error_state(self):
        engine = ElevenLabsTTSEngine()
        with patch("core.elevenlabs_engine._setting", side_effect=_settings(elevenlabs_api_key="")), patch.dict(
            "os.environ", {}, clear=True
        ):
            status = engine.status()
        self.assertEqual(status["state"], "error")
        self.assertIn("API key", status["message"])

    def test_missing_voice_id_is_reported_separately(self):
        engine = ElevenLabsTTSEngine()
        with patch(
            "core.elevenlabs_engine._setting", side_effect=_settings(elevenlabs_voice_id="")
        ):
            status = engine.status()
        self.assertEqual(status["state"], "error")
        self.assertIn("voice", status["message"].lower())

    def test_invalid_key_is_not_re_probed_on_every_status_poll(self):
        engine = ElevenLabsTTSEngine()
        engine._error = "ElevenLabs rejected the API key"
        with patch.object(engine, "_load") as load:
            engine.load_async()
        load.assert_not_called()

    def test_reload_clears_a_sticky_error(self):
        engine = ElevenLabsTTSEngine()
        engine._error = "ElevenLabs rejected the API key"
        with patch.object(engine, "_load") as load:
            engine.reload()
            # reload() starts a background thread; give it a deterministic check
            # by asserting the gate it depends on instead.
            self.assertIsNone(engine._error)
        self.assertFalse(engine._ready)


class RouterTests(unittest.TestCase):
    def test_router_selects_elevenlabs_without_touching_local_engines(self):
        with patch("core.tts_router.selected_engine_name", return_value="elevenlabs"):
            router = TTSEngineRouter()
        self.assertEqual(router.engine_name, "elevenlabs")
        self.assertIsInstance(router._engine, ElevenLabsTTSEngine)


if __name__ == "__main__":
    unittest.main()
