import importlib.util
import json
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from core.higgs_engine import (
    HiggsTTSEngine,
    _language_cleanup,
    _parse_worker_response_line,
    _prepare_reference,
    _translate_inline_tags,
)
from core.higgs_worker import (
    QUANTIZATION_SKIP_MODULES,
    ConcurrencySlots,
    ReferenceCodeCache,
    build_quantization_config,
)
from core.tts_router import TTSEngineRouter


class HiggsPromptTests(unittest.TestCase):
    def test_worker_reply_parser_tolerates_progress_tail_and_prefix(self):
        line = (
            "\rLoading weights 100% "
            'AURIS_HIGGS_JSON:{"ok":true,"event":"ready"}'
            "\rprogress renderer tail"
        )
        self.assertEqual(
            _parse_worker_response_line(line),
            {"ok": True, "event": "ready"},
        )

    def test_existing_omnivoice_nonverbal_tags_are_translated(self):
        text = _translate_inline_tags("Wait. [laughter] Really? [question-oh]")
        self.assertIn("<|sfx:laughter|>Haha", text)
        self.assertIn("<|emotion:surprise|>", text)
        self.assertNotIn("[laughter]", text)

    def test_short_reference_is_mono_and_expanded_to_at_least_four_seconds(self):
        stereo = np.zeros((24_000, 2), dtype=np.float32)
        result = _prepare_reference(stereo, 24_000)
        self.assertEqual(result.ndim, 1)
        self.assertGreaterEqual(len(result), 4 * 24_000)

    def test_prompt_uses_higgs_delivery_controls(self):
        engine = HiggsTTSEngine()

        def setting(key, default):
            return {
                "higgs_prompt_mode": "expressive",
                "higgs_default_emotion": "contentment",
                "higgs_default_style": "none",
                "higgs_default_expressive": "expressive_high",
            }.get(key, default)

        with patch("core.higgs_engine._setting", side_effect=setting):
            prompt = engine._prompt(
                "Hello [sigh]", "female, low pitch, whisper", 1.2, "en", False
            )
        self.assertTrue(prompt.startswith("<|emotion:contentment|>"))
        self.assertIn("<|prosody:expressive_high|>", prompt)
        self.assertIn("<|style:whispering|>", prompt)
        self.assertIn("<|prosody:pitch_low|>", prompt)
        self.assertIn("<|prosody:speed_fast|>", prompt)
        self.assertIn("<|sfx:sigh|>Uh", prompt)

    def test_raw_prompt_matches_friend_app_plain_text_path(self):
        engine = HiggsTTSEngine()
        with patch(
            "core.higgs_engine._setting",
            side_effect=lambda key, default: (
                "raw" if key == "higgs_prompt_mode" else default
            ),
        ):
            prompt = engine._prompt(
                "[surprise-oh] Szia, ez egy rövid magyar teszt.",
                "male, elderly, low pitch, british accent",
                0.85,
                "hu",
                True,
            )
        self.assertEqual(prompt, "Szia, ez egy rövid magyar teszt.")

    def test_hungarian_legacy_pdf_accents_are_repaired(self):
        self.assertEqual(
            _language_cleanup("A bûnözõ õrzi a fõbejáratot.", "hu"),
            "A bűnöző őrzi a főbejáratot.",
        )
        self.assertEqual(_language_cleanup("São João", "pt"), "São João")

    def test_cache_key_changes_with_higgs_sampling(self):
        with patch(
            "core.higgs_engine.HiggsTTSEngine._generation_settings",
            return_value={"temperature": 0.8},
        ):
            first = HiggsTTSEngine.cache_key("Hello", None, None, 1.0)
        with patch(
            "core.higgs_engine.HiggsTTSEngine._generation_settings",
            return_value={"temperature": 1.1},
        ):
            second = HiggsTTSEngine.cache_key("Hello", None, None, 1.0)
        self.assertNotEqual(first, second)


class RouterTests(unittest.TestCase):
    def test_router_selects_higgs_without_importing_it_into_omnivoice_engine(self):
        with patch("core.tts_router.selected_engine_name", return_value="higgs"):
            router = TTSEngineRouter()
        self.assertEqual(router.engine_name, "higgs")
        self.assertIsInstance(router._engine, HiggsTTSEngine)


class HiggsLifecycleTests(unittest.TestCase):
    def test_unload_handles_worker_cleared_during_failed_rpc(self):
        class FakeWorker:
            def __init__(self):
                self.terminated = False

            def poll(self):
                return None if not self.terminated else 1

            def terminate(self):
                self.terminated = True

        engine = HiggsTTSEngine()
        worker = FakeWorker()
        engine._worker = worker

        def failed_rpc(_payload):
            engine._worker = None
            raise RuntimeError("worker exited")

        with patch.object(engine, "_rpc_raw", side_effect=failed_rpc):
            engine.unload()

        self.assertTrue(worker.terminated)
        self.assertIsNone(engine._worker)


class HiggsQuantizationTests(unittest.TestCase):
    def test_quantized_audio_gets_its_own_cache_entries(self):
        with patch(
            "core.higgs_engine._setting",
            side_effect=lambda key, default: default,
        ):
            full = HiggsTTSEngine.cache_key("Hello", None, None, 1.0)
        with patch(
            "core.higgs_engine._setting",
            side_effect=lambda key, default: (
                "4bit" if key == "higgs_quantization" else default
            ),
        ):
            quantized = HiggsTTSEngine.cache_key("Hello", None, None, 1.0)
        self.assertNotEqual(full, quantized)

    def test_full_precision_keys_survive_the_quantization_setting(self):
        """Existing Higgs renders are expensive; 'none' must not re-key them.

        A settings file written before this option existed has no
        higgs_quantization entry at all, and must hash to the same key as an
        explicit 'none'.
        """
        with patch(
            "core.higgs_engine._setting",
            side_effect=lambda key, default: default,
        ):
            absent = HiggsTTSEngine.cache_key("Hello", None, None, 1.0)
        with patch(
            "core.higgs_engine._setting",
            side_effect=lambda key, default: (
                "none" if key == "higgs_quantization" else default
            ),
        ):
            explicit = HiggsTTSEngine.cache_key("Hello", None, None, 1.0)
        self.assertEqual(absent, explicit)

    def test_unknown_mode_falls_back_to_full_precision(self):
        with patch(
            "core.higgs_engine._setting",
            side_effect=lambda key, default: (
                "int3" if key == "higgs_quantization" else default
            ),
        ):
            self.assertEqual(HiggsTTSEngine._quantization(), "none")

    def test_accel_status_reports_what_the_worker_loaded(self):
        engine = HiggsTTSEngine()
        engine._load_metadata = {
            "device": "cuda",
            "dtype": "torch.bfloat16",
            "quantization": "4bit",
            "vram_allocated_gib": 3.6,
            "vram_total_gib": 7.96,
        }
        accel = engine._accel_status()
        self.assertEqual(accel["quantization"], "4bit")
        self.assertIn("4bit", accel["message"])
        self.assertIn("cuda", accel["message"])
        self.assertNotIn("paging", accel["message"])

    def test_accel_status_names_the_vram_overflow(self):
        engine = HiggsTTSEngine()
        engine._load_metadata = {
            "device": "cuda",
            "dtype": "torch.bfloat16",
            "quantization": "none",
            "vram_allocated_gib": 8.67,
            "vram_total_gib": 7.96,
        }
        self.assertIn("paging over PCIe", engine._accel_status()["message"])

    def test_cpu_fallback_is_not_reported_as_cuda(self):
        engine = HiggsTTSEngine()
        engine._load_metadata = {"device": "cpu", "dtype": "torch.float32"}
        message = engine._accel_status()["message"]
        self.assertIn("cpu", message)
        self.assertNotIn("cuda", message)


class HiggsWorkerQuantizationTests(unittest.TestCase):
    def test_full_precision_needs_no_config(self):
        self.assertIsNone(build_quantization_config("none"))
        self.assertIsNone(build_quantization_config(""))

    def test_tied_audio_modules_are_never_quantized(self):
        """audio_head and audio_embedding share storage; quantizing breaks it."""
        self.assertEqual(
            set(QUANTIZATION_SKIP_MODULES), {"audio_head", "audio_embedding"}
        )

    @unittest.skipIf(
        importlib.util.find_spec("bitsandbytes") is None,
        "bitsandbytes is not installed",
    )
    def test_configs_quantize_the_body_only(self):
        eight = build_quantization_config("8bit")
        four = build_quantization_config("4bit")
        self.assertTrue(eight.load_in_8bit)
        self.assertTrue(four.load_in_4bit)
        self.assertEqual(four.bnb_4bit_quant_type, "nf4")
        for config in (eight, four):
            self.assertEqual(
                set(config.llm_int8_skip_modules), set(QUANTIZATION_SKIP_MODULES)
            )

    @unittest.skipUnless(
        importlib.util.find_spec("bitsandbytes") is None,
        "bitsandbytes is installed",
    )
    def test_missing_bitsandbytes_says_how_to_install_it(self):
        with self.assertRaises(RuntimeError) as ctx:
            build_quantization_config("4bit")
        self.assertIn("bitsandbytes", str(ctx.exception))
        self.assertIn("pip install", str(ctx.exception))


class ReferenceCodeCacheTests(unittest.TestCase):
    def setUp(self):
        self.encoded: list[str] = []
        self.cache = ReferenceCodeCache(
            lambda path: self.encoded.append(path) or f"codes:{path}", capacity=2
        )

    def test_a_clip_is_encoded_once_and_reused(self):
        first, cached_first = self.cache.lookup("k1", "/narrator.wav")
        second, cached_second = self.cache.lookup("k1", None)

        self.assertEqual(self.encoded, ["/narrator.wav"])
        self.assertEqual(first, second)
        self.assertFalse(cached_first)
        self.assertTrue(cached_second)

    def test_evicted_key_without_audio_reports_a_miss(self):
        self.cache.lookup("k1", "/a.wav")
        self.cache.lookup("k2", "/b.wav")
        self.cache.lookup("k3", "/c.wav")  # capacity 2 evicts k1

        codes, cached = self.cache.lookup("k1", None)
        self.assertIsNone(codes)
        self.assertFalse(cached)

    def test_use_refreshes_recency(self):
        self.cache.lookup("k1", "/a.wav")
        self.cache.lookup("k2", "/b.wav")
        self.cache.lookup("k1", None)       # k1 becomes the newest
        self.cache.lookup("k3", "/c.wav")   # so k2 is evicted, not k1

        self.assertIn("k1", self.cache)
        self.assertNotIn("k2", self.cache)

    def test_unkeyed_clips_are_encoded_every_time(self):
        self.cache.lookup(None, "/a.wav")
        self.cache.lookup(None, "/a.wav")
        self.assertEqual(self.encoded, ["/a.wav", "/a.wav"])


class ConcurrencySlotsTests(unittest.TestCase):
    def test_normal_requests_run_together(self):
        slots = ConcurrencySlots(2)
        both_inside = threading.Barrier(2, timeout=5)

        def run():
            with slots.acquire():
                both_inside.wait()

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertFalse(any(t.is_alive() for t in threads))

    def test_a_seeded_request_runs_alone(self):
        """A pinned seed must be reproducible, and torch's RNG is global."""
        slots = ConcurrencySlots(2)
        overlapped = []
        active = []
        active_lock = threading.Lock()
        release_exclusive = threading.Event()

        def exclusive():
            with slots.acquire(exclusive=True):
                with active_lock:
                    active.append("seeded")
                release_exclusive.wait(timeout=5)
                with active_lock:
                    active.remove("seeded")

        def normal():
            with slots.acquire():
                with active_lock:
                    overlapped.append(list(active))

        holder = threading.Thread(target=exclusive)
        holder.start()
        while not active:
            time.sleep(0.005)
        waiter = threading.Thread(target=normal)
        waiter.start()
        waiter.join(timeout=0.3)
        self.assertTrue(waiter.is_alive(), "a normal request entered during a seeded one")

        release_exclusive.set()
        holder.join(timeout=5)
        waiter.join(timeout=5)
        self.assertEqual(overlapped, [[]])

    def test_two_seeded_requests_do_not_deadlock(self):
        slots = ConcurrencySlots(3)
        done = []

        def run(tag):
            with slots.acquire(exclusive=True):
                done.append(tag)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(sorted(done), [0, 1])


class HiggsConcurrencyTests(unittest.TestCase):
    @staticmethod
    def _settings(**overrides):
        def setting(key, default):
            return overrides.get(key, default)

        return patch("core.higgs_engine._setting", side_effect=setting)

    def test_auto_stays_serial(self):
        """Overlapping decode loops are not a reliable win; opting in is manual."""
        for quantization in ("none", "4bit", "8bit"):
            with self._settings(higgs_quantization=quantization):
                self.assertEqual(HiggsTTSEngine._concurrency(), 1)

    def test_explicit_setting_wins_and_is_capped(self):
        with self._settings(higgs_concurrency=3, higgs_quantization="none"):
            self.assertEqual(HiggsTTSEngine._concurrency(), 3)
        with self._settings(higgs_concurrency=99):
            self.assertEqual(HiggsTTSEngine._concurrency(), 4)

    def test_generate_many_keeps_input_order_and_indices(self):
        engine = HiggsTTSEngine()
        items = [{"text": f"segment {i}"} for i in range(6)]

        def generate_item(item):
            # Finish in reverse order so ordering cannot come from timing.
            time.sleep(0.02 * (len(items) - int(item["text"].split()[1])))
            return {"cache_key": item["text"]}

        seen: list[tuple[int, str]] = []
        seen_lock = threading.Lock()

        def on_item(index, result):
            with seen_lock:
                seen.append((index, result["cache_key"]))

        with patch.object(engine, "_concurrency", return_value=3), patch.object(
            engine, "_generate_item", side_effect=generate_item
        ):
            results = engine.generate_many(items, on_item=on_item)

        self.assertEqual([r["cache_key"] for r in results], [i["text"] for i in items])
        self.assertEqual(sorted(seen), [(i, f"segment {i}") for i in range(6)])

    def test_generate_many_actually_overlaps(self):
        engine = HiggsTTSEngine()
        items = [{"text": str(i)} for i in range(4)]
        peak = {"now": 0, "max": 0}
        lock = threading.Lock()

        def generate_item(item):
            with lock:
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
            time.sleep(0.05)
            with lock:
                peak["now"] -= 1
            return {"cache_key": item["text"]}

        with patch.object(engine, "_concurrency", return_value=2), patch.object(
            engine, "_generate_item", side_effect=generate_item
        ):
            engine.generate_many(items)
        self.assertEqual(peak["max"], 2)

    def test_a_failing_segment_propagates(self):
        engine = HiggsTTSEngine()
        items = [{"text": str(i)} for i in range(4)]

        def generate_item(item):
            if item["text"] == "2":
                raise RuntimeError("worker died")
            return {"cache_key": item["text"]}

        with patch.object(engine, "_concurrency", return_value=2), patch.object(
            engine, "_generate_item", side_effect=generate_item
        ):
            with self.assertRaises(RuntimeError):
                engine.generate_many(items)


class HiggsRpcMultiplexingTests(unittest.TestCase):
    def setUp(self):
        self.engine = HiggsTTSEngine()
        self.engine._worker = _PipeWorker()

    def test_replies_reach_the_matching_caller(self):
        engine, worker = self.engine, self.engine._worker
        answers: dict[int, dict] = {}

        def call(tag):
            answers[tag] = engine._rpc_raw({"command": "generate", "tag": tag})

        threads = [threading.Thread(target=call, args=(t,)) for t in range(3)]
        for thread in threads:
            thread.start()
        worker.wait_for_requests(3)
        # Answer out of order: the ids, not the arrival order, decide.
        for request in reversed(worker.requests):
            engine._deliver({"ok": True, "id": request["id"], "tag": request["tag"]})
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual({t: a["tag"] for t, a in answers.items()}, {0: 0, 1: 1, 2: 2})

    def test_a_dead_worker_wakes_every_caller(self):
        engine, worker = self.engine, self.engine._worker
        errors: list[Exception] = []

        def call():
            try:
                engine._rpc_raw({"command": "generate"})
            except Exception as exc:  # noqa: BLE001 - the point of the test
                errors.append(exc)

        threads = [threading.Thread(target=call) for _ in range(3)]
        for thread in threads:
            thread.start()
        worker.wait_for_requests(3)
        engine._fail_pending(RuntimeError("Higgs worker exited unexpectedly"))
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(len(errors), 3)
        self.assertTrue(all("exited" in str(e) for e in errors))

    def test_startup_failure_without_an_id_reaches_the_waiting_loader(self):
        engine, worker = self.engine, self.engine._worker
        result = {}

        thread = threading.Thread(
            target=lambda: result.update(engine._rpc_raw({"source": "x"}))
        )
        thread.start()
        worker.wait_for_requests(1)
        engine._deliver({"ok": False, "event": "startup_error", "error": "boom"})
        thread.join(timeout=5)

        self.assertEqual(result.get("error"), "boom")


class _PipeWorker:
    """Captures what the engine writes without running a real process."""

    def __init__(self):
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        self.stdin = self
        self.stdout = self

    def write(self, line):
        with self._lock:
            self.requests.append(json.loads(line))

    def flush(self):
        return None

    def poll(self):
        return None

    def wait_for_requests(self, count, timeout=5):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if len(self.requests) >= count:
                    return
            time.sleep(0.005)
        raise AssertionError(f"only {len(self.requests)}/{count} requests were sent")


class _ExitedWorker:
    """Stand-in for the worker process; already exited, so unload() is a no-op."""

    def poll(self):
        return 0


class HiggsReferenceCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref = os.path.join(self.tmp.name, "narrator.wav")
        # Long enough that _prepare_reference() sends the file as-is; a short
        # clip is expanded into a temp copy first.
        sf.write(self.ref, np.zeros(72_000, dtype=np.float32), 24_000)
        self.engine = HiggsTTSEngine()
        self.engine._ready = True
        self.engine._worker = _ExitedWorker()

    def _synthesize(self, calls, responses):
        real_read = sf.read

        def read(path, *args, **kwargs):
            # The reference clip is read for real; the generated output is not
            # written by a mocked worker, so it is faked.
            if path == self.ref:
                return real_read(path, *args, **kwargs)
            return np.zeros(10, dtype=np.float32), 24_000

        def rpc(payload):
            calls.append(payload)
            return responses.pop(0)

        with patch.object(self.engine, "_rpc_raw", side_effect=rpc), patch(
            "core.higgs_engine.sf.read", side_effect=read
        ):
            return self.engine._synthesize(
                "Hello", None, self.ref, None, 1.0, "en", False
            )

    def test_reference_key_is_stable_for_the_same_clip(self):
        self.assertEqual(
            HiggsTTSEngine._reference_key(self.ref),
            HiggsTTSEngine._reference_key(self.ref),
        )

    def test_reference_key_changes_when_the_clip_changes(self):
        first = HiggsTTSEngine._reference_key(self.ref)
        sf.write(self.ref, np.zeros(48_000, dtype=np.float32), 24_000)
        self.assertNotEqual(first, HiggsTTSEngine._reference_key(self.ref))

    def test_clip_is_sent_once_then_referenced_by_key(self):
        calls: list[dict] = []
        ok = {"ok": True, "reference_cached": True}
        self._synthesize(calls, [dict(ok)])
        self._synthesize(calls, [dict(ok)])

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["reference_audio"], self.ref)
        # Second call carries only the key: the worker already holds the codes.
        self.assertIsNone(calls[1]["reference_audio"])
        self.assertEqual(calls[0]["reference_key"], calls[1]["reference_key"])

    def test_evicted_reference_is_resent_instead_of_failing(self):
        calls: list[dict] = []
        self._synthesize(calls, [{"ok": True, "reference_cached": True}])
        self._synthesize(
            calls,
            [
                {"ok": False, "error_code": "reference_cache_miss", "error": "gone"},
                {"ok": True, "reference_cached": True},
            ],
        )

        self.assertEqual(len(calls), 3)
        self.assertIsNone(calls[1]["reference_audio"])
        self.assertEqual(calls[2]["reference_audio"], self.ref)

    def test_worker_restart_forgets_the_cached_clips(self):
        calls: list[dict] = []
        self._synthesize(calls, [{"ok": True, "reference_cached": True}])
        self.engine.unload()
        self.engine._ready = True
        self.engine._worker = _ExitedWorker()
        self._synthesize(calls, [{"ok": True, "reference_cached": True}])

        self.assertEqual(calls[1]["reference_audio"], self.ref)


if __name__ == "__main__":
    unittest.main()
