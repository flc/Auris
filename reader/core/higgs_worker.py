"""Persistent Higgs inference worker.

This process intentionally runs with a private Transformers package path.
OmniVoice is pinned to Transformers 5.3, while the Higgs community adapter
requires 5.5 or newer; keeping them in separate processes prevents module and
model-class conflicts when users switch engines.
"""

from __future__ import annotations

import collections
import contextlib
import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor


PREFIX = "AURIS_HIGGS_JSON:"
_REPLY_LOCK = threading.Lock()
_PROTOCOL_STDOUT = sys.stdout

# The fused audio embedding and head share storage (see audio_head_shared
# below). Quantizing either one breaks that tying, so both stay in BF16 —
# together they are only ~1.9 GiB, while the transformer body is ~6.8 GiB.
QUANTIZATION_SKIP_MODULES = ["audio_head", "audio_embedding"]

# Encoded reference clips, keyed by the parent's content hash. A locked
# narrator voice is one entry reused by every segment of a book; the spare
# room covers per-character reference clips.
REFERENCE_CACHE_SIZE = 16


def reply(payload: dict) -> None:
    with _REPLY_LOCK:
        # ASCII-only JSON keeps the protocol safe even if a redirected Windows
        # stream is recreated with a legacy locale encoding.
        print(
            PREFIX + json.dumps(payload, ensure_ascii=True),
            file=_PROTOCOL_STDOUT,
            flush=True,
        )


def audio_array(output):
    import numpy as np

    if hasattr(output, "detach"):
        output = output.detach().float().cpu().numpy()
    audio = np.asarray(output, dtype=np.float32).squeeze()
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    if audio.ndim != 1:
        raise RuntimeError(f"Unexpected Higgs output shape: {audio.shape}")
    return audio


def reference_array(output):
    import numpy as np

    audio = np.asarray(output, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1:
        raise RuntimeError(f"Unexpected Higgs reference shape: {audio.shape}")
    return audio


def build_quantization_config(mode: str):
    """Return a BitsAndBytesConfig for ``mode``, or None for full precision.

    A 4.65B BF16 checkpoint needs ~8.7 GiB of weights. On a card that cannot
    hold that, the driver silently pages the overflow through host memory and
    every decoded audio token pays for it over PCIe. Quantizing the body is
    what makes the model resident.
    """
    if mode not in ("8bit", "4bit"):
        return None
    import torch
    from transformers import BitsAndBytesConfig

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"Higgs {mode} quantization needs bitsandbytes. Install it with "
            "reader/.venv/bin/python -m pip install bitsandbytes"
        ) from exc

    if mode == "8bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=QUANTIZATION_SKIP_MODULES,
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=QUANTIZATION_SKIP_MODULES,
    )


class ConcurrencySlots:
    """Bounds concurrent generations and lets a seeded one run alone.

    Requests that pin ``seed`` must be reproducible, and torch's RNG is
    process-global, so those take every slot. Only one thread may collect the
    full set at a time, otherwise two exclusive requests could each hold part
    of it and wait for the other forever.
    """

    def __init__(self, capacity: int):
        self._capacity = max(1, int(capacity))
        self._slots = threading.Semaphore(self._capacity)
        self._exclusive_gate = threading.Lock()

    @contextlib.contextmanager
    def acquire(self, exclusive: bool = False):
        count = self._capacity if exclusive else 1
        gate = self._exclusive_gate if exclusive else contextlib.nullcontext()
        with gate:
            for _ in range(count):
                self._slots.acquire()
            try:
                yield
            finally:
                for _ in range(count):
                    self._slots.release()


class ReferenceCodeCache:
    """Encoded reference clips, keyed by the parent's content hash.

    Without it the adapter re-runs the codec encoder over the same locked
    narrator clip for every sentence of a book.
    """

    def __init__(self, encode, capacity: int = REFERENCE_CACHE_SIZE):
        self._encode = encode
        self._capacity = capacity
        self._entries: collections.OrderedDict = collections.OrderedDict()
        self._lock = threading.Lock()

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._entries

    def lookup(self, key, path):
        """Return ``(codes, was_cached)``; codes is None for an evicted key."""
        with self._lock:
            if key and key in self._entries:
                self._entries.move_to_end(key)
                return self._entries[key], True
            if not path:
                return None, False
        # Encoding runs outside the lock so one slow clip cannot stall the
        # other in-flight segments; a duplicate encode is harmless.
        codes = self._encode(path)
        if key:
            with self._lock:
                self._entries[key] = codes
                self._entries.move_to_end(key)
                while len(self._entries) > self._capacity:
                    self._entries.popitem(last=False)
        return codes, False


def vram_report():
    """Measured allocation after load — the answer to 'does the model fit?'."""
    import torch

    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "vram_allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "vram_total_gib": round(total_bytes / 1024**3, 2),
        "vram_free_gib": round(free_bytes / 1024**3, 2),
    }


def main() -> None:
    # Parent Popen writes UTF-8. Windows redirected stdio otherwise inherits a
    # locale encoding (often cp1250), corrupting Hungarian prompts before they
    # reach the tokenizer.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Reserve the original stdout pipe for framed RPC replies. Libraries such
    # as Transformers/tqdm may use carriage-return progress rendering, which
    # can otherwise become interleaved with a JSON response on Windows.
    sys.stdout = sys.stderr
    import soundfile as sf
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 5):
        raise RuntimeError(
            f"Higgs needs Transformers >=5.5, worker loaded {transformers.__version__}"
        )

    init = json.loads(sys.stdin.readline())
    source = init["source"]
    local_only = bool(init.get("local_only"))
    model_seed = int(init.get("model_seed", 123))
    quantization = str(init.get("quantization") or "none").lower()
    max_concurrency = max(1, int(init.get("max_concurrency", 1)))
    common = {"trust_remote_code": True, "local_files_only": local_only}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if quantization != "none" and device != "cuda":
        raise RuntimeError("Higgs quantization requires CUDA; no GPU is visible.")
    quantization_config = build_quantization_config(quantization)
    # The adapter reports audio_head.weight as missing and initializes it while
    # from_pretrained() constructs the model. Set a stable initialization seed
    # before loading so repeated worker starts remain reproducible.
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    tokenizer = AutoTokenizer.from_pretrained(source, **common)
    if quantization_config is not None:
        # bitsandbytes places the weights itself; a later .to("cuda") on a
        # quantized model raises. device_map pins everything to one GPU so the
        # adapter never has to split across devices.
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=dtype,
            quantization_config=quantization_config,
            device_map={"": 0},
            **common,
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source, dtype=dtype, **common
        ).eval()
        if device == "cuda":
            model = model.to("cuda")
    # Keep the same loading sequence as source/higgs-tts-3-4b/app.py. The
    # Transformers loader applies the model's own weight-tying rules, so an
    # additional generic tie_weights() call is not needed.
    audio_embedding = getattr(getattr(model, "audio_embedding", None), "weight", None)
    audio_head = getattr(getattr(model, "audio_head", None), "weight", None)
    audio_head_shared = bool(
        audio_embedding is not None
        and audio_head is not None
        and audio_embedding.shape == audio_head.shape
        and audio_embedding.data_ptr() == audio_head.data_ptr()
    )
    head_probe = (
        audio_head.detach().reshape(-1)[:8].float().cpu().tolist()
        if audio_head is not None
        else []
    )
    embedding_probe = (
        audio_embedding.detach().reshape(-1)[:8].float().cpu().tolist()
        if audio_embedding is not None
        else []
    )
    if not callable(getattr(model, "generate_speech", None)):
        raise RuntimeError("Selected model has no generate_speech() method")
    sample_rate = int(getattr(model.config, "sample_rate", 24000))
    reply(
        {
            "ok": True,
            "event": "ready",
            "id": init.get("id"),
            "source": source,
            "device": device,
            "dtype": str(dtype),
            "quantization": quantization,
            "max_concurrency": max_concurrency,
            "sample_rate": sample_rate,
            "transformers": transformers.__version__,
            "audio_head_shared": audio_head_shared,
            "audio_head_probe": head_probe,
            "audio_embedding_probe": embedding_probe,
            "model_seed": model_seed,
            **vram_report(),
        }
    )

    def encode_reference(path):
        audio, sr = sf.read(path, always_2d=False)
        waveform = torch.from_numpy(reference_array(audio))
        return model._encode_reference(waveform, int(sr)).cpu()

    reference_cache = ReferenceCodeCache(encode_reference)
    # Pre-encoding reaches into the adapter's internals. If a future revision
    # drops that method, fall back to handing it the waveform per call.
    can_pre_encode = callable(getattr(model, "_encode_reference", None))

    if max_concurrency > 1:
        # The codec is created lazily on first use. Concurrent first requests
        # would race to build it, so it is materialized while nothing runs.
        model.get_audio_codec()

    slots = ConcurrencySlots(max_concurrency)

    def generate_one(request: dict) -> dict:
        seed = int(request.get("seed", -1))
        kwargs = dict(request["generation"])
        ref_path = request.get("reference_audio")
        ref_key = request.get("reference_key")
        cache_hit = False
        if can_pre_encode and (ref_path or ref_key):
            codes, cache_hit = reference_cache.lookup(ref_key, ref_path)
            if codes is None:
                # The parent believed this clip was cached but the entry was
                # evicted. Tell it to resend the audio rather than silently
                # generating in a different voice.
                return {
                    "ok": False,
                    "error": f"Reference clip {ref_key} is not cached",
                    "error_code": "reference_cache_miss",
                }
            kwargs.update(
                {
                    "reference_codes": codes,
                    "reference_text": request.get("reference_text") or None,
                }
            )
        elif ref_path:
            audio, sr = sf.read(ref_path, always_2d=False)
            kwargs.update(
                {
                    "reference_audio": torch.from_numpy(reference_array(audio)),
                    "reference_sample_rate": int(sr),
                    "reference_text": request.get("reference_text") or None,
                }
            )

        # torch's RNG is global, so a request that asks for a specific seed
        # cannot share the model with anything else and stay reproducible.
        with slots.acquire(exclusive=seed >= 0):
            if seed >= 0:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            # Match the known-good direct_speech() path. Cancellation is
            # handled by terminating this isolated process from the parent.
            output = model.generate_speech(request["prompt"], tokenizer, **kwargs)

        audio = audio_array(output)
        sf.write(request["output_path"], audio, sample_rate)
        return {
            "ok": True,
            "event": "generated",
            "output_path": request["output_path"],
            "samples": len(audio),
            "sample_rate": sample_rate,
            "reference_key": ref_key,
            "reference_cached": bool(ref_key) and (cache_hit or ref_key in reference_cache),
        }

    def handle(request: dict) -> None:
        try:
            reply({"id": request.get("id"), **generate_one(request)})
        except Exception as exc:
            reply(
                {
                    "ok": False,
                    "id": request.get("id"),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    pool = ThreadPoolExecutor(
        max_workers=max_concurrency, thread_name_prefix="higgs-gen"
    )
    try:
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line)
                command = request.get("command")
                if command == "shutdown":
                    pool.shutdown(wait=True)
                    reply({"ok": True, "event": "shutdown", "id": request.get("id")})
                    return
                if command != "generate":
                    raise ValueError("Unknown worker command")
                # Dispatch and keep reading: replies carry the request id, so
                # the parent can have several segments in flight at once.
                pool.submit(handle, request)
            except Exception as exc:
                reply(
                    {
                        "ok": False,
                        "id": request.get("id") if isinstance(request, dict) else None,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
    finally:
        pool.shutdown(wait=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        reply(
            {
                "ok": False,
                "event": "startup_error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
