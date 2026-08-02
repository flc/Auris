# Auris

Offline audiobook reader for EPUB, PDF, and TXT with selectable local
OmniVoice, Higgs TTS 3, F5-TTS, or Piper speech, character-aware voices,
per-book narrator control, and synced text highlighting.

Everything runs locally after setup — no API keys, no hosted TTS dependency.
ElevenLabs is available as an opt-in cloud engine for anyone who wants it; the
local engines never call out.

## Screenshots

### Library
![Library](assets/library.png)

### Reader
![Reader](assets/reader.png)

### Voice Studio
![Voice Studio](assets/voice_studio.png)

### Settings
![Settings](assets/settings.png)

## Highlights

- Import EPUB, PDF, and TXT books.
- Detect chapters, prologues, epilogues, forewords, appendices, and parts automatically.
- Generate per-character voices with deterministic assignment.
- Attribute dialogue to characters with OpenAI or a configurable local LM
  Studio, Ollama, llama.cpp, or other OpenAI-compatible endpoint.
- Customize each detected character in Voice Studio.
- Customize the narrator voice per book.
- Preview voices before saving.
- Upload reference WAV files for voice cloning.
- Invalidate stale cached playback automatically when narrator or character voices change.
- Re-render sentences that came back too short for their text, and audition
  alternative takes of any sentence from the reader.
- Export numbered, per-chapter audio as WAV or MP3 and subtitles as ASS or SRT.
- Select all chapters or use print-style selections such as `1,3,5-8`.
- Run from a project-local `.venv` created by the installer.

## Requirements

- Python 3.10 or later
- `ffmpeg` on `PATH` for MP3 export
- OmniVoice model files stored locally
- Optional NVIDIA GPU for faster inference
- On Linux / WSL with an NVIDIA GPU: a C compiler and the CPython headers
  (`sudo apt install -y build-essential python3-dev`). Triton, which the CUDA
  build of PyTorch depends on, compiles kernels on first use — without these,
  install and model load both succeed and only TTS playback fails.

## Installation

```bash
git clone https://github.com/nikhilprasanth/Auris.git
cd Auris
```

Run the installer:

```bash
# Windows
reader\setup.bat

# Linux / macOS
bash reader/setup.sh
```

Or directly:

```bash
python reader/setup.py
```

The installer detects CUDA or CPU, creates `reader/.venv`, installs PyTorch, OmniVoice, spaCy, and the reader dependencies, then downloads the `en_core_web_sm` spaCy model when network access is available.

## Model setup

The OmniVoice weights are not bundled with this repository.

You can either:

- Download them from the Settings page using the built-in Hugging Face downloader.
- Point Settings at an existing local OmniVoice model directory.

The model directory must contain the files OmniVoice expects, such as `config.json` and model weights.

### Higgs TTS 3

Select **Higgs TTS 3 — 4B** in Settings to try the Boson AI model. Auris uses
the Transformers-compatible
`multimodalart/higgs-audio-v3-tts-4b-transformers` adapter and downloads its
model files into the HuggingFace cache on first load. You can also point the
Higgs section at a compatible local snapshot.

Higgs and OmniVoice keep separate settings and run with separate Transformers
versions. The installer puts Higgs' Transformers 5.13 runtime in
`reader/.higgs_runtime`; OmniVoice remains on its compatible 5.3 release.

Higgs supports Hungarian, transcript-assisted zero-shot voice cloning, and
inline emotion, style, prosody, pause, and sound-effect controls. Auris maps its
existing scene speed and expression tags to those controls.

**Quantization.** The BF16 checkpoint is 4.65B parameters and needs ~7.6 GiB of
resident weights. On an 8 GB card that barely fits, so anything else using the
GPU pushes the driver into paging weights through system memory — and because
the decoder runs one full forward pass per audio token, that turns into a PCIe
transfer per token. Setting **Quantization** to 4-bit in the Higgs settings
drops the footprint to ~2.5 GiB and, measured on an RTX 5070 Laptop, makes
decoding about 3.3× faster (6.6 → 22 audio tokens/s). It needs `bitsandbytes`
and CUDA:

    reader/.venv/bin/python -m pip install bitsandbytes

The transformer body is quantized; the fused audio embedding and head stay BF16
because they share storage and quantizing either one would break that tying.
Quantized output differs from full precision, so it uses its own audio cache
entries. The Higgs settings block shows the loaded device, precision and actual
VRAM use after a reload.

**Segments are decoded one at a time**, and measurement says that is the fast
path. The worker can overlap them — threads share the weights and the codec, so
unlike OmniVoice's export replicas it costs almost no VRAM, only the KV cache
is per segment — but a single decode stream already saturates the GPU, and the
parallel loops then contend for Python's GIL. On an RTX 5070 Laptop at 4-bit,
with peak use at 4.7 of 7.96 GiB so nothing was paging:

| Lanes | Throughput | vs serial |
|-------|-----------|-----------|
| 1     | 0.71× realtime | 1.00× |
| 2     | 0.43× realtime | 0.61× |
| 3     | 0.31× realtime | 0.44× |

There is deliberately no setting for it. To experiment on a card that does
leave idle time between token kernels, set `higgs_concurrency` in
`reader/data/settings.json`. Segments that pin a seed always run alone so they
stay reproducible.

Higgs has its own research/non-commercial license with a creator-use grant.
Audiobooks and similar creator media require prominent Boson AI Higgs Audio
attribution, and voice cloning requires the speaker's consent. Review the
[official model card](https://huggingface.co/bosonai/higgs-tts-3-4b) before use.

### F5-TTS — Hungarian

Select **F5-TTS — Hungarian** in Settings to use a Hungarian finetune of
F5-TTS v1 Base. Auris defaults to
[`Maxdorger29/f5-tts-hungarian`](https://huggingface.co/Maxdorger29/f5-tts-hungarian)
(280 h across Common Voice HU 17.0, YodaLingua and CSS10) and downloads its
checkpoint and vocabulary into the HuggingFace cache on first load;
[`sarpba/F5-TTS-Hun`](https://huggingface.co/sarpba/F5-TTS-Hun) works too. Each
checkpoint ships its own `vocab.txt`, so always change the checkpoint and
vocabulary fields together.

Unlike Higgs, this engine runs inside the main environment — F5-TTS leaves
`transformers` unpinned, so it needs no isolated runtime and no worker
subprocess. It wants about 4 GB of VRAM and synthesizes roughly 2.5× faster
than real time on an RTX 3090.

Two limits are worth knowing before switching:

- **No voice design.** F5-TTS can only clone, so a narrator reference WAV
  (5–15 s) *and its exact transcript* are mandatory in Settings. A mismatched
  transcript produces garbled speech rather than an error. Characters without
  their own reference WAV share the narrator's voice — descriptions such as
  `female, young adult` have no effect on this engine.
- **No expression tags.** The 67-token Hungarian vocabulary has no room for
  them, so Auris strips its enrichment tags and keeps only scene speed and
  punctuation-driven pauses.

Auris folds text into that vocabulary before synthesis — lowercasing, spelling
out symbols, repairing legacy PDF vowels (`õ`/`û`) and collapsing punctuation
runs — because F5-TTS maps unknown characters to a space instead of failing.

The Hungarian checkpoints are CC-BY-NC-4.0: personal and research use with
attribution, no commercial use.

### Piper — fast CPU

Select **Piper — fast CPU** in Settings for ONNX synthesis on the processor:
no VRAM, no torch, and about **40× faster than real time** — a 400,000-character
novel in roughly ten minutes. Voices are a few tens of megabytes each and
download from
[`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) on first
use. Hungarian ships three: `hu_HU-anna-medium`, `hu_HU-berta-medium` and
`hu_HU-imre-medium`.

It cannot clone and has no expression control. What it does give you, and the
other local engines do not, is **more than one voice for free**: each Piper
voice is its own model, so Auris casts characters across the configured list
instead of reading everyone in the narrator's timbre.

- The narrator reads all narration and is excluded from character casting, so
  no character shares its voice.
- Characters are assigned deterministically by name — the same character keeps
  its voice across sessions and re-exports.
- With **Match gender** on, a character described as `male` or `female` is cast
  onto a voice of that gender. Auris knows the three Hungarian voices are
  female (anna, berta) and male (imre).
- Uploaded reference clips are ignored; Piper does not clone.

Enrichment tags are stripped before synthesis because espeak-ng pronounces
anything it is handed — `[laughter]` would otherwise be read out as a word.
Piper models run at 22.05 kHz and Auris resamples to the 24 kHz the rest of the
pipeline assumes.

Licensing differs from the rest of the project: the voices come from CC0
datasets, but the `piper-tts` runtime the installer pulls in is
**GPL-3.0-or-later**, unlike every other Auris dependency.

### ElevenLabs (optional, cloud)

Select **ElevenLabs — cloud API** in Settings to synthesize through
[ElevenLabs](https://elevenlabs.io) instead of a local model. Nothing is
downloaded and no VRAM is used; every uncached segment is an HTTP request
billed per character.

Paste an API key and a voice ID in the ElevenLabs section of Settings, then
press **Reload TTS engine**. The key can also come from the
`ELEVENLABS_API_KEY` environment variable, which takes precedence over the
one stored in `data/settings.json`.

Current limitations of this engine:

- One voice reads the whole book. Narrator instructions, per-character voices,
  and reference-clip cloning are ignored — those need ElevenLabs voice IDs,
  which Auris does not map yet.
- Segments are synthesized one request at a time.
- Auris' non-verbal tags (`[laughter]`, `[sigh]`, …) are stripped rather than
  translated to ElevenLabs v3 audio tags.

Generated audio lands in the same WAV cache as the local engines, so replaying
or re-exporting an already-generated chapter costs nothing. A full novel is
several hundred thousand characters — check your plan's quota, shown in the
ElevenLabs settings section after a reload, before exporting one.

## Usage

1. Import a book from the library page.
2. Open the book and start playback from any sentence.
3. Open Voice Studio from the reader sidebar.
4. Adjust character voices or the narrator voice, preview them, then save.
5. Export the current chapter or select chapters with `all`, a range such as
   `2-6`, or a comma-separated expression such as `1,3,7-10`.

### When a sentence reads badly

Every engine occasionally swallows the last few words of a sentence, or lands
the stress somewhere odd. Those are two different problems and Auris handles
them separately.

**Missing words are detected automatically.** After a chapter is generated, each
sentence is compared against the chapter's own median speech rate — nothing is
hardcoded per language, engine or voice — and a sentence whose audio is far too
short for its text is re-rendered. Only the outliers cost extra time. Turn it
off with **Re-render sentences that came out too short** in Settings → Generation
& export.

**Bad stress needs your ear.** While reading, the **🔁 Takes** button in the
playback bar offers alternative readings of the current sentence: generate a
couple, listen, and press *Keep* on the one you prefer. Playback and export use
it from then on. Rejected takes — including any the automatic check threw away —
stay in the list so you can go back.

Takes belong to a segment, so re-enriching a chapter (a text edit, a new
narrator voice, a pronunciation rule) discards them along with the audio they
described.

### Language-model character detection

Open **Settings → Characters & narrator**, enable **Language model**, then choose
either a local OpenAI-compatible server or the OpenAI API.

For OpenAI, add an API key, use **Load models**, and select one of the text models
available to the account. ChatGPT subscriptions and OpenAI API billing are
separate; using this integration requires API access and is billed through the
OpenAI API account.

For a local server, enter its base URL and load the served models. Typical URLs
are:

- LM Studio: `http://127.0.0.1:1234/v1`
- Ollama: `http://127.0.0.1:11434/v1`

For the LM Studio test configuration used during development:

- served model: `unsloth/gemma-4-26b-a4b-it`
- LM Studio context length: `160000` tokens
- Auris request timeout: `600` seconds
- maximum stored characters: `60`
- API key: empty for a normal local server

The context length is configured in LM Studio or Ollama, not in Auris. Auris
deliberately sends one chapter per request because the size of the structured
speaker-assignment response, rather than the model's input context, is normally
the limiting factor.

Recommended setup:

1. Start the local server and load the language model.
2. Open **Settings → Characters & narrator**.
3. Select **Language model — recommended** and the **Local server** provider.
4. Enter the base URL and use **Load models** to select the served model.
5. Set the request timeout and maximum character count. Add an API key only if
   the local server requires one.
6. Use **Test connection**, save the settings, and then import the book.

Character and dialogue-speaker analysis is an import-time background job.
For a local language model, Auris unloads the selected TTS engine first so the
two models do not compete for VRAM. OpenAI analysis leaves local TTS running.
Auris sends numbered text units chapter by chapter,
builds a canonical character roster, and stores every dialogue-to-speaker
assignment with the book. The reader, Voice Studio, playback, and export then
reuse those stored assignments; the TTS engine is loaded lazily only when it is
next needed. If an individual chapter fails, successful chapter results are
kept and the book is marked as partially analyzed instead of discarding the
whole run.

Books imported before enabling Local LLM detection must be deleted and imported
again if they should receive the new speaker assignments. The analysis is not
retroactively started just by changing the setting.

The two files in `test_docs/` were measured end to end against LM Studio with
`unsloth/gemma-4-26b-a4b-it` and a 160,000-token server context:

| Test document | Chapters | Dialogue candidates | Speaker assigned | Coverage | Chapter errors | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Rejto_Jeno-14-karatos-auto.pdf` | 21 | 2,063 | 1,845 | 89.4% | 0 | 6m 36s |
| `14Carat.txt` | 21 | 1,395 | 1,350 | 96.8% | 0 | 7m 56s |

These figures describe the tested model and documents, not a guaranteed score
for every book. Dialogue style, OCR/text extraction quality, model choice, and
model quantization can all change the result. The legacy English-oriented
spaCy/regex detector remains available as a fallback mode.

Exports are saved beneath `reader/exports/<author> - <book_title>/` with
numbered filenames, for example `01_Introduction.mp3`. When MP3 export succeeds,
the temporary WAV file is removed automatically.

On RTX 3090-class GPUs, leave **Settings → Parallel export workers** on
**Auto** or select **2**. Multi-chapter export then loads a second OmniVoice model
temporarily and runs two CUDA streams. If VRAM is insufficient or either
worker fails, Auris automatically continues on the primary model.

## Voice design caveats

OmniVoice does not produce clean output for every voice-design combination. The upstream docs note that some attribute mixes are unreliable, especially without reference audio.

The most fragile cases are youth voices with extreme pitch settings. For example, combinations like `male, teenager, very high pitch, american accent` can degrade into squeaks, bursts, or static instead of intelligible speech.

Auris now tries to stabilize some known-bad combinations during preview and playback by relaxing them to a nearby voice design, but this is still a model limitation, not something the UI can fully solve.

Best results:

- Prefer `young adult` over `teenager` when you do not have reference audio.
- Avoid `very high pitch` and `very low pitch` on `child` and `teenager` voices.
- Upload a clean WAV reference when you need a specific youthful voice.
- Preview before saving.

Reference: `https://github.com/k2-fsa/OmniVoice/blob/master/docs/voice-design.md`

## Offline installs

Local wheels are not used by default.

If you intentionally maintain your own wheel cache, opt in explicitly:

```bash
# Windows
set AURIS_USE_LOCAL_WHEELS=1
reader\setup.bat

# Linux / macOS
AURIS_USE_LOCAL_WHEELS=1 bash reader/setup.sh
```

For a strict offline install:

```bash
# Windows
set AURIS_OFFLINE=1
set AURIS_WHEELS_DIR=E:\path\to\wheels
reader\setup.bat

# Linux / macOS
AURIS_OFFLINE=1 AURIS_WHEELS_DIR=/path/to/wheels bash reader/setup.sh
```

## Project structure

```text
Auris/
|-- README.md
|-- LICENSE
|-- wheels/          ← offline wheel cache (optional)
`-- reader/
    |-- app.py
    |-- setup.py     ← cross-platform installer (called by setup.bat / setup.sh)
    |-- setup.bat    ← Windows installer
    |-- setup.sh     ← Linux / macOS installer
    |-- run.bat      ← Windows launcher
    |-- run.sh       ← Linux / macOS launcher
    |-- requirements.txt
    |-- core/
    |-- static/
    |-- templates/
    `-- data/
```

## Main dependencies

- OmniVoice
- F5-TTS
- Piper (GPL-3.0-or-later)
- Flask
- ebooklib
- PyMuPDF
- spaCy
- pydub
- soundfile
- PyTorch

## Roadmap

### Small language model for emotion classification

The current enrichment pipeline uses regex patterns to decide which non-verbal tag (`[laughter]`, `[surprise-wa]`, `[question-ei]`, etc.) to inject before each TTS segment. It works well when attribution verbs are present in the text ("she gasped", "he scoffed"), but it cannot understand tone, irony, or context that isn't signalled by a keyword.

The plan is to connect to any OpenAI-compatible language model endpoint as an emotion classifier between parsing and TTS synthesis:

- **Connection:** a configurable base URL and API key in Settings, compatible with any OpenAI-spec server — local (Ollama, LM Studio, llama.cpp server) or remote. No runtime library bundled with Auris; the standard `openai` Python client is the only dependency.
- **Model candidates:** **Qwen3-0.8B** (fastest, lowest RAM), **Qwen3-2B** (better reasoning, still lightweight), **Gemma 4 E2B** (Google's 2B edge model), **LFM2.5-1.2B-Instruct** (Liquid AI — strong reasoning efficiency per parameter). Any model the user serves behind an OpenAI-compatible endpoint will work.
- **Input:** the current segment text plus one sentence of surrounding context.
- **Output:** a single tag from the supported set, or `none`. Structured output / JSON mode keeps latency low and parsing trivial.
- **Fallback:** the existing regex engine remains as a zero-latency fallback when no endpoint is configured or the model returns an invalid response.
- **Integration point:** `core/enrichment.py` — the `_select_expression_tag` function would be replaced by a call to the classifier, with the regex result used as a hint in the prompt.
- **UX:** base URL, API key, and model name are set in Settings. Leaving the base URL blank keeps regex-only mode active.

This would fix the main remaining gap: narration sentences that carry emotional weight without any keyword signal, and multi-emotion moments where the current system can only pick one tag.

## License

The Auris source is MIT. See [LICENSE](LICENSE). Models retain their own
licenses; in particular, Higgs TTS 3 and the Hungarian F5-TTS checkpoints
(CC-BY-NC-4.0) are not distributed under the Auris MIT license.

One runtime dependency is not permissively licensed either: the installer
pulls in `piper-tts`, which is **GPL-3.0-or-later**. It is imported only by
`core/piper_engine.py`, and only when Piper is the selected engine. Anyone
redistributing Auris together with its installed environment should check what
that implies for them; Piper's own voices are CC0.
