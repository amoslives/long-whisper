# long-whisper

Transcription pipeline for long audio recordings. Splits audio into overlapping chunks, transcribes each with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), then stitches everything into a single timestamped transcript.

Built for meeting recordings, lectures, and interviews — works well with multilingual audio where speakers switch languages mid-sentence.

## How It Works

1. **Chunk** — splits the audio into 10-minute segments with 30-second overlap using `ffmpeg`
2. **Transcribe** — runs each chunk through faster-whisper on GPU
3. **Stitch** — merges all segments, deduplicating the overlap zones by picking the highest-confidence version
4. **Clean** — removes Whisper hallucination artifacts (repetitive gibberish) and merges word-level fragments into readable sentences

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX A4000 16GB)
- `ffmpeg` and `ffprobe` on PATH

```bash
# System deps (Ubuntu/Debian)
sudo apt install ffmpeg

# Python setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Models are downloaded from HuggingFace on first run and cached at `~/.cache/huggingface/`.

### Initial Prompt

The pipeline uses an optional `prompt.txt` file to prime Whisper with domain-specific vocabulary. This dramatically improves accuracy for jargon, names, and acronyms.

```bash
cp prompt.example.txt prompt.txt
# Edit prompt.txt with vocabulary and phrases from your domain
```

Write a few sentences in the same language mix your meetings use, including key terms, names, and acronyms. The model uses this as context to bias its predictions. If no `prompt.txt` exists, the pipeline runs without a prompt.

## Usage

```bash
source venv/bin/activate
python transcribe_pipeline.py <audio_file> [options]
```

Output is a timestamped text file, one segment per line:

```
[0:08.5 – 0:37.4]  So we'll have a baseline testing of this new system...
[0:38.4 – 1:06.5]  So this AST refill bohinga right now...
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `audio` | *(required)* | Path to the input audio file (mp3, wav, m4a, flac, etc. — anything ffmpeg can read) |
| `-o`, `--output` | `<filename>_transcript.txt` | Path for the output transcript file. If not specified, derived from the input filename |
| `--model` | `large-v3` | Whisper model to use. `large-v3` is recommended — most accurate, produces native-script output (e.g. Devanagari for Hindi), but slower. `large-v3-turbo` is 2-3x faster with romanized output only, at the cost of some accuracy. Other options: `medium`, `small`, `base`, `tiny` |
| `--device-index` | `1` | CUDA GPU index. Use `0` for the first GPU, `1` for the second, etc. |
| `--chunk-size` | `600` | Duration of each audio chunk in seconds. Larger chunks use more VRAM but give Whisper more context. 600s (10 min) is a good balance for 16GB GPUs |
| `--overlap` | `30` | Overlap between consecutive chunks in seconds. The stitcher deduplicates this zone. 30s gives enough margin for clean joins without wasting compute |

## Examples

```bash
# Transcribe a meeting recording (simplest usage)
python transcribe_pipeline.py meeting.mp3

# Save to a specific file
python transcribe_pipeline.py meeting.mp3 -o notes/apr28_meeting.txt

# Use the first GPU instead of the second
python transcribe_pipeline.py meeting.mp3 --device-index 0

# Use the faster turbo variant (2-3x faster, romanized output only)
python transcribe_pipeline.py meeting.mp3 --model large-v3-turbo

# Shorter chunks for a GPU with less VRAM (e.g. 8GB)
python transcribe_pipeline.py meeting.mp3 --chunk-size 300

# Longer overlap for better stitching accuracy
python transcribe_pipeline.py meeting.mp3 --overlap 60

# Combine options
python transcribe_pipeline.py meeting.mp3 -o output.txt --model large-v3 --device-index 0 --chunk-size 300 --overlap 45
```

## Hardcoded Configuration

These are set in the script source (`TRANSCRIBE_OPTS` and `DEFAULTS`) and not exposed as CLI args:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `language` | `"hi"` | Whisper language hint — set to match your audio's primary language |
| `compute_type` | `"float16"` | GPU precision — float16 is fastest on consumer/workstation GPUs |
| `beam_size` | `5` | Beam search width for decoding |
| `repetition_penalty` | `1.2` | Penalizes repeated tokens to reduce Whisper hallucinations |
| `no_repeat_ngram_size` | `5` | Prevents repeating any 5-gram verbatim |
| `vad_filter` | `True` | Silero VAD pre-filters silence before transcription |
| `min_silence_duration_ms` | `500` | Minimum silence gap (ms) before VAD splits a segment |
| `speech_pad_ms` | `400` | Padding added around detected speech (ms) |
| `vad_threshold` | `0.30` | VAD confidence threshold — lower = less aggressive splitting |
| `temperature` | `[0.0, 0.2, 0.4]` | Fallback temperatures if decoding fails at lower temp |
| `log_prob_threshold` | `-0.8` | Segments below this avg log-probability are flagged as low quality |
| `no_speech_threshold` | `0.5` | Segments above this no-speech probability (and below log-prob) are dropped |
| `prompt.txt` | *(see Initial Prompt section)* | Loaded at startup — primes the model with domain vocabulary |

## Post-Processing Pipeline

After transcription, the output goes through three cleaning stages:

1. **Stitch** — sorts all segments by timestamp, deduplicates the overlap zones using similarity matching (>60% text similarity), keeps the version with higher log-probability
2. **Hallucination filter** — detects runs of 3+ consecutive segments with >50% text similarity (Whisper's characteristic repetition artifact) and collapses each run to the single best segment
3. **Fragment merge** — joins adjacent segments shorter than 3 seconds (with gaps under 1.5s) into longer, more readable chunks
