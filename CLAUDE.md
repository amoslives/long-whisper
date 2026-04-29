# Hinglish ASR Transcription Pipeline

Transcription pipeline for long Hinglish (Hindi+English) meeting recordings. Uses faster-whisper with chunking and overlap stitching to handle files of any length.

## How to Run

```bash
# Activate venv
source venv/bin/activate

# Basic usage (output: <filename>_transcript.txt)
python transcribe_pipeline.py recording.mp3

# Custom output path
python transcribe_pipeline.py recording.mp3 -o meeting_notes.txt

# Use the faster turbo variant
python transcribe_pipeline.py recording.mp3 --model large-v3-turbo

# Custom chunk size and overlap
python transcribe_pipeline.py recording.mp3 --chunk-size 300 --overlap 15

# Use a specific GPU
python transcribe_pipeline.py recording.mp3 --device-index 0
```

## Key Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `large-v3` | Whisper model name |
| `--chunk-size` | `600` (10 min) | Audio chunk duration in seconds |
| `--overlap` | `30` | Overlap between chunks in seconds for dedup stitching |
| `--device-index` | `1` | CUDA GPU index (0 or 1) |
| Language | `hi` (Hindi) | Hardcoded in `TRANSCRIBE_OPTS`; covers Hinglish |
| Compute type | `float16` | Hardcoded in `DEFAULTS` |
| `prompt.txt` | Loaded at startup | Domain-specific Hinglish text to prime the model; copy `prompt.example.txt` to `prompt.txt` and customize |

## System Requirements

- Python 3.10+
- CUDA-capable GPU (tested on 2x NVIDIA RTX A4000, 16GB VRAM each)
- `ffmpeg` and `ffprobe` installed and on PATH
- Models are downloaded from HuggingFace on first run and cached at `~/.cache/huggingface/`
