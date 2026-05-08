# Prompt Tester Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool (`test_prompt.py`) that samples random clips from an audio file, transcribes them with a given prompt, prints a per-segment quality report with diagnostic flags, and summarizes overall quality — so users can iterate on their `prompt_<lang>.txt` files before committing to a full transcription run.

**Architecture:** Single-file script that reuses `fmt_ts`, `get_duration`, and `_load_prompt` from `transcribe_pipeline.py` (imported, not copied). The script extracts N random samples via ffmpeg, transcribes each with faster-whisper, flags problems (hallucination loops, low log-prob, high no-speech), and prints a summary report with actionable quality metrics. All model/device/language defaults are inherited from `transcribe_pipeline.py`'s `DEFAULTS` and `TRANSCRIBE_OPTS`.

**Tech Stack:** Python 3.10+, faster-whisper, ffmpeg/ffprobe (existing deps — no new requirements)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `transcribe_pipeline.py` | **Modify** — export `fmt_ts`, `get_duration`, `_load_prompt`, `DEFAULTS`, `TRANSCRIBE_OPTS` (currently all accessible at module level, but guard `main()` so importing doesn't run the CLI) |
| `test_prompt.py` | **Create** — CLI tool: sample extraction, transcription, quality analysis, summary report |
| `sample_test.py` | **Delete** — replaced by `test_prompt.py` |
| `retest_sample1.py` | **Delete** — replaced by `test_prompt.py --at` mode |

---

## Design Decisions

1. **Import from pipeline, don't copy.** `fmt_ts`, `get_duration`, `_load_prompt`, `DEFAULTS`, `TRANSCRIBE_OPTS` are already in `transcribe_pipeline.py`. The test tool imports them. This means changes to model defaults or prompt loading logic propagate automatically.

2. **Two modes in one script:**
   - **Random sampling** (default): `test_prompt.py audio.mp3` — picks N random 5-min clips, transcribes all, prints report.
   - **Targeted re-test**: `test_prompt.py audio.mp3 --at 402,2919` — transcribes only at those offsets. For re-checking specific problem spots after prompt changes.

3. **Quality flags per segment:**
   - `[HALLUCINATION]` — segment text is ≥60% similar to previous segment text (consecutive run ≥ 3)
   - `[LOW_LOGPROB]` — `avg_logprob < -0.8`
   - `[HIGH_NOSPEECH]` — `no_speech_prob > 0.5`

4. **Summary report at the end:** Total segments, flagged segments by type, per-sample segment counts, overall quality verdict (PASS / REVIEW / FAIL based on flagged-segment percentage).

5. **Override transcription params via CLI:** `--repetition-penalty` and `--no-repeat-ngram-size` are the two most commonly tweaked params during prompt iteration, so they get dedicated flags. Other TRANSCRIBE_OPTS stay at pipeline defaults.

---

### Task 1: Make pipeline functions importable

Currently `transcribe_pipeline.py` is already safely guarded by `if __name__ == "__main__"` at the bottom, and all target symbols (`fmt_ts`, `get_duration`, `_load_prompt`, `DEFAULTS`, `TRANSCRIBE_OPTS`) are module-level. But `_load_prompt` is conventionally "private" (leading underscore). Rename it to `load_prompt` so the public API is clear.

**Files:**
- Modify: `transcribe_pipeline.py:29` (rename function)
- Modify: `transcribe_pipeline.py:244` (update call site)

- [ ] **Step 1: Rename `_load_prompt` to `load_prompt`**

In `transcribe_pipeline.py`, change the function definition:

```python
# line 29: change
def _load_prompt(lang):
# to
def load_prompt(lang):
```

And update the call site:

```python
# line 244: change
TRANSCRIBE_OPTS["initial_prompt"] = _load_prompt(args.lang)
# to
TRANSCRIBE_OPTS["initial_prompt"] = load_prompt(args.lang)
```

- [ ] **Step 2: Verify pipeline still works**

Run:
```bash
source venv/bin/activate && python transcribe_pipeline.py --help
```
Expected: Help text prints without error.

- [ ] **Step 3: Verify imports work**

Run:
```bash
source venv/bin/activate && python -c "from transcribe_pipeline import fmt_ts, get_duration, load_prompt, DEFAULTS, TRANSCRIBE_OPTS; print('OK')"
```
Expected: Prints `OK`.

- [ ] **Step 4: Commit**

```bash
git add transcribe_pipeline.py
git commit -m "Rename _load_prompt to load_prompt for public import"
```

---

### Task 2: Create `test_prompt.py` — argument parsing and sample point selection

**Files:**
- Create: `test_prompt.py`

- [ ] **Step 1: Create the script with argument parsing and sample selection logic**

```python
#!/usr/bin/env python3
"""
Test a Whisper prompt against random audio samples before full transcription.

Extracts short clips from an audio file, transcribes each with the current
prompt_<lang>.txt, and reports quality metrics so you can iterate on the
prompt text without waiting for a full pipeline run.

Usage:
    # Random sampling (5 clips, 5 min each)
    python test_prompt.py recording.mp3

    # Custom sample count and duration
    python test_prompt.py recording.mp3 --samples 3 --duration 180

    # Re-test specific offsets after tweaking prompt
    python test_prompt.py recording.mp3 --at 402,2919

    # Override transcription params for A/B testing
    python test_prompt.py recording.mp3 --repetition-penalty 1.4 --no-repeat-ngram-size 4
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher

from faster_whisper import WhisperModel
from transcribe_pipeline import (
    DEFAULTS,
    TRANSCRIBE_OPTS,
    fmt_ts,
    get_duration,
    load_prompt,
)

SIMILARITY_THRESH = 0.6
HALLUCINATION_MIN_RUN = 3


def pick_sample_starts(duration, sample_duration, n_samples, seed=None):
    margin = sample_duration
    max_start = duration - sample_duration - margin
    if max_start <= margin:
        return [0]
    rng = random.Random(seed)
    pool = range(int(margin), int(max_start))
    n = min(n_samples, len(pool))
    return sorted(rng.sample(pool, n))


def extract_clip(audio_path, start_sec, duration, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start_sec), "-i", audio_path,
         "-t", str(duration), "-ac", "1", "-ar", "16000",
         "-loglevel", "error", out_path],
        check=True,
    )


def transcribe_sample(model, clip_path, offset, opts):
    segments, _info = model.transcribe(clip_path, **opts)

    results = []
    for seg in segments:
        results.append(dict(
            start=seg.start + offset,
            end=seg.end + offset,
            text=seg.text.strip(),
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
        ))
    return results


def flag_segments(segments):
    for seg in segments:
        seg["flags"] = []
        if seg["avg_logprob"] < -0.8:
            seg["flags"].append("LOW_LOGPROB")
        if seg["no_speech_prob"] > 0.5:
            seg["flags"].append("HIGH_NOSPEECH")

    for i in range(len(segments)):
        run_len = 0
        if i >= 1:
            j = i
            while j >= 1:
                sim = SequenceMatcher(
                    None,
                    segments[j]["text"].lower(),
                    segments[j - 1]["text"].lower(),
                ).ratio()
                if sim >= SIMILARITY_THRESH:
                    run_len += 1
                    j -= 1
                else:
                    break
            if run_len >= HALLUCINATION_MIN_RUN:
                segments[i]["flags"].append("HALLUCINATION")

    return segments


def print_sample_header(index, total, start_sec, end_sec):
    print(f"\n{'=' * 70}")
    print(f"SAMPLE {index}/{total}: {fmt_ts(start_sec)} – {fmt_ts(end_sec)}")
    print(f"{'=' * 70}")


def print_segment(seg):
    flags_str = ""
    if seg["flags"]:
        flags_str = " [" + "] [".join(seg["flags"]) + "]"
    print(
        f"  [{fmt_ts(seg['start'])} – {fmt_ts(seg['end'])}]"
        f" (lp={seg['avg_logprob']:.2f}, nsp={seg['no_speech_prob']:.2f})"
        f"{flags_str}"
    )
    print(f"    {seg['text']}")


def print_summary(all_sample_results):
    total_segs = 0
    total_hallucination = 0
    total_low_logprob = 0
    total_high_nospeech = 0

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for i, (start, segments) in enumerate(all_sample_results):
        n = len(segments)
        n_flagged = sum(1 for s in segments if s["flags"])
        print(f"  Sample {i + 1} ({fmt_ts(start)}): {n} segments, {n_flagged} flagged")
        total_segs += n
        for s in segments:
            if "HALLUCINATION" in s["flags"]:
                total_hallucination += 1
            if "LOW_LOGPROB" in s["flags"]:
                total_low_logprob += 1
            if "HIGH_NOSPEECH" in s["flags"]:
                total_high_nospeech += 1

    total_flagged = sum(
        1 for _, segs in all_sample_results for s in segs if s["flags"]
    )
    flag_pct = (total_flagged / total_segs * 100) if total_segs else 0

    print(f"\n  Total segments   : {total_segs}")
    print(f"  Flagged segments : {total_flagged} ({flag_pct:.1f}%)")
    print(f"    HALLUCINATION  : {total_hallucination}")
    print(f"    LOW_LOGPROB    : {total_low_logprob}")
    print(f"    HIGH_NOSPEECH  : {total_high_nospeech}")

    if flag_pct == 0:
        verdict = "PASS — no issues detected"
    elif flag_pct < 10:
        verdict = "PASS — minor issues, prompt looks good"
    elif flag_pct < 25:
        verdict = "REVIEW — consider tweaking prompt or repetition_penalty"
    else:
        verdict = "FAIL — significant issues, prompt needs rework"
    print(f"\n  Verdict: {verdict}")


def main():
    p = argparse.ArgumentParser(
        description="Test a Whisper prompt against random audio samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_prompt.py meeting.mp3\n"
            "  python test_prompt.py meeting.mp3 --samples 3 --duration 180\n"
            "  python test_prompt.py meeting.mp3 --at 402,2919\n"
            "  python test_prompt.py meeting.mp3 --repetition-penalty 1.4\n"
        ),
    )
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("--lang", default="hi",
                   help="Language code (default: hi). Loads prompt_<lang>.txt if it exists")
    p.add_argument("--samples", type=int, default=5,
                   help="Number of random samples to test (default: 5)")
    p.add_argument("--duration", type=int, default=300,
                   help="Duration of each sample in seconds (default: 300)")
    p.add_argument("--at", type=str, default=None,
                   help="Comma-separated offsets in seconds to test specific positions (e.g. 402,2919)")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducible sample selection")
    p.add_argument("--model", default=DEFAULTS["model"],
                   help=f"Whisper model (default: {DEFAULTS['model']})")
    p.add_argument("--device-index", type=int, default=DEFAULTS["device_index"],
                   help=f"CUDA device index (default: {DEFAULTS['device_index']})")
    p.add_argument("--repetition-penalty", type=float, default=None,
                   help="Override repetition_penalty (pipeline default used if omitted)")
    p.add_argument("--no-repeat-ngram-size", type=int, default=None,
                   help="Override no_repeat_ngram_size (pipeline default used if omitted)")
    args = p.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: {args.audio} not found", file=sys.stderr)
        sys.exit(1)

    duration = get_duration(args.audio)
    prompt_text = load_prompt(args.lang)

    opts = dict(TRANSCRIBE_OPTS)
    opts["language"] = args.lang
    opts["initial_prompt"] = prompt_text
    if args.repetition_penalty is not None:
        opts["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        opts["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    if args.at:
        starts = [int(s.strip()) for s in args.at.split(",")]
    else:
        starts = pick_sample_starts(duration, args.duration, args.samples, args.seed)

    prompt_status = f"prompt_{args.lang}.txt" if prompt_text else "none"
    print(f"Audio              : {args.audio}")
    print(f"Duration           : {fmt_ts(duration)}")
    print(f"Model              : {args.model}")
    print(f"Language           : {args.lang}")
    print(f"Prompt             : {prompt_status}")
    print(f"Samples            : {len(starts)} x {args.duration}s")
    print(f"repetition_penalty : {opts['repetition_penalty']}")
    print(f"no_repeat_ngram    : {opts['no_repeat_ngram_size']}")

    tmpdir = tempfile.mkdtemp(prefix="whisper_prompt_test_")

    try:
        print(f"\nLoading {args.model} ...")
        model = WhisperModel(
            args.model,
            device=DEFAULTS["device"],
            device_index=args.device_index,
            compute_type=DEFAULTS["compute_type"],
        )

        all_sample_results = []

        for i, start_sec in enumerate(starts):
            end_sec = start_sec + args.duration
            clip_path = os.path.join(tmpdir, f"sample_{i}.mp3")

            print_sample_header(i + 1, len(starts), start_sec, end_sec)
            extract_clip(args.audio, start_sec, args.duration, clip_path)

            segments = transcribe_sample(model, clip_path, start_sec, opts)
            segments = flag_segments(segments)

            for seg in segments:
                print_segment(seg)

            print(f"\n  => {len(segments)} segments")
            all_sample_results.append((start_sec, segments))

        print_summary(all_sample_results)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses arguments correctly**

Run:
```bash
source venv/bin/activate && python test_prompt.py --help
```
Expected: Help text with all arguments listed and example usage in epilog.

- [ ] **Step 3: Commit**

```bash
git add test_prompt.py
git commit -m "Add prompt testing tool for iterating on Whisper prompts"
```

---

### Task 3: Delete old one-off test scripts

The ad-hoc scripts `sample_test.py` and `retest_sample1.py` are now replaced by `test_prompt.py`.

**Files:**
- Delete: `sample_test.py`
- Delete: `retest_sample1.py`

- [ ] **Step 1: Remove the old scripts**

```bash
rm sample_test.py retest_sample1.py
```

- [ ] **Step 2: Commit**

```bash
git add -u sample_test.py retest_sample1.py
git commit -m "Remove ad-hoc test scripts replaced by test_prompt.py"
```

---

### Task 4: Update documentation

**Files:**
- Modify: `CLAUDE.md` — add `test_prompt.py` usage to the "How to Run" section
- Modify: `README.md` — add prompt testing workflow

- [ ] **Step 1: Add test_prompt.py section to CLAUDE.md**

Add after the existing "How to Run" code block:

```markdown
## Testing & Iterating on Prompts

```bash
# Test current prompt against 5 random 5-min samples
python test_prompt.py recording.mp3 --lang hi

# Fewer, shorter samples for quick checks
python test_prompt.py recording.mp3 --samples 3 --duration 180

# Re-test specific problem spots after editing prompt_hi.txt
python test_prompt.py recording.mp3 --at 402,2919

# A/B test with different repetition penalty
python test_prompt.py recording.mp3 --repetition-penalty 1.2
python test_prompt.py recording.mp3 --repetition-penalty 1.6

# Reproducible random selection
python test_prompt.py recording.mp3 --seed 42
```
```

- [ ] **Step 2: Add prompt testing section to README.md**

Add a "Prompt Testing" section describing the workflow:
1. Copy `prompt.example.txt` to `prompt_<lang>.txt`
2. Edit with domain vocabulary
3. Run `test_prompt.py` to check quality
4. Look at the SUMMARY verdict and flagged segments
5. Tweak prompt text and re-run with `--at` on flagged positions
6. Once PASS, run full `transcribe_pipeline.py`

- [ ] **Step 3: Update CLAUDE.md key configuration table**

Add row for `test_prompt.py` to the Key Configuration table:

```markdown
| `test_prompt.py` | Prompt quality tester | Samples random clips and reports transcription quality metrics |
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Document prompt testing workflow in CLAUDE.md and README"
```

---

### Task 5: Smoke test end-to-end

This is not a code task — it's a manual verification that the complete workflow works on a real audio file.

- [ ] **Step 1: Run random sampling mode**

```bash
source venv/bin/activate && python test_prompt.py <any_audio_file> --lang hi --samples 2 --duration 120 --seed 42
```

Verify:
- 2 samples are extracted and transcribed
- Per-segment output shows timestamps, logprob, no-speech-prob, and flags
- Summary prints segment counts, flag counts, and a verdict

- [ ] **Step 2: Run targeted re-test mode**

Pick an offset from the output of step 1 where a flag appeared and run:

```bash
python test_prompt.py <same_audio_file> --lang hi --at <offset_with_flag> --duration 120
```

Verify: Only that single position is transcribed.

- [ ] **Step 3: Run with overridden params**

```bash
python test_prompt.py <same_audio_file> --lang hi --samples 1 --duration 60 --repetition-penalty 1.6 --no-repeat-ngram-size 3
```

Verify: Header shows the overridden values, not the pipeline defaults.

- [ ] **Step 4: Verify pipeline still works unchanged**

```bash
python transcribe_pipeline.py --help
```

Verify: No import errors or regressions from the rename.
