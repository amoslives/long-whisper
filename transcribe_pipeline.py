#!/usr/bin/env python3
"""
Transcription pipeline for long audio files.

Uses faster-whisper large-v3 with chunking, overlap, and
intelligent stitching to produce a single continuous transcript.

Usage:
    python transcribe_pipeline.py <audio_file>
    python transcribe_pipeline.py <audio_file> -o output.txt
    python transcribe_pipeline.py <audio_file> --model large-v3-turbo
    python transcribe_pipeline.py <audio_file> --chunk-size 300 --overlap 15
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from difflib import SequenceMatcher

from faster_whisper import WhisperModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_prompt_file = os.path.join(SCRIPT_DIR, "prompt.txt")
if os.path.exists(_prompt_file):
    with open(_prompt_file, encoding="utf-8") as _f:
        INITIAL_PROMPT = _f.read().strip()
else:
    INITIAL_PROMPT = None

TRANSCRIBE_OPTS = dict(
    language="hi",
    initial_prompt=INITIAL_PROMPT,
    beam_size=5,
    repetition_penalty=1.2,
    no_repeat_ngram_size=5,
    vad_filter=True,
    vad_parameters=dict(
        min_silence_duration_ms=500,
        speech_pad_ms=400,
        threshold=0.30,
    ),
    word_timestamps=True,
    temperature=[0.0, 0.2, 0.4],
    condition_on_previous_text=True,
    log_prob_threshold=-0.8,
    no_speech_threshold=0.5,
)

DEFAULTS = dict(
    model="large-v3",
    device="cuda",
    device_index=1,
    compute_type="float16",
    chunk_size=600,
    overlap=30,
)


def fmt_ts(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:04.1f}"
    return f"{m}:{s:04.1f}"


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"Error: ffprobe failed on {path}", file=sys.stderr)
        sys.exit(1)
    return float(json.loads(r.stdout)["format"]["duration"])


def chunk_audio(input_path, chunk_dir, chunk_s, overlap_s):
    duration = get_duration(input_path)

    if duration <= chunk_s:
        out = os.path.join(chunk_dir, "chunk_000.mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(input_path),
             "-ac", "1", "-ar", "16000", "-loglevel", "error", out],
            check=True,
        )
        return [(out, 0.0, duration)]

    chunks = []
    start = 0.0
    i = 0
    while start < duration:
        out = os.path.join(chunk_dir, f"chunk_{i:03d}.mp3")
        length = min(chunk_s + overlap_s, duration - start)
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(start), "-i", str(input_path),
             "-t", str(length), "-ac", "1", "-ar", "16000",
             "-loglevel", "error", out],
            check=True,
        )
        chunks.append((out, start, min(start + length, duration)))
        start += chunk_s
        i += 1

    return chunks


def transcribe_chunk(model, chunk_path, offset):
    segments, _info = model.transcribe(chunk_path, **TRANSCRIBE_OPTS)

    result = []
    for seg in segments:
        if seg.no_speech_prob > 0.6 and seg.avg_logprob < -0.7:
            continue
        result.append(dict(
            start=seg.start + offset,
            end=seg.end + offset,
            text=seg.text.strip(),
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
        ))
    return result


def text_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def stitch(all_segments):
    if not all_segments:
        return []

    all_segments.sort(key=lambda s: s["start"])

    merged = [all_segments[0]]
    for seg in all_segments[1:]:
        prev = merged[-1]

        is_overlap = seg["start"] < prev["end"] - 0.5
        is_near_dupe = (
            seg["start"] < prev["end"] + 1.0
            and text_similarity(seg["text"], prev["text"]) >= 0.6
        )

        if is_overlap or is_near_dupe:
            if seg.get("avg_logprob", -1) > prev.get("avg_logprob", -1):
                merged[-1] = seg
            continue

        merged.append(seg)

    cleaned = []
    repeat_count = 0
    prev_text = None
    for seg in merged:
        sim = text_similarity(seg["text"], prev_text) if prev_text else 0
        if sim >= 0.6:
            repeat_count += 1
            if repeat_count >= 2:
                continue
        else:
            repeat_count = 0
        prev_text = seg["text"]
        cleaned.append(seg)

    return cleaned


def merge_fragments(segments, min_duration=3.0, max_gap=1.5):
    """Join very short adjacent segments into natural sentence-length chunks."""
    if not segments:
        return segments

    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg["start"] - prev["end"]
        prev_dur = prev["end"] - prev["start"]

        if prev_dur < min_duration and gap <= max_gap:
            prev["end"] = seg["end"]
            prev["text"] = prev["text"] + " " + seg["text"]
            prev["avg_logprob"] = min(prev.get("avg_logprob", -1), seg.get("avg_logprob", -1))
            prev["no_speech_prob"] = max(prev.get("no_speech_prob", 0), seg.get("no_speech_prob", 0))
        else:
            merged.append(dict(seg))

    return merged


def filter_hallucinations(segments, similarity_thresh=0.5, min_run=3):
    """Collapse runs of near-identical segments (Whisper hallucination artifacts)."""
    if len(segments) < min_run:
        return segments

    runs = []
    current_run = [0]
    for i in range(1, len(segments)):
        if text_similarity(segments[i]["text"], segments[i - 1]["text"]) >= similarity_thresh:
            current_run.append(i)
        else:
            runs.append(current_run)
            current_run = [i]
    runs.append(current_run)

    result = []
    for run in runs:
        if len(run) >= min_run:
            best = max(run, key=lambda i: segments[i].get("avg_logprob", -1))
            result.append(segments[best])
        else:
            result.extend(segments[i] for i in run)

    return result


def main():
    p = argparse.ArgumentParser(description="Transcribe long audio with Whisper large-v3")
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("-o", "--output", help="Output transcript path (default: <name>_transcript.txt)")
    p.add_argument("--model", default=DEFAULTS["model"], help=f"Whisper model (default: {DEFAULTS['model']})")
    p.add_argument("--device-index", type=int, default=DEFAULTS["device_index"], help="CUDA device index")
    p.add_argument("--chunk-size", type=int, default=DEFAULTS["chunk_size"], help=f"Chunk duration in seconds (default: {DEFAULTS['chunk_size']})")
    p.add_argument("--overlap", type=int, default=DEFAULTS["overlap"], help=f"Overlap in seconds (default: {DEFAULTS['overlap']})")
    args = p.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: {args.audio} not found", file=sys.stderr)
        sys.exit(1)

    output = args.output or os.path.splitext(os.path.basename(args.audio))[0] + "_transcript.txt"

    duration = get_duration(args.audio)
    n_chunks = max(1, int((duration - 1) // args.chunk_size) + 1)

    print(f"Audio    : {args.audio}")
    print(f"Duration : {fmt_ts(duration)}")
    print(f"Model    : {args.model}")
    print(f"Chunks   : {n_chunks} x {args.chunk_size}s (overlap {args.overlap}s)")
    print(f"Output   : {output}")

    chunk_dir = tempfile.mkdtemp(prefix="whisper_chunks_")

    try:
        # --- Step 1: chunk ---
        print(f"\n[1/3] Splitting audio ...")
        chunks = chunk_audio(args.audio, chunk_dir, args.chunk_size, args.overlap)
        print(f"      {len(chunks)} chunks created")

        # --- Step 2: transcribe ---
        print(f"\n[2/3] Loading {args.model} ...")
        model = WhisperModel(
            args.model,
            device=DEFAULTS["device"],
            device_index=args.device_index,
            compute_type=DEFAULTS["compute_type"],
        )

        all_segments = []
        for i, (chunk_path, offset, end) in enumerate(chunks):
            label = f"[{fmt_ts(offset)} – {fmt_ts(end)}]"
            print(f"      Chunk {i+1}/{len(chunks)} {label} ...", end=" ", flush=True)
            segs = transcribe_chunk(model, chunk_path, offset)
            all_segments.extend(segs)
            print(f"{len(segs)} segments")

        # --- Step 3: stitch & write ---
        print(f"\n[3/3] Stitching {len(all_segments)} raw segments ...")
        stitched = stitch(all_segments)
        print(f"      {len(stitched)} segments after merge")
        dehallucinated = filter_hallucinations(stitched)
        if len(dehallucinated) < len(stitched):
            print(f"      {len(dehallucinated)} segments after hallucination filter")
        final = merge_fragments(dehallucinated)
        if len(final) < len(dehallucinated):
            print(f"      {len(final)} segments after fragment merge")

        lines = []
        for seg in final:
            lines.append(f"[{fmt_ts(seg['start'])} – {fmt_ts(seg['end'])}]  {seg['text']}")

        with open(output, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\nDone — {len(final)} segments written to {output}")

    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
