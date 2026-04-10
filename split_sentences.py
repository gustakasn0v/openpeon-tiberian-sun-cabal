#!/usr/bin/env python3
"""
split_sentences.py

Re-transcribe all files in sounds_new/ with word-level timestamps.
Save enriched JSON, then split any file containing multiple sentences
into individual clips in sounds_new_2/.

A "sentence" is a group of consecutive words ending with a word that
contains terminal punctuation (. ! ?). Short gaps between sentences
(>= GAP_THRESHOLD seconds) also trigger a split even mid-sentence.

Usage:
    python3 split_sentences.py
"""

import json, os, re, subprocess, sys

FFMPEG       = "/nail/home/gus/ffmpeg-master-latest-linux64-lgpl/bin/ffmpeg"
INPUT_DIR    = "sounds_new"
OUTPUT_DIR   = "sounds_new_2"
JSON_OUT     = "sounds_new_word_timestamps.json"
TAIL_PAD_MS  = 80   # ms added to end of each clip
SAMPLE_RATE  = 44100
CHANNELS     = 2
# Minimum inter-word gap (seconds) that counts as a sentence boundary
# even if the preceding word didn't end with terminal punctuation
GAP_THRESHOLD = 0.4


def sanitize(text, max_len=60):
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s[:max_len] if s else ""


def ms_to_ts(ms):
    h  = ms // 3_600_000; ms %= 3_600_000
    m  = ms // 60_000;    ms %= 60_000
    s  = ms // 1_000;     ms %= 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def extract_clip(src, start_ms, end_ms, dst):
    cmd = [
        FFMPEG, "-y",
        "-ss", ms_to_ts(start_ms),
        "-to", ms_to_ts(end_ms + TAIL_PAD_MS),
        "-i",  src,
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-f",  "wav",
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"  WARNING ffmpeg failed for {dst}")
        return False
    return True


def group_into_sentences(words):
    """
    Split a flat word list into sentences.
    A sentence ends when:
      - a word ends with terminal punctuation (. ! ?)
      - OR the gap to the next word is >= GAP_THRESHOLD seconds
    Returns list of word-lists, one per sentence.
    """
    sentences = []
    current = []
    for i, w in enumerate(words):
        current.append(w)
        is_terminal = bool(re.search(r"[.!?]$", w["word"].strip()))
        gap_after = 0.0
        if i + 1 < len(words):
            gap_after = words[i + 1]["start"] - w["end"]
        is_last = (i == len(words) - 1)

        if is_terminal or gap_after >= GAP_THRESHOLD or is_last:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def main():
    # Add ffmpeg dir to PATH so Whisper's internal calls work
    ffmpeg_dir = os.path.dirname(FFMPEG)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    import whisper
    model = whisper.load_model("base")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".WAV"))
    all_data = {}
    counts = {}   # for collision-safe filenames in output dir

    for fname in wav_files:
        src_path = os.path.join(INPUT_DIR, fname)
        print(f"\n--- {fname} ---")

        result = model.transcribe(src_path, word_timestamps=True, verbose=False)

        # Flatten all words across all segments
        all_words = []
        for seg in result["segments"]:
            all_words.extend(seg.get("words", []))

        full_text = result["text"].strip()
        all_data[fname] = {
            "text": full_text,
            "segments": result["segments"],
        }

        sentences = group_into_sentences(all_words)

        if len(sentences) <= 1:
            # Single sentence — copy as-is
            stem = sanitize(full_text) or os.path.splitext(fname)[0]
            counts[stem] = counts.get(stem, 0) + 1
            if counts[stem] > 1:
                stem = f"{stem}_{counts[stem]:02d}"
            out_name = stem + ".WAV"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            start_ms = int(all_words[0]["start"] * 1000) if all_words else 0
            end_ms   = int(all_words[-1]["end"]   * 1000) if all_words else 0
            ok = extract_clip(src_path, start_ms, end_ms, out_path)
            label = "  (single)" if ok else "  (FAILED)"
            print(f"  -> {out_name}{label}")
            print(f"     \"{full_text}\"")
        else:
            print(f"  {len(sentences)} sentences detected, splitting:")
            for sent_words in sentences:
                sent_text = "".join(w["word"] for w in sent_words).strip()
                # Strip trailing punctuation for filename
                stem = sanitize(re.sub(r"[.!?,;]+$", "", sent_text))
                if not stem:
                    stem = "clip"
                counts[stem] = counts.get(stem, 0) + 1
                if counts[stem] > 1:
                    stem = f"{stem}_{counts[stem]:02d}"
                out_name = stem + ".WAV"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                start_ms = int(sent_words[0]["start"] * 1000)
                end_ms   = int(sent_words[-1]["end"]   * 1000)
                ok = extract_clip(src_path, start_ms, end_ms, out_path)
                status = "OK" if ok else "FAILED"
                print(f"  [{status}] {out_name}")
                print(f"       \"{sent_text}\"  ({sent_words[0]['start']:.2f}s - {sent_words[-1]['end']:.2f}s)")

    with open(JSON_OUT, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nEnriched JSON saved to {JSON_OUT}")

    out_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".WAV")]
    print(f"Total output files in {OUTPUT_DIR}/: {len(out_files)}")


if __name__ == "__main__":
    main()
