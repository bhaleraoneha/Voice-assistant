"""Command-line version: python cli.py path/to/audio.wav [more files...]"""
import sys
from pathlib import Path

from pipeline import run_pipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python cli.py <audio_file> [<audio_file> ...]")
        sys.exit(1)
    out_dir = Path(__file__).parent / "outputs"
    history = []
    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        if not Path(path).exists():
            print("File not found, skipping.")
            continue
        r = run_pipeline(path, str(out_dir), history=history)
        print(f"Transcript : {r['transcript'] or '(nothing recognised)'}")
        print(f"Reply      : {r['reply']}")
        print(f"Audio file : {out_dir / r['audio_file'] if r['audio_file'] else 'none'}")
        t = r["timings"]
        slowest = max(("stt", "llm", "tts"), key=lambda k: t[k])
        print(f"Latency    : STT {t['stt']}s | LLM {t['llm']}s | TTS {t['tts']}s | TOTAL {t['total']}s")
        print(f"Bottleneck : {slowest.upper()}")
        if r["fallback"]:
            print("Note       : unclear audio -> graceful fallback reply used")
        if r["error"]:
            print(f"Warning    : {r['error']}")
        if r["transcript"] and not r["fallback"]:
            history += [{"role": "user", "content": r["transcript"]},
                        {"role": "assistant", "content": r["reply"]}]


if __name__ == "__main__":
    main()
