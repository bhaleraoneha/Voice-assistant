"""Free voice pipeline: Speech-to-Text -> LLM -> Text-to-Speech.

  STT : Groq (Whisper large-v3-turbo)   - free API key
  LLM : Groq (Llama 3.3 70B)            - same free key
  TTS : edge-tts (free, no key)         - falls back to gTTS if it fails

Each stage is timed separately. Failures degrade gracefully, never crash.
If a model name is retired, the next model in the list is tried automatically.
"""
import asyncio
import csv
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def _models(env_name, defaults):
    """Models to try in order. A model set in .env goes first."""
    custom = os.getenv(env_name)
    return ([custom] if custom else []) + [m for m in defaults if m != custom]


STT_MODELS = _models("STT_MODEL", ["whisper-large-v3-turbo", "whisper-large-v3"])
LLM_MODELS = _models(
    "LLM_MODEL",
    ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "llama-3.1-8b-instant", "openai/gpt-oss-120b"],
)
EDGE_VOICE = os.getenv("EDGE_VOICE", "en-US-AriaNeural")

SYSTEM_PROMPT = (
    "You are a friendly voice assistant. Your answers are spoken aloud, so keep "
    "them short (1-3 sentences), conversational, and free of markdown or lists."
)

FALLBACK_TEXT = (
    "Sorry, I couldn't quite catch that. Could you please say it again, "
    "a little slower and closer to the microphone?"
)

NO_SPEECH_THRESHOLD = 0.6  # Whisper's "this is probably not speech" probability

LOG_FILE = Path(__file__).parent / "latency_log.csv"
_client = None


def get_client():
    global _client
    if _client is None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        _client = Groq(api_key=key)
    return _client


def _call_with_fallback(models, fn):
    """Try each model in turn; short retry when busy; move on if the model is unavailable."""
    errors = []
    for model in models:
        for attempt in range(2):
            try:
                return fn(model)
            except Exception as e:
                msg = str(e).lower()
                if any(c in msg for c in ("429", "503", "overloaded", "rate limit")) and attempt == 0:
                    time.sleep(1.5)
                    continue
                errors.append(f"{model}: {str(e)[:120]}")
                break  # model not found / no access / other error -> next model
    raise RuntimeError("All models failed -> " + " | ".join(errors))


def _field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def transcribe(audio_path: str) -> str:
    data = Path(audio_path).read_bytes()
    name = Path(audio_path).name

    def run(model):
        return get_client().audio.transcriptions.create(
            file=(name, data),
            model=model,
            response_format="verbose_json",
            temperature=0.0,
        )

    resp = _call_with_fallback(STT_MODELS, run)
    text = (_field(resp, "text", "") or "").strip()

    # Whisper tends to invent text for silence/noise; use its own no-speech score to catch that.
    segments = _field(resp, "segments", None) or []
    if segments:
        probs = [_field(s, "no_speech_prob", 0.0) or 0.0 for s in segments]
        if probs and min(probs) > NO_SPEECH_THRESHOLD:
            return ""
    return text


def ask_llm(user_text: str, history=None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history or []
    messages.append({"role": "user", "content": user_text})

    def run(model):
        r = get_client().chat.completions.create(model=model, messages=messages, max_tokens=800)
        text = (r.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("empty reply")  # e.g. reasoning model ran out of tokens -> try next model
        return text

    return _call_with_fallback(LLM_MODELS, run)


def synthesize(text: str, out_path: str) -> str:
    try:
        import edge_tts

        async def _run():
            await edge_tts.Communicate(text, EDGE_VOICE).save(out_path)

        asyncio.run(_run())
        if os.path.getsize(out_path) > 0:
            return out_path
    except Exception:
        pass
    from gtts import gTTS  # fallback

    gTTS(text=text, lang="en").save(out_path)
    return out_path


def _log(row: dict):
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stt_s", "llm_s", "tts_s", "total_s", "fallback", "transcript"])
        if new:
            w.writeheader()
        w.writerow(row)


def run_pipeline(audio_path: str, out_dir: str, history=None) -> dict:
    """Run the full pipeline. Never raises; errors are reported in the result."""
    out = {
        "transcript": "", "reply": "", "audio_file": None,
        "timings": {"stt": 0.0, "llm": 0.0, "tts": 0.0, "total": 0.0},
        "fallback": False, "error": None,
    }
    t_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)

    # 1) Speech to text
    transcript = ""
    t0 = time.perf_counter()
    try:
        transcript = transcribe(audio_path)
    except Exception as e:
        out["error"] = f"STT failed: {e}"
    out["timings"]["stt"] = time.perf_counter() - t0
    out["transcript"] = transcript

    # 2) LLM (or graceful fallback for unclear audio)
    t0 = time.perf_counter()
    if len(transcript) < 2:
        out["fallback"] = True
        reply = FALLBACK_TEXT
    else:
        try:
            reply = ask_llm(transcript, history) or FALLBACK_TEXT
        except Exception as e:
            out["error"] = f"LLM failed: {e}"
            out["fallback"] = True
            reply = "Sorry, I'm having trouble thinking right now. Please try again."
    out["timings"]["llm"] = time.perf_counter() - t0
    out["reply"] = reply

    # 3) Text to speech
    t0 = time.perf_counter()
    try:
        name = f"reply_{uuid.uuid4().hex[:10]}.mp3"
        synthesize(reply, os.path.join(out_dir, name))
        out["audio_file"] = name
    except Exception as e:
        out["error"] = (out["error"] or "") + f" TTS failed: {e}"
    out["timings"]["tts"] = time.perf_counter() - t0

    out["timings"]["total"] = time.perf_counter() - t_start
    out["timings"] = {k: round(v, 2) for k, v in out["timings"].items()}
    try:
        _log({
            "stt_s": out["timings"]["stt"], "llm_s": out["timings"]["llm"],
            "tts_s": out["timings"]["tts"], "total_s": out["timings"]["total"],
            "fallback": out["fallback"], "transcript": transcript[:80],
        })
    except Exception:
        pass
    return out
