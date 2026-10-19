# Voice Assistant (STT → LLM → TTS) — 100% free stack

A complete voice-in, voice-out pipeline:

1. **Speech-to-Text** – Groq-hosted OpenAI Whisper (`whisper-large-v3-turbo`)
2. **LLM** – Llama 3.3 70B on Groq, with short-conversation memory
3. **Text-to-Speech** – `edge-tts` (free, no key), with `gTTS` as automatic fallback

## Demo

▶️ **Demo recording:** _paste your video/GIF link here_

## Features

- Record live from the microphone **or** upload an audio file (web UI)
- Real LLM replies (not an echo); the last few turns are remembered
- Reply converted to audio and auto-played
- **Per-stage latency** (STT / LLM / TTS / total) shown in the UI and logged to `latency_log.csv`; the slowest stage is flagged as the bottleneck
- **Graceful handling of unclear audio**: Whisper's no-speech score detects silence/noise, and API errors never crash the app — the assistant politely asks you to repeat
- If a model is retired or busy, the app automatically tries a backup model
- Command-line mode for batch testing

## Setup

1. Get a free key at https://console.groq.com/keys
2. Then:

```bash
python -m venv venv
# Windows:   venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env      # Mac/Linux: cp .env.example .env
# edit .env and paste your GROQ_API_KEY
```

## Run

```bash
python app.py              # then open http://127.0.0.1:5000
python cli.py sample.wav   # or CLI mode
```

## Latency findings

Fill in with your own numbers from `latency_log.csv`:

| Stage | Typical time |
|-------|-------------|
| STT (Groq Whisper) | 0.3 – 1 s |
| LLM (Groq Llama) | 0.3 – 1 s |
| TTS (edge-tts) | 1 – 2 s |

Groq's STT and LLM are very fast, so TTS (which generates the full mp3 before playback)
is usually the bottleneck. Streaming the audio would be the next improvement.

## Structure

```
app.py            Flask server + API
pipeline.py       STT -> LLM -> TTS, timing, error handling, model fallback
cli.py            Command-line runner
templates/        Web UI (records mic, converts to WAV in-browser)
```
