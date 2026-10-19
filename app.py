"""Web UI for the voice assistant: record from the mic or upload an audio file."""
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

from pipeline import run_pipeline

BASE = Path(__file__).parent
OUT_DIR = BASE / "outputs"
OUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

HISTORY = []  # simple in-memory conversation memory


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/voice", methods=["POST"])
def voice():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "No audio received."}), 400
    suffix = Path(f.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name
    try:
        result = run_pipeline(tmp_path, str(OUT_DIR), history=HISTORY[-10:])
    finally:
        os.remove(tmp_path)

    if result["transcript"] and not result["fallback"]:
        HISTORY.append({"role": "user", "content": result["transcript"]})
        HISTORY.append({"role": "assistant", "content": result["reply"]})
    if result["audio_file"]:
        result["audio_url"] = f"/outputs/{result['audio_file']}"
    return jsonify(result)


@app.route("/api/reset", methods=["POST"])
def reset():
    HISTORY.clear()
    return jsonify({"ok": True})


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUT_DIR, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
