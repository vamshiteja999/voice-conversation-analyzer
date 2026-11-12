"""Voice conversation analyzer.

Record or upload audio -> Gemini (Vertex AI) produces a timestamped,
speaker-labelled transcript plus a sentiment analysis -> Cloud Text-to-Speech
reads the analysis back.
"""

import io
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai
from google.cloud import texttospeech
from google.genai import types
from pydub import AudioSegment
from starlette.requests import Request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration (all overridable with environment variables) -------------
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "voicellapp")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))

# Credentials: GOOGLE_APPLICATION_CREDENTIALS env var, or
# `gcloud auth application-default login`, or the App Engine service account.
# Never commit a key file to the repository.

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = Path(tempfile.gettempdir()) / "voice-analyzer"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Voice Conversation Analyzer")
if (BASE_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
tts_client = texttospeech.TextToSpeechClient()

ANALYSIS_PROMPT = """
Please analyze this audio and provide:
1. Detailed transcription with:
   - Timestamps in [HH:MM:SS] format
   - Speaker identification (Speaker A, B, etc.)
2. Sentiment analysis for:
   - Overall conversation tone
   - Each speaker's emotional state
   - Key emotional moments
Format the response exactly as:
Transcription:
[timestamp] Speaker: text
Sentiment Analysis:
Overall Tone:
Speaker Analysis:
Key Emotional Moments:
"""

# Formats Gemini accepts directly; anything else is converted to MP3 first.
NATIVE_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "aiff": "audio/aiff",
    "webm": "audio/webm",  # browser MediaRecorder default (Chrome/Firefox)
    "m4a": "audio/m4a",    # Safari recordings, iPhone voice memos
    "mp4": "audio/mp4",
    "opus": "audio/opus",
}


def detect_format(upload: UploadFile) -> str:
    """Best-effort audio format from filename, then content type."""
    if upload.filename and "." in upload.filename:
        return upload.filename.rsplit(".", 1)[-1].lower()
    ctype = (upload.content_type or "").split(";")[0]
    return {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mp4": "m4a",
    }.get(ctype, "webm")


def prepare_audio(data: bytes, fmt: str) -> tuple[bytes, str]:
    """Return (audio_bytes, mime_type) in a format Gemini accepts.

    Browser recordings (WebM/Opus) and other formats are converted to mono
    MP3 with pydub, which requires ffmpeg to be installed.
    """
    if fmt in NATIVE_MIME:
        return data, NATIVE_MIME[fmt]
    try:
        audio = AudioSegment.from_file(io.BytesIO(data), format=fmt)
        audio = audio.set_channels(1).set_frame_rate(44100)
        buf = io.BytesIO()
        audio.export(buf, format="mp3", parameters=["-q:a", "2"])
        return buf.getvalue(), "audio/mpeg"
    except Exception as exc:  # pydub raises several exception types
        logger.exception("Audio conversion failed")
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the audio file ({fmt}). Is ffmpeg installed?",
        ) from exc


def analyze_audio(audio: bytes, mime_type: str) -> str:
    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=[types.Part.from_bytes(data=audio, mime_type=mime_type), ANALYSIS_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_k=40,
            top_p=0.95,
            audio_timestamp=True,
        ),
    )
    if not response.text:
        raise HTTPException(status_code=502, detail="The model returned an empty response")
    return response.text


def split_analysis(text: str) -> tuple[str, str]:
    parts = text.split("Sentiment Analysis:", 1)
    transcription = parts[0].replace("Transcription:", "", 1).strip()
    sentiment = parts[1].strip() if len(parts) > 1 else ""
    return transcription, sentiment


def clean_for_speech(text: str) -> str:
    """Strip markdown so the voice doesn't read out symbols."""
    for token in ("**", "##", "*", "#", "[", "]"):
        text = text.replace(token, "")
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def synthesize(text: str) -> str:
    """Speak the sentiment summary; returns an audio id for /get_audio/{id}."""
    response = tts_client.synthesize_speech(
        input=texttospeech.SynthesisInput(text=clean_for_speech(text)[:4500]),
        voice=texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.95,
        ),
    )
    audio_id = uuid.uuid4().hex
    (AUDIO_DIR / f"{audio_id}.mp3").write_bytes(response.audio_content)
    return audio_id


async def handle_upload(upload: UploadFile) -> dict:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_MB} MB")

    audio, mime = prepare_audio(data, detect_format(upload))
    try:
        analysis = analyze_audio(audio, mime)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini analysis failed")
        raise HTTPException(status_code=502, detail=f"Analysis failed: {exc}") from exc

    transcription, sentiment = split_analysis(analysis)

    audio_url = None
    try:
        audio_url = f"/get_audio/{synthesize(sentiment or analysis)}"
    except Exception:
        # Text results are still useful if speech synthesis fails.
        logger.exception("Text-to-speech failed")

    return {
        "success": True,
        "transcription": transcription,
        "sentiment": sentiment,
        "audio_url": audio_url,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"model": MODEL_NAME})


@app.post("/upload")
async def upload_recording(audio: UploadFile = File(...)):
    """Audio recorded in the browser (WebM/Opus by default)."""
    return await handle_upload(audio)


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """An uploaded audio file (mp3, wav, m4a, ogg, flac, webm...)."""
    return await handle_upload(file)


@app.get("/get_audio/{audio_id}")
async def get_audio(audio_id: str):
    if not audio_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid audio id")
    path = AUDIO_DIR / f"{audio_id}.mp3"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(path, media_type="audio/mpeg", filename="analysis.mp3")


@app.get("/health")
async def health():
    return {"status": "OK", "model": MODEL_NAME}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
