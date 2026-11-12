# Voice Conversation Analyzer (Gemini + FastAPI)

Record a conversation in the browser or upload an audio file. **Gemini on Vertex AI** returns a **timestamped, speaker-labelled transcript** and a **sentiment analysis** of each speaker. **Cloud Text-to-Speech** then reads the analysis back to you.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-backend-009688) ![Vertex AI](https://img.shields.io/badge/Vertex%20AI-Gemini-4285F4) ![App Engine](https://img.shields.io/badge/deploy-App%20Engine-orange)

## What it does

1. **Capture.** Record from the microphone (`MediaRecorder`) or upload mp3 / wav / m4a / ogg / flac / webm.
2. **Understand.** The audio is sent to Gemini as a multimodal input with `audio_timestamp=True`. A structured prompt asks for:
   - a transcription with `[HH:MM:SS]` timestamps and speaker diarization (Speaker A, B, …)
   - the overall tone, each speaker's emotional state, and key emotional moments
3. **Respond.** The analysis is cleaned of markdown and spoken with a Neural2 voice, so you get both a readable and an audible summary.

```
Browser ──audio──► FastAPI /upload or /analyze
                      │  format check (WebM/M4A/MP3/WAV sent natively; others → MP3 via pydub)
                      ▼
               Vertex AI · Gemini  ──►  transcript + sentiment
                      │
                      ▼
               Cloud Text-to-Speech ──►  /get_audio/{id}.mp3
```

## Run locally

**Prerequisites:** Python 3.10+, a Google Cloud project with the **Vertex AI API** and **Cloud Text-to-Speech API** enabled. ffmpeg is optional and only needed for unusual audio formats.

```bash
git clone https://github.com/vamshiteja999/Project3_kaspe.git
cd Project3_kaspe
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

gcloud auth application-default login                # local credentials, no key file
export GOOGLE_CLOUD_PROJECT=your-project-id

uvicorn main:app --reload
```

Open http://127.0.0.1:8000. Interactive API docs are at `/docs`.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | `voicellapp` | GCP project for Vertex AI |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Any audio-capable Gemini model |
| `MAX_UPLOAD_MB` | `20` | Upload size limit |

## Deploy to App Engine

```bash
gcloud app deploy
```

The app uses the App Engine default service account, so no key file is uploaded. Give that account the **Vertex AI User** role.

## API

| Method | Endpoint | Body | Returns |
|---|---|---|---|
| POST | `/upload` | multipart `audio` (browser recording) | `{transcription, sentiment, audio_url}` |
| POST | `/analyze` | multipart `file` (uploaded audio) | same as above |
| GET | `/get_audio/{id}` | – | MP3 of the spoken analysis |
| GET | `/health` | – | `{status, model}` |

## Project structure

```
main.py               FastAPI app: audio handling, Gemini, TTS
templates/index.html  Frontend: recorder, file upload, results
app.yaml              App Engine config
requirements.txt
```

## Design notes

- **Native audio input.** Gemini accepts WebM and M4A directly, so browser recordings need no transcoding (and no ffmpeg on App Engine).
- **Each request gets its own audio file,** so concurrent users don't overwrite each other's responses.
- **Graceful degradation.** If Text-to-Speech fails, the text results are still returned.
- **No secrets in the repo.** Credentials come from Application Default Credentials.
