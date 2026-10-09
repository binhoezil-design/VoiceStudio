import os
import imageio_ffmpeg
os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ.get("PATH", "")
import os
import tempfile
import mlx_whisper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

app = FastAPI(title="VoiceStudio MLX Whisper Server")

@app.get("/health")
async def health():
    return {"status": "ok", "message": "VoiceStudio STT running on MLX Whisper"}

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("mlx-community/whisper-large-v3-turbo"),
    language: str = Form(None),
    response_format: str = Form("json"),
    word_timestamps: bool = Form(False),
):
    tmp_path = None
    try:
        # Save uploaded file temporarily
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe using mlx_whisper
        result = mlx_whisper.transcribe(
            tmp_path,
            path_or_hf_repo=model,
            language=language,
            word_timestamps=word_timestamps,
        )
        
        # Cleanup
        if response_format == "verbose_json" or word_timestamps:
            segments = result.get("segments") or []
            return {
                "text": result.get("text", ""),
                "language": result.get("language", language),
                "segments": segments,
                "words": [word for segment in segments for word in (segment.get("words") or [])],
            }
        return {"text": result.get("text", "")}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/debug")
async def debug():
    import os
    import shutil
    return {"path": os.environ.get("PATH"), "ffmpeg": shutil.which("ffmpeg")}
