"""
NCC CRM — Face Recognition Service (Stateless — no DB required)
Deploy anywhere: Render.com, Railway.app, etc.
Run locally: uvicorn main:app --host 0.0.0.0 --port 8001 --reload
"""

import os, json, base64, io, logging, urllib.request
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Model paths ───────────────────────────────────────────────────────────────
MODELS_DIR      = Path(__file__).parent / "models"
DETECTOR_PATH   = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

DETECTOR_URL   = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
RECOGNIZER_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

CONFIDENCE_THRESHOLD  = float(os.getenv("FACE_THRESHOLD",       "0.50"))
MIN_FACE_AREA_RATIO   = float(os.getenv("MIN_FACE_AREA",         "0.04"))   # face must be ≥4% of frame area
MIN_LIVENESS_MOTION   = float(os.getenv("MIN_LIVENESS_MOTION",   "1.5"))    # mean abs pixel diff threshold
MIN_LIVENESS_FRAMES   = int(os.getenv(  "MIN_LIVENESS_FRAMES",   "3"))      # minimum frames required

# ── Download models on startup ────────────────────────────────────────────────
def download_model(url: str, dest: Path):
    if dest.exists():
        return
    logger.info(f"Downloading {dest.name} ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    logger.info(f"Downloaded {dest.name} ({dest.stat().st_size // 1024} KB)")

download_model(DETECTOR_URL,   DETECTOR_PATH)
download_model(RECOGNIZER_URL, RECOGNIZER_PATH)

# ── Load OpenCV models ────────────────────────────────────────────────────────
_detector   = cv2.FaceDetectorYN.create(str(DETECTOR_PATH),   "", (320, 320), score_threshold=0.6)
_recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_PATH), "")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="NCC Face Recognition Service", version="3.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Helpers ───────────────────────────────────────────────────────────────────
def bytes_to_bgr(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def b64_to_bgr(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return bytes_to_bgr(base64.b64decode(b64_str))

def detect_faces(bgr: np.ndarray):
    h, w = bgr.shape[:2]
    _detector.setInputSize((w, h))
    _, faces = _detector.detect(bgr)
    return faces if faces is not None else []

def get_embedding(bgr: np.ndarray, face_box) -> np.ndarray:
    return _recognizer.feature(_recognizer.alignCrop(bgr, face_box)).flatten()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def check_liveness_frames(frames_b64: list) -> tuple:
    """
    Compare consecutive frames for pixel-level motion.
    A static photo/screen has near-zero frame difference; a live face has visible motion
    from breathing, micro-movements, and eye activity.
    Returns (is_live: bool, message: str).
    """
    if not frames_b64:
        return False, "Liveness frames missing. Please complete the face scan from the attendance page."

    if len(frames_b64) < MIN_LIVENESS_FRAMES:
        return False, f"Insufficient liveness data ({len(frames_b64)} frames). Please repeat the face scan."

    arrays = []
    for b64 in frames_b64[:8]:
        try:
            small = cv2.resize(b64_to_bgr(b64), (80, 60))
            arrays.append(small.astype(np.float32).flatten())
        except Exception:
            pass

    if len(arrays) < 2:
        return False, "Could not process liveness frames. Please try again."

    diffs = [float(np.mean(np.abs(arrays[i + 1] - arrays[i]))) for i in range(len(arrays) - 1)]
    avg_motion = float(np.mean(diffs))
    logger.info(f"Liveness motion score: {avg_motion:.3f} (threshold: {MIN_LIVENESS_MOTION})")

    if avg_motion < MIN_LIVENESS_MOTION:
        return False, "Anti-spoofing check failed — static image detected. Please use a live camera."

    return True, "ok"

# ── Pydantic models ───────────────────────────────────────────────────────────
class KnownFace(BaseModel):
    student_id: int
    encoding:   List[float]

class RecognizeRequest(BaseModel):
    image_base64:    str
    known_faces:     List[KnownFace]        # Laravel passes encodings — no DB needed
    liveness_frames: List[str]   = []       # JPEG frames from client liveness check
    latitude:        Optional[float] = None
    longitude:       Optional[float] = None

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "NCC Face Recognition", "version": "3.1.0"}

@app.get("/health")
def health():
    return {
        "status":     "healthy",
        "detector":   DETECTOR_PATH.exists(),
        "recognizer": RECOGNIZER_PATH.exists(),
    }

@app.post("/api/encode-face")
async def encode_face(image: UploadFile = File(...)):
    """Detect face and return 128-d embedding. Laravel stores it in DB."""
    try:
        bgr   = bytes_to_bgr(await image.read())
        faces = detect_faces(bgr)

        if len(faces) == 0:
            raise HTTPException(422, "No face detected in the image")
        if len(faces) > 1:
            raise HTTPException(422, "Multiple faces detected. Use a single-person photo")

        return {
            "success":  True,
            "encoding": get_embedding(bgr, faces[0]).tolist(),
            "faces":    len(faces),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"encode_face: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/recognize-face")
async def recognize_face(req: RecognizeRequest):
    """
    Match frame against known_faces supplied by Laravel.
    Laravel fetches encodings from DB and sends them here — no DB access needed.
    Security: rejects multiple faces, tiny faces (photo-of-photo), low confidence.
    """
    try:
        # Server-side liveness: verify frame-to-frame motion to reject static photos/screens
        is_live, lv_msg = check_liveness_frames(req.liveness_frames)
        if not is_live:
            return {"success": False, "message": lv_msg}

        bgr   = b64_to_bgr(req.image_base64)
        h, w  = bgr.shape[:2]
        faces = detect_faces(bgr)

        if len(faces) == 0:
            return {"success": False, "message": "No face detected. Please look directly at the camera."}

        if len(faces) > 1:
            return {"success": False, "message": "Multiple faces detected. Only one person should be in frame."}

        # Reject tiny faces — likely a photo of a photo held up to the camera
        face      = faces[0]
        face_area = float(face[2]) * float(face[3])   # width × height from YuNet box
        img_area  = float(w) * float(h)
        if img_area > 0 and (face_area / img_area) < MIN_FACE_AREA_RATIO:
            return {"success": False, "message": "Face too small or too far from camera. Please move closer."}

        if not req.known_faces:
            return {"success": False, "message": "No registered faces in system"}

        query_emb    = get_embedding(bgr, face)
        known_embs   = [np.array(kf.encoding, dtype=np.float32) for kf in req.known_faces]
        similarities = [cosine_similarity(query_emb, k) for k in known_embs]
        best_idx     = int(np.argmax(similarities))
        best_sim     = similarities[best_idx]

        if best_sim < CONFIDENCE_THRESHOLD:
            return {
                "success":    False,
                "message":    f"Face not recognized (confidence {round(best_sim*100)}%). Please try again in better lighting.",
                "confidence": round(best_sim, 3),
            }

        return {
            "success":    True,
            "student_id": req.known_faces[best_idx].student_id,
            "confidence": round(best_sim, 3),
        }
    except Exception as e:
        logger.error(f"recognize_face: {e}")
        raise HTTPException(500, str(e))
