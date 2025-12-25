from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import io
import re
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone

import PyPDF2
from docx import Document

# ---------------- ENV SETUP ----------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("resumate")

# ---------------- MONGODB (OPTIONAL) ----------------
db = None
client = None

mongo_url = os.environ.get("MONGO_URL")
db_name = os.environ.get("DB_NAME")

if mongo_url and db_name:
    try:
        client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=2000)
        db = client[db_name]
        logger.info("MongoDB configured")
    except Exception as e:
        logger.warning(f"MongoDB unavailable, running without DB: {e}")
        db = None
else:
    logger.info("MongoDB env vars not set, running without DB")

# ---------------- APP ----------------
app = FastAPI()
api_router = APIRouter(prefix="/api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODELS ----------------
class ResumeAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score: int
    strengths: List[str]
    missing_sections: List[str]
    improvement_tips: List[str]
    target_role: str
    filename: str
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# ---------------- ANALYSIS LOGIC ----------------
REQUIRED_SECTIONS = ["education", "skills", "projects", "experience"]

def extract_text_from_pdf(b: bytes) -> str:
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(b))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def extract_text_from_docx(b: bytes) -> str:
    try:
        doc = Document(io.BytesIO(b))
        return "\n".join(p.text for p in doc.paragraphs)
    except:
        return ""

def analyze_resume(text: str) -> dict:
    text_l = text.lower()
    found = [s for s in REQUIRED_SECTIONS if s in text_l]
    missing = [s for s in REQUIRED_SECTIONS if s not in found]

    score = min(100, 40 + len(found) * 15)

    strengths = []
    if found:
        strengths.append("Resume contains core sections")
    if "linkedin" in text_l:
        strengths.append("LinkedIn profile included")

    tips = []
    if missing:
        tips.append(f"Add missing sections: {', '.join(missing)}")
    tips += [
        "Quantify achievements",
        "Use action verbs",
        "Keep resume to 1–2 pages",
        "Tailor resume for the role"
    ]

    return {
        "score": score,
        "strengths": strengths,
        "missing_sections": missing,
        "improvement_tips": tips[:5]
    }

# ---------------- ROUTES ----------------
@api_router.get("/")
async def root():
    return {"message": "Resumate API running"}

@api_router.get("/health")
async def health():
    return {"status": "ok"}

@api_router.post("/analyze", response_model=ResumeAnalysisResult)
async def analyze(
    file: UploadFile = File(...),
    target_role: str = "web_developer"
):
    filename = file.filename.lower()

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Upload PDF or DOCX only")

    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, "File too large")

    if filename.endswith(".pdf"):
        text = extract_text_from_pdf(data)
    else:
        text = extract_text_from_docx(data)

    if len(text.strip()) < 50:
        raise HTTPException(400, "Could not extract text")

    analysis = analyze_resume(text)

    result = ResumeAnalysisResult(
        score=analysis["score"],
        strengths=analysis["strengths"],
        missing_sections=analysis["missing_sections"],
        improvement_tips=analysis["improvement_tips"],
        target_role=target_role,
        filename=file.filename
    )

    if db:
        await db.resume_analyses.insert_one(result.model_dump())

    return result

# ---------------- FINALIZE ----------------
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown():
    if client:
        client.close()
