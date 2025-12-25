from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import logging
import io
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List
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

print("🚀 Resumate backend running in REAL SCORING MODE")

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

# ---------------- ROLE MAP ----------------
ROLE_REQUIREMENTS = {
    "ai_ml_intern": {
        "mandatory": ["python", "machine learning"],
        "optional": ["numpy", "pandas", "scikit", "tensorflow", "pytorch"],
        "projects": ["model", "classification", "prediction", "training"]
    },
    "web_developer": {
        "mandatory": ["html", "css", "javascript"],
        "optional": ["react", "api", "node", "tailwind"],
        "projects": ["website", "web app", "frontend", "backend"]
    },
    "data_analyst": {
        "mandatory": ["sql", "python"],
        "optional": ["excel", "pandas", "tableau", "power bi"],
        "projects": ["analysis", "dashboard", "visualization"]
    },
    "cloud_devops": {
        "mandatory": ["linux", "cloud"],
        "optional": ["aws", "docker", "ci/cd", "kubernetes"],
        "projects": ["deployment", "pipeline", "server"]
    }
}

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(b: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(b))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def extract_text_from_docx(b: bytes) -> str:
    doc = Document(io.BytesIO(b))
    return "\n".join(p.text for p in doc.paragraphs)

# ---------------- REAL ANALYSIS ENGINE ----------------
def analyze_resume(text: str, target_role: str) -> dict:
    text_l = text.lower()
    role = ROLE_REQUIREMENTS.get(target_role)

    if not role:
        return {
            "score": 30,
            "strengths": [],
            "missing_sections": [],
            "improvement_tips": ["Target role not supported yet"]
        }

    mandatory = role["mandatory"]
    optional = role["optional"]
    projects = role["projects"]

    found_mandatory = [s for s in mandatory if s in text_l]
    found_optional = [s for s in optional if s in text_l]
    found_projects = [p for p in projects if p in text_l]

    missing_mandatory = [s for s in mandatory if s not in text_l]

    score = 0
    strengths = []
    tips = []

    # -------- Mandatory skills (45) --------
    mandatory_score = int((len(found_mandatory) / len(mandatory)) * 45)
    score += mandatory_score

    if len(found_mandatory) == len(mandatory):
        strengths.append("All mandatory role skills detected")
    else:
        tips.append(f"Missing mandatory skills: {', '.join(missing_mandatory)}")

    # -------- Optional skills (20) --------
    score += min(20, len(found_optional) * 4)
    if found_optional:
        strengths.append("Relevant supporting skills found")

    # -------- Projects (20) --------
    project_score = min(20, len(found_projects) * 7)
    score += project_score

    if found_projects:
        strengths.append("Role-aligned projects detected")
    else:
        tips.append("No role-specific projects found")

    # -------- Resume depth (10) --------
    if len(text) > 1000:
        score += 10
    elif len(text) > 600:
        score += 6
    else:
        tips.append("Resume content is too thin")

    # -------- HARD PENALTIES --------
    if not found_mandatory:
        score = min(score, 45)
        tips.append("Resume not aligned with selected role")

    if not found_projects:
        score -= 10

    score = max(5, min(100, score))

    if score < 50:
        tips.append("Low alignment with target role")

    return {
        "score": score,
        "strengths": strengths,
        "missing_sections": missing_mandatory,
        "improvement_tips": tips[:5]
    }

# ---------------- ROUTES ----------------
@api_router.get("/health")
async def health():
    return {"status": "ok"}

@api_router.post("/analyze", response_model=ResumeAnalysisResult)
async def analyze(file: UploadFile = File(...), target_role: str = "web_developer"):
    filename = file.filename.lower()

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Upload PDF or DOCX only")

    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(400, "File too large")

    text = extract_text_from_pdf(data) if filename.endswith(".pdf") else extract_text_from_docx(data)

    if len(text.strip()) < 80:
        raise HTTPException(400, "Unable to extract meaningful content")

    analysis = analyze_resume(text, target_role)

    return ResumeAnalysisResult(
        score=analysis["score"],
        strengths=analysis["strengths"],
        missing_sections=analysis["missing_sections"],
        improvement_tips=analysis["improvement_tips"],
        target_role=target_role,
        filename=file.filename
    )

app.include_router(api_router)
