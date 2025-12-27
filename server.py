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

class Feedback(BaseModel):
    message: str
    rating: int | None = None
    page: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResumeAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score: int
    skill_match_percentage: int
    project_relevance_percentage: int
    resume_depth_percentage: int
    fit_verdict: str
    strengths: List[str]
    missing_sections: List[str]
    improvement_tips: List[str]
    found_mandatory_skills: List[str]
    found_optional_skills: List[str]
    found_project_indicators: List[str]
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

# ---------------- ANALYSIS HELPERS ----------------
EVIDENCE_WORDS = [
    "experience","worked","developed","built","created","implemented",
    "designed","managed","led","deployed","optimized","automated",
    "project","projects","role","position","job","internship",
    "responsibilities","achieved","delivered","contributed","using",
    "utilized","applied","proficient","expertise","skilled"
]

SECTION_HEADERS = [
    "experience","work experience","employment","skills",
    "technical skills","projects","education","certifications",
    "achievements","summary"
]

def find_skill_with_context(text_lower: str, skill: str, window: int = 100) -> bool:
    import re
    pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
    for match in pattern.finditer(text_lower):
        start = max(0, match.start() - window)
        end = min(len(text_lower), match.end() + window)
        context = text_lower[start:end]
        if any(e in context for e in EVIDENCE_WORDS):
            return True
    return False

def analyze_project_depth(text: str, text_lower: str, keywords: List[str]) -> dict:
    lines = text.split('\n')
    has_project_section = any("project" in l.lower() and len(l.strip()) < 50 for l in lines)
    found = [k for k in keywords if find_skill_with_context(text_lower, k, 150)]
    score = 30 if has_project_section else 0
    score += min(70, len(found) * 20)
    return {"found_projects": found, "has_project_section": has_project_section, "project_detail_score": min(100, score)}

def detect_resume_structure(text: str, text_lower: str) -> dict:
    lines = text.split('\n')
    found = [s for s in SECTION_HEADERS if any(s in l.lower() for l in lines)]
    bullets = sum(1 for l in lines if l.strip().startswith(("-", "•", "*")))
    score = min(100, len(found)*25 + bullets*5)
    return {"found_sections": found, "structure_score": score}

# ---------------- ANALYSIS ENGINE ----------------
def analyze_resume(text: str, target_role: str) -> dict:
    text_l = text.lower()
    role = ROLE_REQUIREMENTS.get(target_role)
    if not role:
        return {"score":30,"skill_match_percentage":0,"project_relevance_percentage":0,
                "resume_depth_percentage":0,"fit_verdict":"Weak Fit",
                "strengths":[],"missing_sections":[],"improvement_tips":[],
                "found_mandatory_skills":[],"found_optional_skills":[],"found_project_indicators":[]}

    mandatory = [s for s in role["mandatory"] if find_skill_with_context(text_l, s)]
    optional = [s for s in role["optional"] if find_skill_with_context(text_l, s)]
    project = analyze_project_depth(text, text_l, role["projects"])
    structure = detect_resume_structure(text, text_l)

    score = min(100, len(mandatory)*20 + len(optional)*5 + project["project_detail_score"]//2 + structure["structure_score"]//4)

    verdict = "Strong Fit" if score>=75 else "Partial Fit" if score>=50 else "Weak Fit"

    return {
        "score": score,
        "skill_match_percentage": int(len(mandatory)/len(role["mandatory"])*100),
        "project_relevance_percentage": project["project_detail_score"],
        "resume_depth_percentage": structure["structure_score"],
        "fit_verdict": verdict,
        "strengths": mandatory,
        "missing_sections": [s for s in role["mandatory"] if s not in mandatory],
        "improvement_tips": [],
        "found_mandatory_skills": mandatory,
        "found_optional_skills": optional,
        "found_project_indicators": project["found_projects"]
    }

# ---------------- ROUTES ----------------
@api_router.get("/health")
async def health():
    return {"status":"ok"}

@api_router.post("/feedback")
async def submit_feedback(feedback: Feedback):
    logger.info(f"Feedback: {feedback.message}")
    return {"status":"success"}

@api_router.post("/analyze", response_model=ResumeAnalysisResult)
async def analyze(file: UploadFile = File(...), target_role: str = "web_developer"):
    data = await file.read()
    text = extract_text_from_pdf(data) if file.filename.endswith(".pdf") else extract_text_from_docx(data)
    analysis = analyze_resume(text, target_role)
    return ResumeAnalysisResult(**analysis, target_role=target_role, filename=file.filename)

app.include_router(api_router)
