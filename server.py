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


class Feedback(BaseModel):
    message: str
    rating: int | None = None
    page: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

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
        if any(e in text_lower[start:end] for e in EVIDENCE_WORDS):
            return True
    return False

def analyze_project_depth(text: str, text_lower: str, project_keywords: List[str]) -> dict:
    lines = text.split("\n")
    has_project_section = any("project" in l.lower() and len(l.strip()) < 50 for l in lines)

    found_projects = [
        k for k in project_keywords if find_skill_with_context(text_lower, k, 150)
    ]

    bullet_lines = [l for l in lines if l.strip().startswith(("•","-","*","◦"))]

    score = 0
    if has_project_section: score += 30
    if len(bullet_lines) >= 3: score += 30
    if found_projects: score += min(40, len(found_projects) * 15)

    return {
        "found_projects": found_projects,
        "has_project_section": has_project_section,
        "project_detail_score": min(100, score)
    }

def detect_resume_structure(text: str, text_lower: str) -> dict:
    import re
    lines = text.split("\n")
    found_sections = []

    for sec in SECTION_HEADERS:
        if any(sec in l.lower() and len(l.strip()) < 50 for l in lines):
            found_sections.append(sec)

    bullet_count = sum(1 for l in lines if l.strip().startswith(("•","-","*","◦")))
    has_dates = bool(re.search(r"\b(20\d{2}|19\d{2})\b", text_lower))

    score = 0
    score += 40 if len(found_sections) >= 3 else 25 if len(found_sections) >= 2 else 10 if found_sections else 0
    score += 30 if bullet_count >= 5 else 15 if bullet_count >= 3 else 0
    score += 30 if has_dates else 0

    return {
        "found_sections": found_sections,
        "structure_score": min(100, score)
    }

# ---------------- ANALYSIS ENGINE ----------------
def analyze_resume(text: str, target_role: str) -> dict:
    text_l = text.lower()
    role = ROLE_REQUIREMENTS.get(target_role)

    if not role:
        return {
            "score": 30,
            "skill_match_percentage": 0,
            "project_relevance_percentage": 0,
            "resume_depth_percentage": 0,
            "fit_verdict": "Weak Fit",
            "strengths": [],
            "missing_sections": [],
            "improvement_tips": ["Target role not supported yet"],
            "found_mandatory_skills": [],
            "found_optional_skills": [],
            "found_project_indicators": []
        }

    mandatory = role["mandatory"]
    optional = role["optional"]
    projects = role["projects"]

    found_mand = [s for s in mandatory if find_skill_with_context(text_l, s)]
    missing = [s for s in mandatory if s not in found_mand]
    mand_ratio = len(found_mand) / len(mandatory)

    mand_score = 40 if mand_ratio == 1 else int(mand_ratio * 25) if mand_ratio >= .5 else int(mand_ratio * 15)
    skill_pct = int(mand_ratio * 100)

    found_opt = [s for s in optional if find_skill_with_context(text_l, s)]
    opt_score = min(20, sum([8,6,4] + [2]*10)[:len(found_opt)])

    proj = analyze_project_depth(text, text_l, projects)
    proj_score = int((proj["project_detail_score"]/100)*25)

    struct = detect_resume_structure(text, text_l)
    length_score = 100 if len(text)>1500 else 70 if len(text)>1000 else 40 if len(text)>600 else 20
    quality_pct = int(struct["structure_score"]*0.6 + length_score*0.4)
    qual_score = int((quality_pct/100)*15)

    total = mand_score + opt_score + proj_score + qual_score
    if not found_mand: total = min(total, 35)
    if missing: total = min(total, 60)
    if not proj["found_projects"]: total = min(total, 55)
    if len(text) < 500: total = int(total * .7)

    total = max(5, min(100, total))
    verdict = "Strong Fit" if total>=75 else "Partial Fit" if total>=50 else "Weak Fit"

    return {
        "score": total,
        "skill_match_percentage": skill_pct,
        "project_relevance_percentage": proj["project_detail_score"],
        "resume_depth_percentage": quality_pct,
        "fit_verdict": verdict,
        "strengths": [],
        "missing_sections": missing,
        "improvement_tips": [],
        "found_mandatory_skills": found_mand,
        "found_optional_skills": found_opt,
        "found_project_indicators": proj["found_projects"]
    }

# ---------------- ROUTES ----------------
@api_router.get("/health")
async def health():
    return {"status": "ok"}

@api_router.post("/feedback")
async def submit_feedback(feedback: Feedback):
    logger.info(f"Feedback: {feedback.message}")
    return {"status": "success"}

@api_router.post("/analyze", response_model=ResumeAnalysisResult)
async def analyze(file: UploadFile = File(...), target_role: str = "web_developer"):
    data = await file.read()
    text = extract_text_from_pdf(data) if file.filename.endswith(".pdf") else extract_text_from_docx(data)
    analysis = analyze_resume(text, target_role)
    return ResumeAnalysisResult(**analysis, target_role=target_role, filename=file.filename)

app.include_router(api_router)
