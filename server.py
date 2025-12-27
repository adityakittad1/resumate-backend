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
    "experience", "worked", "developed", "built", "created", "implemented",
    "designed", "managed", "led", "deployed", "optimized", "automated",
    "project", "projects", "role", "position", "job", "internship",
    "responsibilities", "achieved", "delivered", "contributed", "using",
    "utilized", "applied", "proficient", "expertise", "skilled"
]

SECTION_HEADERS = [
    "experience", "work experience", "employment", "skills", "technical skills",
    "projects", "education", "certifications", "achievements", "summary"
]


def find_skill_with_context(text_lower: str, skill: str, window: int = 100) -> bool:
    import re
    pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
    for match in pattern.finditer(text_lower):
        start = max(0, match.start() - window)
        end = min(len(text_lower), match.end() + window)
        context = text_lower[start:end]
        if any(word in context for word in EVIDENCE_WORDS):
            return True
    return False


def analyze_project_depth(text: str, text_lower: str, project_keywords: List[str]) -> dict:
    lines = text.split('\n')

    has_project_section = any(
        'project' in line.lower() and len(line.strip()) < 50
        for line in lines
    )

    found_projects = []
    for keyword in project_keywords:
        if find_skill_with_context(text_lower, keyword, 150):
            found_projects.append(keyword)

    bullet_lines = [l for l in lines if l.strip().startswith(('•', '-', '*'))]

    score = 0
    if has_project_section:
        score += 30
    if len(bullet_lines) >= 3:
        score += 30
    if found_projects:
        score += min(40, len(found_projects) * 15)

    return {
        "found_projects": found_projects,
        "has_project_section": has_project_section,
        "project_detail_score": min(100, score)
    }


def detect_resume_structure(text: str, text_lower: str) -> dict:
    lines = text.split('\n')
    found_sections = []

    for section in SECTION_HEADERS:
        for line in lines:
            if section in line.lower() and len(line.strip()) < 50:
                found_sections.append(section)
                break

    bullet_count = sum(1 for l in lines if l.strip().startswith(('•', '-', '*')))

    structure_score = 0
    if len(found_sections) >= 3:
        structure_score += 40
    elif len(found_sections) >= 2:
        structure_score += 25
    elif found_sections:
        structure_score += 10

    if bullet_count >= 5:
        structure_score += 30
    elif bullet_count >= 3:
        structure_score += 15

    return {
        "found_sections": found_sections,
        "structure_score": min(100, structure_score)
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
            "improvement_tips": ["Target role not supported"],
            "found_mandatory_skills": [],
            "found_optional_skills": [],
            "found_project_indicators": []
        }

    mandatory = role["mandatory"]
    optional = role["optional"]

    found_mandatory = [s for s in mandatory if find_skill_with_context(text_l, s)]
    mandatory_ratio = len(found_mandatory) / len(mandatory)

    mandatory_score = 40 if mandatory_ratio == 1 else int(mandatory_ratio * 20)
    skill_match_percentage = int(mandatory_ratio * 100)

    found_optional = [s for s in optional if find_skill_with_context(text_l, s)]
    optional_score = min(20, len(found_optional) * 5)

    project_data = analyze_project_depth(text, text_l, role["projects"])
    project_score = int((project_data["project_detail_score"] / 100) * 25)

    structure = detect_resume_structure(text, text_l)
    quality_score = int((structure["structure_score"] / 100) * 15)

    total_score = mandatory_score + optional_score + project_score + quality_score
    total_score = max(5, min(100, total_score))

    fit_verdict = "Strong Fit" if total_score >= 75 else "Partial Fit" if total_score >= 50 else "Weak Fit"

    return {
        "score": total_score,
        "skill_match_percentage": skill_match_percentage,
        "project_relevance_percentage": project_data["project_detail_score"],
        "resume_depth_percentage": structure["structure_score"],
        "fit_verdict": fit_verdict,
        "strengths": ["✓ Resume analyzed successfully"],
        "missing_sections": [],
        "improvement_tips": [],
        "found_mandatory_skills": found_mandatory,
        "found_optional_skills": found_optional,
        "found_project_indicators": project_data["found_projects"]
    }


# ---------------- ROUTES ----------------
@api_router.get("/health")
async def health():
    return {"status": "ok"}


@api_router.post("/feedback")
async def submit_feedback(_: Feedback):
    # FAKE SUCCESS FOR FRONTEND ONLY
    return {"status": "success", "message": "Feedback received"}


@api_router.post("/analyze", response_model=ResumeAnalysisResult)
async def analyze(file: UploadFile = File(...), target_role: str = "web_developer"):
    filename = file.filename.lower()

    if not filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Upload PDF or DOCX only")

    data = await file.read()
    text = extract_text_from_pdf(data) if filename.endswith(".pdf") else extract_text_from_docx(data)

    analysis = analyze_resume(text, target_role)

    return ResumeAnalysisResult(
        score=analysis["score"],
        skill_match_percentage=analysis["skill_match_percentage"],
        project_relevance_percentage=analysis["project_relevance_percentage"],
        resume_depth_percentage=analysis["resume_depth_percentage"],
        fit_verdict=analysis["fit_verdict"],
        strengths=analysis["strengths"],
        missing_sections=analysis["missing_sections"],
        improvement_tips=analysis["improvement_tips"],
        found_mandatory_skills=analysis["found_mandatory_skills"],
        found_optional_skills=analysis["found_optional_skills"],
        found_project_indicators=analysis["found_project_indicators"],
        target_role=target_role,
        filename=file.filename
    )


app.include_router(api_router)
