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
    fit_verdict: str  # "Strong Fit" / "Partial Fit" / "Weak Fit"
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

# Evidence words that indicate real skill usage (not just keyword stuffing)
EVIDENCE_WORDS = [
    "experience", "worked", "developed", "built", "created", "implemented",
    "designed", "managed", "led", "deployed", "optimized", "automated",
    "project", "projects", "role", "position", "job", "internship",
    "responsibilities", "achieved", "delivered", "contributed", "using",
    "utilized", "applied", "proficient", "expertise", "skilled"
]

# Project indicators that show structured project descriptions
PROJECT_INDICATORS = [
    "project", "projects", "developed", "built", "created", "implemented",
    "designed", "deployed", "application", "system", "platform", "tool",
    "website", "app", "dashboard", "model", "algorithm"
]

# Section headers that indicate structured resume
SECTION_HEADERS = [
    "experience", "work experience", "employment", "skills", "technical skills",
    "projects", "education", "certifications", "achievements", "summary"
]

def find_skill_with_context(text_lower: str, skill: str, window: int = 100) -> bool:
    """
    Check if skill appears with evidence words nearby (within window characters).
    This prevents simple keyword stuffing from being rewarded.
    """
    import re
    
    # Find all occurrences of the skill
    pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
    matches = pattern.finditer(text_lower)
    
    for match in matches:
        start = max(0, match.start() - window)
        end = min(len(text_lower), match.end() + window)
        context = text_lower[start:end]
        
        # Check if any evidence word appears in the context window
        if any(evidence in context for evidence in EVIDENCE_WORDS):
            return True
    
    return False

def analyze_project_depth(text: str, text_lower: str, project_keywords: List[str]) -> dict:
    """
    Analyze project quality beyond simple keyword matching.
    Checks for:
    1. Project section headers
    2. Project keywords with descriptions (not just mentions)
    3. Length and detail of project descriptions
    """
    lines = text.split('\n')
    
    # Check for project section
    has_project_section = any(
        'project' in line.lower() and len(line.strip()) < 50
        for line in lines
    )
    
    # Find project keywords with context
    found_projects = []
    for keyword in project_keywords:
        if find_skill_with_context(text_lower, keyword, window=150):
            found_projects.append(keyword)
    
    # Check for project detail indicators (descriptions, bullet points, etc.)
    project_detail_score = 0
    if has_project_section:
        project_detail_score += 30
    
    # Check for structured project descriptions (bullet points, dashes, etc.)
    bullet_lines = [line for line in lines if line.strip().startswith(('•', '-', '*', '◦'))]
    if len(bullet_lines) >= 3:
        project_detail_score += 30
    
    # Check for project description length (meaningful projects have details)
    if len(found_projects) > 0:
        project_detail_score += min(40, len(found_projects) * 15)
    
    return {
        "found_projects": found_projects,
        "has_project_section": has_project_section,
        "project_detail_score": min(100, project_detail_score)
    }

def detect_resume_structure(text: str, text_lower: str) -> dict:
    """
    Analyze resume structure and organization.
    Well-structured resumes indicate professionalism.
    """
    lines = text.split('\n')
    
    # Find section headers
    found_sections = []
    for section in SECTION_HEADERS:
        for line in lines:
            line_clean = line.strip().lower()
            if section in line_clean and len(line_clean) < 50:
                found_sections.append(section)
                break
    
    # Check for bullet points (indicates organized content)
    bullet_count = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '◦')))
    
    # Check for date patterns (indicates timeline/experience)
    import re
    date_pattern = r'\b(20\d{2}|19\d{2})\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b'
    has_dates = bool(re.search(date_pattern, text_lower))
    
    structure_score = 0
    if len(found_sections) >= 3:
        structure_score += 40
    elif len(found_sections) >= 2:
        structure_score += 25
    elif len(found_sections) >= 1:
        structure_score += 10
    
    if bullet_count >= 5:
        structure_score += 30
    elif bullet_count >= 3:
        structure_score += 15
    
    if has_dates:
        structure_score += 30
    
    return {
        "found_sections": found_sections,
        "bullet_count": bullet_count,
        "has_dates": has_dates,
        "structure_score": min(100, structure_score)
    }

# ---------------- ENHANCED ANALYSIS ENGINE ----------------
def analyze_resume(text: str, target_role: str) -> dict:
    """
    Evidence-based, strict resume analysis.
    
    Scoring breakdown (100 points total):
    - Mandatory Skills: 40 points (all-or-nothing approach)
    - Optional Skills: 20 points (diminishing returns, context-aware)
    - Project Depth: 25 points (structured analysis, not just keywords)
    - Resume Quality: 15 points (structure, depth, professionalism)
    
    Philosophy: Real evidence > keyword stuffing
    """
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
    project_keywords = role["projects"]

    # ========================================
    # 1. MANDATORY SKILLS (40 points)
    # All-or-nothing: Missing even one = heavy penalty
    # ========================================
    found_mandatory_with_context = []
    for skill in mandatory:
        if find_skill_with_context(text_l, skill, window=100):
            found_mandatory_with_context.append(skill)
    
    missing_mandatory = [s for s in mandatory if s not in found_mandatory_with_context]
    
    mandatory_ratio = len(found_mandatory_with_context) / len(mandatory)
    
    if mandatory_ratio == 1.0:
        # All mandatory skills found with evidence
        mandatory_score = 40
    elif mandatory_ratio >= 0.5:
        # Missing some mandatory skills = harsh penalty
        mandatory_score = int(mandatory_ratio * 25)  # Max 25 points if missing any
    else:
        # Missing most mandatory skills = severe penalty
        mandatory_score = int(mandatory_ratio * 15)  # Max 15 points
    
    skill_match_percentage = int(mandatory_ratio * 100)

    # ========================================
    # 2. OPTIONAL SKILLS (20 points)
    # Diminishing returns + context validation
    # ========================================
    found_optional_with_context = []
    for skill in optional:
        if find_skill_with_context(text_l, skill, window=100):
            found_optional_with_context.append(skill)
    
    # Diminishing returns: 1st skill = 8pts, 2nd = 6pts, 3rd = 4pts, 4th+ = 2pts each
    optional_score = 0
    for i, skill in enumerate(found_optional_with_context):
        if i == 0:
            optional_score += 8
        elif i == 1:
            optional_score += 6
        elif i == 2:
            optional_score += 4
        else:
            optional_score += 2
        
        if optional_score >= 20:
            break
    
    optional_score = min(20, optional_score)

    # ========================================
    # 3. PROJECT DEPTH (25 points)
    # Structured analysis: sections, descriptions, evidence
    # ========================================
    project_analysis = analyze_project_depth(text, text_l, project_keywords)
    found_projects = project_analysis["found_projects"]
    has_project_section = project_analysis["has_project_section"]
    project_detail_score = project_analysis["project_detail_score"]
    
    # Convert project detail score (0-100) to actual points (0-25)
    project_score = int((project_detail_score / 100) * 25)
    project_relevance_percentage = project_detail_score

    # ========================================
    # 4. RESUME QUALITY (15 points)
    # Structure, depth, professionalism
    # ========================================
    structure_analysis = detect_resume_structure(text, text_l)
    structure_score = structure_analysis["structure_score"]
    
    # Text length quality
    length_score = 0
    if len(text) > 1500:
        length_score = 100
    elif len(text) > 1000:
        length_score = 70
    elif len(text) > 600:
        length_score = 40
    else:
        length_score = 20
    
    # Combine structure and length (weighted average)
    quality_score_percent = int((structure_score * 0.6) + (length_score * 0.4))
    quality_score = int((quality_score_percent / 100) * 15)
    resume_depth_percentage = quality_score_percent

    # ========================================
    # TOTAL SCORE CALCULATION
    # ========================================
    total_score = mandatory_score + optional_score + project_score + quality_score

    # ========================================
    # HARD PENALTIES (stricter than before)
    # ========================================
    
    # PENALTY 1: Missing ANY mandatory skill = cap score
    if len(found_mandatory_with_context) < len(mandatory):
        total_score = min(total_score, 60)  # Cannot exceed 60 if missing mandatory
    
    # PENALTY 2: Missing ALL mandatory skills = severe cap
    if len(found_mandatory_with_context) == 0:
        total_score = min(total_score, 35)  # Cannot exceed 35 if NO mandatory skills
    
    # PENALTY 3: No project evidence = cap score
    if not found_projects:
        total_score = min(total_score, 55)  # Cannot exceed 55 without projects
    
    # PENALTY 4: Thin resume (< 500 chars) = additional penalty
    if len(text) < 500:
        total_score = int(total_score * 0.7)  # 30% penalty for very thin resumes

    # Final bounds
    total_score = max(5, min(100, total_score))

    # ========================================
    # FIT VERDICT
    # ========================================
    if total_score >= 75:
        fit_verdict = "Strong Fit"
    elif total_score >= 50:
        fit_verdict = "Partial Fit"
    else:
        fit_verdict = "Weak Fit"

    # ========================================
    # STRENGTHS & IMPROVEMENT TIPS
    # ========================================
    strengths = []
    tips = []

    # Strengths
    if len(found_mandatory_with_context) == len(mandatory):
        strengths.append(f"✓ All mandatory skills present with evidence ({', '.join(found_mandatory_with_context)})")
    elif len(found_mandatory_with_context) > 0:
        strengths.append(f"✓ Some mandatory skills detected: {', '.join(found_mandatory_with_context)}")
    
    if len(found_optional_with_context) >= 3:
        strengths.append(f"✓ Strong supporting skills: {', '.join(found_optional_with_context[:3])}")
    elif len(found_optional_with_context) > 0:
        strengths.append(f"✓ Additional relevant skills: {', '.join(found_optional_with_context)}")
    
    if has_project_section:
        strengths.append("✓ Dedicated projects section found")
    
    if len(found_projects) >= 2:
        strengths.append(f"✓ Role-relevant project experience: {', '.join(found_projects[:3])}")
    
    if structure_analysis["structure_score"] >= 70:
        strengths.append("✓ Well-structured and organized resume")
    
    if len(text) > 1200:
        strengths.append("✓ Comprehensive resume with good depth")

    # Improvement Tips
    if missing_mandatory:
        tips.append(f"⚠ Missing critical mandatory skills: {', '.join(missing_mandatory)}")
    
    if len(found_optional_with_context) < 2:
        tips.append(f"⚠ Add more supporting skills like: {', '.join(optional[:3])}")
    
    if not found_projects:
        tips.append("⚠ No role-specific project experience found - add relevant projects")
    elif len(found_projects) < 2:
        tips.append("⚠ Limited project evidence - showcase more relevant projects")
    
    if not has_project_section:
        tips.append("⚠ Add a dedicated 'Projects' section to highlight your work")
    
    if structure_analysis["structure_score"] < 50:
        tips.append("⚠ Resume lacks clear structure - add section headers (Experience, Skills, Projects)")
    
    if len(text) < 800:
        tips.append("⚠ Resume is too brief - add more details about your experience and projects")
    
    if len(structure_analysis["found_sections"]) < 2:
        tips.append("⚠ Missing key sections - include Experience, Education, Skills, Projects")
    
    if total_score < 50:
        tips.append("⚠ Resume has low alignment with target role - review job requirements carefully")

    # ========================================
    # RETURN COMPREHENSIVE ANALYSIS
    # ========================================
    return {
        "score": total_score,
        "skill_match_percentage": skill_match_percentage,
        "project_relevance_percentage": project_relevance_percentage,
        "resume_depth_percentage": resume_depth_percentage,
        "fit_verdict": fit_verdict,
        "strengths": strengths[:6],  # Top 6 strengths
        "missing_sections": missing_mandatory,
        "improvement_tips": tips[:6],  # Top 6 tips
        "found_mandatory_skills": found_mandatory_with_context,
        "found_optional_skills": found_optional_with_context,
        "found_project_indicators": found_projects
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
