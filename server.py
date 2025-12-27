from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import logging
import io
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict
import uuid
from datetime import datetime, timezone
import re

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

print("🚀 Resumate backend running in PRODUCTION-GRADE STRICT MODE")

# ---------------- MODELS ----------------
class ResumeAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score: int
    skill_match_percentage: int
    project_relevance_percentage: int
    resume_depth_percentage: int
    consistency_percentage: int  # PRODUCTION FIX: New consistency dimension
    experience_depth_percentage: int  # PRODUCTION FIX: New experience dimension
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

class FeedbackSubmission(BaseModel):
    analysis_id: str
    rating: int = Field(ge=1, le=5)
    comment: str = ""
    helpful: bool = True

# ---------------- ROLE MAP (PRODUCTION FIX: Increased mandatory requirements) ----------------
ROLE_REQUIREMENTS = {
    "ai_ml_intern": {
        "mandatory": ["python", "machine learning", "data"],  # Increased from 2 to 3
        "optional": ["numpy", "pandas", "scikit-learn", "tensorflow", "pytorch", "keras", "jupyter"],
        "projects": ["machine learning model", "neural network", "classification model", "prediction model", "data analysis", "model training"],  # More specific
        "experience_keywords": ["dataset", "trained model", "accuracy", "precision", "recall", "algorithm"]
    },
    "web_developer": {
        "mandatory": ["html", "css", "javascript"],  # Keep as 3
        "optional": ["react", "vue", "angular", "node", "express", "api", "rest", "tailwind", "bootstrap"],
        "projects": ["web application", "website", "web app", "frontend", "backend", "full stack"],  # More specific
        "experience_keywords": ["responsive", "user interface", "component", "routing", "authentication"]
    },
    "data_analyst": {
        "mandatory": ["sql", "python", "data analysis"],  # Increased from 2 to 3
        "optional": ["excel", "pandas", "tableau", "power bi", "matplotlib", "seaborn", "statistics"],
        "projects": ["data analysis", "dashboard", "data visualization", "business intelligence", "report"],
        "experience_keywords": ["insights", "metrics", "kpi", "trend", "correlation"]
    },
    "cloud_devops": {
        "mandatory": ["linux", "cloud", "devops"],  # Increased from 2 to 3
        "optional": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "terraform", "ansible"],
        "projects": ["deployment pipeline", "infrastructure", "automation", "cloud deployment", "container"],
        "experience_keywords": ["deployment", "monitoring", "scaling", "infrastructure as code"]
    }
}

# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(b: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(b))
    return "\n".join(p.extract_text() or "" for p in reader.pages)

def extract_text_from_docx(b: bytes) -> str:
    doc = Document(io.BytesIO(b))
    return "\n".join(p.text for p in doc.paragraphs)

# ---------------- ANALYSIS HELPERS (PRODUCTION FIX: Separated strong/weak evidence) ----------------

# PRODUCTION FIX: Strong evidence = action verbs that show real usage
STRONG_EVIDENCE_WORDS = [
    "developed", "built", "created", "implemented", "designed", "architected",
    "engineered", "deployed", "optimized", "automated", "led", "managed",
    "delivered", "shipped", "launched", "maintained", "integrated", "configured"
]

# PRODUCTION FIX: Weak evidence = passive mentions (lower weight)
WEAK_EVIDENCE_WORDS = [
    "worked", "used", "utilized", "applied", "familiar", "knowledge",
    "experience", "exposure", "responsibilities", "achieved", "contributed"
]

# PRODUCTION FIX: Experience indicators for depth scoring
PROFESSIONAL_INDICATORS = [
    "internship", "intern", "co-op", "full-time", "part-time", "contract",
    "company", "organization", "corporation", "startup", "firm", "role", "position"
]

ACADEMIC_INDICATORS = [
    "university", "college", "course", "coursework", "assignment", "semester",
    "academic", "research", "thesis", "capstone"
]

# Section headers for structure detection
SECTION_HEADERS = [
    "experience", "work experience", "employment", "professional experience",
    "skills", "technical skills", "projects", "education", "certifications",
    "achievements", "summary", "objective"
]

# PRODUCTION FIX: Stricter context window based on skill importance
CONTEXT_WINDOWS = {
    "mandatory": 40,   # Very tight - must be closely associated
    "optional": 50,    # Tight but slightly more lenient
    "projects": 80     # Moderate for project descriptions
}

def find_skill_with_context(text_lower: str, skill: str, window: int, require_strong: bool = False) -> Dict:
    """
    PRODUCTION FIX: Enhanced context validation with evidence quality scoring.
    
    Returns dict with:
    - found: bool
    - evidence_type: "strong" | "weak" | "none"
    - occurrences: int
    """
    pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
    matches = list(pattern.finditer(text_lower))
    
    if not matches:
        return {"found": False, "evidence_type": "none", "occurrences": 0}
    
    strong_evidence_found = False
    weak_evidence_found = False
    
    for match in matches:
        start = max(0, match.start() - window)
        end = min(len(text_lower), match.end() + window)
        context = text_lower[start:end]
        
        # Check for strong evidence
        if any(evidence in context for evidence in STRONG_EVIDENCE_WORDS):
            strong_evidence_found = True
            break
        
        # Check for weak evidence
        if any(evidence in context for evidence in WEAK_EVIDENCE_WORDS):
            weak_evidence_found = True
    
    if require_strong and not strong_evidence_found:
        return {"found": False, "evidence_type": "weak" if weak_evidence_found else "none", "occurrences": len(matches)}
    
    if strong_evidence_found:
        return {"found": True, "evidence_type": "strong", "occurrences": len(matches)}
    elif weak_evidence_found:
        return {"found": True, "evidence_type": "weak", "occurrences": len(matches)}
    else:
        return {"found": False, "evidence_type": "none", "occurrences": len(matches)}

def detect_keyword_stuffing(text: str, skills: List[str]) -> float:
    """
    PRODUCTION FIX: Detect if resume has unnaturally high keyword density.
    Returns penalty multiplier (0.6 to 1.0)
    """
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words)
    
    if total_words < 50:
        return 0.7  # Very short resume = suspicious
    
    # Count how many times skills appear
    skill_mentions = sum(text_lower.count(skill.lower()) for skill in skills)
    
    # Calculate density
    density = skill_mentions / total_words
    
    # Suspicious if more than 8% of words are skill keywords
    if density > 0.08:
        return 0.6  # Heavy penalty for keyword stuffing
    elif density > 0.06:
        return 0.75
    elif density > 0.04:
        return 0.9
    else:
        return 1.0  # No penalty

def analyze_experience_depth(text: str, text_lower: str) -> Dict:
    """
    PRODUCTION FIX: Analyze experience depth and professional vs personal projects.
    
    Returns:
    - has_professional_experience: bool
    - has_internship: bool
    - has_academic_projects: bool
    - experience_months: int (estimated)
    - experience_score: 0-100
    """
    lines = text.split('\n')
    
    # Detect professional experience
    has_professional = any(indicator in text_lower for indicator in PROFESSIONAL_INDICATORS)
    
    # Detect internship specifically
    has_internship = "intern" in text_lower
    
    # Detect academic projects
    has_academic = any(indicator in text_lower for indicator in ACADEMIC_INDICATORS)
    
    # Try to extract duration (years/months)
    duration_patterns = [
        r'(\d+)\+?\s*years?',
        r'(\d+)\+?\s*months?',
        r'(\d{4})\s*-\s*(\d{4})',  # Year ranges
        r'(\d{4})\s*-\s*present',
    ]
    
    total_months = 0
    for pattern in duration_patterns:
        matches = re.findall(pattern, text_lower)
        if 'year' in pattern:
            total_months += sum(int(m) * 12 for m in matches if m.isdigit())
        elif 'month' in pattern:
            total_months += sum(int(m) for m in matches if m.isdigit())
        elif len(matches) > 0 and isinstance(matches[0], tuple):
            # Year range
            for match in matches:
                if match[0].isdigit() and (match[1].isdigit() if len(match) > 1 else True):
                    start_year = int(match[0])
                    end_year = int(match[1]) if len(match) > 1 and match[1].isdigit() else 2025
                    total_months += (end_year - start_year) * 12
    
    # Cap at reasonable maximum (4 years for students)
    total_months = min(total_months, 48)
    
    # Calculate experience score
    experience_score = 0
    
    if has_professional:
        experience_score += 40
    
    if has_internship:
        experience_score += 30
    
    if has_academic:
        experience_score += 15
    
    # Duration bonus
    if total_months >= 24:  # 2+ years
        experience_score += 15
    elif total_months >= 12:  # 1+ year
        experience_score += 10
    elif total_months >= 6:  # 6+ months
        experience_score += 5
    
    experience_score = min(100, experience_score)
    
    return {
        "has_professional_experience": has_professional,
        "has_internship": has_internship,
        "has_academic_projects": has_academic,
        "experience_months": total_months,
        "experience_score": experience_score
    }

def analyze_project_depth(text: str, text_lower: str, project_keywords: List[str], all_skills: List[str]) -> dict:
    """
    PRODUCTION FIX: Stricter project analysis with content validation.
    
    Now checks for:
    1. Actual project section (not just mentions)
    2. Project descriptions with minimum length
    3. Technical details in projects
    4. Skills demonstrated in projects
    """
    lines = text.split('\n')
    
    # PRODUCTION FIX: Find actual project section with content
    project_section_start = -1
    project_section_end = -1
    
    for i, line in enumerate(lines):
        line_clean = line.strip().lower()
        # Check if this is a project header (short line with "project")
        if 'project' in line_clean and len(line_clean) < 30 and i > 0:
            # Verify it's not in the middle of a sentence
            prev_line = lines[i-1].strip()
            if len(prev_line) == 0 or prev_line.endswith(('.', ':', '!')):
                project_section_start = i
                # Find next section or end
                for j in range(i + 1, len(lines)):
                    next_line = lines[j].strip().lower()
                    if any(section in next_line for section in SECTION_HEADERS) and len(next_line) < 30:
                        project_section_end = j
                        break
                if project_section_end == -1:
                    project_section_end = len(lines)
                break
    
    has_project_section = project_section_start != -1
    project_content = ""
    if has_project_section:
        project_content = "\n".join(lines[project_section_start:project_section_end]).lower()
    
    # PRODUCTION FIX: Find project keywords only in project section
    found_projects = []
    for keyword in project_keywords:
        if has_project_section:
            # Must appear in project section with context
            if find_skill_with_context(project_content, keyword, CONTEXT_WINDOWS["projects"])["found"]:
                found_projects.append(keyword)
        else:
            # If no section, require very strong evidence
            result = find_skill_with_context(text_lower, keyword, CONTEXT_WINDOWS["projects"], require_strong=True)
            if result["found"] and result["evidence_type"] == "strong":
                found_projects.append(keyword)
    
    # PRODUCTION FIX: Calculate project score based on real content
    project_detail_score = 0
    
    # Has dedicated section
    if has_project_section:
        project_detail_score += 25
        
        # Check content length in project section
        if len(project_content) > 300:
            project_detail_score += 25
        elif len(project_content) > 150:
            project_detail_score += 15
        elif len(project_content) > 50:
            project_detail_score += 5
    
    # Check for structured descriptions (bullets in project section)
    if has_project_section:
        project_bullets = [line for line in lines[project_section_start:project_section_end] 
                          if line.strip().startswith(('•', '-', '*', '◦'))]
        if len(project_bullets) >= 5:
            project_detail_score += 20
        elif len(project_bullets) >= 3:
            project_detail_score += 10
    
    # PRODUCTION FIX: Check if projects mention technical skills
    skills_in_projects = 0
    if has_project_section:
        for skill in all_skills:
            if skill.lower() in project_content:
                skills_in_projects += 1
    
    if skills_in_projects >= 3:
        project_detail_score += 20
    elif skills_in_projects >= 2:
        project_detail_score += 10
    
    # Reward multiple distinct projects (look for project titles/names)
    if has_project_section:
        # Count lines that look like project titles (capitalized, not too long)
        project_titles = 0
        for line in lines[project_section_start:project_section_end]:
            line_strip = line.strip()
            if (len(line_strip) > 10 and len(line_strip) < 60 and 
                line_strip[0].isupper() and not line_strip.startswith(('•', '-', '*', '◦'))):
                project_titles += 1
        
        if project_titles >= 3:
            project_detail_score += 10
    
    project_detail_score = min(100, project_detail_score)
    
    return {
        "found_projects": found_projects,
        "has_project_section": has_project_section,
        "project_detail_score": project_detail_score,
        "skills_demonstrated_in_projects": skills_in_projects
    }

def detect_resume_structure(text: str, text_lower: str) -> dict:
    """
    PRODUCTION FIX: Stricter structure detection - must be real sections with content.
    """
    lines = text.split('\n')
    
    # PRODUCTION FIX: Find actual section headers (not just mentions)
    found_sections = []
    for section in SECTION_HEADERS:
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            # Must be short line (likely a header)
            if section in line_clean and len(line_clean) < 40:
                # Check if previous line is empty (typical section separator)
                if i > 0 and len(lines[i-1].strip()) == 0:
                    # Check if content follows (next 3 lines not empty)
                    has_content = False
                    for j in range(i+1, min(i+4, len(lines))):
                        if len(lines[j].strip()) > 0:
                            has_content = True
                            break
                    if has_content:
                        found_sections.append(section)
                        break
    
    # Check for bullet points (indicates organized content)
    bullet_count = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '◦')))
    
    # PRODUCTION FIX: Check bullet content quality (not just count)
    meaningful_bullets = 0
    for line in lines:
        if line.strip().startswith(('•', '-', '*', '◦')):
            content = line.strip()[1:].strip()
            # Bullet must have reasonable length
            if len(content) > 20:
                meaningful_bullets += 1
    
    # Check for date patterns (indicates timeline/experience)
    date_pattern = r'\b(20\d{2}|19\d{2})\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b'
    has_dates = bool(re.search(date_pattern, text_lower))
    
    # PRODUCTION FIX: Stricter structure scoring
    structure_score = 0
    if len(found_sections) >= 4:
        structure_score += 35
    elif len(found_sections) >= 3:
        structure_score += 25
    elif len(found_sections) >= 2:
        structure_score += 15
    
    # Reward meaningful bullets, not just any bullets
    if meaningful_bullets >= 8:
        structure_score += 30
    elif meaningful_bullets >= 5:
        structure_score += 20
    elif meaningful_bullets >= 3:
        structure_score += 10
    
    if has_dates:
        structure_score += 20
    
    # PRODUCTION FIX: Check for contact information (email/phone)
    has_contact = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    if has_contact:
        structure_score += 15
    
    structure_score = min(100, structure_score)
    
    return {
        "found_sections": found_sections,
        "bullet_count": bullet_count,
        "meaningful_bullets": meaningful_bullets,
        "has_dates": has_dates,
        "has_contact": has_contact,
        "structure_score": structure_score
    }

def calculate_consistency_score(found_mandatory: List[str], found_optional: List[str], 
                                found_projects: List[str], project_content: str, 
                                text_lower: str) -> Dict:
    """
    PRODUCTION FIX: New dimension - check if claimed skills are actually demonstrated.
    
    Validates:
    1. Skills listed in "Skills" section are used in projects
    2. Skills are mentioned in experience section
    3. No orphaned skills (claimed but never used)
    """
    all_claimed_skills = found_mandatory + found_optional
    
    if len(all_claimed_skills) == 0:
        return {"consistency_score": 0, "orphaned_skills": [], "well_demonstrated_skills": []}
    
    # For each skill, check how many places it appears
    well_demonstrated = []
    orphaned = []
    
    for skill in all_claimed_skills:
        appearances = 0
        
        # Check if in projects (most important)
        if any(proj_keyword in skill.lower() or skill.lower() in proj_keyword.lower() 
               for proj_keyword in found_projects):
            appearances += 2  # Projects count more
        
        # Check mentions in text with strong evidence
        result = find_skill_with_context(text_lower, skill, 50)
        if result["found"] and result["evidence_type"] == "strong":
            appearances += 2
        elif result["found"]:
            appearances += 1
        
        # Check occurrence count
        if result["occurrences"] >= 3:
            appearances += 1
        
        # Classify
        if appearances >= 3:
            well_demonstrated.append(skill)
        elif appearances <= 1:
            orphaned.append(skill)
    
    # Calculate consistency percentage
    if len(all_claimed_skills) > 0:
        consistency_percentage = int((len(well_demonstrated) / len(all_claimed_skills)) * 100)
    else:
        consistency_percentage = 0
    
    # Penalize orphaned skills
    orphaned_ratio = len(orphaned) / max(len(all_claimed_skills), 1)
    if orphaned_ratio > 0.4:  # More than 40% orphaned = major issue
        consistency_percentage = int(consistency_percentage * 0.6)
    elif orphaned_ratio > 0.2:  # More than 20% orphaned
        consistency_percentage = int(consistency_percentage * 0.8)
    
    return {
        "consistency_score": min(100, consistency_percentage),
        "orphaned_skills": orphaned,
        "well_demonstrated_skills": well_demonstrated
    }

# ---------------- ENHANCED ANALYSIS ENGINE (PRODUCTION FIXES APPLIED) ----------------
def analyze_resume(text: str, target_role: str) -> dict:
    """
    PRODUCTION-GRADE STRICT ANALYSIS ENGINE
    
    NEW Scoring breakdown (100 points total):
    - Mandatory Skills: 30 points (stricter validation, evidence quality matters)
    - Optional Skills: 15 points (harsher diminishing returns)
    - Project Depth: 25 points (must have real content, not just keywords)
    - Experience Depth: 10 points (professional > internship > academic)
    - Consistency: 10 points (skills must be demonstrated, not just listed)
    - Resume Quality: 10 points (structure, formatting, completeness)
    
    NEW Penalty System (Multiplicative):
    - Missing mandatory skills: Progressive penalties
    - No projects: ×0.70
    - No professional experience: ×0.85
    - High keyword stuffing: ×0.60 to ×0.90
    - Low consistency: ×0.90
    """
    text_l = text.lower()
    role = ROLE_REQUIREMENTS.get(target_role)

    if not role:
        return {
            "score": 30,
            "skill_match_percentage": 0,
            "project_relevance_percentage": 0,
            "resume_depth_percentage": 0,
            "consistency_percentage": 0,
            "experience_depth_percentage": 0,
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
    all_skills = mandatory + optional

    # ========================================
    # PRODUCTION FIX: Detect keyword stuffing early
    # ========================================
    stuffing_penalty = detect_keyword_stuffing(text, all_skills)

    # ========================================
    # 1. MANDATORY SKILLS (30 points) - STRICTER
    # ========================================
    found_mandatory_details = {}
    for skill in mandatory:
        result = find_skill_with_context(text_l, skill, CONTEXT_WINDOWS["mandatory"])
        found_mandatory_details[skill] = result
    
    # PRODUCTION FIX: Weight by evidence quality
    mandatory_score = 0
    found_mandatory_with_context = []
    
    for skill, details in found_mandatory_details.items():
        if details["found"]:
            found_mandatory_with_context.append(skill)
            if details["evidence_type"] == "strong":
                mandatory_score += (30 / len(mandatory))  # Full credit
            elif details["evidence_type"] == "weak":
                mandatory_score += (30 / len(mandatory)) * 0.6  # Reduced credit
    
    missing_mandatory = [s for s in mandatory if s not in found_mandatory_with_context]
    mandatory_ratio = len(found_mandatory_with_context) / len(mandatory)
    skill_match_percentage = int(mandatory_ratio * 100)

    # ========================================
    # 2. OPTIONAL SKILLS (15 points) - HARSHER
    # ========================================
    found_optional_details = {}
    for skill in optional:
        result = find_skill_with_context(text_l, skill, CONTEXT_WINDOWS["optional"])
        found_optional_details[skill] = result
    
    found_optional_with_context = [s for s, d in found_optional_details.items() if d["found"]]
    
    # PRODUCTION FIX: Harsher diminishing returns
    optional_score = 0
    point_values = [5, 4, 3, 2, 1]  # First skill = 5pts, drastically decreases
    
    for i, skill in enumerate(found_optional_with_context):
        if i < len(point_values):
            points = point_values[i]
        else:
            points = 0.5  # Minimal credit for additional skills
        
        # PRODUCTION FIX: Reduce points for weak evidence
        if found_optional_details[skill]["evidence_type"] == "weak":
            points *= 0.6
        
        optional_score += points
        
        if optional_score >= 15:
            break
    
    optional_score = min(15, optional_score)

    # ========================================
    # 3. PROJECT DEPTH (25 points) - MUCH STRICTER
    # ========================================
    project_analysis = analyze_project_depth(text, text_l, project_keywords, all_skills)
    found_projects = project_analysis["found_projects"]
    has_project_section = project_analysis["has_project_section"]
    project_detail_score = project_analysis["project_detail_score"]
    skills_in_projects = project_analysis["skills_demonstrated_in_projects"]
    
    project_score = int((project_detail_score / 100) * 25)
    project_relevance_percentage = project_detail_score

    # ========================================
    # 4. EXPERIENCE DEPTH (10 points) - NEW
    # ========================================
    experience_analysis = analyze_experience_depth(text, text_l)
    experience_score_percent = experience_analysis["experience_score"]
    experience_score = int((experience_score_percent / 100) * 10)
    experience_depth_percentage = experience_score_percent

    # ========================================
    # 5. CONSISTENCY SCORE (10 points) - NEW
    # ========================================
    consistency_analysis = calculate_consistency_score(
        found_mandatory_with_context,
        found_optional_with_context,
        found_projects,
        "",  # We don't have separate project content here
        text_l
    )
    consistency_percentage = consistency_analysis["consistency_score"]
    consistency_score = int((consistency_percentage / 100) * 10)

    # ========================================
    # 6. RESUME QUALITY (10 points) - REBALANCED
    # ========================================
    structure_analysis = detect_resume_structure(text, text_l)
    structure_score = structure_analysis["structure_score"]
    
    # PRODUCTION FIX: Quality based on multiple factors
    quality_components = []
    
    # Structure
    quality_components.append(structure_score * 0.5)
    
    # Length (but not just raw length - check for substance)
    words = len(text.split())
    if words > 400:
        quality_components.append(100 * 0.3)
    elif words > 250:
        quality_components.append(70 * 0.3)
    elif words > 150:
        quality_components.append(40 * 0.3)
    else:
        quality_components.append(20 * 0.3)
    
    # Unique word count (avoid repetition)
    unique_words = len(set(text.lower().split()))
    uniqueness_ratio = unique_words / max(words, 1)
    if uniqueness_ratio > 0.6:
        quality_components.append(100 * 0.2)
    elif uniqueness_ratio > 0.4:
        quality_components.append(60 * 0.2)
    else:
        quality_components.append(30 * 0.2)
    
    quality_score_percent = int(sum(quality_components))
    quality_score = int((quality_score_percent / 100) * 10)
    resume_depth_percentage = quality_score_percent

    # ========================================
    # TOTAL SCORE CALCULATION (Base Score)
    # ========================================
    base_score = mandatory_score + optional_score + project_score + experience_score + consistency_score + quality_score

    # ========================================
    # PRODUCTION FIX: MULTIPLICATIVE PENALTIES (Order matters!)
    # ========================================
    total_score = base_score
    penalties_applied = []
    
    # PENALTY 1: Keyword stuffing (apply first)
    if stuffing_penalty < 1.0:
        total_score *= stuffing_penalty
        penalties_applied.append(f"Keyword density too high (×{stuffing_penalty})")
    
    # PENALTY 2: Missing mandatory skills (progressive)
    if len(found_mandatory_with_context) == 0:
        total_score *= 0.40  # Missing ALL mandatory = severe
        penalties_applied.append("Missing all mandatory skills (×0.40)")
    elif len(missing_mandatory) >= len(mandatory) * 0.5:
        total_score *= 0.65  # Missing 50%+ mandatory
        penalties_applied.append(f"Missing {len(missing_mandatory)}/{len(mandatory)} mandatory skills (×0.65)")
    elif len(missing_mandatory) > 0:
        total_score *= 0.80  # Missing some mandatory
        penalties_applied.append(f"Missing {len(missing_mandatory)} mandatory skills (×0.80)")
    
    # PENALTY 3: No projects
    if not found_projects:
        total_score *= 0.70
        penalties_applied.append("No relevant project experience (×0.70)")
    
    # PENALTY 4: No professional experience
    if not experience_analysis["has_professional_experience"] and not experience_analysis["has_internship"]:
        total_score *= 0.85
        penalties_applied.append("No professional experience (×0.85)")
    
    # PENALTY 5: Poor consistency (claimed skills not demonstrated)
    if consistency_percentage < 40:
        total_score *= 0.90
        penalties_applied.append("Many skills not demonstrated (×0.90)")
    
    # PENALTY 6: Very thin resume (< 400 chars)
    if len(text) < 400:
        total_score *= 0.65
        penalties_applied.append("Resume too brief (×0.65)")

    # Final bounds
    total_score = int(max(5, min(100, total_score)))

    # ========================================
    # FIT VERDICT (STRICTER THRESHOLDS)
    # ========================================
    if total_score >= 75:
        fit_verdict = "Strong Fit"
    elif total_score >= 55:  # Raised from 50
        fit_verdict = "Partial Fit"
    else:
        fit_verdict = "Weak Fit"

    # ========================================
    # STRENGTHS & IMPROVEMENT TIPS (MORE SPECIFIC)
    # ========================================
    strengths = []
    tips = []

    # Strengths
    if len(found_mandatory_with_context) == len(mandatory):
        strong_count = sum(1 for s in found_mandatory_with_context 
                          if found_mandatory_details[s]["evidence_type"] == "strong")
        if strong_count == len(mandatory):
            strengths.append(f"✓ All mandatory skills demonstrated with strong evidence: {', '.join(found_mandatory_with_context)}")
        else:
            strengths.append(f"✓ All mandatory skills present: {', '.join(found_mandatory_with_context)}")
    elif len(found_mandatory_with_context) > 0:
        strengths.append(f"✓ Found {len(found_mandatory_with_context)}/{len(mandatory)} mandatory skills: {', '.join(found_mandatory_with_context)}")
    
    if len(found_optional_with_context) >= 3:
        strengths.append(f"✓ Strong supporting skills portfolio: {', '.join(found_optional_with_context[:4])}")
    elif len(found_optional_with_context) > 0:
        strengths.append(f"✓ Additional skills: {', '.join(found_optional_with_context[:3])}")
    
    if has_project_section and len(found_projects) >= 2:
        strengths.append(f"✓ Well-documented projects section with {len(found_projects)} relevant projects")
    elif has_project_section:
        strengths.append("✓ Dedicated projects section present")
    
    if skills_in_projects >= 3:
        strengths.append(f"✓ Skills well-integrated in projects ({skills_in_projects} skills demonstrated)")
    
    if experience_analysis["has_professional_experience"]:
        strengths.append("✓ Professional work experience documented")
    elif experience_analysis["has_internship"]:
        strengths.append("✓ Internship experience included")
    
    if consistency_percentage >= 70:
        strengths.append(f"✓ Strong consistency - skills are demonstrated ({consistency_percentage}% validation)")
    
    if structure_analysis["structure_score"] >= 75:
        strengths.append("✓ Excellently structured and organized resume")

    # Improvement Tips (PRODUCTION FIX: More actionable and specific)
    if missing_mandatory:
        tips.append(f"❌ CRITICAL: Add mandatory skills with evidence: {', '.join(missing_mandatory)}")
    
    # Check for weak evidence on mandatory skills
    weak_mandatory = [s for s in found_mandatory_with_context 
                     if found_mandatory_details[s]["evidence_type"] == "weak"]
    if weak_mandatory:
        tips.append(f"⚠️ Strengthen evidence for: {', '.join(weak_mandatory)} (use action verbs like 'developed', 'built', 'implemented')")
    
    if len(found_optional_with_context) < 3:
        missing_optional = [s for s in optional[:5] if s not in found_optional_with_context]
        tips.append(f"⚠️ Add more supporting skills: {', '.join(missing_optional[:3])}")
    
    if not found_projects:
        tips.append(f"❌ CRITICAL: Add relevant projects (e.g., {', '.join(project_keywords[:2])})")
    elif len(found_projects) < 2:
        tips.append(f"⚠️ Add more diverse projects covering: {', '.join([p for p in project_keywords if p not in found_projects][:2])}")
    
    if not has_project_section:
        tips.append("⚠️ Create a dedicated 'Projects' section with detailed descriptions")
    elif project_detail_score < 50:
        tips.append("⚠️ Expand project descriptions - add technologies used, challenges solved, outcomes achieved")
    
    if skills_in_projects < 2:
        tips.append("⚠️ Mention specific technical skills in your project descriptions")
    
    if not experience_analysis["has_professional_experience"] and not experience_analysis["has_internship"]:
        tips.append("⚠️ Add internship or work experience if available")
    
    if consistency_percentage < 50:
        orphaned = consistency_analysis["orphaned_skills"]
        if orphaned:
            tips.append(f"⚠️ These skills lack evidence: {', '.join(orphaned[:3])} - demonstrate them in projects or remove")
    
    if structure_analysis["structure_score"] < 60:
        tips.append("⚠️ Improve resume structure - add clear section headers (Experience, Skills, Projects, Education)")
    
    if structure_analysis["meaningful_bullets"] < 5:
        tips.append("⚠️ Use bullet points to organize your experience and projects")
    
    if len(text.split()) < 250:
        tips.append("⚠️ Resume is too brief - add more details about your work and projects (aim for 300-500 words)")
    
    if stuffing_penalty < 0.9:
        tips.append("⚠️ Reduce keyword repetition - focus on demonstrating skills through detailed descriptions")
    
    if total_score < 55 and not tips:
        tips.append("⚠️ Resume needs significant improvement - review all mandatory requirements and add detailed project descriptions")

    # Ensure tips is never empty
    if not tips:
        tips.append("✓ Solid resume foundation - continue building experience and refining descriptions")

    # ========================================
    # RETURN COMPREHENSIVE ANALYSIS
    # ========================================
    return {
        "score": total_score,
        "skill_match_percentage": skill_match_percentage,
        "project_relevance_percentage": project_relevance_percentage,
        "resume_depth_percentage": resume_depth_percentage,
        "consistency_percentage": consistency_percentage,
        "experience_depth_percentage": experience_depth_percentage,
        "fit_verdict": fit_verdict,
        "strengths": strengths[:6],
        "missing_sections": missing_mandatory,
        "improvement_tips": tips[:8],  # Allow more tips for detailed guidance
        "found_mandatory_skills": found_mandatory_with_context,
        "found_optional_skills": found_optional_with_context,
        "found_project_indicators": found_projects,
        "penalties_applied": penalties_applied,  # For debugging/transparency
        "keyword_stuffing_detected": stuffing_penalty < 1.0,
        "consistency_details": consistency_analysis
    }

# ---------------- ROUTES ----------------
@api_router.get("/health")
async def health():
    return {"status": "ok", "mode": "production-strict"}

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
        consistency_percentage=analysis["consistency_percentage"],
        experience_depth_percentage=analysis["experience_depth_percentage"],
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

@api_router.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    """
    Accept user feedback on analysis results.
    Currently returns success without persistence (no database).
    """
    logger.info(f"Feedback received for analysis {feedback.analysis_id}: rating={feedback.rating}, helpful={feedback.helpful}")
    
    return {
        "success": True,
        "message": "Thank you for your feedback!",
        "feedback_id": str(uuid.uuid4())
    }

app.include_router(api_router)