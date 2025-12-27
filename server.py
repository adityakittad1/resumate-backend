from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException, Body
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import logging
import io
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
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

print("🚀 Resumate backend running in PRODUCTION ATS-LIKE MODE")

# ---------------- MODELS ----------------
class ResumeAnalysisResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    score: int
    skill_match_percentage: int
    project_relevance_percentage: int
    resume_depth_percentage: int
    consistency_percentage: int
    experience_depth_percentage: int
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
    analysis_id: Optional[str] = "unknown"
    rating: Optional[int] = 5
    comment: Optional[str] = ""
    helpful: Optional[bool] = True
    
    class Config:
        extra = "allow"

# ---------------- ROLE MAP ----------------
ROLE_REQUIREMENTS = {
    "ai_ml_intern": {
        "mandatory": ["python", "machine learning", "data"],
        "optional": ["numpy", "pandas", "scikit-learn", "tensorflow", "pytorch", "keras", "jupyter"],
        "projects": ["machine learning model", "neural network", "classification", "prediction", "data analysis", "model training"],
        "experience_keywords": ["dataset", "trained", "accuracy", "algorithm"]
    },
    "web_developer": {
        "mandatory": ["html", "css", "javascript"],
        "optional": ["react", "vue", "angular", "node", "express", "api", "rest", "tailwind", "bootstrap"],
        "projects": ["web application", "website", "web app", "frontend", "backend", "full stack"],
        "experience_keywords": ["responsive", "user interface", "component", "routing"]
    },
    "data_analyst": {
        "mandatory": ["sql", "python", "data"],
        "optional": ["excel", "pandas", "tableau", "power bi", "matplotlib", "seaborn", "statistics"],
        "projects": ["data analysis", "dashboard", "visualization", "business intelligence", "report"],
        "experience_keywords": ["insights", "metrics", "kpi", "trend"]
    },
    "cloud_devops": {
        "mandatory": ["linux", "cloud", "devops"],
        "optional": ["aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins", "terraform"],
        "projects": ["deployment", "infrastructure", "automation", "cloud", "container"],
        "experience_keywords": ["deployment", "monitoring", "scaling"]
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

# Strong evidence = action verbs
STRONG_EVIDENCE_WORDS = [
    "developed", "built", "created", "implemented", "designed", "architected",
    "engineered", "deployed", "optimized", "automated", "led", "managed",
    "delivered", "shipped", "launched", "maintained", "integrated", "configured"
]

# Weak evidence = passive mentions
WEAK_EVIDENCE_WORDS = [
    "worked", "used", "utilized", "applied", "familiar", "knowledge",
    "experience", "exposure", "responsibilities", "achieved", "contributed",
    "proficient", "skilled"
]

# Experience indicators
PROFESSIONAL_INDICATORS = [
    "internship", "intern", "co-op", "full-time", "part-time", "contract",
    "company", "organization", "corporation", "startup", "firm", "role", "position"
]

ACADEMIC_INDICATORS = [
    "university", "college", "course", "coursework", "assignment", "semester",
    "academic", "research", "thesis", "capstone"
]

SECTION_HEADERS = [
    "experience", "work experience", "employment", "professional experience",
    "skills", "technical skills", "projects", "education", "certifications",
    "achievements", "summary", "objective"
]

# REBALANCED: More realistic context windows
CONTEXT_WINDOWS = {
    "mandatory": 80,   # Relaxed from 40
    "optional": 100,   # Relaxed from 50
    "projects": 150    # Relaxed from 80
}

def find_skill_with_context(text_lower: str, skill: str, window: int, require_strong: bool = False) -> Dict:
    """
    Check if skill appears with evidence words nearby.
    Returns: {found: bool, evidence_type: str, occurrences: int}
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
    REBALANCED: Detect extreme keyword stuffing only.
    Returns penalty multiplier (0.7 to 1.0)
    """
    text_lower = text.lower()
    words = text_lower.split()
    total_words = len(words)
    
    if total_words < 50:
        return 0.85  # Reduced penalty for short resumes
    
    # Count skill mentions
    skill_mentions = sum(text_lower.count(skill.lower()) for skill in skills)
    density = skill_mentions / total_words
    
    # REBALANCED: Only penalize extreme cases
    if density > 0.15:  # Raised from 0.08
        return 0.70
    elif density > 0.12:  # Raised from 0.06
        return 0.85
    else:
        return 1.0  # No penalty for normal resumes

def analyze_experience_depth(text: str, text_lower: str) -> Dict:
    """
    Analyze experience depth and type.
    """
    has_professional = any(indicator in text_lower for indicator in PROFESSIONAL_INDICATORS)
    has_internship = "intern" in text_lower
    has_academic = any(indicator in text_lower for indicator in ACADEMIC_INDICATORS)
    
    # Extract duration
    duration_patterns = [
        r'(\d+)\+?\s*years?',
        r'(\d+)\+?\s*months?',
        r'(\d{4})\s*-\s*(\d{4})',
        r'(\d{4})\s*-\s*present',
    ]
    
    total_months = 0
    for pattern in duration_patterns:
        matches = re.findall(pattern, text_lower)
        if 'year' in pattern:
            total_months += sum(int(m) * 12 for m in matches if isinstance(m, str) and m.isdigit())
        elif 'month' in pattern:
            total_months += sum(int(m) for m in matches if isinstance(m, str) and m.isdigit())
        elif len(matches) > 0 and isinstance(matches[0], tuple):
            for match in matches:
                if match[0].isdigit():
                    start_year = int(match[0])
                    end_year = int(match[1]) if len(match) > 1 and match[1].isdigit() else 2025
                    total_months += (end_year - start_year) * 12
    
    total_months = min(total_months, 48)
    
    # Calculate score
    experience_score = 0
    
    if has_professional:
        experience_score += 50  # Increased from 40
    
    if has_internship:
        experience_score += 35  # Increased from 30
    
    if has_academic:
        experience_score += 20  # Increased from 15
    
    # Duration bonus
    if total_months >= 12:
        experience_score += 15
    elif total_months >= 6:
        experience_score += 10
    
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
    Analyze project depth with realistic validation.
    """
    lines = text.split('\n')
    
    # Find project section
    project_section_start = -1
    project_section_end = -1
    
    for i, line in enumerate(lines):
        line_clean = line.strip().lower()
        if 'project' in line_clean and len(line_clean) < 40:
            # Less strict requirements
            project_section_start = i
            for j in range(i + 1, len(lines)):
                next_line = lines[j].strip().lower()
                if any(section in next_line for section in SECTION_HEADERS) and len(next_line) < 40:
                    project_section_end = j
                    break
            if project_section_end == -1:
                project_section_end = len(lines)
            break
    
    has_project_section = project_section_start != -1
    project_content = ""
    if has_project_section:
        project_content = "\n".join(lines[project_section_start:project_section_end]).lower()
    
    # Find project keywords
    found_projects = []
    for keyword in project_keywords:
        if has_project_section:
            if keyword in project_content:
                found_projects.append(keyword)
        else:
            # More lenient for resumes without dedicated section
            result = find_skill_with_context(text_lower, keyword, CONTEXT_WINDOWS["projects"])
            if result["found"]:
                found_projects.append(keyword)
    
    # Calculate project score - REBALANCED
    project_detail_score = 0
    
    if has_project_section:
        project_detail_score += 35  # Increased from 25
        
        # Content length
        if len(project_content) > 200:
            project_detail_score += 30  # Increased
        elif len(project_content) > 100:
            project_detail_score += 20
        elif len(project_content) > 50:
            project_detail_score += 10
    
    # Bullet points in project section
    if has_project_section:
        project_bullets = [line for line in lines[project_section_start:project_section_end] 
                          if line.strip().startswith(('•', '-', '*', '◦'))]
        if len(project_bullets) >= 3:
            project_detail_score += 20
        elif len(project_bullets) >= 1:
            project_detail_score += 10
    
    # Skills in projects
    skills_in_projects = 0
    if has_project_section:
        for skill in all_skills:
            if skill.lower() in project_content:
                skills_in_projects += 1
    
    if skills_in_projects >= 2:
        project_detail_score += 15
    elif skills_in_projects >= 1:
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
    Detect resume structure with balanced validation.
    """
    lines = text.split('\n')
    
    # Find section headers - LESS STRICT
    found_sections = []
    for section in SECTION_HEADERS:
        for i, line in enumerate(lines):
            line_clean = line.strip().lower()
            if section in line_clean and len(line_clean) < 50:
                # Check if content follows
                has_content = False
                for j in range(i+1, min(i+5, len(lines))):
                    if len(lines[j].strip()) > 10:
                        has_content = True
                        break
                if has_content:
                    found_sections.append(section)
                    break
    
    # Bullet points
    bullet_count = sum(1 for line in lines if line.strip().startswith(('•', '-', '*', '◦')))
    
    meaningful_bullets = 0
    for line in lines:
        if line.strip().startswith(('•', '-', '*', '◦')):
            content = line.strip()[1:].strip()
            if len(content) > 15:  # Reduced from 20
                meaningful_bullets += 1
    
    # Date patterns
    date_pattern = r'\b(20\d{2}|19\d{2})\b|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b'
    has_dates = bool(re.search(date_pattern, text_lower))
    
    # REBALANCED structure scoring
    structure_score = 0
    if len(found_sections) >= 4:
        structure_score += 40
    elif len(found_sections) >= 3:
        structure_score += 30
    elif len(found_sections) >= 2:
        structure_score += 20
    elif len(found_sections) >= 1:
        structure_score += 10
    
    if meaningful_bullets >= 5:
        structure_score += 30
    elif meaningful_bullets >= 3:
        structure_score += 20
    elif meaningful_bullets >= 1:
        structure_score += 10
    
    if has_dates:
        structure_score += 20
    
    # Contact info
    has_contact = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text))
    if has_contact:
        structure_score += 10
    
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
                                found_projects: List[str], text_lower: str) -> Dict:
    """
    Check if claimed skills are demonstrated.
    """
    all_claimed_skills = found_mandatory + found_optional
    
    if len(all_claimed_skills) == 0:
        return {"consistency_score": 50, "orphaned_skills": [], "well_demonstrated_skills": []}  # Default to 50 instead of 0
    
    well_demonstrated = []
    orphaned = []
    
    for skill in all_claimed_skills:
        appearances = 0
        
        # Check in projects
        if any(proj_keyword in skill.lower() or skill.lower() in proj_keyword.lower() 
               for proj_keyword in found_projects):
            appearances += 2
        
        # Check mentions with evidence
        result = find_skill_with_context(text_lower, skill, 80)  # Increased window
        if result["found"]:
            if result["evidence_type"] == "strong":
                appearances += 2
            else:
                appearances += 1
        
        # Check occurrence count
        if result["occurrences"] >= 2:  # Reduced from 3
            appearances += 1
        
        # Classify
        if appearances >= 2:  # Reduced threshold from 3
            well_demonstrated.append(skill)
        elif appearances == 0:
            orphaned.append(skill)
    
    # Calculate percentage
    if len(all_claimed_skills) > 0:
        consistency_percentage = int((len(well_demonstrated) / len(all_claimed_skills)) * 100)
    else:
        consistency_percentage = 50
    
    # LESS HARSH orphaned penalty
    orphaned_ratio = len(orphaned) / max(len(all_claimed_skills), 1)
    if orphaned_ratio > 0.5:  # Raised from 0.4
        consistency_percentage = int(consistency_percentage * 0.75)  # Less harsh
    elif orphaned_ratio > 0.3:  # Raised from 0.2
        consistency_percentage = int(consistency_percentage * 0.90)
    
    return {
        "consistency_score": min(100, max(30, consistency_percentage)),  # Floor at 30
        "orphaned_skills": orphaned,
        "well_demonstrated_skills": well_demonstrated
    }

# ---------------- MAIN ANALYSIS ENGINE ----------------
def analyze_resume(text: str, target_role: str) -> dict:
    """
    REBALANCED ATS-LIKE ANALYSIS ENGINE
    
    Scoring (100 points):
    - Mandatory Skills: 35 points (must have with evidence)
    - Optional Skills: 15 points (supporting skills)
    - Project Depth: 20 points (relevant projects)
    - Experience: 10 points (professional > internship > academic)
    - Consistency: 10 points (skills demonstrated)
    - Structure: 10 points (formatting, organization)
    
    Penalties (Less Aggressive):
    - Missing mandatory: Progressive reduction, not multiplicative
    - Other penalties: Additive, not multiplicative
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

    # Detect keyword stuffing
    stuffing_penalty = detect_keyword_stuffing(text, all_skills)

    # ========================================
    # 1. MANDATORY SKILLS (35 points)
    # ========================================
    found_mandatory_details = {}
    for skill in mandatory:
        result = find_skill_with_context(text_l, skill, CONTEXT_WINDOWS["mandatory"])
        found_mandatory_details[skill] = result
    
    mandatory_score = 0
    found_mandatory_with_context = []
    
    for skill, details in found_mandatory_details.items():
        if details["found"]:
            found_mandatory_with_context.append(skill)
            if details["evidence_type"] == "strong":
                mandatory_score += (35 / len(mandatory))
            elif details["evidence_type"] == "weak":
                mandatory_score += (35 / len(mandatory)) * 0.75  # Less harsh - was 0.6
    
    missing_mandatory = [s for s in mandatory if s not in found_mandatory_with_context]
    mandatory_ratio = len(found_mandatory_with_context) / len(mandatory)
    skill_match_percentage = int(mandatory_ratio * 100)

    # ========================================
    # 2. OPTIONAL SKILLS (15 points)
    # ========================================
    found_optional_details = {}
    for skill in optional:
        result = find_skill_with_context(text_l, skill, CONTEXT_WINDOWS["optional"])
        found_optional_details[skill] = result
    
    found_optional_with_context = [s for s, d in found_optional_details.items() if d["found"]]
    
    # REBALANCED diminishing returns
    optional_score = 0
    point_values = [6, 4, 3, 2]  # More generous than before
    
    for i, skill in enumerate(found_optional_with_context):
        if i < len(point_values):
            points = point_values[i]
        else:
            points = 1  # Still give credit
        
        if found_optional_details[skill]["evidence_type"] == "weak":
            points *= 0.75  # Less harsh
        
        optional_score += points
        
        if optional_score >= 15:
            break
    
    optional_score = min(15, optional_score)

    # ========================================
    # 3. PROJECT DEPTH (20 points)
    # ========================================
    project_analysis = analyze_project_depth(text, text_l, project_keywords, all_skills)
    found_projects = project_analysis["found_projects"]
    has_project_section = project_analysis["has_project_section"]
    project_detail_score = project_analysis["project_detail_score"]
    skills_in_projects = project_analysis["skills_demonstrated_in_projects"]
    
    project_score = int((project_detail_score / 100) * 20)
    project_relevance_percentage = project_detail_score

    # ========================================
    # 4. EXPERIENCE DEPTH (10 points)
    # ========================================
    experience_analysis = analyze_experience_depth(text, text_l)
    experience_score_percent = experience_analysis["experience_score"]
    experience_score = int((experience_score_percent / 100) * 10)
    experience_depth_percentage = experience_score_percent

    # ========================================
    # 5. CONSISTENCY SCORE (10 points)
    # ========================================
    consistency_analysis = calculate_consistency_score(
        found_mandatory_with_context,
        found_optional_with_context,
        found_projects,
        text_l
    )
    consistency_percentage = consistency_analysis["consistency_score"]
    consistency_score = int((consistency_percentage / 100) * 10)

    # ========================================
    # 6. RESUME QUALITY (10 points)
    # ========================================
    structure_analysis = detect_resume_structure(text, text_l)
    structure_score = structure_analysis["structure_score"]
    
    quality_components = []
    quality_components.append(structure_score * 0.6)
    
    # Length - more generous
    words = len(text.split())
    if words > 300:
        quality_components.append(100 * 0.3)
    elif words > 200:
        quality_components.append(80 * 0.3)
    elif words > 100:
        quality_components.append(50 * 0.3)
    else:
        quality_components.append(30 * 0.3)
    
    # Uniqueness
    unique_words = len(set(text.lower().split()))
    uniqueness_ratio = unique_words / max(words, 1)
    if uniqueness_ratio > 0.5:
        quality_components.append(100 * 0.1)
    elif uniqueness_ratio > 0.3:
        quality_components.append(70 * 0.1)
    else:
        quality_components.append(40 * 0.1)
    
    quality_score_percent = int(sum(quality_components))
    quality_score = int((quality_score_percent / 100) * 10)
    resume_depth_percentage = quality_score_percent

    # ========================================
    # TOTAL SCORE (Base)
    # ========================================
    base_score = mandatory_score + optional_score + project_score + experience_score + consistency_score + quality_score

    # ========================================
    # REBALANCED PENALTIES (ADDITIVE, NOT MULTIPLICATIVE)
    # ========================================
    total_score = base_score
    penalty_points = 0
    penalties_applied = []
    
    # Keyword stuffing penalty
    if stuffing_penalty < 1.0:
        keyword_penalty = int(base_score * (1 - stuffing_penalty))
        penalty_points += keyword_penalty
        penalties_applied.append(f"Keyword stuffing detected (-{keyword_penalty} pts)")
    
    # Missing mandatory skills penalty - PROGRESSIVE
    if len(found_mandatory_with_context) == 0:
        penalty_points += 30  # Severe but not devastating
        penalties_applied.append("Missing all mandatory skills (-30 pts)")
    elif len(missing_mandatory) >= len(mandatory) * 0.67:
        penalty_points += 20
        penalties_applied.append(f"Missing {len(missing_mandatory)}/{len(mandatory)} mandatory skills (-20 pts)")
    elif len(missing_mandatory) > 0:
        penalty_points += 10
        penalties_applied.append(f"Missing {len(missing_mandatory)} mandatory skills (-10 pts)")
    
    # No projects penalty
    if not found_projects:
        penalty_points += 10
        penalties_applied.append("No relevant projects (-10 pts)")
    
    # No experience penalty
    if not experience_analysis["has_professional_experience"] and not experience_analysis["has_internship"]:
        penalty_points += 5
        penalties_applied.append("No professional experience (-5 pts)")
    
    # Poor consistency penalty
    if consistency_percentage < 30:
        penalty_points += 5
        penalties_applied.append("Low skill consistency (-5 pts)")
    
    # Very thin resume
    if len(text) < 300:
        penalty_points += 5
        penalties_applied.append("Resume too brief (-5 pts)")
    
    total_score = int(base_score - penalty_points)
    total_score = max(10, min(100, total_score))  # Floor at 10, not 5

    # ========================================
    # FIT VERDICT
    # ========================================
    if total_score >= 75:
        fit_verdict = "Strong Fit"
    elif total_score >= 55:
        fit_verdict = "Partial Fit"
    else:
        fit_verdict = "Weak Fit"

    # ========================================
    # STRENGTHS & TIPS
    # ========================================
    strengths = []
    tips = []

    # Strengths
    if len(found_mandatory_with_context) == len(mandatory):
        strengths.append(f"✓ All mandatory skills present: {', '.join(found_mandatory_with_context)}")
    elif len(found_mandatory_with_context) > 0:
        strengths.append(f"✓ Found {len(found_mandatory_with_context)}/{len(mandatory)} mandatory skills: {', '.join(found_mandatory_with_context)}")
    
    if len(found_optional_with_context) >= 3:
        strengths.append(f"✓ Strong supporting skills: {', '.join(found_optional_with_context[:4])}")
    elif len(found_optional_with_context) > 0:
        strengths.append(f"✓ Additional skills: {', '.join(found_optional_with_context[:3])}")
    
    if has_project_section:
        strengths.append(f"✓ Dedicated projects section with {len(found_projects)} relevant projects")
    elif found_projects:
        strengths.append(f"✓ Relevant projects mentioned: {', '.join(found_projects[:2])}")
    
    if experience_analysis["has_professional_experience"]:
        strengths.append("✓ Professional work experience documented")
    elif experience_analysis["has_internship"]:
        strengths.append("✓ Internship experience included")
    
    if consistency_percentage >= 60:
        strengths.append(f"✓ Skills well-demonstrated throughout resume")
    
    if structure_analysis["structure_score"] >= 70:
        strengths.append("✓ Well-structured and organized resume")

    # Tips
    if missing_mandatory:
        tips.append(f"❌ Add missing mandatory skills: {', '.join(missing_mandatory)}")
    
    if len(found_optional_with_context) < 2:
        missing_optional = [s for s in optional[:4] if s not in found_optional_with_context]
        if missing_optional:
            tips.append(f"⚠️ Add supporting skills: {', '.join(missing_optional[:3])}")
    
    if not found_projects:
        tips.append(f"❌ Add relevant projects: {', '.join(project_keywords[:2])}")
    elif len(found_projects) < 2:
        tips.append(f"⚠️ Add more diverse projects")
    
    if not has_project_section:
        tips.append("⚠️ Create a dedicated 'Projects' section")
    
    if not experience_analysis["has_professional_experience"] and not experience_analysis["has_internship"]:
        tips.append("⚠️ Add internship or work experience if available")
    
    if structure_analysis["structure_score"] < 50:
        tips.append("⚠️ Improve structure with clear sections (Experience, Skills, Projects, Education)")
    
    if len(text.split()) < 200:
        tips.append("⚠️ Expand descriptions - add details about responsibilities and achievements")
    
    if consistency_percentage < 40:
        tips.append("⚠️ Demonstrate your skills in project and experience descriptions")

    if not tips:
        tips.append("✓ Strong resume - keep refining and adding new experiences")

    # ========================================
    # RETURN ANALYSIS
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
        "improvement_tips": tips[:8],
        "found_mandatory_skills": found_mandatory_with_context,
        "found_optional_skills": found_optional_with_context,
        "found_project_indicators": found_projects,
        "penalties_applied": penalties_applied,
        "keyword_stuffing_detected": stuffing_penalty < 1.0,
        "consistency_details": consistency_analysis
    }

# ---------------- ROUTES ----------------
@api_router.get("/health")
async def health():
    return {"status": "ok", "mode": "production-ats-balanced"}

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
    """Accept user feedback - lenient validation."""
    try:
        logger.info(f"Feedback: ID={feedback.analysis_id}, Rating={feedback.rating}, Helpful={feedback.helpful}")
    except:
        pass
    
    return {
        "success": True,
        "message": "Thank you for your feedback!",
        "feedback_id": str(uuid.uuid4())
    }

@api_router.post("/feedback/raw")
async def submit_feedback_raw(data: dict = Body(...)):
    """Alternative raw feedback endpoint."""
    try:
        logger.info(f"Raw feedback: {data}")
    except:
        pass
    
    return {
        "success": True,
        "message": "Thank you for your feedback!",
        "feedback_id": str(uuid.uuid4())
    }

app.include_router(api_router)