import truststore 
truststore.inject_into_ssl()
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# ------- Resume screening output schemas -------

class CandidateMatch(BaseModel):
    candidate_id: str
    match_score: int = Field(..., ge=0, le=100)
    strengths: List[str]
    gaps: List[str]
    summary: str

class ScreeningOutput(BaseModel):
    ranked_candidates: List[CandidateMatch]

# ------- Interview questions schemas -------

class InterviewQuestion(BaseModel):
    question: str
    skill_tested: str
    expected_answer_outline: List[str]

class QuestionsOutput(BaseModel):
    candidate_id: str
    questions: List[InterviewQuestion]

# ------- Answer evaluation schemas -------

class AnswerEvaluationItem(BaseModel):
    question: str
    answer: str
    score: int = Field(..., ge=0, le=10)
    feedback: str
    missing_points: List[str]

class AnswerEvaluationOutput(BaseModel):
    candidate_id: str
    overall_score: int = Field(..., ge=0, le=100)
    detailed: List[AnswerEvaluationItem]
    final_verdict: str  # Hire / Strong Consider / No Hire


class WeeklyPlan(BaseModel):
    week: int
    goals: List[str]
    topics: List[str]
    resources: List[str]

# ------- Learning plan schemas -------

class LearningPlan(BaseModel):
    ##candidate_id: str
   ## focus_areas: List[str]
    #----plan_by_week: Dict[str, List[str]]  # e.g., {"Week 1": [...], ...}
 ## practice_projects: List[str]
    ##recommended_resources: List[str]
    
    candidate_id: Optional[str] = None
    summary: Optional[str] = None

    # Accept ANY structure for weeks (dict is safest)
    ###########################plan_by_week: Optional[Dict[str, Any]] = None
    plan_by_week: List[WeeklyPlan]
    
    # Optional fields (AI may skip them)
    practice_projects: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    focus_areas: Optional[List[str]] = None
    ##recommended_resources: Optional[List[str]] = None
    recommended_resources: List[str]