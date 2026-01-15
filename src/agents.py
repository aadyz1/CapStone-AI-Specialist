import truststore 
truststore.inject_into_ssl()
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.schemas import ScreeningOutput, QuestionsOutput, AnswerEvaluationOutput, LearningPlan
from src.rag import retrieve_jd_context, retrieve_resume_context


# Single LLM client shared by all agents
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=OPENAI_MODEL,
    temperature=0,
)

# -----------------------------
# Agent 1: Resume Screening Agent
# -----------------------------
def resume_screening_agent(jd_query: str, candidate_ids: List[str]) -> ScreeningOutput:
    """
    What this agent does:
    - Reads the job description (via retrieval)
    - Reads each candidate resume (via retrieval)
    - Scores match and identifies strengths + gaps

    Why RAG helps:
    - We only fetch the most relevant chunks of the JD and resume.
    - This keeps prompts small and grounded in actual text.
    """
    ## Fetch 6 most relevant parts of the job description
    jd_context = retrieve_jd_context(jd_query, k=6)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict recruitment screening expert. "
         "Score candidate match to the job description. "
         "Return structured output only."),
        ("user",
         "JOB DESCRIPTION CONTEXT:\n{jd_context}\n\n"
         "CANDIDATE RESUME CONTEXT:\n{resume_context}\n\n"
         "Candidate ID: {candidate_id}\n\n"
         "Return a ScreeningOutput with ranked_candidates containing exactly ONE CandidateMatch for this candidate. "
         "Include match_score (0-100), strengths, gaps, summary."),
    ])

    structured_llm = llm.with_structured_output(ScreeningOutput)

    all_matches = []
    for cid in candidate_ids:
        ##Fetch resume chunks relevant to job
        resume_context = retrieve_resume_context(cid, jd_query, k=6)
        ##Send JD + resume to llm
        out = structured_llm.invoke(
            prompt.format_messages(
                jd_context=jd_context,
                resume_context=resume_context,
                candidate_id=cid,
            )
        )
        ## Add results to list.
        all_matches.extend(out.ranked_candidates)

    ##Rank candidates by score (best first)
    all_matches_sorted = sorted(all_matches, key=lambda x: x.match_score, reverse=True)
    ##Return final screening result
    return ScreeningOutput(ranked_candidates=all_matches_sorted)


# -----------------------------
# Agent 2: Interview Question Agent
# -----------------------------
def interview_question_agent(candidate_id: str, jd_query: str, num_questions: int = 6) -> QuestionsOutput:
    """
    What this agent does:
    - Reads the JD context (via retrieval)
    - Generates interview questions based on must-have skills + responsibilities
    - Also generates expected_answer_outline for each question (used later for evaluation)
    """
    jd_context = retrieve_jd_context(jd_query, k=8)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an interviewer designing an interview for this role. "
         "Create practical questions and include what a good answer should contain."
         "Return structured output only."),
        ("user",
         "JOB DESCRIPTION CONTEXT:\n{jd_context}\n\n"
         "Candidate ID: {candidate_id}\n\n"
         "Create {num_questions} interview questions. "
         "Each must include question, skill_tested, and expected_answer_outline."),
    ])

    structured_llm = llm.with_structured_output(QuestionsOutput)

    return structured_llm.invoke(
        prompt.format_messages(
            jd_context=jd_context,
            candidate_id=candidate_id,
            num_questions=num_questions,
        )
    )


# -----------------------------
# Agent 3: Answer Evaluation Agent
# -----------------------------
def answer_evaluation_agent(
    candidate_id: str,
    jd_query: str,
    questions_output: QuestionsOutput,
    answers: Dict[str, str],
) -> AnswerEvaluationOutput:
    """
    What this agent does:
    - Takes questions + expected outlines from Agent 2
    - Takes candidate answers (collected in main.py)
    - Scores each answer 0-10 with feedback and missing points
    - Computes overall score 0-100 and final verdict
    """
    jd_context = retrieve_jd_context(jd_query, k=6)

    q_bundle = []
    for q in questions_output.questions:
        q_bundle.append({
            "question": q.question,
            "skill_tested": q.skill_tested,
            "expected_answer_outline": q.expected_answer_outline,
            "answer": answers.get(q.question, ""),
        })

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict technical evaluator. "
         "Score answers based on correctness, completeness, and relevance to the JD. "
         "Return structured output only."),
        ("user",
         "JOB DESCRIPTION CONTEXT:\n{jd_context}\n\n"
         "Candidate ID: {candidate_id}\n\n"
         "Evaluate these Q&A bundles:\n{q_bundle}\n\n"
         "Rules:\n"
         "- Each answer score 0-10\n"
         "- Provide feedback and missing_points\n"
         "- Compute overall_score (0-100)\n"
         "- final_verdict must be one of: Hire, Strong Consider, No Hire"),
    ])

    structured_llm = llm.with_structured_output(AnswerEvaluationOutput)

    return structured_llm.invoke(
        prompt.format_messages(
            jd_context=jd_context,
            candidate_id=candidate_id,
            q_bundle=q_bundle,
        )
    )


# -----------------------------
# Agent 4: Learning Plan Agent
# -----------------------------
def learning_plan_agent(
    candidate_id: str,
    gaps: List[str],
    eval_output: AnswerEvaluationOutput,
) -> LearningPlan:
    """
    What this agent does:
    - Uses the skill gaps found in screening
    - Uses weak points found in interview evaluation
    - Produces a simple 4-week learning plan with projects + resources
    """
    weak_points = []
    for item in eval_output.detailed:
        if item.score <= 6:
            weak_points.extend(item.missing_points)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a career coach. Create a step-by-step learning plan. "
         "Keep it practical with weekly goals and suggested resources. "
         "Return structured output only."),
        ("user",
         "Candidate ID: {candidate_id}\n\n"
         "Skill gaps from resume screening:\n{gaps}\n\n"
         "Weak points from interview evaluation:\n{weak_points}\n\n"
         "Create: focus_areas, plan_by_week (Week 1..Week 4), practice_projects, recommended_resources URL's along with the label for which the URL is. Label should come first and then the URL for that (example Label: https:abc.com)."),
    ])

    #--structured_llm = llm.with_structured_output(LearningPlan)--

    structured_llm = llm.with_structured_output(
    LearningPlan,
    method="function_calling"
)

    return structured_llm.invoke(
        prompt.format_messages(
            candidate_id=candidate_id,
            gaps=gaps,
            weak_points=weak_points,
        )
    )
