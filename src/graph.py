from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END

from src.schemas import ScreeningOutput, QuestionsOutput, AnswerEvaluationOutput, LearningPlan
from src.agents import (
    resume_screening_agent,
    interview_question_agent,
    answer_evaluation_agent,
    learning_plan_agent,
)

# This is the shared "state" that flows through the graph.
# Each node (agent step) reads from state and writes new fields into state.
class RecruitState(TypedDict):
    # Inputs
    jd_query: str
    candidate_ids: List[str]

    # Agent outputs / intermediate values
    screening: Optional[ScreeningOutput]
    selected_candidate_id: Optional[str]

    questions: Optional[QuestionsOutput]
    answers: Optional[Dict[str, str]]
    evaluation: Optional[AnswerEvaluationOutput]
    learning_plan: Optional[LearningPlan]


# -------------------------
# Graph nodes (each node = one step/agent)
# -------------------------

def node_screen_resumes(state: RecruitState) -> RecruitState:
    screening = resume_screening_agent(state["jd_query"], state["candidate_ids"])
    top_candidate = screening.ranked_candidates[0].candidate_id if screening.ranked_candidates else None
    return {**state, "screening": screening, "selected_candidate_id": top_candidate}


def node_generate_questions(state: RecruitState) -> RecruitState:
    cid = state["selected_candidate_id"]
    questions = interview_question_agent(cid, state["jd_query"], num_questions=6)
    return {**state, "questions": questions}


def node_evaluate_answers(state: RecruitState) -> RecruitState:
    cid = state["selected_candidate_id"]
    evaluation = answer_evaluation_agent(cid, state["jd_query"], state["questions"], state["answers"])
    return {**state, "evaluation": evaluation}


def node_learning_plan(state: RecruitState) -> RecruitState:
    cid = state["selected_candidate_id"]

    # Find gaps from screening for selected candidate
    gaps = []
    if state.get("screening"):
        for c in state["screening"].ranked_candidates:
            if c.candidate_id == cid:
                gaps = c.gaps
                break

    plan = learning_plan_agent(cid, gaps, state["evaluation"])
    return {**state, "learning_plan": plan}


def build_recruitment_graph():
    """
    Builds the LangGraph workflow.

    Sequence:
    screen_resumes -> generate_questions -> evaluate_answers -> learning_plan -> END
    """
    graph = StateGraph(RecruitState)

    graph.add_node("screen_resumes", node_screen_resumes)
    graph.add_node("generate_questions", node_generate_questions)
    graph.add_node("evaluate_answers", node_evaluate_answers)
    graph.add_node("learning_plan", node_learning_plan)

    graph.set_entry_point("screen_resumes")
    graph.add_edge("screen_resumes", "generate_questions")
    graph.add_edge("generate_questions", "evaluate_answers")
    graph.add_edge("evaluate_answers", "learning_plan")
    graph.add_edge("learning_plan", END)

    return graph.compile()
