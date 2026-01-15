import truststore 
truststore.inject_into_ssl()
import os
import json


from src.ingest import ingest_all
from src.graph import build_recruitment_graph

DATA_DIR = "./data"
JD_PATH = os.path.join(DATA_DIR, "jd.txt")
RESUMES_DIR = os.path.join(DATA_DIR, "resumes")


def list_candidate_ids(resumes_dir: str):
    ids = []
    for f in os.listdir(resumes_dir):
        lower = f.lower()
        if lower.endswith(".txt") or lower.endswith(".docx") or lower.endswith(".doc") or lower.endswith(".pdf"):
            ids.append(os.path.splitext(f)[0])
    return sorted(ids)


def main():
    # 1) Ingest JD + resumes into Chroma
    print("Ingesting JD + resumes into Chroma ...")
    ingest_all(JD_PATH, RESUMES_DIR)
    print("Ingestion complete.\n")

    # 2) Build LangGraph app
    app = build_recruitment_graph()

    # 3) Initial state
    candidate_ids = list_candidate_ids(RESUMES_DIR)
    jd_query = "Find best candidate fit for this job and identify gaps."

    state = {
        "jd_query": jd_query,
        "candidate_ids": candidate_ids,
        "screening": None,
        "selected_candidate_id": None,
        "questions": None,
        "answers": None,
        "evaluation": None,
        "learning_plan": None,
    }

    # 4) Run first part to get screening + questions
    print("Running screening + question generation...")
    partial = app.invoke({**state, "answers": {}})

    screening = partial["screening"]
    selected = partial["selected_candidate_id"]
    questions = partial["questions"]

    print("\n=== Ranked Candidates ===")
    for c in screening.ranked_candidates:
        print(f"- {c.candidate_id}: {c.match_score}/100")
        print(f"  Strengths: {', '.join(c.strengths[:3])} ...")
        print(f"  Gaps: {', '.join(c.gaps[:3])} ...\n")

    print(f"Selected top candidate: {selected}\n")

    print("=== Interview Questions ===")
    for i, q in enumerate(questions.questions, 1):
        print(f"{i}. {q.question} (Skill: {q.skill_tested})")

    # 5) Collect answers from user (simulating candidate answers)
    print("\nNow enter candidate answers (press Enter after each answer).\n")
    answers = {}
    for q in questions.questions:
        ans = input(f"Answer for: {q.question}\n> ")
        answers[q.question] = ans

    # 6) Run evaluation + learning plan
    final = app.invoke({
        **state,
        "screening": screening,
        "selected_candidate_id": selected,
        "questions": questions,
        "answers": answers,
    })

    evaluation = final["evaluation"]
    plan = final["learning_plan"]

    print("\n=== Evaluation ===")
    print(f"Overall Score: {evaluation.overall_score}/100")
    print(f"Verdict: {evaluation.final_verdict}\n")

    for item in evaluation.detailed:
        print(f"Q: {item.question}")
        print(f"Score: {item.score}/10")
        print(f"Feedback: {item.feedback}")
        if item.missing_points:
            print(f"Missing: {', '.join(item.missing_points)}")
        print("-" * 50)

    print("\n=== Learning Plan ===")
    print("Focus Areas:", ", ".join(plan.focus_areas))
    ##for week, tasks in plan.plan_by_week.items():
    for index, tasks in enumerate(plan.plan_by_week, start=1):
        week = f"Week {index}"
        print(f"\n{week}:")
        for t in tasks:
            print(f"- {t}")

    print("\nPractice Projects:")
    for p in plan.practice_projects:
        print(f"- {p}")

    print("\nResources:")
    for r in plan.recommended_resources:
        print(f"- {r}")

    # Save a JSON report (useful for capstone submission)
    with open("final_output.json", "w", encoding="utf-8") as f:
        json.dump({
            "screening": screening.model_dump(),
            "questions": questions.model_dump(),
            "evaluation": evaluation.model_dump(),
            "learning_plan": plan.model_dump(),
        }, f, indent=2)

    print("\nSaved final_output.json")


if __name__ == "__main__":
    main()
