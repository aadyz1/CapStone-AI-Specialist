import truststore  
truststore.inject_into_ssl()
import os
import json
import streamlit as st


from src.ingest import ingest_all
from src.graph import build_recruitment_graph

APP_TITLE = "Recruitment Multi-Agent System (LangChain + LangGraph + RAG + Chroma)"

DATA_DIR = "./data"
JD_PATH = os.path.join(DATA_DIR, "jd.txt")
RESUMES_DIR = os.path.join(DATA_DIR, "resumes")

# =========================================================
# directory where streamlit_app.py is located
# =========================================================
APP_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FINAL_OUTPUT_PATH = os.path.join(APP_ROOT_DIR, "final_output.json")


def ensure_data_folders():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESUMES_DIR, exist_ok=True)


def save_uploaded_file(uploaded_file, dest_path: str):
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


def clear_resumes_folder():
    if not os.path.exists(RESUMES_DIR):
        return
    for fn in os.listdir(RESUMES_DIR):
        fp = os.path.join(RESUMES_DIR, fn)
        if os.path.isfile(fp):
            os.remove(fp)


def list_candidate_ids():
    if not os.path.exists(RESUMES_DIR):
        return []
    ids = []
    for f in os.listdir(RESUMES_DIR):
        lower = f.lower()
        if lower.endswith(".txt") or lower.endswith(".docx") or lower.endswith(".doc") or lower.endswith(".pdf"):
            ids.append(os.path.splitext(f)[0])
    return sorted(ids)


def render_candidate_card(c):
    st.markdown(f"### {c.candidate_id} — **{c.match_score}/100**")
    st.write("**Summary:**", c.summary)
    st.write("**Strengths:**")
    st.write("\n".join([f"- {s}" for s in c.strengths]))
    st.write("**Gaps:**")
    st.write("\n".join([f"- {g}" for g in c.gaps]))
    st.divider()


def render_evaluation(evaluation):
    st.subheader("Evaluation Result (Agent 3)")
    st.metric("Overall Score", f"{evaluation.overall_score}/100")
    st.write("**Verdict:**", evaluation.final_verdict)
    st.divider()

    for item in evaluation.detailed:
        st.markdown(f"**Q:** {item.question}")
        st.write("**Answer:**", item.answer)
        st.write("**Score:**", f"{item.score}/10")
        st.write("**Feedback:**", item.feedback)
        if item.missing_points:
            st.write("**Missing points:**")
            st.write("\n".join([f"- {m}" for m in item.missing_points]))
        st.divider()


def render_learning_plan(plan):
    st.subheader("Learning Plan (Agent 4)")

    if hasattr(plan, "focus_areas") and plan.focus_areas:
        st.write("**Focus Areas:**", ", ".join(plan.focus_areas))
    st.divider()

    st.write("### Weekly Plan")

    for index, weekly_plan in enumerate(plan.plan_by_week, start=1):
        if isinstance(weekly_plan, dict):
            week_number = weekly_plan.get("week", index)
            goals = weekly_plan.get("goals", [])
        else:
            week_number = getattr(weekly_plan, "week", index)
            goals = getattr(weekly_plan, "goals", [])

        st.markdown(f"**Week {week_number}**")
        st.markdown("**Goals:**")
        for goal in goals:
            st.write(f"- {goal}")
        st.divider()

    st.write("### Practice Projects")
    for p in getattr(plan, "practice_projects", []):
        st.write(f"- {p}")

    st.write("### Recommended Resources")
    for r in getattr(plan, "recommended_resources", []):
        st.write(f"- {r}")

    
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Upload resumes, automatically ingest with backend JD, "
        "screen candidates, generate interview questions, evaluate answers, "
        "and create a learning plan."
    )

    ensure_data_folders()

    st.session_state.setdefault("screening", None)
    st.session_state.setdefault("questions", None)
    st.session_state.setdefault("selected_candidate_id", None)
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("evaluation", None)
    st.session_state.setdefault("learning_plan", None)
    st.session_state.setdefault("final_json", None)

    left, right = st.columns([1, 1])

    with left:
        st.header("1) Upload Resumes")

        resume_files = st.file_uploader(
            "Upload Resumes (txt/docx/doc/pdf) — multiple files allowed",
            type=["txt","docx","doc","pdf"],
            accept_multiple_files=True
        )

        if resume_files:
            clear_resumes_folder()
            for rf in resume_files:
                safe_name = rf.name.replace(" ", "_")
                save_uploaded_file(rf, os.path.join(RESUMES_DIR, safe_name))

            try:
                ingest_all(JD_PATH, RESUMES_DIR)
                st.success("Resumes uploaded and JD + resumes ingested automatically.")
            except Exception as e:
                st.exception(e)

        st.divider()

        st.header("2) Run Screening + Generate Questions")
        jd_query = st.text_input(
            "Query used for retrieval + reasoning",
            value="Find best candidate fit for this job and identify gaps."
        )

        if st.button("Run Screening", type="primary", use_container_width=True):
            try:
                candidate_ids = list_candidate_ids()
                if not candidate_ids:
                    st.error("No resumes found.")
                else:
                    app = build_recruitment_graph()

                    state = {
                        "jd_query": jd_query,
                        "candidate_ids": candidate_ids,
                        "screening": None,
                        "selected_candidate_id": None,
                        "questions": None,
                        "answers": {},
                        "evaluation": None,
                        "learning_plan": None,
                    }

                    partial = app.invoke(state)

                    st.session_state.screening = partial["screening"]
                    st.session_state.selected_candidate_id = partial["selected_candidate_id"]
                    st.session_state.questions = partial["questions"]
                    st.session_state.answers = {}
                    st.session_state.evaluation = None
                    st.session_state.learning_plan = None
                    st.session_state.final_json = None

                    st.success("Screening + questions generated. See results on the right.")
            except Exception as e:
                st.exception(e)

    with right:
        st.header("Results")

        if st.session_state.screening is None:
            st.info("Run screening to see ranked candidates and interview questions.")
            return

        st.subheader("Ranked Candidates (Agent 1)")
        for c in st.session_state.screening.ranked_candidates:
            render_candidate_card(c)

        all_ids = [c.candidate_id for c in st.session_state.screening.ranked_candidates]
        if not all_ids:
            st.warning("No candidates found.")
            return

        st.subheader("Select Candidate")
        selected = st.selectbox(
            "Candidate ID",
            options=all_ids,
            index=all_ids.index(st.session_state.selected_candidate_id) if st.session_state.selected_candidate_id in all_ids else 0
        )
        st.session_state.selected_candidate_id = selected

        st.divider()

        st.subheader("Interview Q&A (Agent 2)")
        questions_obj = st.session_state.questions

        if questions_obj is None or not questions_obj.questions:
            st.warning("No questions generated yet.")
            return

        # =====================================================
        # ✅ FIX: Use st.form to prevent rerun on every keystroke
        # =====================================================
        with st.form("interview_form"):
            answers = {}
            for i, q in enumerate(questions_obj.questions, 1):
                st.markdown(f"""
**Q{i}. {q.question}**
*Skill:* {q.skill_tested}
""")
                answers[q.question] = st.text_area(
                    label=f"Answer {i}",
                    value=st.session_state.answers.get(q.question, ""),
                    height=120
                )
                with st.expander("Expected Answer Outline"):
                    st.write("\n".join([f"- {p}" for p in q.expected_answer_outline]))

            submitted = st.form_submit_button("Save Answers")

            if submitted:
                st.session_state.answers = answers
                st.success("Answers saved. Click 'Evaluate + Learning Plan'.")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Evaluate + Learning Plan", type="primary", use_container_width=True):
                try:
                    app = build_recruitment_graph()

                    state = {
                        "jd_query": jd_query,
                        "candidate_ids": list_candidate_ids(),
                        "screening": st.session_state.screening,
                        "selected_candidate_id": st.session_state.selected_candidate_id,
                        "questions": st.session_state.questions,
                        "answers": st.session_state.answers,
                        "evaluation": None,
                        "learning_plan": None,
                    }

                    final = app.invoke(state)

                    st.session_state.evaluation = final["evaluation"]
                    st.session_state.learning_plan = final["learning_plan"]

                    st.session_state.final_json = {
                        "screening": st.session_state.screening.model_dump(),
                        "questions": st.session_state.questions.model_dump(),
                        "evaluation": st.session_state.evaluation.model_dump(),
                        "learning_plan": st.session_state.learning_plan.model_dump(),
                    }

                    with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
                        json.dump(st.session_state.final_json, f, indent=2)

                    st.success("Evaluation + learning plan ready below.")
                except Exception as e:
                    st.exception(e)

        with col2:
            if st.session_state.final_json is not None:
                st.download_button(
                    "Download JSON report",
                    data=json.dumps(st.session_state.final_json, indent=2),
                    file_name="final_output.json",
                    mime="application/json",
                    use_container_width=True
                )

        if st.session_state.evaluation is not None:
            render_evaluation(st.session_state.evaluation)

        if st.session_state.learning_plan is not None:
            render_learning_plan(st.session_state.learning_plan)


if __name__ == "__main__":
    main()
