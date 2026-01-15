import truststore 
truststore.inject_into_ssl()
from dotenv import load_dotenv
load_dotenv()

import json
import pandas as pd
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
# We imported the new metrics here
from ragas.metrics import answer_relevancy, faithfulness, context_precision

print("RAGAS EVALUATION STARTED")

# ---------------------------------------------
# 1. Load system output (The Homework)
# ---------------------------------------------
with open("final_output.json", "r") as f:
    data = json.load(f)

# The "Experts" who will do the grading
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

final_report = {}

# =================================================
# 2. Candidate Answer Evaluation (The Ragas Triad)
# =================================================
questions = data["questions"]["questions"]
evaluation = data["evaluation"]["detailed"]

# Create the "Cheat Sheet" (Mapping questions to expected outlines)
outline_map = {
    q["question"]: " ".join(q["expected_answer_outline"])
    for q in questions
}

records = []
for ev in evaluation:
    # Only grade if the AI actually gave an answer
    if ev["answer"].strip():
        ref = outline_map.get(ev["question"])
        
        # Faithfulness needs 'contexts'. 
        # We look for it in your file, or use the reference as the 'truth' source.
        contexts = ev.get("retrieved_contexts", [ref]) 
        
        if ref:
            records.append({
                "question": ev["question"],
                "answer": ev["answer"],
                "contexts": contexts,  # New: Needed for Faithfulness
                "reference": ref       # New: Needed for Precision
            })

if records:
    # Turn the list of boxes into a neat spreadsheet
    dataset = Dataset.from_list(records)
    
    # Run the 3-point inspection
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy, faithfulness, context_precision],
        llm=llm,
        embeddings=embeddings
    )
    
    ragas_df = ragas_result.to_pandas()
    
    # Save the technical row-by-row scores to CSV immediately
    ##ragas_df.to_csv("detailed_per_question_scores.csv", index=False)
    
    # Get the average scores (0.0 to 1.0)
    avg_relevancy = float(ragas_df["answer_relevancy"].mean())
    avg_faithfulness = float(ragas_df.get("faithfulness", 0).mean())
    avg_precision = float(ragas_df.get("context_precision", 0).mean())
    
    # Combine them for a total score out of 5
    total_avg = (avg_relevancy + avg_faithfulness + avg_precision) / 3
    candidate_score = round(total_avg * 5, 1)
else:
    avg_relevancy = avg_faithfulness = avg_precision = 0.0
    candidate_score = 0.0

# Write the "Report Card" for the candidates
final_report["candidate_answer_evaluation"] = {
    "score": candidate_score,
    "what_was_evaluated": "Relevancy, Faithfulness (No Hallucinations), and Context Precision.",
    "detailed_metrics": {
        "answer_relevancy": round(avg_relevancy, 3),
        "faithfulness": round(avg_faithfulness, 3),
        "context_precision": round(avg_precision, 3)
    },
    "strengths": [
        "Uses Faithfulness to ensure AI isn't making up facts",
        "Uses Context Precision to ensure search results are accurate"
    ],
    "weaknesses": [
        "If faithfulness is low, the model is ignoring the provided data",
        "Several candidate answers were empty or incomplete"
    ],
    "justification": (
        f"The candidate scored {candidate_score}/5. "
        "This is based on how well they answered the prompt, how factual they were "
        "relative to the documents, and how precise the retrieved information was."
    )
}

# =================================================
# 3. Screening Agent Evaluation (LLM-as-a-Judge)
# =================================================
screening_prompt = f"""
Evaluate the screening agent.
Return ONLY valid JSON:
{{
  "score": 1_to_5,
  "fairness": "low | medium | high",
  "consistency": "low | medium | high",
  "strengths": [ "..."],
  "improvement_areas": [ "..."],
  "justification": "..."
}}
DATA: {json.dumps(data["screening"]["ranked_candidates"], indent=2)}
"""
final_report["screening_agent_evaluation"] = json.loads(llm.invoke(screening_prompt).content)

# =================================================
# 4. Question Generation Agent Evaluation
# =================================================
question_prompt = f"""
Evaluate the question generation agent.
Return ONLY valid JSON:
{{
  "score": 1_to_5,
  "coverage": "poor | fair | good",
  "difficulty_balance": "poor | fair | good",
  "missing_topics": [ "..."],
  "justification": "..."
}}
DATA: {json.dumps(data["questions"]["questions"], indent=2)}
"""
final_report["question_generation_agent_evaluation"] = json.loads(llm.invoke(question_prompt).content)

# =================================================
# 5. Evaluation / Feedback Agent Evaluation
# =================================================
feedback_prompt = f"""
Evaluate the interview evaluation agent.
Return ONLY valid JSON:
{{
  "score": 1_to_5,
  "feedback_quality": "low | medium | high",
  "actionability": "low | medium | high",
  "strengths": [ "..."],
  "weaknesses": [ "..."],
  "justification": "..."
}}
DATA: {json.dumps(data["evaluation"]["detailed"], indent=2)}
"""
final_report["evaluation_agent_evaluation"] = json.loads(llm.invoke(feedback_prompt).content)

# =================================================
# 6. Learning Plan Agent Evaluation
# =================================================
learning_prompt = f"""
Evaluate the learning plan agent.
Return ONLY valid JSON:
{{
  "score": 1_to_5,
  "alignment_with_gaps": "low | medium | high",
  "progression_quality": "poor | fair | good",
  "practical_value": "low | medium | high",
  "justification": "..."
}}
DATA: {json.dumps(data["learning_plan"], indent=2)}
"""
final_report["learning_plan_agent_evaluation"] = json.loads(llm.invoke(learning_prompt).content)

# =================================================
# 7. Overall System Summary
# =================================================
overall_prompt = f"""
Given the following evaluations, write a clear summary for non-technical readers.
Return ONLY valid JSON:
{{
  "overall_score": number,
  "summary_for_non_technical_readers": "...",
  "system_strengths": [ "..."],
  "system_weaknesses": [ "..."],
  "final_verdict": "Not Ready | Partially Ready | Ready"
}}
DATA: {json.dumps(final_report, indent=2)}
"""
overall_eval = json.loads(llm.invoke(overall_prompt).content)
final_report["overall_system_evaluation"] = overall_eval

# =================================================
# 8. Save final report (JSON AND CSV)
# =================================================

# Save the JSON file exactly as before
with open("recuruitment_system_evaluation.json", "w") as f:
    json.dump(final_report, f, indent=2)

# --- NEW: GENERATE THE CSV REPORT ---
# This part flattens the deep JSON structure into a table row
flat_report = pd.json_normalize(final_report)
flat_report.to_csv("recuruitment_system_evaluation.csv", index=False)

print("EVALUATION COMPLETED")
print("Saved to recuruitment_system_evaluation.json")
print("Saved to recuruitment_system_evaluation.csv")
##print("Saved detailed technical scores to detailed_per_question_scores.csv")
