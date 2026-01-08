# Recruitment Multi-Agent (LangChain + LangGraph + RAG + Chroma)

This is an end-to-end, runnable capstone-style project that:
1) screens resumes vs a Job Description (JD) and finds gaps
2) generates JD-based interview questions
3) evaluates candidate answers
4) generates a learning plan to close gaps

## Quick start

### 1) Create venv and install deps
```bash
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Configure env
Copy `.env.example` to `.env` and add your OpenAI key:
```bash
cp .env.example .env
```

### 3) Put your documents
- Replace `data/jd.txt` with your JD
- Replace / add resumes in `data/resumes/*.txt`

### 4) Run
```bash
python -m src.main
```

Outputs will be printed and also saved to `final_output.json`.

## Notes
- This project uses OpenAI embeddings + chat model by default.
- You can swap LLM/provider later (Ollama, Anthropic, etc.).
