import truststore 
truststore.inject_into_ssl()
import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Where Chroma persists vectors to disk
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

# Collections inside Chroma (think of them like tables)
JD_COLLECTION = "job_description"
RESUME_COLLECTION = "resumes"
