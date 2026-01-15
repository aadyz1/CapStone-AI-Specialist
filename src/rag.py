import truststore 
truststore.inject_into_ssl()
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma


from src.config import CHROMA_DIR, JD_COLLECTION, RESUME_COLLECTION, OPENAI_API_KEY


def get_vectorstores():
    """
    Loads existing Chroma collections.
    (They are created/populated in ingest.py)
    """
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    jd_vs = Chroma(
        collection_name=JD_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    resume_vs = Chroma(
        collection_name=RESUME_COLLECTION,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    return jd_vs, resume_vs


def retrieve_jd_context(query: str, k: int = 5) -> str:
    """
    RAG step: retrieve the most relevant JD chunks for a query.
    """
    jd_vs, _ = get_vectorstores()
    docs = jd_vs.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])


def retrieve_resume_context(candidate_id: str, query: str, k: int = 5) -> str:
    """
    RAG step: retrieve the most relevant resume chunks for ONE candidate.
    Filtering prevents mixing multiple resumes.
    """
    _, resume_vs = get_vectorstores()
    docs = resume_vs.similarity_search(
        query,
        k=k,
        filter={"candidate_id": candidate_id},
    )
    return "\n\n".join([d.page_content for d in docs])
