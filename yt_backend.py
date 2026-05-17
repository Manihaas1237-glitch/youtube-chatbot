from functools import lru_cache

import os
from dotenv import load_dotenv
load_dotenv()

try:
    import streamlit as st
    if "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass
    
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def fetch_transcript(video_id: str) -> str:
    ytt_api = YouTubeTranscriptApi()
    fetched = ytt_api.fetch(video_id, languages=["en"])
    return " ".join(snippet.text for snippet in fetched)


def build_vector_store(transcript: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)


@lru_cache(maxsize=8)
def _get_vector_store(video_id: str) -> FAISS:
    # Combines fetch + embed so the cache key is just the video ID.
    # Subsequent calls for the same ID skip the API and embedding entirely.
    transcript = fetch_transcript(video_id)
    return build_vector_store(transcript)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoderReranker:
    # Lazy-loaded once; the cross-encoder model download only happens on first call.
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    return CrossEncoderReranker(model=model, top_n=4)


def build_chain(vector_store: FAISS):
    # MMR: fetch_k=20 candidate chunks, then pick k=6 diverse ones for the reranker.
    # lambda_mult=0.5 balances relevance vs. diversity (0=max diversity, 1=pure similarity).
    base_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.5},
    )

    retriever = ContextualCompressionRetriever(
        base_compressor=_get_reranker(),
        base_retriever=base_retriever,
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, just say you don't know.

{context}
Question: {question}""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })

    return parallel_chain | prompt | llm | parser


def load_video(video_id: str):
    vector_store = _get_vector_store(video_id)
    return build_chain(vector_store)


if __name__ == "__main__":
    video_id = "Gfr50f6ZBvo"

    try:
        chain = load_video(video_id)
        answer = chain.invoke("Can you summarize the video?")
        print(answer)
    except TranscriptsDisabled:
        print("No captions available for this video.")
