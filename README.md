# YouTube Chatbot

A RAG-powered chatbot that lets you ask questions about any YouTube video instead of watching the whole thing.

## How it works

1. Paste a YouTube URL in the sidebar
2. The app fetches the video transcript and builds a searchable index
3. Ask any question — the bot retrieves the most relevant parts of the transcript and answers using GPT-4o-mini

## Tech stack

- **Streamlit** — chat UI
- **LangChain** — RAG pipeline
- **FAISS** — vector store for transcript chunks
- **OpenAI** — embeddings (`text-embedding-3-small`) and LLM (`gpt-4o-mini`)
- **MMR retrieval** — reduces redundancy in retrieved chunks
- **Cross-encoder reranker** — `ms-marco-MiniLM-L-6-v2` reranks chunks before sending to the LLM

## Setup

```bash
git clone https://github.com/Manihaas1237-glitch/youtube-chatbot.git
cd youtube-chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_key_here
```

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501`, paste a YouTube URL in the sidebar, and start chatting.
