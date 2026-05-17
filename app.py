import re
import streamlit as st
from youtube_transcript_api import TranscriptsDisabled
from yt_backend import load_video


def extract_video_id(url: str) -> str | None:
    # Handles: youtu.be/ID, youtube.com/watch?v=ID, or bare ID
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return None


st.set_page_config(page_title="YouTube Chatbot", page_icon="▶️")
st.title("YouTube Chatbot")

# --- Sidebar: load video ---
with st.sidebar:
    st.header("Load a Video")
    url_input = st.text_input("YouTube URL or video ID", placeholder="https://youtube.com/watch?v=...")

    if st.button("Load Video", use_container_width=True):
        video_id = extract_video_id(url_input)
        if not video_id:
            st.error("Could not parse a video ID from that URL.")
        elif video_id == st.session_state.get("current_video_id"):
            st.info("This video is already loaded.")
        else:
            with st.spinner("Fetching transcript and building index…"):
                try:
                    st.session_state.chain = load_video(video_id)
                    st.session_state.current_video_id = video_id
                    st.session_state.messages = []
                    st.success(f"Loaded `{video_id}`")
                except TranscriptsDisabled:
                    st.error("No captions available for this video.")
                except Exception as e:
                    st.error(f"Failed to load video: {e}")

    if st.session_state.get("current_video_id"):
        st.caption(f"Active video: `{st.session_state.current_video_id}`")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if question := st.chat_input("Ask something about the video…"):
    if not st.session_state.get("chain"):
        st.warning("Load a YouTube video from the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = st.session_state.chain.invoke(question)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
