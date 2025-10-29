import streamlit as st
from app.query_engine import answer_with_gpt5

st.set_page_config(page_title="JosefGPT Local", layout="wide")
st.title("🧠 JosefGPT — Hybrid Reasoning Chat")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_area("💬 Your question", "", height=100)

if st.button("Ask") and user_input.strip():
    with st.spinner("Thinking..."):
        answer = answer_with_gpt5(user_input)
        st.session_state.history.append((user_input, answer))

for q, a in reversed(st.session_state.history):
    st.markdown(f"**🧍‍♂️ You:** {q}")
    st.markdown(f"**🤖 JosefGPT:** {a}")
    st.markdown("---")
