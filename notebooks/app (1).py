# file: app.py
import streamlit as st

st.title("LLM Chat — Minimal Demo (no API)")

# 1) init history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything 😊"}
    ]

# 2) render past messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 3) simple local “responder” (placeholder for a real LLM)
def local_reply(user_text: str) -> str:
    user_text_l = user_text.lower().strip()
    if "python" in user_text_l:
        return "Python is a high-level, general-purpose programming language."
    if "streamlit" in user_text_l:
        return "Streamlit lets you build Python web apps quickly—perfect for data and AI demos."
    if user_text_l.endswith("?"):
        return "Great question! What do you think the answer might be?"
    return f"You said: {user_text} (try asking about Python or Streamlit!)"

# 4) handle new input
if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # “generate” assistant reply
    reply = local_reply(prompt)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    with st.chat_message("assistant"):
        st.markdown(reply)
/