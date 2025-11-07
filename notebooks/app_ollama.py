import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Local CSV Chat (Ollama)", page_icon="📚")
st.title("📚 Chat with your CSV — 100% Local (Ollama)")

# ---------- Sidebar settings ----------
with st.sidebar:
    st.header("Settings")
    model = st.text_input("Ollama model", value="phi3",
                          help="Examples: llama3, llama3:8b-instruct, phi3, mistral:instruct")
    top_k = st.slider("Top-k rows as context", 1, 10, 3)
    max_rows = st.number_input("Limit rows (speed)", 100, 100000, 1000, step=100)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max new tokens", 32, 1024, 256, 32)
    if st.button("🔄 Reset chat"):
        st.session_state.messages = [{"role": "assistant", "content": "New chat started!"}]
        st.rerun()

# ---------- CSV upload ----------
file = st.file_uploader("Upload a CSV to start", type=["csv"])
if file is None:
    st.info("Upload a CSV (e.g., students.csv, products.csv) to begin.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_df(_file, _max_rows):
    df = pd.read_csv(_file)
    return df.head(_max_rows).copy() if len(df) > _max_rows else df

df = load_df(file, max_rows)
st.success(f"Loaded {len(df):,} rows × {len(df.columns)} cols")
st.dataframe(df.head(10), use_container_width=True)

# Choose columns to index
text_cols = st.multiselect(
    "Columns to build searchable text (pick titles/notes/description/key fields):",
    options=list(df.columns),
    default=list(df.columns[: min(3, len(df.columns))])
)
if not text_cols:
    st.warning("Select at least one column for the retriever.")
    st.stop()

# ---------- Build TF-IDF retriever ----------
@st.cache_data(show_spinner=False)
def build_corpus_vectors(_df: pd.DataFrame, _cols):
    text_series = _df[_cols].astype(str).apply(lambda r: " | ".join(r.values), axis=1)
    vec = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(text_series.values)
    return text_series, vec, X

text_series, vectorizer, X = build_corpus_vectors(df, text_cols)

def retrieve(query: str, k: int):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask something about your CSV and I'll ground my answer on matching rows."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Ollama call (local) ----------
OLLAMA_CHAT_URL = "http://localhost:11434/api/generate"  # using /generate for simple prompting

def stream_from_ollama(prompt: str):
    try:
        with requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            },
            stream=True,
            timeout=0xFFFF,
        ) as r:
            r.raise_for_status()
            full = ""
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    token = data["response"]
                    full += token
                    yield token
                if data.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield "⚠️ Cannot reach Ollama at http://localhost:11434. Is `ollama serve` running?"
    except Exception as e:
        yield f"⚠️ Error: {e}"

# ---------- Prompt template ----------
SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. Answer ONLY using the provided CSV CONTEXT rows. "
    "If the answer is not in the context, say you cannot find it."
)

def build_prompt(user_q: str, rows_md: str) -> str:
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"QUESTION:\n{user_q}\n\n"
        f"CONTEXT (CSV rows):\n{rows_md}\n\n"
        f"ANSWER:"
    )

# ---------- Handle user question ----------
if user_q := st.chat_input("Ask a question about the CSV…"):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Searching relevant rows…"):
            idxs, scores = retrieve(user_q, top_k)
            top_rows = df.iloc[idxs]
            st.caption("Top-matching rows (used as context):")
            st.dataframe(top_rows, use_container_width=True)

            # Compact context block
            rows_md = "\n".join(
                f"- ROW {i}: " + " | ".join(f"{c}={str(top_rows.iloc[i][c])}" for c in text_cols)
                for i in range(len(top_rows))
            )

        with st.spinner("Generating answer (local model)…"):
            prompt = build_prompt(user_q, rows_md)
            placeholder = st.empty()
            acc = ""
            for tok in stream_from_ollama(prompt):
                acc += tok
                placeholder.markdown(acc)

    st.session_state.messages.append({"role": "assistant", "content": acc})
