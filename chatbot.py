import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="AI Answering Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Intelligent AI Chatbot")
st.write("Ask me any question!")

# -----------------------------
# Load Model (Cached)
# -----------------------------

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Chat Memory
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# User Input
# -----------------------------

user_input = st.text_input("You:", placeholder="Type your question here...")

if st.button("Send") and user_input:

    prompt = f"Answer clearly and accurately:\n{user_input}"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# -----------------------------
# Display Chat
# -----------------------------

for role, text in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")
