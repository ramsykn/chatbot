import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------
# Page Setup
# -----------------------------

st.set_page_config(
    page_title="AI Bank Assistant",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 AI Bank Services Assistant")
st.write("Ask any question related to banking services.")

# -----------------------------
# Load Model
# -----------------------------

@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# -----------------------------
# Response Generator
# -----------------------------

def generate_response(user_input):

    # Restrict to banking domain
    prompt = f"""
You are a professional banking assistant.
Only answer questions related to banking services such as accounts, loans, cards, transactions, balance, statements, online banking, KYC, ATM, and finance.
If the question is not related to banking, politely say you only answer banking-related questions.

Question: {user_input}

Answer clearly and professionally:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_length=256,
        temperature=0.5,   # lower = more factual
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

# -----------------------------
# Chat Memory
# -----------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Enter your banking question:")

if st.button("Submit") and user_input:
    answer = generate_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", answer))

# Display conversation
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🏦 Assistant:** {text}")
