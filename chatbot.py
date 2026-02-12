import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 Intelligent Chatbot (6GB RAM Optimized)")

# ----------------------------
# Load Model
# ----------------------------

@st.cache_resource
def load_model():
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        load_in_4bit=True  # reduces memory usage
    )

    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# Generate Response
# ----------------------------

def generate_response(question):
    prompt = f"<|user|>\n{question}\n<|assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("<|assistant|>")[-1].strip()

    return response

# ----------------------------
# Chat Memory
# ----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

if st.button("Send") and user_input:
    answer = generate_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", answer))

for role, text in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")
