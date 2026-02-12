import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Page Setup
# -------------------------

st.set_page_config(page_title="Mistral AI Chatbot", page_icon="🤖")
st.title("🤖 Mistral-7B Intelligent Chatbot")
st.write("Ask me anything.")

# -------------------------
# Load Model (Cached)
# -------------------------

@st.cache_resource
def load_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------
# Generate Response
# -------------------------

def generate_response(question):
    prompt = f"<s>[INST] {question} [/INST]"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("[/INST]")[-1].strip()
    
    return response

# -------------------------
# Chat Memory
# -------------------------

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:")

if st.button("Send") and user_input:
    answer = generate_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", answer))

# -------------------------
# Display Chat
# -------------------------

for role, text in st.session_state.history:
    if role == "You":
        st.markdown(f"**🧑 You:** {text}")
    else:
        st.markdown(f"**🤖 Bot:** {text}")
