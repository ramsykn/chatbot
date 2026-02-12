import streamlit as st
import wikipedia
from transformers import pipeline

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Accurate QA Chatbot", page_icon="🤖")

st.title("🤖 Knowledge-Based QA Chatbot")
st.write("Ask factual questions. Answers are grounded from Wikipedia.")

# -------------------------
# Load QA Model
# -------------------------

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_qa_model()

# -------------------------
# Function to Get Answer
# -------------------------

def get_answer(question):
    try:
        # Search Wikipedia
        search_results = wikipedia.search(question)
        page = wikipedia.page(search_results[0])
        context = page.content[:2000]  # limit for speed
        
        result = qa_pipeline(question=question, context=context)
        
        return result["answer"]
    
    except Exception:
        return "Sorry, I couldn't find reliable information."

# -------------------------
# Chat Interface
# -------------------------

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask your question:")

if st.button("Submit") and user_input:
    answer = get_answer(user_input)
    st.session_state.history.append((user_input, answer))

# Display Chat
for question, answer in st.session_state.history:
    st.markdown(f"**🧑 You:** {question}")
    st.markdown(f"**🤖 Bot:** {answer}")
