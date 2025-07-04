import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import numpy as np
import pickle

st.set_page_config(page_title="Honda City QA", layout="wide")

st.title("🚗 Honda City Manual Q&A")

# Load models (using Streamlit cache to avoid reloading)
@st.cache_resource
def load_models():
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    generator_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    generator_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return retriever_model, generator_tokenizer, generator_model

retriever_model, generator_tokenizer, generator_model = load_models()

# Load chunks and FAISS index
chunks = pickle.load(open("chunks.pkl", "rb"))
index = faiss.read_index("index.faiss")

# user input
question = st.text_input("Ask me anything about the Honda City manual:")

if question:
    # retrieve
    query_embedding = retriever_model.encode([question]).astype('float32')
    distances, indices = index.search(query_embedding, 3)
    top_chunks = [chunks[i] for i in indices[0]]

    # generate
    context = top_chunks[0]

    prompt = f"""
    You are a helpful Honda City car manual assistant.
    Use the context below to answer the question with as much detail as possible.
    First, think step by step through the problem, then present a numbered list of steps in the final answer.
    If you don't know, say "I don't know".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    inputs = generator_tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = generator_model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    top_k=50,
    top_p=0.9
)


    answer = generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    with st.expander("Show retrieval context"):
     st.write(context)

     st.markdown(f"**Answer:** {answer}")

