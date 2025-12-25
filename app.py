import os
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# 1. Setup
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("--- STATUS: Loading AI Model... ---")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("--- STATUS: Model Ready! ---")

def get_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file.name)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def rag_pipeline(pdf_file, user_question):
    if not pdf_file:
        return "⚠️ Please upload a PDF file first.", "" # Return empty string to clear input
    
    try:
        # --- A. Process PDF ---
        raw_text = get_text_from_pdf(pdf_file)
        chunks = split_text(raw_text)
        
        if not chunks:
            return "Could not extract text from PDF.", ""

        # --- B. Search Engine ---
        embeddings = embedder.encode(chunks)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        # --- C. Search ---
        question_embedding = embedder.encode([user_question])
        distances, indices = index.search(question_embedding, k=3)
        retrieved_context = "\n\n".join([chunks[i] for i in indices[0]])

        # --- D. Ask Llama 3.1 ---
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant. Answer strictly based on the context below:\n\n{retrieved_context}"
                },
                {
                    "role": "user",
                    "content": user_question,
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0,
        )
        
        answer = chat_completion.choices[0].message.content
        return answer, ""  # <--- MAGIC FIX: Return answer AND empty string

    except Exception as e:
        return f"Error: {str(e)}", ""

# --- UI SETUP ---
with gr.Blocks(theme=gr.themes.Soft(), title="RAG Based ChatBot") as demo:
    gr.Markdown("""
    # 🤖 RAG Based ChatBot
    Upload a PDF document and ask questions about its content.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Document")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])

        with gr.Column(scale=2):
            gr.Markdown("### 2. Ask Questions")
            chatbot_output = gr.Textbox(label="AI Answer", lines=10, interactive=False)
            user_input = gr.Textbox(label="Your Question", placeholder="e.g., What are the main requirements?")
            
            with gr.Row():
                clear_btn = gr.ClearButton([user_input, chatbot_output])
                submit_btn = gr.Button("Submit", variant="primary")

    # --- EVENT LISTENERS ---
    # We now map outputs to TWO things: [chatbot_output, user_input]
    # This means: "Put the answer in the big box, and put emptiness in the small box."
    
    submit_btn.click(
        fn=rag_pipeline, 
        inputs=[pdf_input, user_input], 
        outputs=[chatbot_output, user_input] 
    )
    
    user_input.submit(
        fn=rag_pipeline, 
        inputs=[pdf_input, user_input], 
        outputs=[chatbot_output, user_input]
    )

if __name__ == "__main__":
    demo.launch()