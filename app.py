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
if not os.getenv("GROQ_API_KEY"):
    print("⚠️ WARNING: GROQ_API_KEY is missing!")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("--- STATUS: Loading AI Model... ---")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
print("--- STATUS: Model Ready! ---")

def get_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file.name)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return ""

def split_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

def rag_pipeline(pdf_file, user_question, chat_history):
    if chat_history is None:
        chat_history = []

    if not user_question.strip():
        return "", chat_history

    # 1. Add User Question to History (UI State)
    chat_history.append({"role": "user", "content": user_question})

    if not pdf_file:
        chat_history.append({"role": "assistant", "content": "⚠️ Please upload a PDF file first."})
        return "", chat_history

    try:
        # --- A. Process PDF ---
        raw_text = get_text_from_pdf(pdf_file)
        if not raw_text:
            chat_history.append({"role": "assistant", "content": "⚠️ Error: The PDF appears to be empty."})
            return "", chat_history
            
        chunks = split_text(raw_text)
        
        # --- B. Search Engine ---
        embeddings = embedder.encode(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        
        question_embedding = embedder.encode([user_question])
        distances, indices = index.search(question_embedding, k=3)
        retrieved_context = "\n\n".join([chunks[i] for i in indices[0]])

        # --- C. BUILD CLEAN API MESSAGES (CRITICAL FIX) ---
        # We must create a NEW list for Groq that DOES NOT contain Gradio's 'metadata'
        
        system_prompt = {
            "role": "system", 
            "content": f"You are a helpful assistant. Use the context below to answer. If the answer is not in the context, say 'I don't know based on this document'.\n\nCONTEXT:\n{retrieved_context}"
        }
        
        # Start with System Prompt
        api_messages = [system_prompt]
        
        # Loop through history and ONLY copy 'role' and 'content'
        for msg in chat_history:
            api_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })

        # --- D. Call API ---
        chat_completion = client.chat.completions.create(
            messages=api_messages,
            model="llama-3.1-8b-instant",
            temperature=0.5,
        )
        
        bot_answer = chat_completion.choices[0].message.content
        
        # 2. Add Bot Answer to History (UI State)
        chat_history.append({"role": "assistant", "content": bot_answer})
        
        return "", chat_history

    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"System Error: {str(e)}"})
        return "", chat_history

# --- UI SETUP ---
with gr.Blocks(title="RAG Chatbot with Memory") as demo:
    gr.Markdown("""
    # 🤖 RAG Chatbot (With Memory)
    Upload a PDF and ask questions. I can now remember what we discussed earlier!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Upload Document")
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])

        with gr.Column(scale=2):
            gr.Markdown("### 2. Chat Interface")
            
            # Use standard Chatbot. 
            # Note: We removed 'type="messages"' to support your version, 
            # but we are manually feeding it dictionaries in the function above.
            chatbot_output = gr.Chatbot(label="Conversation", height=500)
            
            user_input = gr.Textbox(label="Your Question", placeholder="Ask something...")
            
            with gr.Row():
                clear_btn = gr.ClearButton([user_input, chatbot_output])
                submit_btn = gr.Button("Submit", variant="primary")

    # --- EVENT LISTENERS ---
    submit_btn.click(
        fn=rag_pipeline, 
        inputs=[pdf_input, user_input, chatbot_output], 
        outputs=[user_input, chatbot_output] 
    )
    
    user_input.submit(
        fn=rag_pipeline, 
        inputs=[pdf_input, user_input, chatbot_output], 
        outputs=[user_input, chatbot_output]
    )

if __name__ == "__main__":
    demo.launch()