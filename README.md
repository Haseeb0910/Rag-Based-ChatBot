Markdown

# 📚 RAG Chatbot with Conversational Memory & Multi-Format Support

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Gradio](https://img.shields.io/badge/Frontend-Gradio-orange)
![Llama 3](https://img.shields.io/badge/Model-Llama%203.1-purple)
![License](https://img.shields.io/badge/License-MIT-green)

A robust **Retrieval-Augmented Generation (RAG)** application designed to bridge the gap between static documents and dynamic AI conversation. This chatbot allows users to upload **PDF** and **Word (DOCX)** documents, automatically summarizes them, and engages in context-aware dialogue using **Llama 3** (via Groq) and **FAISS** vector search.

---

## 🌟 Live Demo
**[Try the App on Hugging Face Spaces](https://huggingface.co/spaces/haseeb0910/Rag-Based-ChatBot)**

---

## ✨ Key Features

* **📂 Multi-Format Ingestion:** Seamlessly processes both **PDF** and **DOCX** files using custom text extraction pipelines.
* **🧠 Conversational Memory:** Unlike standard RAG bots, this system maintains chat history, allowing for follow-up questions (e.g., *"Explain that point further"*) without losing context.
* **📝 Auto-Summarization:** Automatically generates a concise **3-sentence summary** immediately upon file upload, giving users a quick overview before they even ask a question.
* **⚡ Ultra-Fast Inference:** Powered by **Llama 3.1-8b-instant** via the **Groq API**, delivering near real-time responses.
* **🔍 Semantic Search:** Utilizes `sentence-transformers` and **FAISS** (Facebook AI Similarity Search) to retrieve highly relevant context chunks.
* **🛡️ Robust Error Handling:** Includes a sanitized pipeline to handle API metadata conflicts and empty file errors gracefully.

---

## 🛠️ Tech Stack

* **LLM Engine:** [Groq API](https://groq.com/) (Llama 3.1-8b-instant)
* **Interface:** [Gradio](https://gradio.app/) (Block-based UI with Markdown support)
* **Vector Database:** FAISS (CPU)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Document Processing:** `pypdf` (PDFs), `python-docx` (Word Docs)
* **Environment Management:** `python-dotenv`

---

## 🚀 Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)[YOUR_USERNAME]/rag-chatbot.git
cd rag-chatbot
2. Create a Virtual Environment (Recommended)
Bash

python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Set Up Environment Variables
Create a .env file in the root directory and add your Groq API key:

Code snippet

GROQ_API_KEY=gsk_your_actual_api_key_here
(You can get a free API key from console.groq.com)

5. Run the Application
Bash

python app.py
The app will launch in your browser at http://127.0.0.1:7860.

📂 Project Structure
rag-chatbot/
├── app.py                  # Main application logic (Gradio + RAG pipeline)
├── requirements.txt        # List of dependencies
├── .env                    # API keys (not uploaded to GitHub)
├── .gitignore              # Files to ignore (venv, .env, __pycache__)
└── README.md               # Project documentation

Special thanks to the open-source community behind Hugging Face and Gradio.

📜 License
This project is licensed under the MIT License - feel free to use it for your own learning!
