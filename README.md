# 🤖 AI Backend Assistant

A production-style AI backend service that combines:

* 🧠 LLM (OpenAI)
* 🗂️ Retrieval-Augmented Generation (RAG)
* 📄 PDF ingestion
* ⚡ FastAPI backend
* 🔧 Tool-based AI agent

---

## 🚀 Features

* Ask backend/system design questions
* Upload PDFs and query them with AI
* Multi-step AI agent with tool usage
* Persistent vector database (ChromaDB)

---

## 🏗️ Architecture

Client → FastAPI → Agent → RAG → LLM → Response

---

## 📦 Tech Stack

* Python
* FastAPI
* OpenAI API
* ChromaDB
* PyPDF

---

## ⚙️ Setup

```bash
git clone <your-repo-url>
cd ai-backend-assistant

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```text
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Run Server

```bash
uvicorn api:app --reload
```

---

## 📄 API Endpoints

### Ask AI

POST `/ask`

```json
{
  "question": "Design a scalable system"
}
```

---

### Upload PDF

POST `/upload`

Upload a PDF file → automatically indexed for querying

---

## 🧠 Example Questions

* "What does the document say about DynamoDB?"
* "Design a scalable backend system"
* "Explain partition key best practices"

---

## 💼 Resume Description

Built a FastAPI-based AI backend service with a multi-step agent architecture, tool orchestration, and Retrieval-Augmented Generation (RAG) using ChromaDB and PDF ingestion.

---
