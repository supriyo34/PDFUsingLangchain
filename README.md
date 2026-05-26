# PDF RAG Chatbot using Gemini + ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot built with:

- Gemini 2.5 Flash
- LangChain
- ChromaDB
- Sentence Transformers
- PDF document processing

## Features

✔ Upload PDF files  
✔ Split documents into chunks  
✔ Generate embeddings  
✔ Store embeddings in ChromaDB  
✔ Semantic search  
✔ Gemini-powered answers  

---

## Project Architecture

PDF → Chunking → Embeddings → ChromaDB → Retrieval → Gemini → Answer

---

## Installation

Clone repository:

```bash
git clone https://github.com/yourusername/rag-pdf-chatbot.git
```

Move into project:

```bash
cd rag-pdf-chatbot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create .env:

```env
GOOGLE_API_KEY=your_api_key_here
```

Run project:

```bash
python app.py
```

---

## Example

Input:

```text
Ask: What is the leave policy?
```

Output:

```text
Employees receive 15 paid leaves annually.
```

---

## Tech Stack

Python  
LangChain  
Gemini API  
Sentence Transformers  
ChromaDB  
