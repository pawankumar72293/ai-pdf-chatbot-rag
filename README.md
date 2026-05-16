# AI PDF Chatbot (RAG)

An AI-powered PDF Question Answering system built with Python, Streamlit, LangChain, ChromaDB, sentence-transformers, and Google Gemini.

## Features

- Upload any PDF document
- Automatically extract and chunk text
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Store vectors in ChromaDB
- Ask questions in natural language
- Retrieve relevant chunks using semantic similarity
- Generate answers using Google Gemini
- View retrieved chunks and vector previews

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- sentence-transformers
- Google Gemini API

## Project Structure

```text
ai-pdf-chatbot-rag/
│── app.py
│── README.md
│── requirements.txt
│── .gitignore
│── .env.example
│── LICENSE
│── data/
│── vector_db/
```

## Installation

```bash
git clone https://github.com/your-username/ai-pdf-chatbot-rag.git
cd ai-pdf-chatbot-rag

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Run the Application

```bash
streamlit run app.py
```

## How It Works

1. Upload a PDF.
2. Text is extracted and split into chunks.
3. Embeddings are generated locally.
4. Chunks are stored in ChromaDB.
5. User asks a question.
6. Relevant chunks are retrieved.
7. Gemini generates the final answer.

## Resume Description

Built an AI-powered PDF Chatbot using Python, Streamlit, LangChain, ChromaDB, sentence-transformers, and Google Gemini to implement Retrieval-Augmented Generation (RAG) over uploaded PDF documents.

## License

MIT License.
