LLM(RAG)-Based Document Processing System
Overview

This project is an LLM-powered system for processing and querying PDF documents. It allows users to upload PDF documents, extract text, generate embeddings, retrieve relevant content, and ask questions using a Retrieval-Augmented Generation (RAG) pipeline. The system is built with a FastAPI backend and a React frontend.

Features
Document Upload and Processing: Upload PDF files, extract text using pdfplumber, and chunk the text for efficient querying.
Vector Embeddings: Generate embeddings using Google's Gemini embedding model for semantic search.
Hybrid Retrieval: Combines semantic similarity and keyword matching to retrieve relevant document chunks.
Question Answering: Uses Gemini 3.6 Flash for reasoning and response generation based on retrieved document context.
Structured Responses: Returns JSON responses containing decision, amount, justification, matched clauses, and highlights.
Local Fallback: Automatically uses a rule-based fallback when Gemini generation fails due to quota limits, rate limits, API errors, or invalid responses.
In-Memory Storage: Simple in-memory storage for document chunks and embeddings.
RESTful API: FastAPI-based endpoints for document upload and querying.
React Frontend: User interface built with React and Vite.
Tech Stack
Backend:
FastAPI: Web framework for building APIs.
pdfplumber: PDF text extraction.
LangChain: Text splitting and processing.
Google GenAI SDK: Gemini API integration.
NumPy: Vector operations and similarity search.
Frontend:
React: UI library.
Vite: Build tool and development server.
Other:
Python 3.x
Node.js
LLM Model Used
Embeddings: gemini-embedding-2-preview (Google Gemini)
Reasoning/Question Answering: gemini-3.6-flash (Google Gemini)
Setup Instructions
Prerequisites
Python 3.10+
Node.js
Gemini API Key
Backend Setup

Navigate to the backend directory:

cd backend

Install dependencies:

pip install -r requirements.txt

Set the Gemini API key:

Windows PowerShell:

$env:GEMINI_API_KEY="your_api_key_here"

Run the server:

uvicorn main:app --reload

The API will be available at http://localhost:8000.

Frontend Setup

Navigate to the frontend directory:

cd frontend

Install dependencies:

npm install

Run the development server:

npm run dev

The frontend will be available at http://localhost:5173.

Usage
Start both backend and frontend servers.
Use the frontend to upload a PDF document.
Ask questions about the document content.
The system retrieves relevant document chunks and generates a structured response.
API Endpoints
GET /: Health check.
POST /upload/: Upload and process a PDF document.
POST /query/: Ask a question about the document and receive a structured JSON response.
Project Structure
backend/: FastAPI application.
frontend/: React application.
data/: Sample data files.
.gitignore: Git ignore configuration.
Contributing

Contributions are welcome! Please open an issue or submit a pull request.

License

This project is licensed under the MIT License.
