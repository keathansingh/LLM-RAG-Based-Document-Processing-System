# LLM(RAG)-Based Document Processing System

## Overview

This project is an LLM-powered system for processing and querying PDF documents using a Retrieval-Augmented Generation (RAG) pipeline. It allows users to upload PDF documents, extract text, generate embeddings, retrieve relevant content, and ask questions through a FastAPI backend and React frontend.

## Features

- **Document Upload and Processing**: Upload PDF files, extract text using pdfplumber, and chunk the text using LangChain.
- **Vector Embeddings**: Generate embeddings using Google's Gemini embedding model for semantic search.
- **Hybrid Retrieval**: Combines semantic similarity and keyword matching to retrieve relevant document chunks.
- **Question Answering**: Uses Gemini 3.6 Flash for reasoning and response generation based on retrieved document context.
- **Structured Responses**: Returns JSON responses containing decision, amount, justification, matched clauses, and highlights.
- **Local Fallback**: Automatically uses a rule-based fallback when Gemini generation fails due to quota limits, rate limits, API errors, or invalid responses.
- **In-Memory Storage**: Stores document chunks and embeddings in memory for simple and fast retrieval.
- **RESTful API**: FastAPI-based backend for document upload and querying.
- **React Frontend**: User interface built with React and Vite.

## Tech Stack

- **Backend**:
  - FastAPI: Web framework for building APIs.
  - pdfplumber: PDF text extraction.
  - LangChain: Text splitting and processing.
  - Google GenAI SDK: Gemini embeddings and LLM interaction.
  - NumPy: Vector operations and similarity search.

- **Frontend**:
  - React: UI library.
  - Vite: Build tool and development server.
  - JavaScript
  - CSS

- **Other**:
  - Python 3.x
  - Node.js
  - npm

## LLM Model Used

- **Embeddings**: `gemini-embedding-2-preview` (Google Gemini)
- **Reasoning/Question Answering**: `gemini-3.6-flash` (Google Gemini)

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js
- npm
- Gemini API Key

### Backend Setup

1. Navigate to the `backend` directory:

    ````bash
    cd backend
    ````

2. Install dependencies:

    ````bash
    pip install -r requirements.txt
    ````

3. Set the Gemini API key.

   Windows PowerShell:

    ````powershell
    $env:GEMINI_API_KEY="your_api_key_here"
    ````

4. Run the server:

    ````bash
    uvicorn main:app --reload
    ````

   The API will be available at `http://localhost:8000`.

### Frontend Setup

1. Navigate to the `frontend` directory:

    ````bash
    cd frontend
    ````

2. Install dependencies:

    ````bash
    npm install
    ````

3. Run the development server:

    ````bash
    npm run dev
    ````

   The frontend will be available at `http://localhost:5173`.

## Usage

1. Start both the backend and frontend servers.
2. Use the frontend to upload a PDF document.
3. The system extracts and processes the document.
4. Ask questions about the uploaded document.
5. The system retrieves relevant document chunks and generates a structured response.

## API Endpoints

- `GET /`: Health check.
- `POST /upload/`: Upload and process a PDF document.
- `POST /query/`: Ask a question about the document and receive a structured JSON response.

## Project Structure

- `backend/`: FastAPI application and RAG processing logic.
- `frontend/`: React and Vite application.
- `data/`: Sample document data.
- `.gitignore`: Git ignore configuration.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
