# LLM(RAG)-Based Document Processing System

## Overview
This project is an LLM-powered system for processing and querying insurance policy documents. It allows users to upload PDF documents, extract text, generate embeddings, and ask questions about the content using a Retrieval-Augmented Generation (RAG) pipeline. The system is built with a FastAPI backend for document processing and a React frontend for user interaction.

## Features
- **Document Upload and Processing**: Upload PDF files, extract text using pdfplumber, and chunk the text for efficient querying.
- **Vector Embeddings**: Generate embeddings using OpenAI's text-embedding-3-small model for semantic search.
- **Question Answering**: Ask questions about the uploaded document using a RAG pipeline with GPT-4-turbo for reasoning and response generation.
- **In-Memory Storage**: Simple in-memory data store for document chunks and embeddings (suitable for hackathons; use a database for production).
- **RESTful API**: FastAPI-based backend with endpoints for upload, embedding, and querying.
- **React Frontend**: User interface built with React and Vite for easy interaction.

## Tech Stack
- **Backend**:
  - FastAPI: Web framework for building APIs.
  - pdfplumber: PDF text extraction.
  - LangChain: Text splitting and processing.
  - OpenAI API: For embeddings and LLM interactions.
  - NumPy: Vector operations for similarity search.
- **Frontend**:
  - React: UI library.
  - Vite: Build tool and development server.
- **Other**:
  - Python 3.x
  - Node.js (for frontend)

## LLM Model Used
- **Embeddings**: text-embedding-3-small (OpenAI)
- **Reasoning/Question Answering**: gpt-4-turbo (OpenAI)

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+
- OpenAI API Key (set as environment variable `OPENAI_API_KEY`)

### Backend Setup
1. Navigate to the `backend` directory:
   ```
   cd backend
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set the OpenAI API key:
   ```
   export OPENAI_API_KEY=your_api_key_here
   ```
4. Run the server:
   ```
   uvicorn main:app --reload
   ```
   The API will be available at `http://localhost:8000`.

### Frontend Setup
1. Navigate to the `frontend` directory:
   ```
   cd frontend
   ```
2. Install dependencies:
   ```
   npm install
   ```
3. Run the development server:
   ```
   npm run dev
   ```
   The frontend will be available at `http://localhost:5173`.

## Usage
1. Start both backend and frontend servers.
2. Use the frontend to upload a PDF document.
3. Generate embeddings for the document.
4. Ask questions about the document content.

## API Endpoints
- `GET /`: Health check.
- `POST /upload/`: Upload and process a PDF document.
- `POST /embed/`: Generate embeddings for the uploaded document.
- `POST /query/`: Ask a question about the document (returns JSON response with decision, justification, etc.).

## Project Structure
- `backend/`: FastAPI application.
- `frontend/`: React application.
- `data/`: Sample data files.
- `scratch/`: Verification and testing scripts.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
