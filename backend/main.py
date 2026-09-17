from fastapi import FastAPI, UploadFile, File, HTTPException
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import os
from pydantic import BaseModel
import json
import re
from google import genai
from google.genai import types


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Insurance Policy Q&A System",
    description=(
        "A Gemini-powered RAG system to answer "
        "questions about insurance policies."
    ),
)


# =========================================================
# DOCUMENT STORE
# =========================================================

document_store = {
    "filename": "",
    "text": "",
    "chunks": [],
    "embeddings": np.array([]),
}


# =========================================================
# GEMINI CONFIGURATION
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:

    print(
        "WARNING: GEMINI_API_KEY environment variable "
        "is not set."
    )

    gemini_client = None

else:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "Gemini client initialized successfully."
        )

    except Exception as e:

        print(
            f"Error initializing Gemini client: {e}"
        )

        gemini_client = None


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def read_root():

    return {
        "status": "ok",
        "message": "Welcome to the Insurance Q&A System!"
    }


# =========================================================
# UPLOAD DOCUMENT
# =========================================================

@app.post(
    "/upload/",
    summary="Upload and Process a Document"
)
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only .pdf files are supported for now."
        )


    temp_dir = "/tmp/insurancedocs"

    os.makedirs(
        temp_dir,
        exist_ok=True
    )


    file_path = os.path.join(
        temp_dir,
        file.filename
    )


    try:

        # -------------------------------------------------
        # Save uploaded PDF
        # -------------------------------------------------

        with open(file_path, "wb") as f:

            contents = await file.read()

            f.write(contents)


        # -------------------------------------------------
        # Extract text from PDF
        # -------------------------------------------------

        full_text = ""


        with pdfplumber.open(file_path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:

                    full_text += (
                        page_text + "\n"
                    )


        if not full_text.strip():

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract any text from the PDF."
                )
            )


        # -------------------------------------------------
        # Split document into chunks
        # -------------------------------------------------

        text_splitter = RecursiveCharacterTextSplitter(

            chunk_size=500,

            chunk_overlap=50,

            length_function=len,

        )


        chunk_texts = (
            text_splitter.split_text(
                full_text
            )
        )


        # -------------------------------------------------
        # Store document
        # -------------------------------------------------

        document_store["filename"] = (
            file.filename
        )

        document_store["text"] = (
            full_text
        )

        document_store["chunks"] = (
            chunk_texts
        )

        document_store["embeddings"] = (
            np.array([])
        )


        print(
            f"\nDocument uploaded: "
            f"{file.filename}"
        )

        print(
            f"Total chunks: "
            f"{len(chunk_texts)}"
        )


        return {

            "message": (
                f"Successfully uploaded and processed "
                f"'{file.filename}'"
            ),

            "filename": file.filename,

            "num_chunks": len(chunk_texts),

        }


    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"An error occurred: "
                f"{str(e)}"
            )

        )


    finally:

        if os.path.exists(file_path):

            os.remove(file_path)


# =========================================================
# CREATE GEMINI EMBEDDINGS
# =========================================================

@app.post(
    "/embed/",
    summary="Create Embeddings for the Uploaded Document"
)
def embed_document():

    if gemini_client is None:

        raise HTTPException(

            status_code=500,

            detail=(
                "Gemini client is not initialized. "
                "Is GEMINI_API_KEY set?"
            )

        )


    if not document_store["chunks"]:

        raise HTTPException(

            status_code=404,

            detail=(
                "No document has been uploaded. "
                "Please use the /upload endpoint first."
            )

        )


    if (

        document_store.get("embeddings") is not None

        and

        document_store["embeddings"].size > 0

    ):

        return {

            "message": (
                "Embeddings have already been created "
                "for the current document."
            )

        }


    try:

        # -------------------------------------------------
        # IMPORTANT:
        # Create ONE Gemini embedding per chunk.
        # -------------------------------------------------

        embeddings = []


        for chunk_number, chunk in enumerate(
            document_store["chunks"],
            start=1
        ):

            print(
                f"Creating embedding "
                f"{chunk_number}/"
                f"{len(document_store['chunks'])}..."
            )


            response = (
                gemini_client.models.embed_content(

                    model="gemini-embedding-2-preview",

                    contents=chunk,

                )
            )


            if (
                not response.embeddings
                or
                not response.embeddings[0].values
            ):

                raise ValueError(
                    f"Gemini returned no embedding "
                    f"for chunk {chunk_number}."
                )


            embeddings.append(
                response.embeddings[0].values
            )


        # -------------------------------------------------
        # Convert embeddings to NumPy array
        # -------------------------------------------------

        document_store["embeddings"] = (
            np.array(
                embeddings,
                dtype=float
            )
        )


        print(
            f"\nCreated {len(embeddings)} embeddings."
        )

        print(
            f"Embedding shape: "
            f"{document_store['embeddings'].shape}"
        )


        return {

            "message": (
                f"Successfully created "
                f"{len(embeddings)} embeddings for "
                f"'{document_store['filename']}'."
            ),

            "num_embeddings": len(embeddings),

            "embedding_shape": (
                document_store["embeddings"].shape
            ),

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Gemini embedding error: "
                f"{str(e)}"
            )

        )


# =========================================================
# QUERY REQUEST MODEL
# =========================================================

class QueryRequest(BaseModel):

    query: str

    top_k: int = 5


# =========================================================
# COSINE SIMILARITY
# =========================================================

def cosine_similarity(v1, v2):

    dot_product = np.dot(
        v1,
        v2
    )


    norm_v1 = np.linalg.norm(
        v1
    )


    norm_v2 = np.linalg.norm(
        v2
    )


    if norm_v1 == 0 or norm_v2 == 0:

        return 0.0


    return (
        dot_product
        /
        (norm_v1 * norm_v2)
    )


# =========================================================
# KEYWORD EXTRACTION
# =========================================================

def extract_keywords(text):

    words = re.findall(
        r"\b[a-zA-Z0-9₹]+\b",
        text.lower()
    )


    stop_words = {

        "the",
        "is",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "under",
        "this",
        "that",
        "with",
        "what",
        "does",
        "do",
        "are",
        "can",
        "will",
        "be",
        "my",
        "i",
        "it",
        "policy",

    }


    keywords = {

        word

        for word in words

        if word not in stop_words

        and len(word) > 2

    }


    return keywords


# =========================================================
# KEYWORD SCORE
# =========================================================

def keyword_score(
    query,
    chunk
):

    query_keywords = (
        extract_keywords(query)
    )


    chunk_keywords = (
        extract_keywords(chunk)
    )


    if not query_keywords:

        return 0.0


    matching_keywords = (
        query_keywords.intersection(
            chunk_keywords
        )
    )


    return (
        len(matching_keywords)
        /
        len(query_keywords)
    )


# =========================================================
# HYBRID RETRIEVAL
# =========================================================

def hybrid_retrieval(
    query,
    query_embedding,
    top_k
):

    # -----------------------------------------------------
    # Semantic similarity
    # -----------------------------------------------------

    semantic_scores = np.array(

        [

            cosine_similarity(

                query_embedding,

                chunk_embedding

            )

            for chunk_embedding
            in document_store["embeddings"]

        ],

        dtype=float

    )


    # -----------------------------------------------------
    # Keyword matching
    # -----------------------------------------------------

    keyword_scores = np.array(

        [

            keyword_score(
                query,
                chunk
            )

            for chunk
            in document_store["chunks"]

        ],

        dtype=float

    )


    number_of_chunks = (
        len(document_store["chunks"])
    )


    # -----------------------------------------------------
    # Safety checks
    # -----------------------------------------------------

    if number_of_chunks == 0:

        return (
            np.array([], dtype=int),
            np.array([])
        )


    if (
        len(semantic_scores)
        !=
        number_of_chunks
    ):

        raise ValueError(

            "Number of embeddings does not match "
            "number of document chunks. "

            f"Chunks: {number_of_chunks}, "

            f"Embeddings: {len(semantic_scores)}"

        )


    if (
        len(keyword_scores)
        !=
        number_of_chunks
    ):

        raise ValueError(

            "Number of keyword scores does not match "
            "number of document chunks."

        )


    # -----------------------------------------------------
    # Normalize semantic scores
    # -----------------------------------------------------

    minimum_score = (
        semantic_scores.min()
    )


    maximum_score = (
        semantic_scores.max()
    )


    if maximum_score != minimum_score:

        normalized_semantic = (

            semantic_scores
            -
            minimum_score

        ) / (

            maximum_score
            -
            minimum_score

        )

    else:

        normalized_semantic = (
            np.zeros_like(
                semantic_scores
            )
        )


    # -----------------------------------------------------
    # Hybrid score
    #
    # 70% semantic similarity
    # 30% keyword matching
    # -----------------------------------------------------

    combined_scores = (

        0.70 * normalized_semantic

        +

        0.30 * keyword_scores

    )


    # -----------------------------------------------------
    # Rank chunks
    # -----------------------------------------------------

    ranked_indices = np.argsort(
        combined_scores
    )[::-1]


    top_k = min(

        max(
            1,
            top_k
        ),

        number_of_chunks

    )


    selected_indices = (
        ranked_indices[:top_k]
    )


    # -----------------------------------------------------
    # Debug output
    # -----------------------------------------------------

    print(
        "\n=============================================="
    )

    print(
        "HYBRID RETRIEVAL RESULTS"
    )

    print(
        "=============================================="
    )


    for rank, index in enumerate(

        selected_indices,

        start=1

    ):

        index = int(index)


        print(
            f"\n--- Rank {rank} ---"
        )


        print(
            f"Chunk Index: {index}"
        )


        print(
            f"Semantic Score: "
            f"{semantic_scores[index]:.4f}"
        )


        print(
            f"Keyword Score: "
            f"{keyword_scores[index]:.4f}"
        )


        print(
            f"Combined Score: "
            f"{combined_scores[index]:.4f}"
        )


        print(
            "Chunk:"
        )


        print(
            document_store["chunks"][index]
        )


    print(
        "\n=============================================="
    )

    print(
        "RETRIEVED CHUNKS"
    )

    print(
        "=============================================="
    )


    for rank, index in enumerate(

        selected_indices,

        start=1

    ):

        index = int(index)


        print(
            f"\n--- Chunk {rank} ---"
        )


        print(
            document_store["chunks"][index]
        )


    print(
        "\n==============================================\n"
    )


    return (
        selected_indices,
        combined_scores
    )


# =========================================================

# =========================================================
# LOCAL RULE-BASED FALLBACK
# =========================================================

def normalize_for_match(text):
    text = text.lower()
    text = text.replace("₹", "rs ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def local_rule_based_fallback(query, relevant_chunks):
    """
    Independent fallback. It never calls Gemini.
    It only uses the policy chunks already retrieved.
    """

    query_norm = normalize_for_match(query)
    query_keywords = extract_keywords(query)

    candidates = []

    for chunk in relevant_chunks:
        overlap = query_keywords.intersection(
            extract_keywords(chunk)
        )

        if overlap:
            candidates.append(
                (len(overlap), chunk)
            )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    # Explicit exclusions only.
    exclusion_patterns = [
        r"\bnot covered\b",
        r"\bdoes not cover\b",
        r"\bdo not cover\b",
        r"\bexcluded\b",
        r"\bnot payable\b",
        r"\bnot eligible\b",
    ]

    for _, chunk in candidates:
        normalized_chunk = normalize_for_match(chunk)

        if any(
            re.search(pattern, normalized_chunk)
            for pattern in exclusion_patterns
        ):
            return {
                "decision": "rejected",
                "amount": "As per policy",
                "justification": (
                    "The retrieved policy text contains an explicit "
                    "exclusion relevant to the requested item. "
                    "Gemini generation was unavailable, so the "
                    "local policy-evidence fallback was used."
                ),
                "matched_clauses": [{
                    "clause_id": "Retrieved policy clause",
                    "text": chunk,
                    "document": document_store["filename"]
                }],
                "highlights": [{
                    "type": "system",
                    "text": "Local fallback used because Gemini generation failed."
                }]
            }

    # Explicit coverage/value evidence.
    coverage_patterns = [
        r"\bcovers?\b",
        r"\bcovered\b",
        r"\bcoverage\b",
        r"\bbenefit\b",
        r"\bmaximum\b",
        r"\blimit\b",
        r"\bwaiting period\b",
        r"\bper policy year\b",
        r"\bper hospitalization\b",
    ]

    numeric_question = any(
        term in query_norm
        for term in [
            "how much",
            "how many",
            "amount",
            "limit",
            "maximum",
            "sum insured",
            "waiting period",
            "days",
            "months"
        ]
    )

    for _, chunk in candidates:
        normalized_chunk = normalize_for_match(chunk)

        if not any(
            re.search(pattern, normalized_chunk)
            for pattern in coverage_patterns
        ):
            continue

        money_matches = re.findall(
            r"(?:₹|rs\.?|n)\s?[\d,]+(?:\.\d+)?",
            chunk,
            flags=re.IGNORECASE
        )

        period_matches = re.findall(
            r"\b\d+\s*(?:days?|months?|hours?|years?)\b",
            chunk,
            flags=re.IGNORECASE
        )

        if numeric_question and not (
            money_matches or period_matches
        ):
            continue

        amount = "As per policy"

        if money_matches:
            amount = money_matches[0]
            if amount.lower().startswith("n"):
                amount = "₹" + amount[1:].lstrip()
        elif period_matches:
            amount = period_matches[0]

        return {
            "decision": "approved",
            "amount": amount,
            "justification": (
                "The retrieved policy text contains information "
                "supporting the requested coverage or policy value. "
                "Gemini generation was unavailable, so the "
                "local policy-evidence fallback was used."
            ),
            "matched_clauses": [{
                "clause_id": "Retrieved policy clause",
                "text": chunk,
                "document": document_store["filename"]
            }],
            "highlights": [{
                "type": "system",
                "text": "Local fallback used because Gemini generation failed."
            }]
        }

    # No explicit evidence: do not infer rejection.
    return {
        "decision": "more_info_needed",
        "amount": "N/A",
        "justification": (
            "The retrieved policy context does not contain sufficient "
            "explicit information to determine whether the requested "
            "item is covered or excluded. The system did not infer "
            "an exclusion from the absence of information."
        ),
        "matched_clauses": [
            {
                "clause_id": "Retrieved policy context",
                "text": chunk,
                "document": document_store["filename"]
            }
            for _, chunk in candidates[:3]
        ],
        "highlights": [{
            "type": "system",
            "text": "Local fallback used because Gemini generation failed."
        }]
    }



# QUERY DOCUMENT
# =========================================================

@app.post(
    "/query/",
    summary="Ask a Question About the Document"
)
async def query_document(
    request: QueryRequest
):

    # -----------------------------------------------------
    # Validate embeddings
    # -----------------------------------------------------

    if (
        document_store["embeddings"].size
        == 0
    ):

        raise HTTPException(

            status_code=404,

            detail=(
                "No embeddings found. "
                "Please upload and embed a document first."
            )

        )


    if gemini_client is None:

        raise HTTPException(

            status_code=500,

            detail=(
                "Gemini client is not initialized. "
                "Is GEMINI_API_KEY set?"
            )

        )


    # =====================================================
    # QUERY EMBEDDING
    # =====================================================

    try:

        response = (
            gemini_client.models.embed_content(

                model="gemini-embedding-2-preview",

                contents=request.query,

            )
        )


        if (
            not response.embeddings
            or
            not response.embeddings[0].values
        ):

            raise ValueError(
                "Gemini returned no embedding for the query."
            )


        query_embedding = np.array(

            response.embeddings[0].values,

            dtype=float

        )


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Failed to embed query using Gemini: "
                f"{str(e)}"
            )

        )


    # =====================================================
    # HYBRID RETRIEVAL
    # =====================================================

    try:

        (
            top_k_indices,
            combined_scores

        ) = hybrid_retrieval(

            query=request.query,

            query_embedding=query_embedding,

            top_k=request.top_k,

        )

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=(
                f"Retrieval error: "
                f"{str(e)}"
            )

        )


    if len(top_k_indices) == 0:

        raise HTTPException(

            status_code=404,

            detail="No relevant document chunks found."

        )


    # =====================================================
    # RETRIEVAL CONFIDENCE
    # =====================================================

    best_score = float(

        combined_scores[
            int(top_k_indices[0])
        ]

    )


    print(
        f"\nBest retrieval score: "
        f"{best_score:.4f}"
    )


    # =====================================================
    # FALLBACK RETRIEVAL
    # =====================================================

    FALLBACK_THRESHOLD = 0.20


    if best_score < FALLBACK_THRESHOLD:

        print(
            "\n⚠️ LOW RETRIEVAL CONFIDENCE"
        )


        print(
            "Activating full-document fallback..."
        )


        # Use complete document
        # instead of only top-k chunks.

        relevant_chunks = (
            document_store["chunks"]
        )


        context = (
            document_store["text"]
        )


        retrieval_mode = (
            "full_document_fallback"
        )


    else:

        relevant_chunks = [

            document_store["chunks"][int(i)]

            for i in top_k_indices

        ]


        context = (
            "\n\n---\n\n".join(
                relevant_chunks
            )
        )


        retrieval_mode = (
            "hybrid_retrieval"
        )


    # =====================================================
    # GEMINI PROMPT
    # =====================================================

    prompt_template = f"""
You are an expert insurance claim analyst.

You are answering a question about an insurance policy.

IMPORTANT:

Use ONLY the information contained in the policy
document context provided below.

Do NOT invent policy information.

If the answer is present anywhere in the provided
document context, use it.

Do NOT say "more_info_needed" simply because the first
retrieved clause does not contain the answer.

You must consider ALL provided policy context.

-------------------------------------------------------
USER QUERY
-------------------------------------------------------

"{request.query}"

-------------------------------------------------------
POLICY DOCUMENT
-------------------------------------------------------

Document:
{document_store['filename']}

-------------------------------------------------------
POLICY CONTEXT
-------------------------------------------------------

{context}

-------------------------------------------------------
INSTRUCTIONS
-------------------------------------------------------

1. Carefully search ALL provided policy context.

2. Identify the clause or clauses that directly answer
   the user's question.

3. Determine the decision as one of:

   "approved"

   "rejected"

   "more_info_needed"

4. Return "rejected" ONLY when the policy context explicitly
   states that the requested treatment, service, expense, or
   claim is excluded, not covered, not payable, or ineligible.

5. Do NOT treat the absence of a benefit from a list of covered
   expenses as an explicit exclusion.

6. If the policy does not contain enough explicit information
   to determine coverage or exclusion, return "more_info_needed".

7. Every "rejected" decision must be supported by an explicit
   exclusion statement in the matched clause.

8. Never infer an exclusion merely because something is not
   mentioned. Never invent policy information.

9. If the question asks for a policy limit, maximum
   benefit, amount, waiting period, eligibility,
   exclusion, or other specific value, look throughout
   the entire provided context before deciding that the
   information is unavailable.

5. If the requested amount is explicitly stated in the
   policy, return that amount.

6. If the amount cannot be determined from the policy,
   return:

   "As per policy"

7. The justification must explain the answer using the
   policy text.

8. matched_clauses must contain the clauses that support
   the answer.

9. Do not include clauses that are unrelated to the
   question.

10. Respond with ONE valid JSON object only.

11. Do not use Markdown.

12. Follow this exact structure:

{{
    "decision": "approved",
    "amount": "string",
    "justification": "string",
    "matched_clauses": [
        {{
            "clause_id": "string",
            "text": "string",
            "document": "{document_store['filename']}"
        }}
    ],
    "highlights": []
}}

Now provide the final answer as JSON.
"""


    # =====================================================
    # GEMINI GENERATION
    # =====================================================

    print(
        f"\nGeneration mode: {retrieval_mode}"
    )

    try:
        gemini_response = gemini_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt_template,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        response_content = (
            gemini_response.text
            if gemini_response
            else ""
        )

        if not response_content:
            raise ValueError(
                "Gemini returned an empty response."
            )

        json_response = json.loads(response_content)

        print("\nGemini generation succeeded.")
        return json_response

    except json.JSONDecodeError as e:
        print(f"\nGemini returned invalid JSON: {e}")
        print("Activating LOCAL rule-based fallback...")

        fallback_result = local_rule_based_fallback(
            request.query,
            relevant_chunks,
        )

        print("LOCAL fallback succeeded.")
        return fallback_result

    except Exception as e:
        error_message = str(e)

        print(
            f"\nGemini generation error: {error_message}"
        )

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
            or "quota" in error_message.lower()
            or "rate limit" in error_message.lower()
        ):
            print("Gemini quota/rate limit reached.")
        else:
            print("Gemini generation failed.")

        print("Activating LOCAL rule-based fallback...")

        # Never call Gemini a second time here.
        fallback_result = local_rule_based_fallback(
            request.query,
            relevant_chunks,
        )

        print("LOCAL fallback succeeded.")
        return fallback_result
