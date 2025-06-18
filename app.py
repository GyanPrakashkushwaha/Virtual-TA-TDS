from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import asyncio
import json
import base64

# Your existing imports
from embed import (get_embeddings, generate_answer, describe_base64_image)
from config import OPEN_API_KEY
from helper import (load_embeddings, extract_europe1_urls, image_url_to_base64)
from get_answer import find_similar_content, parse_llm_response

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class QueryResponse(BaseModel):
    answer: str
    links: list

# Initialize FastAPI app
app = FastAPI(title="RAG Query API", description="Simple API for querying with optional images")

# Optional: Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embeddings once at startup
discourse_embeddings = load_embeddings('embeddings/discourse_embeddings.npz')
markdown_embeddings = load_embeddings('embeddings/markdown_embeddings.npz')

async def process_query(question: str, image_base64: Optional[str] = None):
    """
    Modified version of your result_response function
    """
    try:
        CONTEXT = 10
        
        # Handle image processing
        if image_base64:
            # If image is provided directly as base64
            img_description = await describe_base64_image(image_base64, OPEN_API_KEY, 3, question=question)
            question = question + " " + img_description
        else:
            # Try to extract URLs from question if no direct image provided
            try:
                url = extract_europe1_urls(question)
                if url:
                    base64_img = image_url_to_base64(url[0])
                    img_description = await describe_base64_image(base64_img, OPEN_API_KEY, 3, question=question)
                    question = question + " " + img_description
            except:
                # Continue without image if extraction fails
                pass
        
        # Generate embeddings
        embedding_response = await get_embeddings(question, OPEN_API_KEY)
        
        # Find relevant content
        relevant_results = find_similar_content(embedding_response, CONTEXT, discourse_embeddings, markdown_embeddings)
        
        # Generate answer
        answer = await generate_answer(OPEN_API_KEY, question, relevant_results)
        llm_response = parse_llm_response(answer)
        
        return llm_response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main endpoint for processing queries with optional images
    """
    try:
        result = await process_query(request.question, request.image)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "message": "API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
