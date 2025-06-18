from google.genai import Client, types
from helper import RateLimiter, bytes_to_data_url
import re
import asyncio
import base64
from config import IMG_GENERATION_PROMPT
import httpx

rate_limiter = RateLimiter(requests_per_minute=50, requests_per_second=60)

# rate_limiter = AsyncLimiter(5, 60)  # 5 requests per 60 seconds

async def get_embeddings(text, api_key, model="gemini-embedding-exp-03-07", max_tries=10):
    # api_key = api_keys[0]
    # client = Client(api_key=api_key)
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    timeout = httpx.Timeout(30.0, connect=10.0)
    
    for attempt in range(max_tries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post("https://aipipe.org/openai/v1/embeddings", headers=headers, json=payload)
                response.raise_for_status()  # Raise an exception for bad status codes
                return response.json()["data"][0]["embedding"]  # Return JSON directly
            # rate_limiter.wait()
            # response = client.models.embed_content(
            #     model=model,
            #     contents=text
            # )
            # return response.embeddings[0].values
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                # api_key = api_keys[1]
                await asyncio.sleep(wait_time)
            elif attempt == max_tries - 1:
                print(f"Failed to get embeddings after {max_tries} attempts: {e}")
                raise
            else:
                print(f'attempt {attempt + 1} failed with error: {e}, retrying...')
                await asyncio.sleep(1)


async def describe_base64_image(base64_image, api_key, max_tries, question):
    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(base64_image)
    image_data_url = bytes_to_data_url(image_bytes)
    # client = Client(api_key=api_key)
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMG_GENERATION_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_data_url}}
                    ]
                }
            ]
        }
    for attempt in range(max_tries):
        try:
            async with httpx.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    # print(response)
                    result = await response.json()
                    image_description = result["choices"][0]["message"]["content"]
                    return image_description
            # response = client.models.generate_content(
            #     model='gemini-2.0-flash',
            #     contents=[
            #         types.Part.from_bytes(
            #             data=image_bytes,
            #             mime_type='image/jpeg',  # or 'image/png' if your image is PNG
            #         ),
            #         IMG_GENERATION_PROMPT
            #     ]
            # )
            # return response.text
        
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                # api_key = api_keys[1]
                await asyncio.sleep(wait_time)
            elif attempt == max_tries - 1:
                print(f"Failed to get embeddings after {max_tries} attempts: {e}")
                raise
            else:
                print(f'attempt {attempt + 1} failed with error: {e}, retrying...')
                await asyncio.sleep(1)

async def generate_answer(API_KEY,  question, relevant_results, max_tries=5):
    context = ""
    for result in relevant_results:
        source_type = "Discourse post" if result["source"] == "discourse" else "Documentation"
        context += f"\n\n{source_type} (URL: {result['url']}):\n{result['contents'][:1500]}"
    
    prompt = f"""Answer the following question based ONLY on the provided context. 
    If you cannot answer the question based on the context, say "I don't have enough information to answer this question."

    Context:
    {context}

    Question: {question}

    Return your response in this exact format:
    1. A comprehensive yet concise answer
    2. A "Sources:" section that lists the URLs and relevant text snippets you used to answer

    Sources must be in this exact format:
    Sources:
    1. URL: [exact_url_1], Text: [brief quote or description]
    2. URL: [exact_url_2], Text: [brief quote or description]

    Make sure the URLs are copied exactly from the context without any changes.
    """
    headers = {
            "Authorization": API_KEY,
            "Content-Type": "application/json"
        }
    payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
    timeout = httpx.Timeout(60.0, connect=10.0)
    
    for attempt in range(max_tries):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post('https://aipipe.org/openai/v1/chat/completions', headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            if "rate limit" in str(e).lower() or "quota" in str(e).lower():
                wait_time = 2 ** attempt
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                # api_key = api_keys[1]
                await asyncio.sleep(wait_time)
            elif attempt == max_tries - 1:
                print(f"Failed to get embeddings after {max_tries} attempts: {e}")
                raise
            else:
                print(f'attempt {attempt + 1} failed with error: {e}, retrying...')
                await asyncio.sleep(1)
                

def get_chunks(text, chunk_overlap: int = 100, max_embedding_chars: int = 8000):
    if not text:
        return []
    
    chunks = []
    
    # Clean up whitespace and newlines, preserving meaningful paragraph breaks
    text = re.sub(r'\n+', '\n', text)  # Replace multiple newlines with single
    text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single
    text = text.strip()
    
    # Apply embedding limit constraint - ensure chunk_size doesn't exceed embedding limit
    # effective_chunk_size = min(chunk_size, max_embedding_chars)
    
    # If text is very short, return it as a single chunk
    if len(text) <= max_embedding_chars:
        return [text]
    
    # Split text by paragraphs for more meaningful chunks
    paragraphs = text.split('\n')
    current_chunk = ""
    
    for i, para in enumerate(paragraphs):
        # If this paragraph alone exceeds chunk size, we need to split it further
        if len(para) > max_embedding_chars:
            # If we have content in the current chunk, store it first
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            
            # Split the paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sentence_chunk = ""
            
            for sentence in sentences:
                # If single sentence exceeds chunk size, split by chunks with overlap
                if len(sentence) > max_embedding_chars:
                    # If we have content in the sentence chunk, store it first
                    if sentence_chunk:
                        chunks.append(sentence_chunk.strip())
                        sentence_chunk = ""
                    
                    # Process the long sentence in chunks with embedding limit consideration
                    for j in range(0, len(sentence), max_embedding_chars - chunk_overlap):
                        sentence_part = sentence[j:j + max_embedding_chars]
                        if sentence_part:
                            chunks.append(sentence_part.strip())
                
                # If adding this sentence would exceed chunk size, save current and start new
                elif len(sentence_chunk) + len(sentence) > max_embedding_chars and sentence_chunk:
                    chunks.append(sentence_chunk.strip())
                    sentence_chunk = sentence
                else:
                    # Add to current sentence chunk
                    if sentence_chunk:
                        sentence_chunk += " " + sentence
                    else:
                        sentence_chunk = sentence
            
            # Add any remaining sentence chunk
            if sentence_chunk:
                chunks.append(sentence_chunk.strip())
            
        # Normal paragraph handling - if adding would exceed chunk size
        elif len(current_chunk) + len(para) > max_embedding_chars and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with this paragraph
            current_chunk = para
        else:
            # Add to current chunk
            if current_chunk:
                current_chunk += " " + para
            else:
                current_chunk = para
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Verify we have chunks and apply overlap between chunks
    if chunks:
        # Create new chunks list with proper overlap
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            current_chunk = chunks[i]
            
            # If the previous chunk ends with a partial sentence, find where it begins
            if len(prev_chunk) > chunk_overlap:
                # Find a good breaking point for overlap
                overlap_start = max(0, len(prev_chunk) - chunk_overlap)
                # Try to find sentence boundary
                sentence_break = prev_chunk.rfind('. ', overlap_start)
                if sentence_break != -1 and sentence_break > overlap_start:
                    overlap = prev_chunk[sentence_break+2:]
                    if overlap and not current_chunk.startswith(overlap):
                        # Ensure the overlapped chunk doesn't exceed embedding limit
                        proposed_chunk = overlap + " " + current_chunk
                        if len(proposed_chunk) <= max_embedding_chars:
                            current_chunk = proposed_chunk
                        # If it would exceed limit, truncate the overlap
                        else:
                            available_space = max_embedding_chars - len(current_chunk) - 1
                            if available_space > 0:
                                truncated_overlap = overlap[:available_space]
                                current_chunk = truncated_overlap + " " + current_chunk
                
            overlapped_chunks.append(current_chunk)
        
        # Final validation: ensure no chunk exceeds embedding limit
        validated_chunks = []
        for chunk in overlapped_chunks:
            if len(chunk) <= max_embedding_chars:
                validated_chunks.append(chunk)
            else:
                # If a chunk still exceeds limit, split it further
                for j in range(0, len(chunk), max_embedding_chars - chunk_overlap):
                    subchunk = chunk[j:j + max_embedding_chars]
                    if subchunk:
                        validated_chunks.append(subchunk.strip())
        
        return validated_chunks
    
    return chunks
