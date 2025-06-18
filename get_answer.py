import re
from helper import cosine_similarity


def find_similar_content(query_embedding, MAX_SIMILAR_TEXT, discourse_data, markdown_data):
    results = []
    
    # Search discourse chunks
    print("Searching discourse chunks for similar content...")
    embeddings = discourse_data['embeddings']
    contents = discourse_data['chunks']
    urls = discourse_data['original_urls']
    
    for i, embedding in enumerate(embeddings):            
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity >= 0.5:
                results.append({
                    "source": "discourse",
                    "url": urls[i] if i < len(urls) else "",
                    "contents": contents[i] if i < len(contents) else "",
                    "similarity": similarity
                })
    
    # Search markdown chunks
    print("Searching markdown chunks for similar contents...")
    embeddings = markdown_data['embeddings']
    contents = markdown_data['chunks']
    urls = markdown_data['original_urls']
    
    for i, embedding in enumerate(embeddings):            
            similarity = cosine_similarity(query_embedding, embedding)
            if similarity >= 0.5:
                results.append({
                    "source": "markdown",
                    "url": urls[i] if i < len(urls) else "",
                    "contents": contents[i] if i < len(contents) else "",
                    "similarity": similarity
                })
    
    # Sort by similarity and return top results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:MAX_SIMILAR_TEXT]



def parse_llm_response(response):
    try:
        # Split by "Sources:" or similar headings
        parts = response.split("Sources:", 1)
        if len(parts) == 1:
            for heading in ["Source:", "References:", "Reference:"]:
                if heading in response:
                    parts = response.split(heading, 1)
                    break

        answer = parts[0].strip()
        links = []

        if len(parts) > 1:
            sources_text = parts[1].strip()
            source_lines = sources_text.split("\n")
            for line in source_lines:
                line = line.strip()
                if not line:
                    continue
                # Remove list markers
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                # Extract URL and text
                url_match = re.search(
                    r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)',
                    line, re.IGNORECASE)
                text_match = re.search(
                    r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|"(.*?)"|Text:\s*"(.*?)"|text:\s*"(.*?)"',
                    line, re.IGNORECASE)
                if url_match:
                    url = next((g for g in url_match.groups() if g), "")
                    url = url.strip()
                    text = "Source reference"
                    if text_match:
                        text_value = next((g for g in text_match.groups() if g), "")
                        if text_value:
                            text = text_value.strip()
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})

        return {
            "answer": answer,
            "links": links
        }
    except Exception as e:
        return {
            "answer": "Error parsing the response from the language model.",
            "links": []
        }
