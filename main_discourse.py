from pathlib import Path
import numpy as np
from tqdm import tqdm
from embed import get_chunks, get_embeddings
from config import APIS_LIST, OPEN_API_KEY
from extract_text import clean_html
import asyncio
import ast
import json, os
from helper import read_json_file, extract_europe1_urls, load_text_file


# apis_list = ast.literal_eval(APIS_LIST)
async def process_save_discourse():
    files = [*Path("raw-data/Discourse-data").glob("*.json")]
    all_chunks = []
    all_embeddings = []
    all_original_urls = []
    total_chunks = 0
    file_chunks = {}
    file_urls = {}
    
    # Check for existing embeddings and load them
    existing_count = 0
    resume_file = 'discourse_embeddings_temp.txt'
    if os.path.exists(resume_file):
        print(f"📂 Loading from {resume_file}...")
        saved_data = load_text_file(resume_file)
        all_chunks_loaded = saved_data["chunks"]
        all_embeddings_loaded = saved_data["embeddings"] 
        all_original_urls_loaded = saved_data["original_urls"]
        
        # Count only valid (non-null) embeddings
        valid_count = 0
        for i, embedding in enumerate(all_embeddings_loaded):
            if embedding[0] and len(str(embedding[0]).strip()) > 1:
                valid_count += 1
            else:
                break
        
        # Keep only valid entries
        all_chunks = all_chunks_loaded[:valid_count]
        all_embeddings = all_embeddings_loaded[:valid_count]
        all_original_urls = all_original_urls_loaded[:valid_count]
        existing_count = valid_count

        print(f"✅ Loaded {existing_count} valid chunks (filtered out null embeddings)")


    # First pass: extract content and count chunks for all files
    # count_api_calls = existing_count
    print("🔍 Analyzing files and extracting content...")
    img_descriptions =  load_text_file('embeddings\img_description.txt')
    for file_path in files:
        data = read_json_file(file_path)
        posts = data.get('post_stream', {}).get('posts', [])
        topic_id = data.get('id')
        topic_slug = data.get('slug', '')
        
        complete_post = ''
        for post in posts:
            content = post.get('cooked', '')
            question_img_url = extract_europe1_urls(content) if extract_europe1_urls(content) else None
            clean_content = clean_html(content)
            complete_post += clean_content
            
            if question_img_url:
                try:
                    complete_post += img_descriptions[question_img_url[0]]
                except:
                    continue
            
        chunks = get_chunks(complete_post)
        topic_url = f"https://discourse.onlinedegree.iitm.ac.in/t/{topic_slug}/{topic_id}"
        
        file_chunks[file_path] = chunks
        file_urls[file_path] = topic_url
        total_chunks += len(chunks)
        print(f"File: {file_path.name}, Chunks: {len(chunks)}")
    
    print(f"Total chunks to process: {total_chunks}")
    
    # Calculate which file and chunk to start from
    chunks_to_skip = existing_count
    start_file_index = 0
    start_chunk_index = 0
    
    # Find the starting file and chunk position
    file_list = list(file_chunks.items())
    for i, (file_path, chunks) in enumerate(file_list):
        if chunks_to_skip >= len(chunks):
            chunks_to_skip -= len(chunks)
        else:
            start_file_index = i
            start_chunk_index = chunks_to_skip
            break
    
    processed_count = existing_count
    
    # Second pass: process chunks starting from the correct position
    print(f"🚀 Starting processing from file index {start_file_index}, chunk index {start_chunk_index}")
    
    with tqdm(total=total_chunks, initial=existing_count, desc="Processing Chunks") as pbar:
        for i in range(start_file_index, len(file_list)):
            file_path, chunks = file_list[i]
            topic_url = file_urls[file_path]
            # print(topic_url)
            
         # Start from the correct chunk index for the first file, 0 for others
            chunk_start = start_chunk_index if i == start_file_index else 0
            
            for j in range(chunk_start, len(chunks)):
                chunk = chunks[j]
                # print(F'================================ \n {chunk} \n ================================')
                try:
                    all_chunks.append([chunk])
                    all_original_urls.append([topic_url])
                    
                    # Create embeddings for chunks
                    embedding = await get_embeddings(chunk, OPEN_API_KEY)
                    
                    all_embeddings.append([embedding])
                    
                    processed_count += 1
                    pbar.set_postfix({"file": file_path.name, "chunk": processed_count})
                    
                    print(f'✅ Embedding generated for "{file_path.name}" - chunk: "{chunk[:50]}..."')
                    print(f'📊 First 5 embedding values: {embedding[:5]} for topic: "{topic_url}"')
                    print("=" * 70)
                    
                except Exception as e:
                    print(f"❌ Error getting embedding for chunk in {file_path}: {e}")
                    # Add empty embedding to maintain alignment
                    # all_embeddings.append([None])
                finally:
                    temp_data = {
                        "chunks": all_chunks,
                        "embeddings": all_embeddings,
                        "original_urls": all_original_urls,
                            "metadata": {
                        "total_chunks": len(all_chunks),
                        "total_files_processed": len(files),
                        "processing_complete": True
                        }
                    }
                    with open("discourse_embeddings_temp_2.txt", "w", encoding="utf-8") as f:
                        f.write(str(temp_data))
                    
                    pbar.update(1)
                    


    # Final save after processing all files
    print("💾 Saving final results...")
    data_safe = {
        "chunks": all_chunks,
        "embeddings": all_embeddings,
        "original_urls": all_original_urls,
        "metadata": {
            "total_chunks": len(all_chunks),
            "total_files_processed": len(files),
            "processing_complete": True
        }
    }

    # Save in multiple formats for redundancy
    with open("discourse_embeddings_safe.json", "w", encoding="utf-8") as f:
        json.dump(data_safe, f, indent=2, ensure_ascii=False)
        
    np.savez("discourse_embeddings.npz", 
             chunks=all_chunks, 
             embeddings=all_embeddings, 
             original_urls=all_original_urls)
    


if __name__ == "__main__":
    asyncio.run(process_save_discourse())
    print("🎉 Processing complete. Embeddings saved to 'discourse_embeddings.npz' and 'discourse_embeddings_safe.json'.")
