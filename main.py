from pathlib import Path
import numpy as np
from tqdm import tqdm
from embed import get_chunks, get_embeddings
from config import GEMINI_API_KEY, OPEN_API_KEY
from extract_text import extract_text_from_markdown
import asyncio
import signal
import json, sys, os


def setup_interrupt_handler(chunks_list, embeddings_list, urls_list):
    def signal_handler(signum, frame):
        print("\n⚠️ Ctrl+C detected! Saving data...")
        
        # Save current progress
        data = {
            "chunks": chunks_list,
            "embeddings": embeddings_list,
            "original_urls": urls_list,
            "metadata": {"total_chunks": len(chunks_list), "emergency_save": True}
        }
        
        with open("emergency_save.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved {len(chunks_list)} chunks to 'emergency_save.json'")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

async def process_save_markdown():
    files = [*Path("raw-data/Markdown-data").glob("*.md")]
    all_chunks = []
    all_embeddings = []
    all_original_urls = []
    total_chunks = 0
    file_chunks = {}
    file_urls = {}
    
    existing_count = 0
    if os.path.exists("emergency_save_me.json"):
        print("📂 Loading from emergency_save_me.json...")
        with open("emergency_save_me.json", "r", encoding="utf-8") as f:  # Fixed filename
            saved_data = json.load(f)
            all_chunks_loaded = saved_data["chunks"]
            all_embeddings_loaded = saved_data["embeddings"] 
            all_original_urls_loaded = saved_data["original_urls"]
            
            # Count only valid (non-null) embeddings
            valid_count = 0
            for i, embedding in enumerate(all_embeddings_loaded):
                if embedding[0] and len(str(embedding[0]).strip()) > 1:  # Simplified validation
                    valid_count += 1
                else:
                    break
            
            # Keep only valid entries
            all_chunks = all_chunks_loaded[:valid_count]
            all_embeddings = all_embeddings_loaded[:valid_count]
            all_original_urls = all_original_urls_loaded[:valid_count]
            existing_count = valid_count - 2

            print(f"✅ Loaded {existing_count} valid chunks (filtered out null embeddings)")

    # First pass: extract content and count chunks
    for file_path in files:
        content, original_url = extract_text_from_markdown(file_path)
        chunks = get_chunks(content)
        file_chunks[file_path] = chunks
        file_urls[file_path] = original_url
        total_chunks += len(chunks)
        print(f"File: {file_path.name}, Chunks: {len(chunks)}")
    
    print(f"Total chunks created: {total_chunks}")
    
    # **FIXED**: Calculate which file and chunk to start from
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
    with tqdm(total=total_chunks, initial=existing_count, desc="Processing Chunks") as pbar:
        for i in range(start_file_index, len(file_list)):
            file_path, chunks = file_list[i]
            original_url = file_urls[file_path]
            
            # Start from the correct chunk index for the first file, 0 for others
            chunk_start = start_chunk_index if i == start_file_index else 0
            
            for j in range(chunk_start, len(chunks)):
                chunk = chunks[j]
                try:
                    embedding = await get_embeddings(chunk, api_key= OPEN_API_KEY)
                    all_chunks.append([chunk])  # Simplified - removed unnecessary list wrapping
                    all_embeddings.append([embedding])  # Simplified
                    all_original_urls.append([original_url])  # Simplified
                    processed_count += 1
                    pbar.set_postfix({"file": file_path.name, "chunk": processed_count})
                    print(f'{chunk} is created of url {original_url} and first few embeddings are: {embedding[:6]}')
                except Exception as e:
                    print(f"Error processing chunk in {file_path}: {e}")
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
                    with open("discourse_embeddings_temp.txt", "w", encoding="utf-8") as f:
                        f.write(str(temp_data))
                    pbar.update(1)
                    
    
    # Save once at the end
    data_safe = {
        "chunks": all_chunks,
        "embeddings": all_embeddings,
        "original_urls": all_original_urls,
    }
    
    with open("markdown_embeddings_safe.json", "w", encoding="utf-8") as f:
        json.dump(data_safe, f, indent=2, ensure_ascii=False)
        
    np.savez("markdown_embeddings.npz", 
             chunks=all_chunks, 
             embeddings=all_embeddings, 
             original_urls=all_original_urls)
    
if __name__ == "__main__":
    asyncio.run(process_save_markdown())
    print("Processing complete. Embeddings saved to 'markdown_embeddings.npz'.")  # Fixed syntax
