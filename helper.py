import time
import requests
import base64
import re
import json, os
import ast
import numpy as np


class RateLimiter:
    def __init__(self, requests_per_minute: int = 300, requests_per_second: int = 100):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0
    
    def wait(self):
        curr_time = time.time()
        
        # per-second rate limiting
        time_since_last_request = curr_time - self.last_request_time
        if time_since_last_request < (1 / self.requests_per_second):
            time.sleep((1 / self.requests_per_second) - time_since_last_request)
        
        # per-minute rate limiting
        curr_time = time.time()
        self.request_times = [t for t in self.request_times if t > curr_time - 60]
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (curr_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                curr_time = time.time()
                self.request_times = [t for t in self.request_times if t > curr_time - 60]
                
        self.request_times.append(curr_time)
        self.last_request_time = curr_time


def image_url_to_base64(image_url, format_hint=None):
    if image_url.startswith("file://"):
        # Local file path
        file_path = image_url[7:]  # Remove 'file://'
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return None
        try:
            with open(file_path, "rb") as f:
                img_bytes = f.read()
            base64_str = base64.b64encode(img_bytes).decode()
            return base64_str
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    else:
        # HTTP(S) URL
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/129.0.0.0 Safari/537.36"
            ),
            "Referer": "https://europe1.discourse-cdn.com/"
        }
        try:
            response = requests.get(image_url, headers=headers)
            response.raise_for_status()
            img_bytes = response.content
            base64_str = base64.b64encode(img_bytes).decode()
            return base64_str
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {image_url}: {e}")
            return None
        except Exception as e:
            print(f"Other error for {image_url}: {e}")
            return None
def extract_europe1_urls(text):
    # Regex pattern to find URLs starting with 'https://europe1'
    pattern = r'https://europe1[^"\s]+'
    urls = re.findall(pattern, text)
    return urls

def read_json_file(file):
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def load_embeddings(file_path):
    try:
        return np.load(file_path)
    except Exception as e:
        print(e)


def load_text_file(file_path):
    """Load embeddings from txt file containing str(dict) format"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Convert string representation back to dictionary
        saved_data = ast.literal_eval(content)
        return saved_data
    except Exception as e:
        print(f"Error loading from {file_path}: {e}")
        return None

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2[0])
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product/(norm_vec1 * norm_vec2)

def bytes_to_data_url(image_bytes, mime_type="image/png"):
    base64_encoded = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded}"