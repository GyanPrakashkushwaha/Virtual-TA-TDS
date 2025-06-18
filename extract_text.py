import frontmatter
import re
from markdown import markdown
from bs4 import BeautifulSoup

def extract_text_from_markdown(filepath):
    # Load markdown file and extract frontmatter
    post = frontmatter.load(filepath)
    # title = post.get('title', '')
    original_url = post.get('original_url', '')
    # Remove downloaded_at if present (not needed)
    
    # Get markdown content
    content = post.content

    # Remove navigation links (e.g., [Previous ...], [Next ...])
    # This regex removes lines starting with [Previous or [Next (case-insensitive)
    content = re.sub(r'^\[(Previous|Next)[\s\S]*?\]\([^\)]*\)\s*', '', content, flags=re.MULTILINE|re.IGNORECASE)
    
    # Convert markdown to HTML, then HTML to plain text
    html = markdown(content)
    soup = BeautifulSoup(html, "html.parser")
    plain_text = soup.get_text(separator='\n')

    return plain_text.strip(), original_url


def clean_html(html_content):
    if not html_content:
        return ""
    
    # Use BeautifulSoup to parse HTML
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    
    # Convert to text and clean up whitespace
    text = soup.get_text(separator=' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


