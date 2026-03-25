"""
Chunk arXiv papers: combine title + abstract, clean text, prepare for embedding
One chunk per paper (title+abstract is optimal for 512-dim embeddings)
"""

import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import PAPERS_JSON, CHUNKS_JSON

def clean_text(text: str) -> str:
    """Clean LaTeX artifacts and normalize whitespace"""
    if not text:
        return ""
    
    # Remove LaTeX math expressions (simple patterns)
    text = re.sub(r'\$[^$]*\$', '', text)  # $...$
    text = re.sub(r'\\\(.*?\\\)', '', text)  # \(...\)
    text = re.sub(r'\\\[.*?\\\]', '', text)  # \[...\]
    
    # Remove common LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # \command{...}
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # \command
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single space
    text = re.sub(r'\n+', ' ', text)  # Newlines → space
    text = re.sub(r'\t+', ' ', text)  # Tabs → space
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def create_chunks():
    """Create text chunks from papers"""
    print("Creating text chunks from papers...")
    
    # Load papers
    if not PAPERS_JSON.exists():
        raise FileNotFoundError(f"Run fetch_papers.py first! {PAPERS_JSON} not found.")
    
    with open(PAPERS_JSON, 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"   Loaded {len(papers)} papers")
    
    chunks = []
    
    for paper in tqdm(papers, desc="Chunking papers"):
        # Combine title + abstract
        title = clean_text(paper.get('title', ''))
        abstract = clean_text(paper.get('abstract', ''))
        
        # Skip if no title or abstract
        if not title and not abstract:
            continue
        
        # Create chunk text
        if title and abstract:
            chunk_text = f"{title}. {abstract}"
        elif title:
            chunk_text = title
        else:
            chunk_text = abstract
        
        # Skip very short chunks
        if len(chunk_text.strip()) < 50:
            continue
        
        # Create chunk object
        chunk = {
            "chunk_id": paper['id'],
            "text": chunk_text,
            "title": title,
            "abstract": abstract,
            "authors": paper.get('authors', []),
            "categories": paper.get('categories', []),
            "primary_category": paper.get('primary_category', ''),
            "published": paper.get('published', ''),
            "arxiv_id": paper['id'],
            "doi": paper.get('doi'),
            "text_length": len(chunk_text)
        }
        
        chunks.append(chunk)
    
    # Save chunks
    CHUNKS_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    with open(CHUNKS_JSON, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {CHUNKS_JSON}")
    
    # Print statistics
    print("\nChunk statistics:")
    lengths = [c['text_length'] for c in chunks]
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Avg length: {sum(lengths)/len(lengths):.0f} chars")
    print(f"   Min length: {min(lengths)} chars")
    print(f"   Max length: {max(lengths)} chars")
    
    # Category distribution
    category_counts = {}
    for chunk in chunks:
        cat = chunk['primary_category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nPrimary categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {cat}: {count}")
    
    print(f"\nChunking complete! Ready for embedding.")

if __name__ == "__main__":
    create_chunks()
