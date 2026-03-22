"""
Generate BM25 sparse vectors in Endee's native format
Each sparse vector: {"indices": [term_id, ...], "values": [weight, ...]}
"""

import json
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNKS_JSON, SPARSE_VECTORS_JSON

# Download NLTK data if needed
def download_nltk_data():
    """Download required NLTK data"""
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    
    try:
        word_tokenize('test')
    except LookupError:
        print("Downloading NLTK punkt...")
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK punkt_tab...")
        nltk.download('punkt_tab')

def preprocess_text(text: str) -> list[str]:
    """Preprocess text for BM25: tokenize, lowercase, remove stopwords"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short tokens
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return tokens

def build_vocabulary(all_tokens: list[list[str]]) -> dict[str, int]:
    """Build vocabulary mapping term -> term_id"""
    vocab = {}
    term_id = 0
    
    print("Building vocabulary...")
    
    for tokens in tqdm(all_tokens, desc="Processing documents"):
        for token in tokens:
            if token not in vocab:
                vocab[token] = term_id
                term_id += 1
    
    print(f"Vocabulary size: {len(vocab)} terms")
    return vocab

def compute_bm25_weights(bm25_model: BM25Okapi, doc_tokens: list[list[str]], vocab: dict[str, int]) -> list[dict]:
    """Compute BM25 weights for each document"""
    sparse_vectors = []
    
    print("Computing BM25 weights...")
    
    for i, tokens in enumerate(tqdm(doc_tokens, desc="Computing weights")):
        # Get BM25 scores for this document against all documents
        scores = bm25_model.get_scores(tokens)
        
        # Get BM25 score for this document specifically
        doc_score = scores[i]
        
        # Create sparse vector with term frequencies weighted by BM25
        term_counts = {}
        for token in tokens:
            if token in vocab:
                term_id = vocab[token]
                term_counts[term_id] = term_counts.get(term_id, 0) + 1
        
        # Convert to Endee format: indices and values
        if term_counts:
            indices = list(term_counts.keys())
            # Use term frequency * BM25 relevance as weight
            values = [count * doc_score / len(tokens) for count in term_counts.values()]
            
            sparse_vectors.append({
                "indices": indices,
                "values": values
            })
        else:
            # Empty sparse vector
            sparse_vectors.append({
                "indices": [],
                "values": []
            })
    
    return sparse_vectors

def main():
    """Main sparse vector generation"""
    print("Starting BM25 sparse vector generation...")
    
    # Download NLTK data
    download_nltk_data()
    
    # Load chunks
    if not CHUNKS_JSON.exists():
        raise FileNotFoundError(f"Run chunk_text.py first! {CHUNKS_JSON} not found.")
    
    with open(CHUNKS_JSON, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Preprocess all texts
    print("Preprocessing texts...")
    all_tokens = []
    texts = [chunk['text'] for chunk in chunks]
    
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = preprocess_text(text)
        all_tokens.append(tokens)
    
    # Build vocabulary
    vocab = build_vocabulary(all_tokens)
    
    # Create BM25 model
    print("Creating BM25 model...")
    bm25_model = BM25Okapi(all_tokens)
    
    # Compute sparse vectors
    sparse_vectors = compute_bm25_weights(bm25_model, all_tokens, vocab)
    
    # Save sparse vectors
    SPARSE_VECTORS_JSON.parent.mkdir(parents=True, exist_ok=True)
    
    with open(SPARSE_VECTORS_JSON, 'w', encoding='utf-8') as f:
        json.dump(sparse_vectors, f, indent=2)
    
    print(f"Saved {len(sparse_vectors)} sparse vectors to {SPARSE_VECTORS_JSON}")
    
    # Save vocabulary for query processing
    vocab_file = SPARSE_VECTORS_JSON.parent / "vocabulary.json"
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)
    
    print(f"Saved vocabulary to {vocab_file}")
    
    # Print statistics
    print("\nSparse vector statistics:")
    non_empty = [v for v in sparse_vectors if v['indices']]
    avg_terms = np.mean([len(v['indices']) for v in non_empty])
    max_terms = max([len(v['indices']) for v in sparse_vectors])
    
    print(f"Total vectors: {len(sparse_vectors)}")
    print(f"Non-empty vectors: {len(non_empty)}")
    print(f"Avg non-zero terms: {avg_terms:.1f}")
    print(f"Max non-zero terms: {max_terms}")
    print(f"Vocabulary size: {len(vocab)}")
    
    print(f"\nSparse vector generation complete!")

if __name__ == "__main__":
    main()
