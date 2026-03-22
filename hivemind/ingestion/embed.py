"""
Generate dense embeddings using Voyage AI voyage-3-lite model
512-dimensional vectors optimized for retrieval
"""

import json
import numpy as np
import sys
import time
from pathlib import Path
from tqdm import tqdm
import voyageai

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    VOYAGE_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM, 
    EMBED_BATCH_SIZE, CHUNKS_JSON, EMBEDDINGS_NPY,
    validate_config
)

def init_voyage_client():
    """Initialize Voyage AI client"""
    if not VOYAGE_API_KEY:
        raise ValueError("VOYAGE_API_KEY not set in .env file")
    
    voyageai.api_key = VOYAGE_API_KEY
    return voyageai.Client()

def embed_texts_batch(client, texts: list, batch_size: int = 128) -> np.ndarray:
    """Embed texts in batches using Voyage AI"""
    all_embeddings = []
    
    print(f"Embedding {len(texts)} texts with {EMBEDDING_MODEL}...")
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i:i + batch_size]
        
        try:
            # Voyage AI embedding with document input type
            result = client.embed(
                batch_texts,
                model=EMBEDDING_MODEL,
                input_type="document"
            )
            
            batch_embeddings = np.array(result.embeddings)
            
            all_embeddings.append(batch_embeddings)
            
            print(f"Batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}: {len(batch_texts)} texts")
            
            # No rate limiting needed for Voyage AI (much more generous)
            
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            # Retry with smaller batch if needed
            if len(batch_texts) > 10:
                print("Retrying with smaller batch...")
                smaller_batch = batch_texts[:10]
                try:
                    result = client.embed(
                        smaller_batch,
                        model=EMBEDDING_MODEL,
                        input_type="document"
                    )
                    batch_embeddings = np.array(result.embeddings)
                    all_embeddings.append(batch_embeddings)
                    print(f"Smaller batch succeeded: {len(smaller_batch)} texts")
                except Exception as e2:
                    print(f"Smaller batch also failed: {e2}")
                    # Add zero embeddings to maintain array shape
                    zero_batch = np.zeros((len(smaller_batch), EMBEDDING_DIM))
                    all_embeddings.append(zero_batch)
            else:
                # Add zero embeddings to maintain array shape
                zero_batch = np.zeros((len(batch_texts), EMBEDDING_DIM))
                all_embeddings.append(zero_batch)
    
    # Concatenate all embeddings
    final_embeddings = np.vstack(all_embeddings)
    
    print(f"Embeddings shape: {final_embeddings.shape}")
    print(f"   Expected: ({len(texts)}, {EMBEDDING_DIM})")
    
    # Validate dimensions
    if final_embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension mismatch: got {final_embeddings.shape[1]}, expected {EMBEDDING_DIM}")
    
    return final_embeddings

def main():
    """Main embedding generation"""
    print("Starting Voyage AI embedding process...")
    
    # Validate configuration
    validate_config()
    
    # Initialize Voyage AI client
    client = init_voyage_client()
    print(f"Voyage AI client initialized with model: {EMBEDDING_MODEL}")
    
    # Load chunks
    if not CHUNKS_JSON.exists():
        raise FileNotFoundError(f"Run chunk_text.py first! {CHUNKS_JSON} not found.")
    
    with open(CHUNKS_JSON, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Extract texts for embedding
    texts = [chunk['text'] for chunk in chunks]
    
    # Initialize Voyage AI client
    client = init_voyage_client()
    
    # Generate embeddings
    embeddings = embed_texts_batch(client, texts, EMBED_BATCH_SIZE)
    
    # Save embeddings
    EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_NPY, embeddings)
    
    print(f"Saved embeddings to {EMBEDDINGS_NPY}")
    
    # Verify the save
    loaded_embeddings = np.load(EMBEDDINGS_NPY)
    assert loaded_embeddings.shape == embeddings.shape, "Saved embeddings don't match!"
    print(f"Embedding complete! {len(embeddings)} vectors ready for sparse processing.")
    
    # Print embedding statistics
    print("\nEmbedding statistics:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Average embedding norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.3f}")
    
    print(f"\nVoyage AI embedding complete!")

if __name__ == "__main__":
    main()
