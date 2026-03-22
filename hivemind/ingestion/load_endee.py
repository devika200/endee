"""
Load data into Endee: create indexes, insert vectors via MessagePack, verify
Creates 5 indexes: knowledge_base (hybrid), knowledge_base_fp32/fp16/int8 (dense only), session_memory
"""

import json
import numpy as np
import msgpack
import requests
import sys
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    ENDEE_HOST, ENDEE_AUTH_TOKEN,
    KNOWLEDGE_BASE_INDEX, KNOWLEDGE_BASE_FP32_INDEX, KNOWLEDGE_BASE_FP16_INDEX, KNOWLEDGE_BASE_INT8_INDEX,
    SESSION_MEMORY_INDEX,
    EMBEDDING_DIM, DISTANCE_METRIC, QUANTIZATION, SPARSE_MODEL,
    CHUNKS_JSON, EMBEDDINGS_NPY, SPARSE_VECTORS_JSON,
    INSERT_BATCH_SIZE, validate_config
)

class EndeeClient:
    """Simple Endee REST API client"""
    
    def __init__(self, host: str, auth_token: str = ""):
        self.host = host.rstrip('/')
        self.auth_token = auth_token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
    
    def _request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request with error handling"""
        url = f"{self.host}/api/v1{endpoint}"
        
        # Handle custom headers
        if 'headers' in kwargs:
            # Merge with default headers
            merged_headers = self.headers.copy()
            merged_headers.update(kwargs.pop('headers'))
            kwargs['headers'] = merged_headers
        else:
            kwargs['headers'] = self.headers
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"   Response: {e.response.text}")
            raise
    
    def create_index(self, name: str, dim: int, metric: str, quant: str, sparse_model: str = "none", m: int = 16, ef_con: int = 128):
        """Create a new index"""
        # Convert quant format for Endee API
        precision_map = {
            "fp32": "float32",
            "fp16": "float16", 
            "int8": "int8",
            "int16": "int16",
            "binary": "binary"
        }
        precision = precision_map.get(quant, quant)
        
        # Convert sparse model format for Endee API
        sparse_map = {
            "none": "None",
            "default": "default",
            "endee_bm25": "endee_bm25"
        }
        sparse = sparse_map.get(sparse_model.lower(), sparse_model)
        
        payload = {
            "index_name": name,
            "dim": dim,
            "space_type": metric,
            "M": m,
            "ef_con": ef_con,
            "precision": precision,
            "sparse_model": sparse
        }
        
        print(f"Creating index: {name}")
        print(f"   Config: dim={dim}, metric={metric}, quant={quant}, sparse={sparse_model}")
        
        response = self._request("POST", "/index/create", json=payload)
        print(f"Index {name} created successfully")
        try:
            return response.json()
        except:
            # Handle plain text response
            return {"status": "success", "message": response.text}
    
    def insert_vectors(self, index_name: str, vectors: list, use_msgpack: bool = True):
        """Insert vectors into index (MessagePack or JSON)"""
        if use_msgpack:
            # Use MessagePack for faster insertion
            headers = self.headers.copy()
            headers["Content-Type"] = "application/msgpack"
            data = msgpack.packb(vectors)
            print(f"Inserting {len(vectors)} vectors via MessagePack...")
            response = self._request("POST", f"/index/{index_name}/vector/insert", data=data, headers=headers)
        else:
            # Use JSON (fallback)
            print(f"Inserting {len(vectors)} vectors via JSON...")
            response = self._request("POST", f"/index/{index_name}/vector/insert", json=vectors)
        
        return response
    
    def get_index_info(self, index_name: str):
        """Get index statistics"""
        response = self._request("GET", f"/index/{index_name}/info")
        return response.json()
    
    def list_indexes(self):
        """List all indexes"""
        response = self._request("GET", "/index/list")
        return response.json()
    
    def create_index_if_not_exists(self, name: str, dim: int, metric: str, quant: str, sparse_model: str = "none", m: int = 16, ef_con: int = 128):
        """Create a new index only if it doesn't exist"""
        # Check if index already exists
        try:
            indexes = self.list_indexes()
            existing_names = [idx['name'] for idx in indexes.get('indexes', [])]
            if name in existing_names:
                print(f"Index {name} already exists, skipping creation")
                return {"status": "exists", "message": "Index already exists"}
        except:
            pass  # Continue with creation if list fails
        
        # Create the index
        return self.create_index(name, dim, metric, quant, sparse_model, m, ef_con)
    
    def backup_index(self, index_name: str, backup_name: str = None):
        """Create backup of index"""
        if not backup_name:
            backup_name = f"{index_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        payload = {"name": backup_name}
        response = self._request("POST", f"/index/{index_name}/backup", json=payload)
        print(f"Backup created: {backup_name}")
        return response.json()

def load_data():
    """Load all data files"""
    print("Loading data files...")
    
    # Load chunks
    with open(CHUNKS_JSON, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Load embeddings
    embeddings = np.load(EMBEDDINGS_NPY)
    
    # Load sparse vectors
    with open(SPARSE_VECTORS_JSON, 'r', encoding='utf-8') as f:
        sparse_vectors = json.load(f)
    
    print(f"   Chunks: {len(chunks)}")
    print(f"   Embeddings: {embeddings.shape}")
    print(f"   Sparse vectors: {len(sparse_vectors)}")
    
    # Validate data consistency
    assert len(chunks) == len(embeddings) == len(sparse_vectors), "Data size mismatch!"
    
    return chunks, embeddings, sparse_vectors

def create_hybrid_vector_objects(chunks: list, embeddings: np.ndarray, sparse_vectors: list) -> list:
    """Create hybrid vector objects for Endee insertion"""
    vectors = []
    
    for i, chunk in enumerate(chunks):
        # Create filter payload
        year = int(chunk.get('published', '2020')[:4]) if chunk.get('published') else 2020
        category = chunk.get('primary_category', 'unknown')
        has_code = 'code' in chunk.get('abstract', '').lower() or 'implementation' in chunk.get('abstract', '').lower()
        
        filter_payload = {
            "year": year,
            "category": category,
            "has_code": has_code
        }
        
        # Create metadata payload
        meta_payload = {
            "title": chunk.get('title', ''),
            "authors": chunk.get('authors', []),
            "arxiv_id": chunk.get('arxiv_id', ''),
            "abstract_snippet": chunk.get('abstract', '')[:200] + "..." if len(chunk.get('abstract', '')) > 200 else chunk.get('abstract', '')
        }
        
        vector_obj = {
            "id": chunk['chunk_id'],
            "vector": embeddings[i].tolist(),
            "sparse_indices": sparse_vectors[i]['indices'],
            "sparse_values": sparse_vectors[i]['values'],
            "filter": json.dumps(filter_payload),
            "meta": json.dumps(meta_payload)
        }
        
        vectors.append(vector_obj)
    
    return vectors

def create_dense_vector_objects(chunks: list, embeddings: np.ndarray) -> list:
    """Create dense-only vector objects for evaluation indexes"""
    vectors = []
    
    for i, chunk in enumerate(chunks):
        # Same filter and metadata as hybrid
        year = int(chunk.get('published', '2020')[:4]) if chunk.get('published') else 2020
        category = chunk.get('primary_category', 'unknown')
        has_code = 'code' in chunk.get('abstract', '').lower() or 'implementation' in chunk.get('abstract', '').lower()
        
        filter_payload = {
            "year": year,
            "category": category,
            "has_code": has_code
        }
        
        meta_payload = {
            "title": chunk.get('title', ''),
            "authors": chunk.get('authors', []),
            "arxiv_id": chunk.get('arxiv_id', ''),
            "abstract_snippet": chunk.get('abstract', '')[:200] + "..." if len(chunk.get('abstract', '')) > 200 else chunk.get('abstract', '')
        }
        
        vector_obj = {
            "id": chunk['chunk_id'],
            "vector": embeddings[i].tolist(),
            "filter": json.dumps(filter_payload),
            "meta": json.dumps(meta_payload)
        }
        
        vectors.append(vector_obj)
    
    return vectors

def main():
    """Main loading process"""
    print("Starting Endee data loading...")
    
    # Validate configuration
    validate_config()
    
    # Initialize client
    client = EndeeClient(ENDEE_HOST, ENDEE_AUTH_TOKEN)
    
    # Check Endee health
    try:
        response = client._request("GET", "/health")
        print("Endee server is healthy")
    except Exception as e:
        print("Cannot connect to Endee at {ENDEE_HOST}: {e}")
        print("   Make sure Endee is running on localhost:8080")
        return
    
    # Load data
    chunks, embeddings, sparse_vectors = load_data()
    
    # Create vector objects
    print("Creating vector objects...")
    hybrid_vectors = create_hybrid_vector_objects(chunks, embeddings, sparse_vectors)
    dense_vectors = create_dense_vector_objects(chunks, embeddings)
    
    # Create indexes
    print("\nCreating indexes...")
    
    # Knowledge base indexes
    client.create_index_if_not_exists(KNOWLEDGE_BASE_INDEX, EMBEDDING_DIM, DISTANCE_METRIC, QUANTIZATION, SPARSE_MODEL)
    client.create_index_if_not_exists(KNOWLEDGE_BASE_FP32_INDEX, EMBEDDING_DIM, DISTANCE_METRIC, "fp32", "none")
    client.create_index_if_not_exists(KNOWLEDGE_BASE_FP16_INDEX, EMBEDDING_DIM, DISTANCE_METRIC, "fp16", "none")
    client.create_index_if_not_exists(KNOWLEDGE_BASE_INT8_INDEX, EMBEDDING_DIM, DISTANCE_METRIC, "int8", "none")
    
    # Session memory index
    client.create_index_if_not_exists(SESSION_MEMORY_INDEX, EMBEDDING_DIM, DISTANCE_METRIC, "fp16", "none")
    
    # Insert data into indexes
    print("\nInserting data into indexes...")
    
    # Hybrid knowledge base
    print(f"\n{KNOWLEDGE_BASE_INDEX} (hybrid):")
    for i in tqdm(range(0, len(hybrid_vectors), INSERT_BATCH_SIZE), desc="Inserting hybrid vectors"):
        batch = hybrid_vectors[i:i + INSERT_BATCH_SIZE]
        client.insert_vectors(KNOWLEDGE_BASE_INDEX, batch, use_msgpack=False)
    
    # Dense-only evaluation indexes
    dense_indexes = [
        (KNOWLEDGE_BASE_FP32_INDEX, "FP32"),
        (KNOWLEDGE_BASE_FP16_INDEX, "FP16"),
        (KNOWLEDGE_BASE_INT8_INDEX, "INT8")
    ]
    
    for index_name, label in dense_indexes:
        print(f"\n{index_name} ({label}):")
        for i in tqdm(range(0, len(dense_vectors), INSERT_BATCH_SIZE), desc=f"Inserting {label} vectors"):
            batch = dense_vectors[i:i + INSERT_BATCH_SIZE]
            client.insert_vectors(index_name, batch, use_msgpack=False)
    
    # Verify indexes
    print("\nVerifying indexes...")
    all_indexes = [KNOWLEDGE_BASE_INDEX, KNOWLEDGE_BASE_FP32_INDEX, KNOWLEDGE_BASE_FP16_INDEX, KNOWLEDGE_BASE_INT8_INDEX, SESSION_MEMORY_INDEX]
    
    for index_name in all_indexes:
        info = client.get_index_info(index_name)
        print(f"\n{index_name}:")
        print(f"   Total elements: {info.get('total_elements', 'N/A')}")
        print(f"   Dimension: {info.get('dimension', 'N/A')}")
        print(f"   Quantization: {info.get('quant_level', 'N/A')}")
        print(f"   Sparse model: {info.get('sparse_model', 'N/A')}")
    
    # Create backup
    print(f"\nCreating backup of {KNOWLEDGE_BASE_INDEX}...")
    client.backup_index(KNOWLEDGE_BASE_INDEX)
    
    print(f"\nLoading complete! All indexes are ready.")
    print(f"Ready to run the HiveMind assistant!")

if __name__ == "__main__":
    main()
