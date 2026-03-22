"""
Endee Hybrid Search: embed query, generate sparse vector, call Endee API
Supports dense, sparse, and hybrid search with filters
"""

import json
import numpy as np
import msgpack
import requests
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    ENDEE_HOST, ENDEE_AUTH_TOKEN,
    KNOWLEDGE_BASE_INDEX, SESSION_MEMORY_INDEX,
    EMBEDDING_DIM, DEFAULT_PREFILTER_THRESHOLD,
    VOYAGE_API_KEY, EMBEDDING_MODEL, TOP_K_RETRIEVAL
)
import voyageai
from .router import QueryRouter, RouterResult

@dataclass
class SearchResult:
    """Single search result from Endee"""
    id: str
    score: float
    title: str
    authors: List[str]
    arxiv_id: str
    abstract_snippet: str
    year: int
    category: str
    has_code: bool

@dataclass
class SearchResponse:
    """Complete search response"""
    results: List[SearchResult]
    total_time_ms: float
    query_type: str
    dense_weight: float
    sparse_weight: float
    filters: List[Dict]

class EndeeSearcher:
    """Handles search operations with Endee"""
    
    def __init__(self):
        self.host = ENDEE_HOST.rstrip('/')
        self.auth_token = ENDEE_AUTH_TOKEN
        self.router = QueryRouter()
        
        # Initialize Voyage AI client
        if VOYAGE_API_KEY:
            voyageai.api_key = VOYAGE_API_KEY
            self.voyage_client = voyageai.Client()
        else:
            self.voyage_client = None
            print("WARNING: VOYAGE_API_KEY not set - embedding disabled")
        
        # Load vocabulary for sparse vectors
        self.vocab = self._load_vocabulary()
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/msgpack"  # Endee returns msgpack
        }
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
    
    def _load_vocabulary(self) -> Dict[str, int]:
        """Load vocabulary for sparse vector generation"""
        vocab_file = Path(__file__).parent.parent / "data" / "processed" / "vocabulary.json"
        
        if vocab_file.exists():
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            print(f"Loaded vocabulary: {len(vocab)} terms")
            return vocab
        else:
            print("WARNING: Vocabulary file not found - sparse search disabled")
            return {}
    
    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Embed query using Voyage AI"""
        if not self.voyage_client:
            return None
        
        try:
            result = self.voyage_client.embed(
                [query],
                model=EMBEDDING_MODEL,
                input_type="query"
            )
            return np.array(result.embeddings[0])
        except Exception as e:
            print(f"ERROR: Query embedding failed: {e}")
            return None
    
    def _generate_sparse_vector(self, query: str) -> Tuple[List[int], List[float]]:
        """Generate sparse BM25 vector for query"""
        if not self.vocab:
            return [], []
        
        # Simple tokenization (matches preprocessing in sparse.py)
        import re
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        
        try:
            # Download NLTK data if needed
            import nltk
            try:
                stopwords.words('english')
            except LookupError:
                nltk.download('stopwords')
            try:
                word_tokenize('test')
            except LookupError:
                nltk.download('punkt')
            
            # Preprocess query
            query_lower = query.lower()
            query_lower = re.sub(r'[^a-zA-Z0-9\s]', ' ', query_lower)
            tokens = word_tokenize(query_lower)
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            
            # Map to vocabulary
            indices = []
            values = []
            
            token_counts = {}
            for token in tokens:
                if token in self.vocab:
                    term_id = self.vocab[token]
                    token_counts[term_id] = token_counts.get(term_id, 0) + 1
            
            if token_counts:
                indices = list(token_counts.keys())
                # Simple TF weighting
                total_tokens = len(tokens)
                values = [count / total_tokens for count in token_counts.values()]
            
            return indices, values
            
        except Exception as e:
            print(f"ERROR: Sparse vector generation failed: {e}")
            return [], []
    
    def _call_endee_search(self, index_name: str, dense_vector: np.ndarray, 
                          sparse_indices: List[int], sparse_values: List[float],
                          k: int, filters: List[Dict], dense_weight: float, 
                          sparse_weight: float) -> List[SearchResult]:
        """Call Endee search API"""
        # Build search payload
        payload = {
            "k": k,
            "filter_params": {
                "prefilter_threshold": DEFAULT_PREFILTER_THRESHOLD,
                "boost_percentage": 10
            }
        }
        
        # Add dense vector if available
        if dense_vector is not None:
            payload["vector"] = dense_vector.tolist()
        
        # Add sparse vector if available
        if sparse_indices and sparse_values:
            payload["sparse_indices"] = sparse_indices
            payload["sparse_values"] = sparse_values
        
        # Add filters if any
        if filters:
            payload["filter"] = json.dumps(filters)
        
        # Add weight parameters for hybrid search
        if dense_vector is not None and sparse_indices:
            # For true hybrid search, Endee uses both vectors
            pass  # Endee handles weighting internally
        elif dense_vector is not None:
            # Dense-only search
            pass
        elif sparse_indices:
            # Sparse-only search (remove dense vector)
            payload.pop("vector", None)
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.host}/api/v1/index/{index_name}/search",
                headers=self.headers,
                json=payload
            )
            
            response.raise_for_status()
            
            # Parse MessagePack response
            response_data = msgpack.unpackb(response.content, raw=False)
            
            # Endee returns a list of results, not a dict with "results" key
            results = []
            for item in response_data[:k]:
                # item format: [score, id, meta_bytes, filter_str, ?, ?]
                if len(item) >= 4:
                    score = item[0]
                    doc_id = item[1]
                    meta_bytes = item[2]
                    filter_str = item[3]
                    
                    # Parse metadata
                    try:
                        if isinstance(meta_bytes, bytes):
                            meta = json.loads(meta_bytes.decode('utf-8'))
                        else:
                            meta = json.loads(meta_bytes)
                    except:
                        meta = {}
                    
                    # Parse filter data
                    try:
                        filter_data = json.loads(filter_str) if isinstance(filter_str, str) else {}
                    except:
                        filter_data = {}
                    
                    result = SearchResult(
                        id=doc_id,
                        score=score,
                        title=meta.get('title', ''),
                        authors=meta.get('authors', []),
                        arxiv_id=meta.get('arxiv_id', ''),
                        abstract_snippet=meta.get('abstract_snippet', ''),
                        year=filter_data.get('year', 2020),
                        category=filter_data.get('category', ''),
                        has_code=filter_data.get('has_code', False)
                    )
                    results.append(result)
            
            search_time = (time.time() - start_time) * 1000
            
            return results, search_time
            
        except Exception as e:
            print(f"ERROR: Endee search failed: {e}")
            return [], 0.0
    
    def search(self, query: str, k: int = TOP_K_RETRIEVAL, 
               search_memory: bool = False) -> SearchResponse:
        """Perform hybrid search"""
        # Route the query
        routing_result = self.router.classify_query(query)
        
        # Generate embeddings
        dense_vector = self._embed_query(query)
        sparse_indices, sparse_values = self._generate_sparse_vector(query)
        
        # Choose index
        index_name = SESSION_MEMORY_INDEX if search_memory else KNOWLEDGE_BASE_INDEX
        
        # Perform search
        start_time = time.time()
        results, search_time = self._call_endee_search(
            index_name=index_name,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            k=k,
            filters=routing_result.filters,
            dense_weight=routing_result.dense_weight,
            sparse_weight=routing_result.sparse_weight
        )
        total_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=results,
            total_time_ms=total_time,
            query_type=routing_result.query_type,
            dense_weight=routing_result.dense_weight,
            sparse_weight=routing_result.sparse_weight,
            filters=routing_result.filters
        )
    
    def search_memory(self, query: str, session_id: str, k: int = 3) -> List[SearchResult]:
        """Search session memory with session filter"""
        # Generate embeddings
        dense_vector = self._embed_query(query)
        sparse_indices, sparse_values = self._generate_sparse_vector(query)
        
        # Add session filter
        filters = [{"session_id": {"$eq": session_id}}]
        
        # Search memory index
        results, _ = self._call_endee_search(
            index_name=SESSION_MEMORY_INDEX,
            dense_vector=dense_vector,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
            k=k,
            filters=filters,
            dense_weight=0.7,  # Favor semantic matching for memory
            sparse_weight=0.3
        )
        
        return results

def main():
    """Test the searcher"""
    import time
    
    searcher = EndeeSearcher()
    
    test_queries = [
        "What are the latest advances in transformer models?",
        "How does LoRA fine-tuning work?",
        "Computer vision object detection 2023",
        "BERT vs GPT comparison"
    ]
    
    print("Testing Endee Search")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        response = searcher.search(query, k=5)
        
        print(f"Type: {response.query_type}")
        print(f"Time: {response.total_time_ms:.1f}ms")
        print(f"Results: {len(response.results)}")
        
        for i, result in enumerate(response.results[:3], 1):
            print(f"\n{i}. {result.title}")
            print(f"   Score: {result.score:.3f}")
            print(f"   Year: {result.year}, Category: {result.category}")
            print(f"   Abstract: {result.abstract_snippet[:100]}...")

if __name__ == "__main__":
    main()
