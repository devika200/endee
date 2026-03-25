"""
Session Memory Management: store/retrieve conversation history
Uses Endee session_memory index with timestamp and session_id filters
"""

import json
import time
import msgpack
import requests
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    ENDEE_HOST, ENDEE_AUTH_TOKEN,
    SESSION_MEMORY_INDEX, MEMORY_EXPIRY_DAYS,
    VOYAGE_API_KEY, EMBEDDING_MODEL, TOP_K_MEMORY,
    validate_config
)
import voyageai
from .search import SearchResult

@dataclass
class MemoryEntry:
    """Single memory entry (Q&A pair)"""
    session_id: str
    query: str
    answer: str
    timestamp: datetime
    memory_id: str

@dataclass
class MemoryResult:
    """Memory search result"""
    query: str
    answer: str
    timestamp: datetime
    score: float
    session_id: str

class SessionMemory:
    """Manages conversation session memory in Endee"""
    
    def __init__(self):
        self.host = ENDEE_HOST.rstrip('/')
        self.auth_token = ENDEE_AUTH_TOKEN
        
        # Initialize Voyage AI client for embeddings
        if VOYAGE_API_KEY:
            voyageai.api_key = VOYAGE_API_KEY
            self.voyage_client = voyageai.Client()
        else:
            self.voyage_client = None
            print("WARNING: VOYAGE_API_KEY not set - memory embedding disabled")
        
        # Setup headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/msgpack"
        }
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
    
    def _embed_text(self, text: str) -> Optional[List[float]]:
        """Embed text using Voyage AI"""
        if not self.voyage_client:
            return None
        
        try:
            result = self.voyage_client.embed(
                [text],
                model=EMBEDDING_MODEL,
                input_type="document"
            )
            return result.embeddings[0]
        except Exception as e:
            print(f"ERROR: Memory embedding failed: {e}")
            return None
    
    def _call_endee(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request to Endee"""
        try:
            response = requests.request(
                method, 
                f"{self.host}/api/v1{endpoint}",
                headers=self.headers,
                **kwargs
            )
            response.raise_for_status()
            return response
        except Exception as e:
            print(f"ERROR: Endee request failed: {e}")
            raise
    
    def write_memory(self, session_id: str, query: str, answer: str) -> bool:
        """Store a Q&A pair in session memory"""
        # Create memory text (query + answer)
        memory_text = f"Q: {query} A: {answer}"
        
        # Generate embedding
        embedding = self._embed_text(memory_text)
        if not embedding:
            print("WARNING: Could not embed memory text - skipping storage")
            return False
        
        # Create timestamp
        timestamp = datetime.now()
        timestamp_str = timestamp.isoformat()
        
        # Create memory ID
        memory_id = f"{session_id}_{int(timestamp.timestamp())}"
        
        # Create filter payload
        filter_payload = {
            "session_id": session_id,
            "timestamp": int(timestamp.timestamp())
        }
        
        # Create metadata payload
        meta_payload = {
            "query": query,
            "answer": answer,
            "session_id": session_id,
            "timestamp": timestamp_str
        }
        
        # Create vector object
        vector_obj = {
            "id": memory_id,
            "vector": embedding,
            "filter": json.dumps(filter_payload),
            "meta": json.dumps(meta_payload)
        }
        
        try:
            # Insert into Endee
            response = self._call_endee(
                "POST",
                f"/index/{SESSION_MEMORY_INDEX}/vector/insert",
                json=[vector_obj]
            )
            
            print(f"Stored memory: {memory_id}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to store memory: {e}")
            return False
    
    def read_memory(self, session_id: str, query: str, k: int = TOP_K_MEMORY) -> List[MemoryResult]:
        """Retrieve relevant memories for a session"""
        # Embed query
        embedding = self._embed_text(query)
        if not embedding:
            return []
        
        # Create search payload
        payload = {
            "vector": embedding,
            "k": k,
            "filter": json.dumps([{"session_id": {"$eq": session_id}}]),
            "filter_params": {
                "prefilter_threshold": 1000,  # Lower threshold for memory
                "boost_percentage": 5
            }
        }
        
        try:
            # Search memory index
            response = self._call_endee(
                "POST",
                f"/index/{SESSION_MEMORY_INDEX}/search",
                json=payload
            )
            
            # Parse MessagePack response
            result_data = msgpack.unpackb(response.content, raw=False)
            
            # Convert to MemoryResult objects
            memories = []
            for item in result_data.get('results', []):
                meta = json.loads(item.get('meta', '{}'))
                
                memory = MemoryResult(
                    query=meta.get('query', ''),
                    answer=meta.get('answer', ''),
                    timestamp=datetime.fromisoformat(meta.get('timestamp', '')),
                    score=item.get('score', 0.0),
                    session_id=meta.get('session_id', '')
                )
                memories.append(memory)
            
            # Sort by score (descending)
            memories.sort(key=lambda x: x.score, reverse=True)
            
            print(f"Retrieved {len(memories)} memories for session {session_id}")
            return memories
            
        except Exception as e:
            print(f"ERROR: Failed to read memories: {e}")
            return []
    
    def cleanup_old_memories(self, session_id: str = None, days: int = MEMORY_EXPIRY_DAYS) -> int:
        """Delete old memories (by session or all expired)"""
        cutoff_time = datetime.now() - timedelta(days=days)
        cutoff_timestamp = int(cutoff_time.timestamp())
        
        # Create filter
        if session_id:
            filter_payload = [{
                "session_id": {"$eq": session_id},
                "timestamp": {"$lt": cutoff_timestamp}
            }]
        else:
            filter_payload = [{
                "timestamp": {"$lt": cutoff_timestamp}
            }]
        
        try:
            # Delete by filter
            response = self._call_endee(
                "DELETE",
                f"/index/{SESSION_MEMORY_INDEX}/vectors/delete",
                json={"filter": filter_payload}
            )
            
            deleted_count = int(response.text)
            print(f"Deleted {deleted_count} old memories")
            return deleted_count
            
        except Exception as e:
            print(f"ERROR: Failed to cleanup memories: {e}")
            return 0
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a specific session"""
        # This would require a custom endpoint or counting logic
        # For now, return basic info
        try:
            # Get index info
            response = self._call_endee("GET", f"/index/{SESSION_MEMORY_INDEX}/info")
            info = response.json()
            
            return {
                "session_id": session_id,
                "total_memories": info.get('total_elements', 0),
                "memory_expiry_days": MEMORY_EXPIRY_DAYS
            }
            
        except Exception as e:
            print(f"ERROR: Failed to get session stats: {e}")
            return {"session_id": session_id, "total_memories": 0, "error": str(e)}
    
    def list_sessions(self) -> List[str]:
        """List all active sessions (would need custom implementation)"""
        # This is a placeholder - would need to scan all memories
        # For now, return empty list
        print("Session listing not implemented yet")
        return []

def main():
    """Test session memory functionality"""
    print("Testing Session Memory")
    print("=" * 60)
    
    try:
        memory = SessionMemory()
        session_id = "test_session_001"
        
        # Test writing memories
        print("\nWriting test memories...")
        memory.write_memory(session_id, "What is BERT?", "BERT is a transformer model pre-trained on large text corpora.")
        memory.write_memory(session_id, "How does attention work?", "Attention mechanisms allow models to focus on relevant parts of the input.")
        memory.write_memory(session_id, "What are transformers?", "Transformers are neural network architectures based on attention mechanisms.")
        
        # Test reading memories
        print("\nReading memories...")
        query = "transformer models"
        memories = memory.read_memory(session_id, query, k=3)
        
        for i, memory_result in enumerate(memories, 1):
            print(f"\n{i}. Score: {memory_result.score:.3f}")
            print(f"   Q: {memory_result.query}")
            print(f"   A: {memory_result.answer[:100]}...")
            print(f"   Time: {memory_result.timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Test cleanup
        print(f"\nTesting cleanup (removing memories older than {MEMORY_EXPIRY_DAYS} days)...")
        deleted = memory.cleanup_old_memories(session_id, days=0)  # Delete all test memories
        print(f"   Deleted {deleted} test memories")
        
        # Test session stats
        print(f"\nSession stats:")
        stats = memory.get_session_stats(session_id)
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"ERROR: Memory test failed: {e}")
        print("   Make sure Endee is running and session_memory index exists")

if __name__ == "__main__":
    main()
