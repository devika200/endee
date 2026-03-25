"""
LLM Generation: Groq Llama-3.3-70b-versatile for RAG answer generation
Constructs prompt with context + memory + query, returns answer with sources
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, validate_config
from retrieval.search import SearchResult
from retrieval.memory import MemoryResult

try:
    from groq import Groq
except ImportError:
    print("groq package not installed. Run: pip install groq")
    sys.exit(1)

@dataclass
class LLMResponse:
    """Complete LLM response"""
    answer: str
    sources: List[Dict]
    context_used: int
    memory_used: int
    tokens_used: int
    model: str

class GroqGenerator:
    """Groq LLM wrapper for RAG generation"""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not set in .env file")
        
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL
    
    def _build_context_text(self, results: List[SearchResult], max_context: int = 3) -> str:
        """Build context text from search results"""
        if not results:
            return "No relevant papers found."
        
        context_parts = []
        for i, result in enumerate(results[:max_context], 1):
            context_part = f"""
Paper {i}:
Title: {result.title}
Authors: {', '.join(result.authors[:3])}{' et al.' if len(result.authors) > 3 else ''}
Year: {result.year}
Category: {result.category}
Abstract: {result.abstract_snippet}
ArXiv ID: {result.arxiv_id}
Score: {result.score:.3f}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_memory_text(self, memories: List[MemoryResult]) -> str:
        """Build memory text from conversation history"""
        if not memories:
            return "No relevant conversation history."
        
        memory_parts = []
        for i, memory in enumerate(memories, 1):
            memory_part = f"""
Memory {i}:
Previous Question: {memory.query}
Previous Answer: {memory.answer}
Time: {memory.timestamp.strftime('%Y-%m-%d %H:%M')}
Relevance: {memory.score:.3f}
"""
            memory_parts.append(memory_part)
        
        return "\n".join(memory_parts)
    
    def _build_prompt(self, query: str, context_text: str, memory_text: str) -> str:
        """Build complete RAG prompt"""
        prompt = f"""You are a helpful research assistant specializing in machine learning and computer science papers. Answer the user's question based only on the provided context and conversation memory.

CONTEXT FROM RELEVANT PAPERS:
{context_text}

RELEVANT CONVERSATION HISTORY:
{memory_text}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer based primarily on the provided paper context
2. If multiple papers are relevant, synthesize information across them
3. Mention specific papers by title when referencing their content
4. Include arXiv IDs for easy reference
5. If the context doesn't fully answer the question, acknowledge limitations
6. Keep answers concise but comprehensive (2-4 paragraphs)
7. Use conversation history to provide personalized follow-ups if relevant

ANSWER:"""
        
        return prompt
    
    def generate_answer(self, query: str, search_results: List[SearchResult], 
                       memory_results: List[MemoryResult] = None) -> LLMResponse:
        """Generate answer using RAG"""
        if memory_results is None:
            memory_results = []
        
        # Build context and memory
        context_text = self._build_context_text(search_results, max_context=3)
        memory_text = self._build_memory_text(memory_results)
        
        # Build prompt
        prompt = self._build_prompt(query, context_text, memory_text)
        
        try:
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful research assistant. Answer questions based only on the provided context."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3,  # Lower temperature for factual answers
                max_tokens=1000,  # Reasonable length for research answers
                top_p=0.9,
                stream=False
            )
            
            answer = completion.choices[0].message.content
            tokens_used = completion.usage.total_tokens
            
            # Extract sources
            sources = []
            for result in search_results[:3]:  # Top 3 sources
                sources.append({
                    "title": result.title,
                    "authors": result.authors,
                    "arxiv_id": result.arxiv_id,
                    "year": result.year,
                    "score": result.score,
                    "url": f"https://arxiv.org/abs/{result.arxiv_id}"
                })
            
            return LLMResponse(
                answer=answer,
                sources=sources,
                context_used=len(search_results),
                memory_used=len(memory_results),
                tokens_used=tokens_used,
                model=self.model
            )
            
        except Exception as e:
            print(f"LLM generation failed: {e}")
            # Fallback response
            return LLMResponse(
                answer=f"I apologize, but I encountered an error generating the answer: {str(e)}. Please try again.",
                sources=[],
                context_used=0,
                memory_used=0,
                tokens_used=0,
                model=self.model
            )
    
    def generate_simple_answer(self, query: str, context: str) -> str:
        """Generate simple answer without full RAG structure"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Answer based on the provided context."
                    },
                    {
                        "role": "user",
                        "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"Simple generation failed: {e}")
            return f"Error: {str(e)}"

def main():
    """Test the LLM generator"""
    print("Testing LLM Generator")
    print("=" * 60)
    
    try:
        generator = GroqGenerator()
        
        # Create dummy search results
        dummy_results = [
            SearchResult(
                id="test1",
                score=0.85,
                title="Attention is All You Need",
                authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                arxiv_id="1706.03762",
                abstract_snippet="The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
                year=2017,
                category="cs.LG",
                has_code=True
            ),
            SearchResult(
                id="test2",
                score=0.78,
                title="BERT: Pre-training of Deep Bidirectional Transformers",
                authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee"],
                arxiv_id="1810.04805",
                abstract_snippet="We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers...",
                year=2018,
                category="cs.CL",
                has_code=True
            )
        ]
        
        # Create dummy memory results
        dummy_memories = [
            MemoryResult(
                query="What is attention mechanism?",
                answer="Attention mechanisms allow neural networks to focus on specific parts of the input sequence when producing outputs.",
                timestamp=None,
                score=0.92,
                session_id="test_session"
            )
        ]
        
        test_query = "How do transformer models use attention mechanisms?"
        
        print(f"Query: {test_query}")
        print(f"Context: {len(dummy_results)} papers")
        print(f"Memory: {len(dummy_memories)} memories")
        print("-" * 40)
        
        # Generate answer
        response = generator.generate_answer(test_query, dummy_results, dummy_memories)
        
        print(f"\nAnswer:")
        print(response.answer)
        
        print(f"\nGeneration stats:")
        print(f"   Model: {response.model}")
        print(f"   Tokens used: {response.tokens_used}")
        print(f"   Context used: {response.context_used}")
        print(f"   Memory used: {response.memory_used}")
        
        print(f"\nSources:")
        for i, source in enumerate(response.sources, 1):
            print(f"   {i}. {source['title']} ({source['arxiv_id']})")
            print(f"      Score: {source['score']:.3f}, Year: {source['year']}")
        
    except Exception as e:
        print(f"ERROR: LLM test failed: {e}")
        print("   Make sure GROQ_API_KEY is set in .env file")

if __name__ == "__main__":
    main()
