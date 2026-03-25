"""
Voyage AI Reranker: rerank top-k results for better relevance
Takes top-20 from Endee, returns top-5 reranked with scores
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import VOYAGE_API_KEY, RERANKER_MODEL, TOP_K_RERANKED
import voyageai
from .search import SearchResult

@dataclass
class RerankResult:
    """Reranked search result"""
    result: SearchResult
    endee_score: float
    rerank_score: float
    rank_improvement: int  # How many positions moved up

class VoyageReranker:
    """Voyage AI reranker wrapper"""
    
    def __init__(self):
        if not VOYAGE_API_KEY:
            raise ValueError("VOYAGE_API_KEY not set in .env file")
        
        voyageai.api_key = VOYAGE_API_KEY
        self.client = voyageai.Client()
    
    def rerank(self, query: str, results: List[SearchResult], 
               top_k: int = TOP_K_RERANKED) -> List[RerankResult]:
        """Rerank search results using Voyage AI"""
        if not results:
            return []
        
        # Prepare documents for reranking
        documents = []
        original_indices = []
        
        for i, result in enumerate(results):
            # Use title + abstract snippet for reranking
            doc_text = f"{result.title}. {result.abstract_snippet}"
            documents.append(doc_text)
            original_indices.append(i)
        
        try:
            # Call Voyage AI rerank API
            rerank_response = self.client.rerank(
                query=query,
                documents=documents,
                model=RERANKER_MODEL,
                top_k=min(top_k, len(documents))
            )
            
            # Process rerank results
            reranked = []
            
            for rerank_item in rerank_response.results:
                original_idx = original_indices[rerank_item.index]
                original_result = results[original_idx]
                
                # Calculate rank improvement
                rank_improvement = original_idx - rerank_item.index
                
                rerank_result = RerankResult(
                    result=original_result,
                    endee_score=original_result.score,
                    rerank_score=rerank_item.relevance_score,
                    rank_improvement=rank_improvement
                )
                
                reranked.append(rerank_result)
            
            print(f"Reranked {len(results)} to {len(reranked)} results")
            
            # Show improvements
            improvements = [r.rank_improvement for r in reranked if r.rank_improvement > 0]
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                print(f"   Avg rank improvement: {avg_improvement:.1f} positions")
            
            return reranked
            
        except Exception as e:
            print(f"ERROR: Reranking failed: {e}")
            # Fallback to original results
            return [RerankResult(
                result=result,
                endee_score=result.score,
                rerank_score=result.score,
                rank_improvement=0
            ) for result in results[:top_k]]
    
    def compare_scores(self, original_results: List[SearchResult], 
                       reranked_results: List[RerankResult]) -> Dict:
        """Compare Endee vs rerank scores"""
        comparison = {
            "total_original": len(original_results),
            "total_reranked": len(reranked_results),
            "avg_endee_score": 0.0,
            "avg_rerank_score": 0.0,
            "score_correlation": 0.0,
            "top_5_overlap": 0
        }
        
        if not reranked_results:
            return comparison
        
        # Calculate average scores
        comparison["avg_endee_score"] = sum(r.endee_score for r in reranked_results) / len(reranked_results)
        comparison["avg_rerank_score"] = sum(r.rerank_score for r in reranked_results) / len(reranked_results)
        
        # Calculate correlation (simple Pearson)
        endee_scores = [r.endee_score for r in reranked_results]
        rerank_scores = [r.rerank_score for r in reranked_results]
        
        if len(endee_scores) > 1:
            import numpy as np
            correlation = np.corrcoef(endee_scores, rerank_scores)[0, 1]
            comparison["score_correlation"] = correlation if not np.isnan(correlation) else 0.0
        
        # Calculate top-5 overlap
        original_top_5 = set(r.id for r in original_results[:5])
        reranked_top_5 = set(r.result.id for r in reranked_results[:5])
        comparison["top_5_overlap"] = len(original_top_5 & reranked_top_5)
        
        return comparison

def main():
    """Test the reranker"""
    # Create dummy search results for testing
    dummy_results = [
        SearchResult(
            id="test1",
            score=0.85,
            title="Attention is All You Need",
            authors=["Vaswani et al."],
            arxiv_id="1706.03762",
            abstract_snippet="The dominant sequence transduction models...",
            year=2017,
            category="cs.LG",
            has_code=True
        ),
        SearchResult(
            id="test2", 
            score=0.82,
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin et al."],
            arxiv_id="1810.04805",
            abstract_snippet="We introduce a new language representation model...",
            year=2018,
            category="cs.CL",
            has_code=True
        ),
        SearchResult(
            id="test3",
            score=0.78,
            title="GPT-3: Language Models are Few-Shot Learners",
            authors=["Brown et al."],
            arxiv_id="2005.14165",
            abstract_snippet="Recent work has demonstrated substantial gains...",
            year=2020,
            category="cs.CL",
            has_code=True
        )
    ]
    
    try:
        reranker = VoyageReranker()
        
        query = "How do transformer models work in NLP?"
        print(f"Testing reranker for query: {query}")
        print("=" * 60)
        
        print(f"\nOriginal results ({len(dummy_results)}):")
        for i, result in enumerate(dummy_results, 1):
            print(f"{i}. {result.title} (score: {result.score:.3f})")
        
        reranked = reranker.rerank(query, dummy_results, top_k=3)
        
        print(f"\nReranked results ({len(reranked)}):")
        for i, rerank_result in enumerate(reranked, 1):
            print(f"{i}. {rerank_result.result.title}")
            print(f"   Endee: {rerank_result.endee_score:.3f} to Rerank: {rerank_result.rerank_score:.3f}")
            if rerank_result.rank_improvement > 0:
                print(f"   Moved up {rerank_result.rank_improvement} positions")
        
        # Show comparison
        comparison = reranker.compare_scores(dummy_results, reranked)
        print(f"\nComparison:")
        print(f"   Avg Endee score: {comparison['avg_endee_score']:.3f}")
        print(f"   Avg Rerank score: {comparison['avg_rerank_score']:.3f}")
        print(f"   Score correlation: {comparison['score_correlation']:.3f}")
        print(f"   Top-5 overlap: {comparison['top_5_overlap']}/5")
        
    except Exception as e:
        print(f"ERROR: Reranker test failed: {e}")
        print("   Make sure VOYAGE_API_KEY is set in .env file")

if __name__ == "__main__":
    main()
