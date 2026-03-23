"""
Query Router: corpus-aware IDF-based weight calculation for hybrid search
Grounded in Mandikal et al. 2024, Mala et al. 2025, and Hsu & Tzeng 2025
"""

import re
import json
import math
import string
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import VOCABULARY_PATH, ARXIV_YEAR_START, ARXIV_YEAR_END, TOP_K_RETRIEVAL

@dataclass
class RouterResult:
    """Result of query routing with corpus-aware weights"""
    query_type: str        # KEYWORD / CONCEPTUAL / HYBRID
    dense_weight: float    
    sparse_weight: float   
    filters: List[Dict]    
    confidence: float      
    avg_query_idf: float   
    specificity: float     # z-score
    explanation: str       # human readable for Streamlit UI

@dataclass  
class CorpusStats:
    """Corpus statistics loaded from vocabulary.json"""
    idf_scores: Dict[str, float]
    avg_idf: float
    max_idf: float
    min_idf: float
    std_idf: float
    total_docs: int

class QueryRouter:
    """Corpus-aware query router using IDF-based weight calculation"""
    
    # Weight boundaries grounded in Mandikal et al. 2024
    DENSE_MAX  = 0.70
    DENSE_MIN  = 0.30
    SPARSE_MAX = 0.70
    SPARSE_MIN = 0.30
    
    def __init__(self, vocabulary_path: str = None):
        """Initialize router with corpus statistics"""
        self.vocab_path = vocabulary_path or VOCABULARY_PATH
        self.corpus_stats = self._load_corpus_stats(self.vocab_path)
        self._compile_patterns()
        
        # Download NLTK data if needed
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            word_tokenize('test')
        except LookupError:
            nltk.download('punkt')
    
    def _load_corpus_stats(self, path: str) -> CorpusStats:
        """Load corpus statistics from vocabulary.json"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)
            
            return CorpusStats(
                idf_scores=vocab_data.get('idf_scores', {}),
                avg_idf=vocab_data.get('avg_idf', 5.0),
                max_idf=vocab_data.get('max_idf', 10.0),
                min_idf=vocab_data.get('min_idf', 1.0),
                std_idf=vocab_data.get('std_idf', 2.0),
                total_docs=vocab_data.get('total_docs', 10000)
            )
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"WARNING: Could not load vocabulary from {path}: {e}")
            print("Using fallback corpus statistics")
            return CorpusStats(
                idf_scores={},
                avg_idf=5.0,
                max_idf=10.0,
                min_idf=1.0,
                std_idf=2.0,
                total_docs=10000
            )
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency"""
        # Model name patterns for boost
        self.model_patterns = [
            r'\bBERT\b', r'\bGPT\b', r'\bLLaMA\b', r'\bMistral\b',
            r'\bLoRA\b', r'\bQLoRA\b', r'\bRLHF\b', r'\bT5\b',
            r'\bViT\b', r'\bResNet\b', r'\bTransformer\b', r'\bAttention\b'
        ]
        self.model_regex = re.compile('|'.join(self.model_patterns), re.IGNORECASE)
        
        # Temporal patterns (DISABLED - all papers from 2026)
        # self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        # self.temporal_words = re.compile(r'\b(recent|latest|new|current|modern|state-of-the-art|SOTA)\b', re.IGNORECASE)
        
        # Category patterns
        self.category_patterns = {
            'cs.CL': [r'\bNLP\b', r'\bnatural language\b', r'\bmachine translation\b', r'\bsentiment\b'],
            'cs.CV': [r'\bcomputer vision\b', r'\bimage\b', r'\bvision\b', r'\bdetection\b', r'\bsegmentation\b'],
            'cs.LG': [r'\bmachine learning\b', r'\blearning\b', r'\bneural network\b', r'\bdeep learning\b'],
            'cs.AI': [r'\bartificial intelligence\b', r'\bAI\b', r'\bagent\b', r'\breinforcement learning\b']
        }
        self.category_regex = {
            cat: re.compile('|'.join(patterns), re.IGNORECASE)
            for cat, patterns in self.category_patterns.items()
        }
    
    def _preprocess(self, text: str) -> List[str]:
        """Preprocess text identically to sparse.py"""
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
    
    def _compute_avg_idf(self, tokens: List[str]) -> float:
        """Compute average IDF of query tokens"""
        if not tokens:
            return self.corpus_stats.avg_idf
        
        idf_sum = 0.0
        for token in tokens:
            # Use actual IDF if known, corpus average if unknown
            idf_sum += self.corpus_stats.idf_scores.get(token, self.corpus_stats.avg_idf)
        
        return idf_sum / len(tokens)
    
    def _idf_to_sparse_weight(self, avg_idf: float) -> float:
        """Convert average IDF to sparse weight using z-score normalization"""
        # Compute z-score normalization (corpus-relative)
        z = (avg_idf - self.corpus_stats.avg_idf) / (self.corpus_stats.std_idf + 1e-8)
        
        # Map z-score to sparse weight
        sparse_weight = 0.5 + (z * 0.1)
        
        # Clip to valid range
        sparse_weight = max(self.SPARSE_MIN, min(self.SPARSE_MAX, sparse_weight))
        
        return sparse_weight
    
    def _apply_model_pattern_boost(self, query: str, sparse_weight: float) -> float:
        """Apply boost for known model name patterns"""
        matches = len(self.model_regex.findall(query))
        boost = min(matches * 0.03, 0.09)  # Max +0.09 boost
        
        boosted_weight = sparse_weight + boost
        return min(self.SPARSE_MAX, boosted_weight)
    
    def _extract_year_filter(self, query: str) -> Optional[Dict]:
        """Extract year range filter from temporal patterns"""
        # DISABLED: All papers are from 2026, year filtering provides no value
        return None
    
    def _extract_category_filters(self, query: str) -> List[Dict]:
        """Extract category filters from query"""
        matching_categories = []
        
        for cat, regex in self.category_regex.items():
            if regex.search(query):
                matching_categories.append(cat)
        
        if not matching_categories:
            return []
        
        if len(matching_categories) == 1:
            return [{"category": {"$eq": matching_categories[0]}}]
        else:
            return [{"category": {"$in": matching_categories}}]
    
    def _determine_query_type(self, sparse_weight: float, dense_weight: float) -> str:
        """Determine query type label for display"""
        if sparse_weight >= 0.60:
            return "KEYWORD"
        elif dense_weight >= 0.60:
            return "CONCEPTUAL"
        else:
            return "HYBRID"
    
    def _build_explanation(self, query: str, avg_query_idf: float, specificity: float, 
                          sparse_weight: float, dense_weight: float, filters: List[Dict]) -> str:
        """Build human-readable explanation"""
        parts = []
        
        # Explain specificity
        if specificity > 0.5:
            parts.append(f"Query contains rare terms (avg IDF: {avg_query_idf:.2f}) -> favoring sparse search")
        elif specificity < -0.5:
            parts.append(f"Query contains common terms (avg IDF: {avg_query_idf:.2f}) -> favoring dense search")
        else:
            parts.append(f"Query has average term specificity (avg IDF: {avg_query_idf:.2f}) -> balanced search")
        
        # Explain model boost
        if self.model_regex.search(query):
            parts.append("Model name patterns detected -> boosted sparse weight")
        
        # Explain filters
        if filters:
            filter_types = []
            for f in filters:
                if "category" in f:
                    filter_types.append("category filter")
            parts.append(f"Applied: {', '.join(filter_types)}")
        
        return " | ".join(parts)
    
    def classify_query(self, query: str) -> RouterResult:
        """Classify query and compute corpus-aware weights"""
        # Preprocess query
        tokens = self._preprocess(query)
        
        # Compute average IDF
        avg_query_idf = self._compute_avg_idf(tokens)
        
        # Compute specificity (z-score)
        specificity = (avg_query_idf - self.corpus_stats.avg_idf) / (self.corpus_stats.std_idf + 1e-8)
        
        # Convert to sparse weight
        sparse_weight = self._idf_to_sparse_weight(avg_query_idf)
        
        # Apply model pattern boost
        sparse_weight = self._apply_model_pattern_boost(query, sparse_weight)
        
        # Compute dense weight
        dense_weight = 1.0 - sparse_weight
        
        # Extract filters
        filters = []
        
        # Temporal filters (year only, no weight change)
        year_filter = self._extract_year_filter(query)
        if year_filter:
            filters.append(year_filter)
        
        # Category filters (no weight change)
        category_filters = self._extract_category_filters(query)
        filters.extend(category_filters)
        
        # Determine query type
        query_type = self._determine_query_type(sparse_weight, dense_weight)
        
        # Compute confidence
        deviation = abs(sparse_weight - 0.5)
        confidence = min(deviation / 0.2, 1.0)
        
        # Build explanation
        explanation = self._build_explanation(query, avg_query_idf, specificity, 
                                            sparse_weight, dense_weight, filters)
        
        return RouterResult(
            query_type=query_type,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            filters=filters,
            confidence=confidence,
            avg_query_idf=avg_query_idf,
            specificity=specificity,
            explanation=explanation
        )
    
    def explain_routing(self, query: str) -> str:
        """Convenience method for Streamlit to get explanation"""
        result = self.classify_query(query)
        return result.explanation

def main():
    """Test the query router with various query types"""
    print("Testing Corpus-Aware Query Router")
    print("=" * 60)
    
    router = QueryRouter()
    
    # Print corpus stats
    print(f"Corpus Statistics:")
    print(f"   Vocabulary size: {len(router.corpus_stats.idf_scores)}")
    print(f"   Average IDF: {router.corpus_stats.avg_idf:.3f}")
    print(f"   Max IDF: {router.corpus_stats.max_idf:.3f}")
    print(f"   Min IDF: {router.corpus_stats.min_idf:.3f}")
    print(f"   Std IDF: {router.corpus_stats.std_idf:.3f}")
    print(f"   Total documents: {router.corpus_stats.total_docs}")
    print()
    
    # Test queries
    test_queries = [
        ("LoRA QLoRA fine-tuning memory efficiency", "KEYWORD"),
        ("How do neural networks learn representations?", "CONCEPTUAL"),
        ("Recent advances in diffusion models", "HYBRID"),
        ("NLP sentiment analysis techniques", "CATEGORY"),
        ("BERT vs GPT performance comparison", "KEYWORD_BOOST"),
        ("What are the benefits of self-supervised learning?", "CONCEPTUAL")
    ]
    
    for query, expected_type in test_queries:
        result = router.classify_query(query)
        
        print(f"Query: {query}")
        print(f"  Type: {result.query_type} (expected: {expected_type})")
        print(f"  Dense: {result.dense_weight:.3f}, Sparse: {result.sparse_weight:.3f}")
        print(f"  Specificity (z-score): {result.specificity:.2f}")
        print(f"  Avg Query IDF: {result.avg_query_idf:.2f} vs Corpus: {router.corpus_stats.avg_idf:.2f}")
        print(f"  Confidence: {result.confidence:.2f}")
        if result.filters:
            print(f"  Filters: {result.filters}")
        print(f"  Explanation: {result.explanation}")
        print("-" * 40)

if __name__ == "__main__":
    main()
