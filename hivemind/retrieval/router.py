"""
Query Router: classify queries into types and determine search parameters
KEYWORD, CONCEPTUAL, TEMPORAL, HYBRID: weights + filters
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import DEFAULT_PREFILTER_THRESHOLD, DEFAULT_BOOST_PERCENTAGE

@dataclass
class RouterResult:
    """Result of query routing"""
    query_type: str
    dense_weight: float
    sparse_weight: float
    filters: List[Dict]
    confidence: float

class QueryRouter:
    """Routes queries to optimal search configuration"""
    
    def __init__(self):
        # Keyword patterns for technical terms
        self.model_patterns = [
            r'\b(BERT|GPT|LLaMA|T5|ViT|ResNet|Transformer|LSTM|GRU|RNN|CNN|SVM|RF|XGBoost)\b',
            r'\b(LoRA|QLoRA|AdaLoRA|Adapter|Prefix-tuning|Prompt-tuning)\b',
            r'\b(Adam|SGD|RMSprop|AdaGrad|AdamW)\b',
            r'\b(ReLU|GELU|Swish|Sigmoid|Tanh|Softmax)\b',
            r'\b(Attention|Self-attention|Multi-head|Cross-attention)\b',
            r'\b(Embedding|Word2Vec|GloVe|FastText|BERT|RoBERTa)\b'
        ]
        
        # Conceptual patterns
        self.conceptual_patterns = [
            r'\b(why|how|what is|explain|describe|compare|relationship between)\b',
            r'\b(advantage|disadvantage|pro|con|benefit|drawback)\b',
            r'\b(principle|theory|concept|idea|approach|methodology)\b'
        ]
        
        # Temporal patterns
        self.temporal_patterns = [
            r'\b(20\d{2})\b',  # Years 2000-2099
            r'\b(recent|latest|current|state-of-the-art|SOTA)\b',
            r'\b(new|emerging|trending|breakthrough)\b',
            r'\b(last|past|previous|earlier)\b'
        ]
        
        # Category patterns
        self.category_patterns = {
            'cs.LG': r'\b(machine learning|ML|deep learning|neural network|AI)\b',
            'cs.CL': r'\b(NLP|natural language|text|language|translation|sentiment)\b',
            'cs.CV': r'\b(computer vision|image|visual|object detection|segmentation)\b',
            'cs.AI': r'\b(artificial intelligence|AI|reasoning|planning|knowledge)\b'
        }
    
    def classify_query(self, query: str) -> RouterResult:
        """Classify query and return routing parameters"""
        query_lower = query.lower()
        
        # Initialize scores
        keyword_score = 0
        conceptual_score = 0
        temporal_score = 0
        
        # Check keyword patterns
        for pattern in self.model_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                keyword_score += 1
        
        # Check conceptual patterns
        for pattern in self.conceptual_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                conceptual_score += 1
        
        # Check temporal patterns
        for pattern in self.temporal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                temporal_score += 1
        
        # Determine query type
        scores = {
            'KEYWORD': keyword_score,
            'CONCEPTUAL': conceptual_score,
            'TEMPORAL': temporal_score
        }
        
        query_type = max(scores, key=scores.get)
        max_score = scores[query_type]
        
        # If no strong signals, default to HYBRID
        if max_score == 0:
            query_type = 'HYBRID'
            confidence = 0.5
        else:
            confidence = min(max_score / 3.0, 1.0)  # Normalize to 0-1
        
        # Determine weights based on type
        if query_type == 'KEYWORD':
            dense_weight = 0.3
            sparse_weight = 0.7
        elif query_type == 'CONCEPTUAL':
            dense_weight = 0.8
            sparse_weight = 0.2
        elif query_type == 'TEMPORAL':
            dense_weight = 0.5
            sparse_weight = 0.5
        else:  # HYBRID
            dense_weight = 0.5
            sparse_weight = 0.5
        
        # Generate filters
        filters = []
        
        # Temporal filters
        if query_type == 'TEMPORAL':
            year_filter = self._extract_year_filter(query)
            if year_filter:
                filters.append(year_filter)
        
        # Category filters
        category_filter = self._extract_category_filter(query)
        if category_filter:
            filters.append(category_filter)
        
        return RouterResult(
            query_type=query_type,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            filters=filters,
            confidence=confidence
        )
    
    def _extract_year_filter(self, query: str) -> Dict:
        """Extract year range filter from query"""
        # Find all years in the query
        years = re.findall(r'\b(20\d{2})\b', query)
        
        if not years:
            # Check for temporal keywords
            if any(word in query.lower() for word in ['recent', 'latest', 'current', 'new', 'emerging']):
                current_year = 2024  # Could make this dynamic
                return {"year": {"$gte": current_year - 2, "$lte": current_year}}
            elif any(word in query.lower() for word in ['past', 'previous', 'earlier']):
                return {"year": {"$lte": 2020}}
            return None
        
        # If specific years mentioned, create range around them
        years = [int(year) for year in years]
        min_year, max_year = min(years), max(years)
        
        # Add +/- 1 year buffer
        return {"year": {"$gte": max(2019, min_year - 1), "$lte": min(2024, max_year + 1)}}
    
    def _extract_category_filter(self, query: str) -> Dict:
        """Extract category filter from query"""
        query_lower = query.lower()
        
        for category, pattern in self.category_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                return {"category": {"$eq": category}}
        
        return None
    
    def explain_routing(self, query: str, result: RouterResult) -> str:
        """Generate explanation of routing decision"""
        explanation = f"Query type: {result.query_type} (confidence: {result.confidence:.2f})\n"
        explanation += f"Weights: dense={result.dense_weight:.1f}, sparse={result.sparse_weight:.1f}\n"
        
        if result.filters:
            explanation += f"Filters: {len(result.filters)} applied\n"
            for filter_dict in result.filters:
                for field, condition in filter_dict.items():
                    explanation += f"  - {field}: {condition}\n"
        else:
            explanation += "No filters applied\n"
        
        return explanation

def main():
    """Test the query router"""
    router = QueryRouter()
    
    test_queries = [
        "What is the relationship between attention mechanisms and transformer models?",
        "BERT vs GPT-4 performance comparison 2023",
        "How does LoRA fine-tuning work?",
        "Recent advances in computer vision object detection",
        "Explain the advantages of Adam optimizer over SGD",
        "2022 NLP sentiment analysis techniques"
    ]
    
    print("Testing Query Router")
    print("=" * 60)
    
    for query in test_queries:
        result = router.classify_query(query)
        print(f"\nQuery: {query}")
        print(router.explain_routing(query, result))
        print("-" * 40)

if __name__ == "__main__":
    main()
