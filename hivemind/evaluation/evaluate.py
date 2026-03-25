"""
Comprehensive Evaluation System for HiveMind + Endee
Uses semantic relevance scoring for accurate metrics
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass

from retrieval.search import EndeeSearcher
from retrieval.reranker import VoyageReranker
from config import (
    KNOWLEDGE_BASE_INDEX, KNOWLEDGE_BASE_FP32_INDEX, 
    KNOWLEDGE_BASE_FP16_INDEX, KNOWLEDGE_BASE_INT8_INDEX
)

@dataclass
class EvalQuery:
    """Evaluation query with semantic relevance criteria"""
    query: str
    description: str
    expected_keywords: List[str]
    irrelevant_keywords: List[str]

class HiveMindEvaluator:
    """Comprehensive evaluation system for HiveMind"""
    
    def __init__(self):
        self.searcher = EndeeSearcher()
        self.reranker = VoyageReranker()
        
        # Comprehensive test queries covering ML/AI domains
        self.eval_queries = [
            EvalQuery(
                query="attention mechanism",
                description="Papers about attention mechanisms in neural networks",
                expected_keywords=["attention", "transformer", "self-attention", "multi-head", "query", "key", "value"],
                irrelevant_keywords=["protein", "dna", "medical", "clinical", "health"]
            ),
            EvalQuery(
                query="transformer models",
                description="Papers about transformer architecture and models",
                expected_keywords=["transformer", "bert", "gpt", "encoder", "decoder", "attention"],
                irrelevant_keywords=["cnn", "resnet", "image", "vision", "protein"]
            ),
            EvalQuery(
                query="BERT language model",
                description="Papers specifically about BERT model",
                expected_keywords=["bert", "bidirectional", "pre-training", "nlp", "language model"],
                irrelevant_keywords=["gpt", "image", "vision", "protein", "medical"]
            ),
            EvalQuery(
                query="computer vision",
                description="Papers about computer vision and image processing",
                expected_keywords=["computer vision", "image", "cnn", "object detection", "segmentation"],
                irrelevant_keywords=["nlp", "text", "language", "medical", "protein"]
            ),
            EvalQuery(
                query="reinforcement learning",
                description="Papers about reinforcement learning algorithms",
                expected_keywords=["reinforcement learning", "rl", "agent", "policy", "reward", "q-learning"],
                irrelevant_keywords=["supervised", "classification", "nlp", "attention", "protein"]
            ),
            EvalQuery(
                query="machine learning algorithms",
                description="Papers about ML algorithms and methods",
                expected_keywords=["machine learning", "algorithm", "classification", "regression", "clustering"],
                irrelevant_keywords=["protein", "medical", "attention", "transformer", "dna"]
            ),
            EvalQuery(
                query="neural network",
                description="Papers about neural networks and deep learning",
                expected_keywords=["neural network", "deep learning", "neural", "layers", "architecture"],
                irrelevant_keywords=["protein", "dna", "medical", "clinical", "health"]
            ),
            EvalQuery(
                query="GPT model",
                description="Papers about GPT models and large language models",
                expected_keywords=["gpt", "large language model", "llm", "generative", "transformer"],
                irrelevant_keywords=["bert", "cnn", "image", "vision", "protein"]
            ),
            EvalQuery(
                query="deep learning",
                description="Papers about deep learning architectures",
                expected_keywords=["deep learning", "neural network", "architecture", "layers", "backpropagation"],
                irrelevant_keywords=["protein", "medical", "classical", "statistical", "traditional"]
            ),
            EvalQuery(
                query="natural language processing",
                description="Papers about NLP and text processing",
                expected_keywords=["nlp", "natural language", "text", "tokenization", "language model"],
                irrelevant_keywords=["image", "vision", "protein", "medical", "cnn"]
            )
        ]
        
        # Test configurations
        self.configs = {
            "dense_fp32": {
                "index": KNOWLEDGE_BASE_FP32_INDEX,
                "use_dense": True,
                "use_sparse": False,
                "description": "Dense FP32"
            },
            "dense_fp16": {
                "index": KNOWLEDGE_BASE_FP16_INDEX,
                "use_dense": True,
                "use_sparse": False,
                "description": "Dense FP16"
            },
            "dense_int8": {
                "index": KNOWLEDGE_BASE_INT8_INDEX,
                "use_dense": True,
                "use_sparse": False,
                "description": "Dense INT8"
            },
            "sparse_only": {
                "index": KNOWLEDGE_BASE_INDEX,
                "use_dense": False,
                "use_sparse": True,
                "description": "Sparse BM25"
            },
            "hybrid": {
                "index": KNOWLEDGE_BASE_INDEX,
                "use_dense": True,
                "use_sparse": True,
                "description": "Hybrid (Dense+Sparse)"
            },
            "hybrid_rerank": {
                "index": KNOWLEDGE_BASE_INDEX,
                "use_dense": True,
                "use_sparse": True,
                "use_rerank": True,
                "description": "Hybrid + Rerank"
            }
        }
    
    def calculate_semantic_relevance(self, title: str, query: EvalQuery) -> float:
        """Calculate relevance score based on keyword matching"""
        title_lower = title.lower()
        
        # Count expected keywords
        expected_count = sum(1 for kw in query.expected_keywords if kw.lower() in title_lower)
        
        # Count irrelevant keywords (penalty)
        irrelevant_count = sum(1 for kw in query.irrelevant_keywords if kw.lower() in title_lower)
        
        # Calculate relevance score (0 to 1)
        if expected_count == 0:
            base_score = 0.0
        else:
            base_score = min(expected_count / 3, 1.0)  # Cap at 1.0
        
        # Apply penalty for irrelevant keywords
        if irrelevant_count > 0:
            base_score *= max(0.1, 1.0 - (irrelevant_count * 0.3))
        
        return base_score
    
    def calculate_metrics(self, relevance_scores: List[float]) -> Dict[str, float]:
        """Calculate evaluation metrics from relevance scores"""
        if not relevance_scores:
            return {'recall_at_5': 0.0, 'recall_at_10': 0.0, 'precision_at_5': 0.0, 'ndcg_at_5': 0.0}
        
        # Binary relevance (score > 0.3)
        relevant_docs = [i for i, s in enumerate(relevance_scores) if s > 0.3]
        
        # Recall@k
        recall_at_5 = len([i for i in relevant_docs if i < 5]) / min(5, len(relevance_scores))
        recall_at_10 = len(relevant_docs) / min(10, len(relevance_scores))
        
        # Precision@5
        precision_at_5 = len([i for i in relevant_docs if i < 5]) / min(5, len(relevance_scores))
        
        # NDCG@5
        ndcg_at_5 = 0.0
        for i, score in enumerate(relevance_scores[:5], 1):
            ndcg_at_5 += score / np.log2(i + 1)  # DCG formula
        
        # Normalize by ideal DCG (all scores = 1.0)
        ideal_dcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(6, len(relevance_scores) + 1)))
        ndcg_at_5 = ndcg_at_5 / ideal_dcg if ideal_dcg > 0 else 0.0
        
        return {
            'recall_at_5': recall_at_5,
            'recall_at_10': recall_at_10,
            'precision_at_5': precision_at_5,
            'ndcg_at_5': ndcg_at_5,
            'avg_relevance': np.mean(relevance_scores),
            'relevant_found': len(relevant_docs)
        }
    
    def evaluate_config(self, config_name: str, config: Dict) -> List[Dict]:
        """Evaluate a single configuration"""
        print(f"\nEvaluating: {config['description']}")
        results = []
        
        for eval_query in self.eval_queries:
            try:
                # Perform search
                start_time = time.time()
                response = self.searcher.search(eval_query.query, k=20, search_memory=False)
                latency = (time.time() - start_time) * 1000
                
                # Rerank if needed
                if config.get("use_rerank") and self.reranker and response.results:
                    response.results = [r.result for r in self.reranker.rerank(
                        eval_query.query, response.results[:10], top_k=10
                    )]
                
                # Calculate relevance scores
                relevance_scores = []
                for result in response.results[:10]:
                    relevance = self.calculate_semantic_relevance(result.title, eval_query)
                    relevance_scores.append(relevance)
                
                # Calculate metrics
                metrics = self.calculate_metrics(relevance_scores)
                
                result = {
                    'query': eval_query.query,
                    'config': config_name,
                    'results_found': len(response.results),
                    'latency_ms': latency,
                    'relevance_scores': relevance_scores,
                    **metrics
                }
                
                results.append(result)
                
                print(f"  {eval_query.query}: R@5={metrics['recall_at_5']:.3f}, P@5={metrics['precision_at_5']:.3f}, NDCG@5={metrics['ndcg_at_5']:.3f}")
                
            except Exception as e:
                print(f"  ERROR in {eval_query.query}: {e}")
                continue
        
        return results
    
    def run_evaluation(self) -> pd.DataFrame:
        """Run full evaluation"""
        print("=" * 80)
        print("HIVEMIND + ENDEE EVALUATION")
        print("=" * 80)
        print(f"Evaluating {len(self.eval_queries)} queries across {len(self.configs)} configurations")
        print(f"Queries cover: attention, transformers, BERT, computer vision, RL, ML, neural networks, GPT")
        
        all_results = []
        
        for config_name, config in self.configs.items():
            config_results = self.evaluate_config(config_name, config)
            all_results.extend(config_results)
        
        return pd.DataFrame(all_results)
    
    def generate_summary(self, df: pd.DataFrame):
        """Generate comprehensive evaluation summary"""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Aggregate by configuration
        summary = df.groupby('config').agg({
            'recall_at_5': 'mean',
            'recall_at_10': 'mean',
            'precision_at_5': 'mean',
            'ndcg_at_5': 'mean',
            'avg_relevance': 'mean',
            'latency_ms': 'mean',
            'relevant_found': 'mean'
        }).round(3)
        
        # Sort by recall@5
        summary = summary.sort_values('recall_at_5', ascending=False)
        
        print("\nPERFORMANCE RANKING:")
        for i, (config, row) in enumerate(summary.iterrows(), 1):
            print(f"{i}. {config}:")
            print(f"   Recall@5: {row['recall_at_5']:.3f}")
            print(f"   Recall@10: {row['recall_at_10']:.3f}")
            print(f"   Precision@5: {row['precision_at_5']:.3f}")
            print(f"   NDCG@5: {row['ndcg_at_5']:.3f}")
            print(f"   Avg Relevance: {row['avg_relevance']:.3f}")
            print(f"   Latency: {row['latency_ms']:.1f}ms")
            print(f"   Avg Relevant: {row['relevant_found']:.1f}/10")
            print()
        
        # Performance interpretation
        best_recall = summary.iloc[0]['recall_at_5']
        print(f"PERFORMANCE ASSESSMENT:")
        if best_recall >= 0.7:
            print("  EXCELLENT: High recall (>70%)")
        elif best_recall >= 0.5:
            print("  GOOD: Moderate recall (50-70%)")
        elif best_recall >= 0.3:
            print("  ACCEPTABLE: Low recall (30-50%)")
        else:
            print("  POOR: Very low recall (<30%)")
        
        # Key findings
        print(f"\nKEY FINDINGS:")
        print(f"  Best recall: {best_recall:.3f}")
        print(f"  Best configuration: {summary.index[0]}")
        print(f"  Fastest configuration: {summary.sort_values('latency_ms').index[0]}")
        
        # Hybrid vs Dense comparison
        if 'dense_fp16' in summary.index and 'hybrid' in summary.index:
            dense_recall = summary.loc['dense_fp16', 'recall_at_5']
            hybrid_recall = summary.loc['hybrid', 'recall_at_5']
            dense_latency = summary.loc['dense_fp16', 'latency_ms']
            hybrid_latency = summary.loc['hybrid', 'latency_ms']
            
            if dense_recall > 0:
                recall_improvement = ((hybrid_recall - dense_recall) / dense_recall) * 100
                latency_diff = hybrid_latency - dense_latency
                print(f"\nHYBRID vs DENSE FP16:")
                print(f"  Recall change: {recall_improvement:+.1f}%")
                print(f"  Latency change: {latency_diff:+.1f}ms")
        
        return summary
    
    def create_plots(self, df: pd.DataFrame, eval_dir: Path, timestamp: str):
        """Create evaluation plots"""
        # Plot 1: Recall Comparison
        plt.figure(figsize=(12, 6))
        recall_summary = df.groupby('config')[['recall_at_5', 'recall_at_10']].mean()
        
        x = range(len(recall_summary))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], recall_summary['recall_at_5'], width, 
                label='Recall@5', alpha=0.8, color='skyblue')
        plt.bar([i + width/2 for i in x], recall_summary['recall_at_10'], width,
                label='Recall@10', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Configuration')
        plt.ylabel('Recall')
        plt.title('Recall Comparison Across Configurations')
        plt.xticks(x, recall_summary.index, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot1_file = eval_dir / f"recall_comparison_{timestamp}.png"
        plt.savefig(plot1_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Performance Trade-off
        plt.figure(figsize=(10, 6))
        perf_summary = df.groupby('config')[['recall_at_5', 'latency_ms']].mean()
        
        plt.scatter(perf_summary['latency_ms'], perf_summary['recall_at_5'], 
                   s=100, alpha=0.7, c='red')
        
        for i, (config, row) in enumerate(perf_summary.iterrows()):
            plt.annotate(config, (row['latency_ms'], row['recall_at_5']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.xlabel('Average Latency (ms)')
        plt.ylabel('Recall@5')
        plt.title('Performance Trade-off: Recall vs Latency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot2_file = eval_dir / f"performance_tradeoff_{timestamp}.png"
        plt.savefig(plot2_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot1_file, plot2_file
    
    def save_results(self, df: pd.DataFrame, summary: pd.DataFrame):
        """Save evaluation results"""
        eval_dir = Path(__file__).parent
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = eval_dir / f"evaluation_results_{timestamp}.csv"
        df.to_csv(results_file, index=False)
        
        # Save summary
        summary_file = eval_dir / f"evaluation_summary_{timestamp}.csv"
        summary.to_csv(summary_file)
        
        # Create plots
        plot1_file, plot2_file = self.create_plots(df, eval_dir, timestamp)
        
        print(f"\nFILES GENERATED:")
        print(f"  Detailed results: {results_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Recall plot: {plot1_file}")
        print(f"  Trade-off plot: {plot2_file}")
        
        return results_file, summary_file

def main():
    """Run comprehensive evaluation"""
    evaluator = HiveMindEvaluator()
    
    try:
        # Run evaluation
        df = evaluator.run_evaluation()
        
        if df.empty:
            print("No evaluation completed successfully")
            return
        
        # Generate summary
        summary = evaluator.generate_summary(df)
        
        # Save results
        evaluator.save_results(df, summary)
        
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE!")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
