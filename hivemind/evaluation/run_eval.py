"""
Evaluation Runner: benchmark 6 configurations on SciFact dataset
Measures recall@k, latency, and generates comparison plots
"""

import json
import time
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    KNOWLEDGE_BASE_INDEX, KNOWLEDGE_BASE_FP32_INDEX, KNOWLEDGE_BASE_FP16_INDEX, KNOWLEDGE_BASE_INT8_INDEX,
    EVAL_DATASET, EVAL_NUM_QUERIES, EVAL_RESULTS_FILE,
    validate_config
)
from retrieval.search import EndeeSearcher
from retrieval.reranker import VoyageReranker

class EvaluationRunner:
    """Runs comprehensive evaluation of different search configurations"""
    
    def __init__(self):
        self.searcher = EndeeSearcher()
        self.reranker = VoyageReranker()
        
        # Define evaluation configurations
        self.configs = {
            "dense_fp32_only": {
                "index": KNOWLEDGE_BASE_FP32_INDEX,
                "use_dense": True,
                "use_sparse": False,
                "use_rerank": False,
                "description": "Dense FP32 only"
            },
            "dense_fp16_only": {
                "index": KNOWLEDGE_BASE_FP16_INDEX,
                "use_dense": True,
                "use_sparse": False,
                "use_rerank": False,
                "description": "Dense FP16 only"
            },
            "dense_int8_only": {
                "index": KNOWLEDGE_BASE_INT8_INDEX,
                "use_dense": True,
                "use_sparse": False,
                "use_rerank": False,
                "description": "Dense INT8 only"
            },
            "sparse_only": {
                "index": KNOWLEDGE_BASE_INDEX,
                "use_dense": False,
                "use_sparse": True,
                "use_rerank": False,
                "description": "Sparse BM25 only"
            },
            "hybrid_fp16": {
                "index": KNOWLEDGE_BASE_INDEX,
                "use_dense": True,
                "use_sparse": True,
                "use_rerank": False,
                "description": "Hybrid FP16"
            },
            "hybrid_fp16_rerank": {
                "index": KNOWLEDGE_BASE_INDEX,
                "use_dense": True,
                "use_sparse": True,
                "use_rerank": True,
                "description": "Hybrid FP16 + Rerank"
            },
            "hybrid_idf_router_rerank": {
                "index": KNOWLEDGE_BASE_INDEX,
                "use_dense": True,
                "use_sparse": True,
                "use_rerank": True,
                "use_idf_router": True,
                "description": "Hybrid IDF Router + Rerank"
            }
        }
    
    def load_domain_dataset(self, num_queries: int = EVAL_NUM_QUERIES) -> Tuple[List[str], List[set]]:
        """Load domain-specific ML/AI evaluation queries with ground truth"""
        print(f"Loading domain-specific ML/AI evaluation dataset...")
        
        # Curated queries with expected relevant keywords/topics
        # Each query has keywords that should appear in relevant papers
        eval_data = [
            {
                "query": "transformer attention mechanism neural networks",
                "keywords": ["transformer", "attention", "self-attention", "multi-head"],
                "categories": ["cs.LG", "cs.CL"]
            },
            {
                "query": "BERT language model pre-training fine-tuning",
                "keywords": ["bert", "language model", "pre-training", "fine-tuning"],
                "categories": ["cs.CL"]
            },
            {
                "query": "computer vision object detection convolutional networks",
                "keywords": ["object detection", "vision", "convolutional", "cnn"],
                "categories": ["cs.CV"]
            },
            {
                "query": "neural network optimization gradient descent",
                "keywords": ["optimization", "gradient", "training", "neural network"],
                "categories": ["cs.LG"]
            },
            {
                "query": "generative adversarial networks GAN image synthesis",
                "keywords": ["gan", "generative", "adversarial"],
                "categories": ["cs.CV", "cs.LG"]
            },
            {
                "query": "reinforcement learning policy gradient deep Q-learning",
                "keywords": ["reinforcement learning", "policy", "q-learning"],
                "categories": ["cs.LG", "cs.AI"]
            },
            {
                "query": "natural language processing sentiment analysis",
                "keywords": ["nlp", "sentiment", "text classification"],
                "categories": ["cs.CL"]
            },
            {
                "query": "image segmentation semantic segmentation deep learning",
                "keywords": ["segmentation", "semantic", "image"],
                "categories": ["cs.CV"]
            },
            {
                "query": "recurrent neural networks LSTM sequence modeling",
                "keywords": ["lstm", "recurrent", "rnn", "sequence"],
                "categories": ["cs.LG", "cs.CL"]
            },
            {
                "query": "transfer learning domain adaptation few-shot learning",
                "keywords": ["transfer learning", "domain adaptation", "few-shot"],
                "categories": ["cs.LG"]
            },
            {
                "query": "graph neural networks node classification",
                "keywords": ["graph", "gnn", "node"],
                "categories": ["cs.LG"]
            },
            {
                "query": "vision transformer ViT image classification",
                "keywords": ["vision transformer", "vit", "image classification"],
                "categories": ["cs.CV"]
            },
            {
                "query": "contrastive learning self-supervised representation",
                "keywords": ["contrastive", "self-supervised", "representation"],
                "categories": ["cs.LG", "cs.CV"]
            },
            {
                "query": "neural architecture search AutoML",
                "keywords": ["architecture search", "nas", "automl"],
                "categories": ["cs.LG"]
            },
            {
                "query": "diffusion models image generation stable diffusion",
                "keywords": ["diffusion", "generation", "denoising"],
                "categories": ["cs.CV", "cs.LG"]
            },
            {
                "query": "large language models GPT scaling laws",
                "keywords": ["language model", "gpt", "scaling"],
                "categories": ["cs.CL", "cs.LG"]
            },
            {
                "query": "meta-learning learning to learn",
                "keywords": ["meta-learning", "learning to learn"],
                "categories": ["cs.LG"]
            },
            {
                "query": "multimodal learning vision language models CLIP",
                "keywords": ["multimodal", "vision language", "clip"],
                "categories": ["cs.CV", "cs.CL"]
            },
            {
                "query": "neural machine translation encoder decoder",
                "keywords": ["translation", "encoder", "decoder"],
                "categories": ["cs.CL"]
            },
            {
                "query": "adversarial robustness adversarial examples",
                "keywords": ["adversarial", "robustness", "attack"],
                "categories": ["cs.LG", "cs.CV"]
            }
        ]
        
        # Limit to requested number
        eval_data = eval_data[:num_queries]
        
        queries = [item["query"] for item in eval_data]
        
        # For ground truth, we'll use keyword matching
        # Store keywords for each query to check relevance
        self.eval_keywords = {i: item["keywords"] for i, item in enumerate(eval_data)}
        self.eval_categories = {i: item["categories"] for i, item in enumerate(eval_data)}
        
        # Placeholder for relevant docs - will be determined by keyword matching
        relevant_docs = [set() for _ in queries]
        
        print(f"Loaded {len(queries)} domain-specific queries")
        return queries, relevant_docs
    
    def calculate_recall_at_k(self, retrieved_results: List, query_idx: int, k: int) -> float:
        """Calculate recall@k using keyword matching for relevance"""
        if not retrieved_results:
            return 0.0
        
        # Get expected keywords and categories for this query
        expected_keywords = self.eval_keywords.get(query_idx, [])
        expected_categories = self.eval_categories.get(query_idx, [])
        
        if not expected_keywords:
            return 0.0
        
        top_k = retrieved_results[:k]
        
        # Count how many top_k results are relevant
        relevant_found = sum(
            1 for result in top_k
            if any(kw.lower() in result.title.lower() 
                   for kw in expected_keywords)
            and (result.category in expected_categories 
                 if expected_categories else True)
        )
        
        # Total relevant = all results that match keywords (not just top-k)
        total_relevant = sum(
            1 for result in retrieved_results  # full result set
            if any(kw.lower() in result.title.lower() 
                   for kw in expected_keywords)
            and (result.category in expected_categories 
                 if expected_categories else True)
        )
        
        if total_relevant == 0:
            return 0.0
        
        # recall@k = relevant found in top-k / total relevant
        return relevant_found / total_relevant
    
    def evaluate_config(self, config_name: str, config: Dict, queries: List[str], 
                       relevant_docs: List[set]) -> Dict:
        """Evaluate a single configuration"""
        print(f"\n Evaluating: {config['description']}")
        
        results = {
            "config": config_name,
            "description": config["description"],
            "recall@1": [],
            "recall@5": [],
            "recall@10": [],
            "latency_ms": []
        }
        
        for i, query in enumerate(tqdm(queries, desc=f"Testing {config_name}")):
            # Perform search
            start_time = time.time()
            
            # Modify searcher behavior based on config
            if not config["use_dense"] and not config["use_sparse"]:
                # Skip invalid config
                continue
            
            # Pass config flags to searcher
            search_response = self.searcher.search(
                query          = query,
                k              = 20,
                index_name     = config["index"],
                use_dense      = config.get("use_dense", True),
                use_sparse     = config.get("use_sparse", False),
                use_idf_router = config.get("use_idf_router", False)
            )
            
            search_results = search_response.results if search_response else []
            
            # Apply reranking if needed
            if config["use_rerank"] and search_results:
                reranked = self.reranker.rerank(query, search_results, top_k=10)
                search_results = [r.result for r in reranked]
            
            search_time = (time.time() - start_time) * 1000
            
            # Calculate metrics using keyword-based relevance
            results["recall@1"].append(self.calculate_recall_at_k(search_results, i, 1))
            results["recall@5"].append(self.calculate_recall_at_k(search_results, i, 5))
            results["recall@10"].append(self.calculate_recall_at_k(search_results, i, 10))
            results["latency_ms"].append(search_time)
        
        # Save latency list before overwriting
        latency_list = results["latency_ms"].copy()
        
        # Calculate averages for recall metrics
        for metric in ["recall@1", "recall@5", "recall@10"]:
            if results[metric]:
                results[metric] = np.mean(results[metric])
            else:
                results[metric] = 0.0
        
        # Calculate percentiles for latency from saved list
        if latency_list:
            results["p50_latency"] = float(np.percentile(latency_list, 50))
            results["p95_latency"] = float(np.percentile(latency_list, 95))
        else:
            results["p50_latency"] = 0.0
            results["p95_latency"] = 0.0
        
        print(f"   Recall@5: {results['recall@5']:.3f}")
        print(f"   Latency: {results['p50_latency']:.1f}ms")
        
        return results
    
    def run_evaluation(self) -> List[Dict]:
        """Run full evaluation across all configurations"""
        print("Starting comprehensive evaluation...")
        
        # Load dataset
        queries, relevant_docs = self.load_domain_dataset()
        
        if not queries:
            raise ValueError("No evaluation queries available")
        
        # Evaluate each configuration
        all_results = []
        
        for config_name, config in self.configs.items():
            try:
                result = self.evaluate_config(config_name, config, queries, relevant_docs)
                all_results.append(result)
            except Exception as e:
                print(f"Failed to evaluate {config_name}: {e}")
                continue
        
        return all_results
    
    def save_results(self, results: List[Dict]):
        """Save evaluation results to CSV"""
        # Create DataFrame
        df_data = []
        for result in results:
            df_data.append({
                "config": result["config"],
                "description": result["description"],
                "recall@1": result["recall@1"],
                "recall@5": result["recall@5"],
                "recall@10": result["recall@10"],
                "p50_latency_ms": result["p50_latency"],
                "p95_latency_ms": result["p95_latency"]
            })
        
        df = pd.DataFrame(df_data)
        
        # Save to CSV
        EVAL_RESULTS_FILE.parent.mkdir(exist_ok=True)
        df.to_csv(EVAL_RESULTS_FILE, index=False)
        
        print(f"Results saved to {EVAL_RESULTS_FILE}")
        
        return df
    
    def create_plots(self, df: pd.DataFrame):
        """Generate evaluation plots"""
        plots_dir = EVAL_RESULTS_FILE.parent
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Recall@5 comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(df["description"], df["recall@5"], color='skyblue', alpha=0.8)
        plt.title('Recall@5 Comparison Across Configurations', fontsize=14, fontweight='bold')
        plt.xlabel('Configuration', fontsize=12)
        plt.ylabel('Recall@5', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, df["recall@5"]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot1_path = plots_dir / "recall_comparison.png"
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Recall vs Latency Pareto frontier
        plt.figure(figsize=(10, 6))
        
        # Color code by configuration type
        colors = []
        for config in df["config"]:
            if "dense" in config and "only" in config:
                colors.append('red' if 'fp32' in config else 'orange' if 'fp16' in config else 'yellow')
            elif "sparse" in config:
                colors.append('green')
            elif "hybrid" in config:
                colors.append('blue' if "rerank" not in config else 'purple')
            else:
                colors.append('gray')
        
        scatter = plt.scatter(df["p50_latency_ms"], df["recall@5"], 
                             s=100, c=colors, alpha=0.7, edgecolors='black')
        
        # Add labels for each point
        for i, row in df.iterrows():
            plt.annotate(row["config"], (row["p50_latency_ms"], row["recall@5"]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.title('Recall@5 vs Latency Pareto Frontier', fontsize=14, fontweight='bold')
        plt.xlabel('p50 Latency (ms)', fontsize=12)
        plt.ylabel('Recall@5', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Dense FP32'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Dense FP16'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='Dense INT8'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Sparse Only'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Hybrid'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Hybrid + Rerank')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plot2_path = plots_dir / "recall_latency_pareto.png"
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved:")
        print(f"   {plot1_path}")
        print(f"   {plot2_path}")
    
    def print_summary(self, df: pd.DataFrame):
        """Print evaluation summary"""
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        # Sort by recall@5
        df_sorted = df.sort_values("recall@5", ascending=False)
        
        print(f"\nBEST CONFIGURATIONS (by Recall@5):")
        for i, row in df_sorted.iterrows():
            print(f"   {i+1}. {row['description']}: {row['recall@5']:.3f} recall, {row['p50_latency_ms']:.1f}ms")
        
        # Find best trade-off
        df_sorted['score'] = df_sorted['recall@5'] / (df_sorted['p50_latency_ms'] / 100)  # Normalize latency
        best_tradeoff = df_sorted.loc[df_sorted['score'].idxmax()]
        
        print(f"\nBEST TRADE-OFF (recall/latency):")
        print(f"   {best_tradeoff['description']}")
        print(f"   Recall@5: {best_tradeoff['recall@5']:.3f}")
        print(f"   Latency: {best_tradeoff['p50_latency_ms']:.1f}ms")
        
        # Key findings
        hybrid_fp16 = df[df['config'] == 'hybrid_fp16']
        dense_fp16 = df[df['config'] == 'dense_fp16_only']
        
        if not hybrid_fp16.empty and not dense_fp16.empty:
            hybrid_recall = hybrid_fp16['recall@5'].iloc[0]
            dense_recall = dense_fp16['recall@5'].iloc[0]
            improvement = ((hybrid_recall - dense_recall) / dense_recall) * 100
            
            print(f"\nKEY FINDINGS:")
            print(f"   Hybrid search improves recall@5 by {improvement:.1f}% over dense-only FP16")
            print(f"   Hybrid latency: {hybrid_fp16['p50_latency_ms'].iloc[0]:.1f}ms")
            print(f"   Dense-only latency: {dense_fp16['p50_latency_ms'].iloc[0]:.1f}ms")
        
        print("\n" + "="*80)

def main():
    """Run evaluation"""
    print("Starting HiveMind Evaluation")
    print("="*80)
    
    # Validate configuration
    validate_config()
    
    # Create evaluation runner
    runner = EvaluationRunner()
    
    try:
        # Run evaluation
        results = runner.run_evaluation()
        
        if not results:
            print("No evaluation results generated")
            return
        
        # Save results
        df = runner.save_results(results)
        
        # Create plots
        runner.create_plots(df)
        
        # Print summary
        runner.print_summary(df)
        
        print(f"\nEvaluation complete! Check {EVAL_RESULTS_FILE.parent} for results.")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
