"""
Router Quality Analysis: Analyze IDF-aware router behavior
Measures weight distribution, IDF correlation, and routing decisions
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass

from retrieval.router import QueryRouter

@dataclass
class RouterAnalysis:
    """Analysis of router behavior for a single query"""
    query: str
    expected_type: str
    
    # Router output
    dense_weight: float
    sparse_weight: float
    query_type: str
    specificity: float
    avg_query_idf: float
    confidence: float
    
    # Filters
    has_category_filter: bool
    num_filters: int

class RouterAnalyzer:
    """Analyze IDF-aware router quality"""
    
    def __init__(self):
        self.router = QueryRouter()
        
        # Test queries categorized by expected type
        self.test_queries = {
            "keyword": [
                "BERT fine-tuning techniques",
                "GPT-3 few-shot learning",
                "LoRA QLoRA parameter efficient",
                "ResNet architecture computer vision",
                "Transformer attention mechanism",
                "CLIP vision language model",
                "T5 text-to-text transfer",
                "ViT vision transformer",
                "DALL-E image generation",
                "Stable Diffusion latent space",
                "LLaMA open source model",
                "Mistral mixture of experts",
                "Whisper speech recognition",
                "SAM segment anything model",
                "YOLO object detection",
                "EfficientNet neural architecture",
                "MobileNet edge deployment",
                "BERT vs RoBERTa comparison",
                "GPT-2 GPT-3 scaling laws",
                "LoRA vs full fine-tuning"
            ],
            "conceptual": [
                "How do neural networks learn representations?",
                "What is the attention mechanism?",
                "Why does self-supervised learning work?",
                "How does backpropagation work?",
                "What are the benefits of transfer learning?",
                "How do transformers process sequences?",
                "What is gradient descent optimization?",
                "Why do large language models emerge abilities?",
                "How does batch normalization help training?",
                "What causes mode collapse in GANs?",
                "How do convolutional layers extract features?",
                "What is the vanishing gradient problem?",
                "Why does dropout prevent overfitting?",
                "How do residual connections help deep networks?",
                "What is the role of positional encoding?",
                "How does beam search improve generation?",
                "What are the trade-offs of model compression?",
                "Why does pre-training improve downstream tasks?",
                "How do vision transformers differ from CNNs?",
                "What causes catastrophic forgetting?"
            ],
            "category": [
                "NLP sentiment analysis techniques",
                "Computer vision object detection methods",
                "Natural language processing transformers",
                "Image segmentation deep learning",
                "Machine learning classification algorithms",
                "Reinforcement learning policy gradient",
                "Text generation language models",
                "Image recognition convolutional networks",
                "Speech recognition acoustic models",
                "Video understanding temporal modeling",
                "Question answering reading comprehension",
                "Machine translation neural networks",
                "Image captioning vision language",
                "Named entity recognition NLP",
                "Semantic segmentation computer vision",
                "Dialogue systems conversational AI",
                "Document classification text mining",
                "Face recognition deep learning",
                "Pose estimation computer vision",
                "Text summarization abstractive"
            ],
            "mixed": [
                "Recent advances in diffusion models",
                "State-of-the-art object detection",
                "Latest transformer architectures",
                "Modern neural network optimization",
                "Current trends in self-supervised learning",
                "New approaches to few-shot learning",
                "Recent progress in multimodal learning",
                "Latest developments in neural architecture search",
                "Current methods for model compression",
                "Recent work on continual learning",
                "New techniques for domain adaptation",
                "Latest research on adversarial robustness",
                "Current approaches to explainable AI",
                "Recent advances in meta-learning",
                "New methods for zero-shot learning",
                "Latest work on neural scaling laws",
                "Current research on efficient transformers",
                "Recent progress in vision-language models",
                "New approaches to prompt engineering",
                "Latest developments in chain-of-thought reasoning"
            ]
        }
    
    def analyze_query(self, query: str, expected_type: str) -> RouterAnalysis:
        """Analyze router behavior on a single query"""
        result = self.router.classify_query(query)
        
        # Check for category filters
        has_category_filter = any('category' in f for f in result.filters)
        num_filters = len(result.filters)
        
        return RouterAnalysis(
            query=query,
            expected_type=expected_type,
            dense_weight=result.dense_weight,
            sparse_weight=result.sparse_weight,
            query_type=result.query_type,
            specificity=result.specificity,
            avg_query_idf=result.avg_query_idf,
            confidence=result.confidence,
            has_category_filter=has_category_filter,
            num_filters=num_filters
        )
    
    def run_analysis(self) -> pd.DataFrame:
        """Run full router analysis"""
        print("=" * 80)
        print("IDF-AWARE ROUTER QUALITY ANALYSIS")
        print("=" * 80)
        print(f"Analyzing IDF-aware router behavior")
        print(f"Testing {sum(len(queries) for queries in self.test_queries.values())} queries")
        print(f"Corpus: {self.router.corpus_stats.total_docs} docs, {len(self.router.corpus_stats.idf_scores)} terms")
        print(f"Corpus avg IDF: {self.router.corpus_stats.avg_idf:.2f} (std: {self.router.corpus_stats.std_idf:.2f})")
        print()
        
        all_analyses = []
        
        for query_type, queries in self.test_queries.items():
            print(f"\nTesting {query_type} queries ({len(queries)} queries)...")
            
            for query in queries:
                analysis = self.analyze_query(query, query_type)
                all_analyses.append(analysis)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'query': a.query,
                'expected_type': a.expected_type,
                'dense_weight': a.dense_weight,
                'sparse_weight': a.sparse_weight,
                'query_type': a.query_type,
                'specificity': a.specificity,
                'avg_query_idf': a.avg_query_idf,
                'confidence': a.confidence,
                'has_category_filter': a.has_category_filter,
                'num_filters': a.num_filters
            }
            for a in all_analyses
        ])
        
        return df
    
    def generate_summary(self, df: pd.DataFrame):
        """Generate analysis summary"""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Weight distribution analysis
        print("\nWEIGHT DISTRIBUTION:")
        print(f"Sparse weights:")
        print(f"  Unique values: {len(df['sparse_weight'].unique())}")
        print(f"  Range: [{df['sparse_weight'].min():.2f}, {df['sparse_weight'].max():.2f}]")
        print(f"  Mean: {df['sparse_weight'].mean():.3f}")
        print(f"  Std dev: {df['sparse_weight'].std():.3f}")
        
        print(f"\nDense weights:")
        print(f"  Range: [{df['dense_weight'].min():.2f}, {df['dense_weight'].max():.2f}]")
        print(f"  Mean: {df['dense_weight'].mean():.3f}")
        print(f"  Std dev: {df['dense_weight'].std():.3f}")
        
        # IDF statistics
        print(f"\nIDF STATISTICS:")
        print(f"  Avg query IDF range: [{df['avg_query_idf'].min():.2f}, {df['avg_query_idf'].max():.2f}]")
        print(f"  Mean query IDF: {df['avg_query_idf'].mean():.2f}")
        print(f"  Corpus avg IDF: {self.router.corpus_stats.avg_idf:.2f}")
        print(f"  Avg specificity (z-score): {df['specificity'].mean():.2f}")
        print(f"  Specificity range: [{df['specificity'].min():.2f}, {df['specificity'].max():.2f}]")
        
        # Confidence analysis
        print(f"\nCONFIDENCE SCORES:")
        print(f"  Mean confidence: {df['confidence'].mean():.2f}")
        print(f"  High confidence (>0.7): {(df['confidence'] > 0.7).sum()} queries ({(df['confidence'] > 0.7).mean() * 100:.1f}%)")
        print(f"  Low confidence (<0.3): {(df['confidence'] < 0.3).sum()} queries ({(df['confidence'] < 0.3).mean() * 100:.1f}%)")
        
        # Filter analysis
        print(f"\nFILTER ANALYSIS:")
        print(f"  Queries with category filters: {df['has_category_filter'].sum()} ({df['has_category_filter'].mean() * 100:.1f}%)")
        print(f"  Avg filters per query: {df['num_filters'].mean():.2f}")
        
        # Query type distribution
        print(f"\nQUERY TYPE DISTRIBUTION:")
        type_counts = df['query_type'].value_counts()
        for qtype, count in type_counts.items():
            print(f"  {qtype}: {count} queries ({count/len(df)*100:.1f}%)")
        
        # Per-category analysis
        print(f"\nPER-CATEGORY ANALYSIS:")
        for category in df['expected_type'].unique():
            cat_df = df[df['expected_type'] == category]
            print(f"\n  {category.upper()}:")
            print(f"    Avg sparse weight: {cat_df['sparse_weight'].mean():.3f}")
            print(f"    Avg dense weight: {cat_df['dense_weight'].mean():.3f}")
            print(f"    Avg query IDF: {cat_df['avg_query_idf'].mean():.2f}")
            print(f"    Avg specificity: {cat_df['specificity'].mean():.2f}")
            print(f"    Avg confidence: {cat_df['confidence'].mean():.2f}")
            print(f"    Category filters: {cat_df['has_category_filter'].sum()}/{len(cat_df)}")
    
    def create_plots(self, df: pd.DataFrame):
        """Create visualization plots"""
        eval_dir = Path(__file__).parent
        
        # Plot 1: Weight distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sparse weight histogram
        ax1.hist(df['sparse_weight'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_title('IDF Router: Sparse Weight Distribution', fontweight='bold')
        ax1.set_xlabel('Sparse Weight')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(df['sparse_weight'].mean(), color='darkblue', linestyle='--', 
                   label=f'Mean: {df["sparse_weight"].mean():.3f}')
        ax1.legend()
        
        # Dense weight histogram
        ax2.hist(df['dense_weight'], bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('IDF Router: Dense Weight Distribution', fontweight='bold')
        ax2.set_xlabel('Dense Weight')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(df['dense_weight'].mean(), color='darkgreen', linestyle='--',
                   label=f'Mean: {df["dense_weight"].mean():.3f}')
        ax2.legend()
        
        plt.tight_layout()
        plot1_file = eval_dir / "router_weight_distribution.png"
        plt.savefig(plot1_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: IDF vs Sparse Weight
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(df['avg_query_idf'], df['sparse_weight'], 
                            c=df['specificity'], cmap='RdYlGn_r', 
                            s=100, alpha=0.6, edgecolors='black')
        plt.colorbar(scatter, label='Specificity (z-score)')
        plt.xlabel('Average Query IDF', fontsize=12)
        plt.ylabel('Sparse Weight', fontsize=12)
        plt.title('IDF-Aware Router: Query IDF vs Sparse Weight', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add corpus average IDF line
        corpus_avg = self.router.corpus_stats.avg_idf
        plt.axvline(corpus_avg, color='red', linestyle='--', 
                   label=f'Corpus Avg IDF: {corpus_avg:.2f}')
        plt.legend()
        
        plt.tight_layout()
        plot2_file = eval_dir / "router_idf_vs_weight.png"
        plt.savefig(plot2_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Weights by category
        plt.figure(figsize=(10, 6))
        categories = df['expected_type'].unique()
        sparse_weights = [df[df['expected_type'] == cat]['sparse_weight'].mean() 
                         for cat in categories]
        dense_weights = [df[df['expected_type'] == cat]['dense_weight'].mean() 
                        for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, sparse_weights, width, label='Sparse', 
                       color='blue', alpha=0.7, edgecolor='black')
        bars2 = plt.bar(x + width/2, dense_weights, width, label='Dense',
                       color='green', alpha=0.7, edgecolor='black')
        
        plt.title('Average Weights by Query Category', fontsize=14, fontweight='bold')
        plt.xlabel('Query Category', fontsize=12)
        plt.ylabel('Average Weight', fontsize=12)
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot3_file = eval_dir / "router_weights_by_category.png"
        plt.savefig(plot3_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nPLOTS GENERATED:")
        print(f"  {plot1_file}")
        print(f"  {plot2_file}")
        print(f"  {plot3_file}")
    
    def save_results(self, df: pd.DataFrame):
        """Save analysis results"""
        eval_dir = Path(__file__).parent
        results_file = eval_dir / "router_analysis.csv"
        
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")

def main():
    """Run router analysis"""
    analyzer = RouterAnalyzer()
    
    try:
        # Run analysis
        df = analyzer.run_analysis()
        
        # Generate summary
        analyzer.generate_summary(df)
        
        # Create plots
        analyzer.create_plots(df)
        
        # Save results
        analyzer.save_results(df)
        
        print("\n" + "=" * 80)
        print("ROUTER ANALYSIS COMPLETE!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
