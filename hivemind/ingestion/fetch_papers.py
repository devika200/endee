"""
Fetch arXiv papers using the arxiv Python library
Primary data source: arXiv public API (no download required)
"""

import sys
import arxiv
import time
import json
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pickle

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    ARXIV_CATEGORIES, PAPERS_PER_CATEGORY, YEAR_MIN, YEAR_MAX,
    ARXIV_DELAY_SECONDS, PAPERS_JSON
)

def fetch_category_papers(category: str, max_results: int) -> list:
    """Fetch papers from a specific arXiv category"""
    papers = []
    checkpoint_file = PAPERS_JSON.parent / f"checkpoint_{category}.pkl"
    
    # Check for existing checkpoint
    if checkpoint_file.exists():
        print(f"Resuming {category} from checkpoint...")
        with open(checkpoint_file, 'rb') as f:
            papers = pickle.load(f)
        print(f"Already have {len(papers)} papers")
    
    print(f"Fetching {category} papers...")
    
    # Create search query for this category
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Configure client with better rate limiting
    client = arxiv.Client(
        page_size=100,           # Smaller pages to avoid timeouts
        delay_seconds=ARXIV_DELAY_SECONDS,  # Use config delay (10 seconds)
        num_retries=3            # Retry on failures
    )
    
    print(f"Fetching papers with {ARXIV_DELAY_SECONDS}s delay between requests...")
    
    try:
        results = list(client.results(search))
    except Exception as e:
        print(f"Error fetching {category}: {e}")
        print(f"Retrying with shorter delay...")
        
        # Fallback: try with shorter delay and smaller batch
        client = arxiv.Client(
            page_size=50,
            delay_seconds=5,
            num_retries=2
        )
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=min(500, max_results),  # Smaller batch
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        results = list(client.results(search))
    
    print(f"Found {len(results)} total papers in {category}")
    
    # Process results with progress bar
    processed_count = 0
    for result in tqdm(results, desc=f"Processing {category}"):
        try:
            # Check if we already have this paper
            paper_id = result.entry_id.split('/')[-1]
            if any(p['id'] == paper_id for p in papers):
                continue
            
            # Filter by year
            if not (YEAR_MIN <= result.published.year <= YEAR_MAX):
                continue
            
            paper = {
                "id": paper_id,
                "title": result.title,
                "abstract": result.summary,
                "authors": [author.name for author in result.authors],
                "categories": result.categories,
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "doi": result.doi,
                "primary_category": result.primary_category
            }
            
            papers.append(paper)
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing paper {paper_id}: {e}")
            continue
        
        # Save checkpoint every 100 papers
        if len(papers) % 100 == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(papers, f)
            print(f"Checkpoint saved: {len(papers)} papers")
        
        # No additional sleep needed - arxiv.Client handles rate limiting
        # Only save checkpoint more frequently
    
    # Clean up checkpoint
    if checkpoint_file.exists():
        checkpoint_file.unlink()
    
    print(f"{category}: {processed_count} papers fetched and processed")
    return papers

def deduplicate_papers(all_papers: list) -> list:
    """Remove duplicate papers across categories"""
    seen_ids = set()
    deduplicated = []
    
    for paper in all_papers:
        if paper['id'] not in seen_ids:
            seen_ids.add(paper['id'])
            deduplicated.append(paper)
    
    print(f"Deduplicated: {len(all_papers)} → {len(deduplicated)} papers")
    return deduplicated

def fetch_all_papers(target_count=10000):
    """Fetch all papers in one efficient query and categorize locally"""
    papers = []
    category_counts = {cat: 0 for cat in ARXIV_CATEGORIES}
    
    print(f"Fetching {target_count} papers from all ML categories...")
    
    # Single query for all ML categories
    search = arxiv.Search(
        query="cat:(cs.LG OR cs.CL OR cs.CV OR cs.AI)",
        max_results=target_count,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    # Configure client for large batch
    client = arxiv.Client(
        page_size=1000,      # Large pages for efficiency
        delay_seconds=ARXIV_DELAY_SECONDS,
        num_retries=3
    )
    
    try:
        results = list(client.results(search))
        print(f"Found {len(results)} papers from arXiv")
    except Exception as e:
        print(f"Error with large batch: {e}")
        print(f"Trying smaller batch size...")
        
        # Fallback: smaller batch
        client = arxiv.Client(page_size=500, delay_seconds=10, num_retries=2)
        search = arxiv.Search(
            query="cat:(cs.LG OR cs.CL OR cs.CV OR cs.AI)",
            max_results=min(5000, target_count)
        )
        results = list(client.results(search))
        print(f"Fallback: Found {len(results)} papers")
    
    # Process and categorize papers
    processed_count = 0
    year_filtered = 0
    category_filtered = 0
    balance_filtered = 0
    
    # Debug: Check first few results
    print(f"Debug: Checking first 3 results...")
    for i, result in enumerate(results[:3]):
        print(f"Result {i+1}:")
        print(f"     Year: {result.published.year}")
        print(f"     Primary Category: {result.primary_category}")
        print(f"     Categories: {result.categories}")
    
    for result in tqdm(results, desc="Processing papers"):
        try:
            # Filter by year
            if not (YEAR_MIN <= result.published.year <= YEAR_MAX):
                year_filtered += 1
                continue
                
            # Get primary category
            primary_cat = result.primary_category
            if primary_cat not in ARXIV_CATEGORIES:
                category_filtered += 1
                continue
                
            # Balance categories (don't exceed ~2750 per category)
            max_per_cat = (target_count // len(ARXIV_CATEGORIES) + 750)
            if category_counts[primary_cat] >= max_per_cat:
                balance_filtered += 1
                continue
                
            paper = {
                "id": result.entry_id.split('/')[-1],
                "title": result.title,
                "abstract": result.summary,
                "authors": [author.name for author in result.authors],
                "categories": result.categories,
                "published": result.published.isoformat(),
                "updated": result.updated.isoformat(),
                "doi": result.doi,
                "primary_category": primary_cat
            }
            
            papers.append(paper)
            category_counts[primary_cat] += 1
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing paper: {e}")
            continue
    
    print(f"Category distribution:")
    for cat, count in category_counts.items():
        print(f"{cat}: {count} papers")
    
    # Debug: Show filtering statistics
    print(f"Debug Statistics:")
    print(f"Year filtered: {year_filtered}")
    print(f"Category filtered: {category_filtered}")
    print(f"   Balance filtered: {balance_filtered}")
    print(f"   Total processed: {processed_count}")
    print(f"   Total results: {len(results)}")
    
    print(f"Successfully processed {processed_count} papers")
    return papers

def main():
    """Main fetch process"""
    print("Starting arXiv paper fetch...")
    print(f"   Target: 10,000 papers from {', '.join(ARXIV_CATEGORIES)}")
    print(f"   Years: {YEAR_MIN}-{YEAR_MAX}")
    print(f"   Rate delay: {ARXIV_DELAY_SECONDS}s between requests")
    
    # Use optimized single query approach
    papers = fetch_all_papers(10000)
    
    # Save results
    PAPERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(PAPERS_JSON, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2)
    
    print(f"Saved {len(papers)} papers to {PAPERS_JSON}")
    
    # Show final statistics
    category_stats = {}
    for paper in papers:
        cat = paper['primary_category']
        category_stats[cat] = category_stats.get(cat, 0) + 1
    
    print(f"Final category distribution:")
    for cat in ARXIV_CATEGORIES:
        count = category_stats.get(cat, 0)
        print(f"   {cat}: {count} papers")
    
    print(f"Total papers: {len(papers)}")
    print(f"Average per category: {len(papers)//len(ARXIV_CATEGORIES)}")
    
    if len(papers) < 8000:
        print(f"Warning: Only got {len(papers)} papers, might want to run again")
    else:
        print(f"Success: Got sufficient papers for HiveMind!")
    
    print(f"\nFetch complete! {len(papers)} papers ready for processing.")

if __name__ == "__main__":
    main()
