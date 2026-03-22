"""
HiveMind Streamlit App: AI Research Paper Assistant
3 pages: Search, Index Dashboard, Evaluation Results
"""

import streamlit as st
import sys
import time
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import all components
from config import (
    ENDEE_HOST, validate_config,
    EVAL_RESULTS_FILE, MEMORY_EXPIRY_DAYS
)
from retrieval.search import EndeeSearcher
from retrieval.reranker import VoyageReranker
from retrieval.memory import SessionMemory
from generation.llm import GroqGenerator
from evaluation.run_eval import EvaluationRunner

# Page configuration
st.set_page_config(
    page_title="HiveMind - AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.result-card {
    background-color: white;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e1e5e9;
    margin: 0.5rem 0;
    color: #1a1a1a;
}
.result-card strong {
    color: #000000;
    font-weight: 600;
}
.memory-card {
    background-color: #f8f9fa;
    padding: 0.75rem;
    border-radius: 0.25rem;
    border-left: 3px solid #6c757d;
    margin: 0.25rem 0;
    font-size: 0.9em;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'searcher' not in st.session_state:
        try:
            st.session_state.searcher = EndeeSearcher()
        except Exception as e:
            st.error(f"Failed to initialize searcher: {e}")
            st.session_state.searcher = None
    
    if 'reranker' not in st.session_state:
        try:
            st.session_state.reranker = VoyageReranker()
        except Exception as e:
            st.warning(f"Reranker not available: {e}")
            st.session_state.reranker = None
    
    if 'memory' not in st.session_state:
        try:
            st.session_state.memory = SessionMemory()
        except Exception as e:
            st.warning(f"Memory not available: {e}")
            st.session_state.memory = None
    
    if 'generator' not in st.session_state:
        try:
            st.session_state.generator = GroqGenerator()
        except Exception as e:
            st.error(f"Failed to initialize generator: {e}")
            st.session_state.generator = None
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

def check_endee_connection():
    """Check if Endee is running"""
    try:
        import requests
        response = requests.get(f"{ENDEE_HOST}/api/v1/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def page_search():
    """Search page: main RAG interface"""
    st.header("🔍 Research Paper Search")
    
    # Check Endee connection
    if not check_endee_connection():
        st.error("❌ Cannot connect to Endee server. Make sure Endee is running on localhost:8080")
        st.info("To start Endee: `docker run -p 8080:8080 endeeio/endee-server:latest`")
        return
    
    # Session controls
    with st.sidebar:
        st.subheader("Session Controls")
        
        # Session ID display
        st.text(f"Session ID: {st.session_state.session_id}")
        
        # New session button
        if st.button("Start New Session"):
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.conversation_history = []
            st.rerun()
        
        # Conversation history
        st.subheader("Conversation History")
        for i, (q, a) in enumerate(st.session_state.conversation_history[-5:], 1):
            with st.expander(f"Q{i}: {q[:50]}..."):
                st.text_area(f"Q: {q}", height=50, disabled=True)
                st.text_area(f"A: {a[:200]}...", height=100, disabled=True)
        
        # Memory cleanup
        if st.session_state.memory and st.button("Clear Old Memories"):
            deleted = st.session_state.memory.cleanup_old_memories(days=MEMORY_EXPIRY_DAYS)
            st.success(f"Cleaned {deleted} old memories")
    
    # Main search interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Enter your research question:",
            placeholder="e.g., How do transformer models use attention mechanisms?",
            key="search_query"
        )
    
    with col2:
        st.write("")
        search_button = st.button("🔎 Search", type="primary", use_container_width=True)
    
    if search_button and query:
        if not st.session_state.searcher:
            st.error("Searcher not initialized")
            return
        
        if not st.session_state.generator:
            st.error("Generator not initialized")
            return
        
        # Show router decision
        with st.expander("🧭 Query Routing Analysis", expanded=True):
            routing_result = st.session_state.searcher.router.classify_query(query)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Query Type", routing_result.query_type)
            with col2:
                st.metric("Dense Weight", f"{routing_result.dense_weight:.1f}")
            with col3:
                st.metric("Sparse Weight", f"{routing_result.sparse_weight:.1f}")
            
            if routing_result.filters:
                st.write("Filters applied:")
                for filter_dict in routing_result.filters:
                    for field, condition in filter_dict.items():
                        st.code(f"{field}: {condition}")
        
        # Perform search
        with st.spinner("🔍 Searching knowledge base..."):
            search_response = st.session_state.searcher.search(query, k=20)
        
        # Show Endee results
        with st.expander("📚 Endee Search Results (Top 20)", expanded=False):
            st.write(f"Found {len(search_response.results)} results in {search_response.total_time_ms:.1f}ms")
            
            for i, result in enumerate(search_response.results[:10], 1):
                with st.container():
                    st.markdown(f"""
                    <div class="result-card">
                        <strong>{i}. {result.title}</strong><br>
                        <small>Score: {result.score:.3f} | Year: {result.year} | Category: {result.category}</small><br>
                        <small>Authors: {', '.join(result.authors[:3])}{' et al.' if len(result.authors) > 3 else ''}</small><br>
                        <small>📄 <a href="https://arxiv.org/abs/{result.arxiv_id}" target="_blank">{result.arxiv_id}</a></small><br>
                        <p>{result.abstract_snippet}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Rerank results
        reranked_results = []
        if st.session_state.reranker and search_response.results:
            with st.spinner("🔄 Reranking results..."):
                reranked = st.session_state.reranker.rerank(query, search_response.results, top_k=5)
                reranked_results = [r.result for r in reranked]
        
        # Show reranked results
        if reranked_results:
            with st.expander("⭐ Reranked Results (Top 5)", expanded=True):
                for i, result in enumerate(reranked_results, 1):
                    st.markdown(f"""
                    <div class="result-card">
                        <strong>{i}. {result.title}</strong><br>
                        <small>📄 <a href="https://arxiv.org/abs/{result.arxiv_id}" target="_blank">{result.arxiv_id}</a></small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Reranking not available")
        
        # Retrieve memory
        memory_results = []
        if st.session_state.memory:
            with st.spinner("🧠 Retrieving relevant memories..."):
                memory_results = st.session_state.memory.read_memory(
                    st.session_state.session_id, query, k=3
                )
        
        # Show memory used
        if memory_results:
            with st.expander("🧠 Relevant Conversation Memory", expanded=False):
                for memory in memory_results:
                    st.markdown(f"""
                    <div class="memory-card">
                        <strong>Q:</strong> {memory.query}<br>
                        <strong>A:</strong> {memory.answer[:150]}...<br>
                        <small>Score: {memory.score:.3f} | Time: {memory.timestamp.strftime('%Y-%m-%d %H:%M')}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Generate answer
        with st.spinner("🤖 Generating answer..."):
            results_to_use = reranked_results if reranked_results else search_response.results[:5]
            llm_response = st.session_state.generator.generate_answer(
                query, results_to_use, memory_results
            )
        
        # Show final answer
        st.markdown("---")
        st.subheader("🤖 Answer")
        st.markdown(f"""
        <div class="metric-card">
            {llm_response.answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Show sources
        if llm_response.sources:
            st.subheader("📚 Sources")
            for i, source in enumerate(llm_response.sources, 1):
                st.markdown(f"""
                **{i}.** [{source['title']}]({source['url']})  
                *{', '.join(source['authors'][:3])}{' et al.' if len(source['authors']) > 3 else ''}*  
                Year: {source['year']} | Score: {source['score']:.3f}
                """)
        
        # Store in memory
        if st.session_state.memory:
            st.session_state.memory.write_memory(
                st.session_state.session_id, query, llm_response.answer
            )
        
        # Add to conversation history
        st.session_state.conversation_history.append((query, llm_response.answer))
        
        # Show generation stats
        with st.expander("📊 Generation Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Context Used", llm_response.context_used)
            with col2:
                st.metric("Memory Used", llm_response.memory_used)
            with col3:
                st.metric("Tokens Used", llm_response.tokens_used)
            with col4:
                st.metric("Model", llm_response.model)

def page_dashboard():
    """Index dashboard page"""
    st.header("📊 Index Dashboard")
    
    if not check_endee_connection():
        st.error("❌ Cannot connect to Endee server")
        return
    
    # Get index statistics
    try:
        import requests
        headers = {}
        if st.session_state.searcher and st.session_state.searcher.auth_token:
            headers["Authorization"] = f"Bearer {st.session_state.searcher.auth_token}"
        
        # List indexes
        response = requests.get(f"{ENDEE_HOST}/api/v1/index/list", headers=headers)
        if response.status_code == 200:
            indexes = response.json().get('indexes', [])
            
            # Display index table
            if indexes:
                st.subheader("📈 Index Statistics")
                
                index_data = []
                for index in indexes:
                    index_data.append({
                        "Name": index.get("name", "N/A"),
                        "Elements": index.get("total_elements", 0),
                        "Dimension": index.get("dimension", "N/A"),
                        "Quantization": index.get("quant_level", "N/A"),
                        "Sparse Model": index.get("sparse_model", "N/A"),
                        "Created": datetime.fromtimestamp(index.get("created_at", 0)).strftime("%Y-%m-%d")
                    })
                
                df = pd.DataFrame(index_data)
                st.dataframe(df, use_container_width=True)
                
                # Index details
                st.subheader("🔍 Index Details")
                selected_index = st.selectbox("Select index for details:", [idx["name"] for idx in indexes])
                
                if selected_index:
                    # Get detailed info
                    info_response = requests.get(f"{ENDEE_HOST}/api/v1/index/{selected_index}/info", headers=headers)
                    if info_response.status_code == 200:
                        info = info_response.json()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Elements", info.get("total_elements", 0))
                            st.metric("Dimension", info.get("dimension", "N/A"))
                            st.metric("Space Type", info.get("space_type_str", "N/A"))
                        
                        with col2:
                            st.metric("Quantization", info.get("quant_level", "N/A"))
                            st.metric("M Parameter", info.get("M", "N/A"))
                            st.metric("EF Construction", info.get("ef_con", "N/A"))
            else:
                st.info("No indexes found. Run the ingestion pipeline first.")
        
        # Server stats
        st.subheader("🖥️ Server Statistics")
        stats_response = requests.get(f"{ENDEE_HOST}/api/v1/stats", headers=headers)
        if stats_response.status_code == 200:
            stats = stats_response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Version", stats.get("version", "N/A"))
            with col2:
                st.metric("Uptime", f"{stats.get('uptime', 0)}s")
        
        # Backup management
        st.subheader("💾 Backup Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Create Backup"):
                try:
                    backup_response = requests.post(
                        f"{ENDEE_HOST}/api/v1/index/knowledge_base/backup",
                        json={"name": f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"},
                        headers=headers
                    )
                    if backup_response.status_code == 200:
                        st.success("Backup created successfully")
                    else:
                        st.error("Failed to create backup")
                except Exception as e:
                    st.error(f"Backup failed: {e}")
        
        with col2:
            if st.button("List Backups"):
                try:
                    backup_response = requests.get(f"{ENDEE_HOST}/api/v1/backups", headers=headers)
                    if backup_response.status_code == 200:
                        backups = backup_response.json()
                        if backups:
                            st.write("Available backups:")
                            for backup in backups:
                                st.write(f"- {backup}")
                        else:
                            st.info("No backups found")
                except Exception as e:
                    st.error(f"Failed to list backups: {e}")
        
    except Exception as e:
        st.error(f"Failed to get index information: {e}")

def page_evaluation():
    """Evaluation results page"""
    st.header("📊 Evaluation Results")
    
    # Check if evaluation results exist
    if EVAL_RESULTS_FILE.exists():
        try:
            df = pd.read_csv(EVAL_RESULTS_FILE)
            
            # Results table
            st.subheader("📈 Performance Comparison")
            
            # Format the table for better display
            display_df = df.copy()
            display_df['Recall@5'] = display_df['recall@5'].apply(lambda x: f"{x:.3f}")
            display_df['Latency (ms)'] = display_df['p50_latency_ms'].apply(lambda x: f"{x:.1f}")
            
            # Highlight best configuration
            best_idx = df['recall@5'].idxmax()
            display_df.loc[best_idx, 'Recall@5'] = f"**{display_df.loc[best_idx, 'Recall@5']}** ⭐"
            
            st.dataframe(display_df[['config', 'description', 'Recall@5', 'Latency (ms)']], 
                        use_container_width=True)
            
            # Key findings
            st.subheader("🔍 Key Findings")
            
            # Find best configurations
            best_recall = df.loc[df['recall@5'].idxmax()]
            fastest = df.loc[df['p50_latency_ms'].idxmin()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🏆 Best Recall</h4>
                    <p><strong>{best_recall['description']}</strong></p>
                    <p>Recall@5: {best_recall['recall@5']:.3f}</p>
                    <p>Latency: {best_recall['p50_latency_ms']:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>⚡ Fastest</h4>
                    <p><strong>{fastest['description']}</strong></p>
                    <p>Recall@5: {fastest['recall@5']:.3f}</p>
                    <p>Latency: {fastest['p50_latency_ms']:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Hybrid vs Dense comparison
            hybrid_row = df[df['config'] == 'hybrid_fp16']
            dense_fp16_row = df[df['config'] == 'dense_fp16_only']
            
            if not hybrid_row.empty and not dense_fp16_row.empty:
                hybrid_recall = hybrid_row['recall@5'].iloc[0]
                dense_recall = dense_fp16['recall@5'].iloc[0]
                improvement = ((hybrid_recall - dense_recall) / dense_recall) * 100
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎯 Hybrid Search Impact</h4>
                    <p>Hybrid search improves recall@5 by <strong>{improvement:.1f}%</strong> over dense-only FP16</p>
                    <p>Hybrid latency: {hybrid_row['p50_latency_ms'].iloc[0]:.1f}ms</p>
                    <p>Dense-only latency: {dense_fp16_row['p50_latency_ms'].iloc[0]:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Load plots if they exist
            plots_dir = EVAL_RESULTS_FILE.parent
            recall_plot = plots_dir / "recall_comparison.png"
            pareto_plot = plots_dir / "recall_latency_pareto.png"
            
            if recall_plot.exists():
                st.subheader("📊 Recall@5 Comparison")
                st.image(str(recall_plot), use_column_width=True)
            
            if pareto_plot.exists():
                st.subheader("📈 Recall vs Latency Pareto Frontier")
                st.image(str(pareto_plot), use_column_width=True)
            
            # Run evaluation button
            if st.button("🚀 Run New Evaluation"):
                with st.spinner("Running comprehensive evaluation..."):
                    try:
                        runner = EvaluationRunner()
                        results = runner.run_evaluation()
                        if results:
                            df = runner.save_results(results)
                            runner.create_plots(df)
                            runner.print_summary(df)
                            st.success("Evaluation completed! Refresh to see results.")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")
        
        except Exception as e:
            st.error(f"Failed to load evaluation results: {e}")
    else:
        st.info("No evaluation results found. Run the evaluation first:")
        st.code("python evaluation/run_eval.py")
        
        # Quick run button
        if st.button("🚀 Run Evaluation Now"):
            with st.spinner("Running comprehensive evaluation..."):
                try:
                    runner = EvaluationRunner()
                    results = runner.run_evaluation()
                    if results:
                        df = runner.save_results(results)
                        runner.create_plots(df)
                        runner.print_summary(df)
                        st.success("Evaluation completed! Refresh to see results.")
                        st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

def main():
    """Main app entry point"""
    # Initialize session state
    init_session_state()
    
    # Sidebar navigation
    st.sidebar.title("🧠 HiveMind")
    st.sidebar.markdown("AI Research Paper Assistant")
    
    page = st.sidebar.radio("Navigate", ["🔍 Search", "📊 Dashboard", "📈 Evaluation"])
    
    # Connection status
    if check_endee_connection():
        st.sidebar.success("✅ Endee Connected")
    else:
        st.sidebar.error("❌ Endee Not Connected")
        st.sidebar.info("Run: `docker run -p 8080:8080 endeeio/endee-server:latest`")
    
    # Page routing
    if page == "🔍 Search":
        page_search()
    elif page == "📊 Dashboard":
        page_dashboard()
    elif page == "📈 Evaluation":
        page_evaluation()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("Built with ❤️ using Endee Vector Database")

if __name__ == "__main__":
    main()
