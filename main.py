import streamlit as st
import pandas as pd
import json
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from PIL import Image
import matplotlib.pyplot as plt
import time

# Set page configuration
st.set_page_config(
    page_title="Interview Prep Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - Improved
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 0;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
        margin-top: 0;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 0.5rem 2rem 0 rgba(58, 59, 69, 0.2);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #424242;
        font-weight: 500;
    }
    .difficulty-easy {
        color: #4CAF50;
        font-weight: 600;
    }
    .difficulty-medium {
        color: #FF9800;
        font-weight: 600;
    }
    .difficulty-hard {
        color: #F44336;
        font-weight: 600;
    }
    .tips-container {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .gemini-response {
        background-color: #f1f8e9;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #8BC34A;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        color: #333;
    }
    .search-result {
        border-left: 4px solid #1E88E5;
        padding-left: 10px;
        margin-bottom: 8px;
    }
    .search-container {
        background-color: #f5f7fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .section-header {
        background-color: #1E88E5;
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .analysis-btn {
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        transition: background-color 0.3s;
    }
    .analysis-btn:hover {
        background-color: #388E3C;
    }
    .history-table {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
    div.block-container {
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/streamlit/streamlit/master/lib/streamlit/static/favicon.png", width=100)
    st.title("Interview Prep Hub")
    st.markdown("---")
    st.write("Prepare for your tech interviews with data-driven insights and AI assistance.")

    # Database file selection
    st.subheader("Database Configuration")
    db_path = st.text_input("Database Path", "C:/Users/jhaad/Downloads/rag_application/database/database (1).json")

    # API key input
    st.subheader("API Configuration")
    api_key = st.text_input("Google Gemini API Key", "AIzaSyDRWSutSiXKGyjX40lvBuIlnNANEgcsCDY", type="password")

    # Model selection
    st.subheader("Model Settings")
    embedding_model = st.selectbox(
        "Embedding Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L6-v2"]
    )

    gemini_model = st.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-flash"]
    )

    st.markdown("---")
    st.info("Developed by Interview Prep Team")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'bm25_matrix' not in st.session_state:
    st.session_state.bm25_matrix = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'index' not in st.session_state:
    st.session_state.index = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'gemini' not in st.session_state:
    st.session_state.gemini = None
if 'company_analysis' not in st.session_state:
    st.session_state.company_analysis = {}
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_search' not in st.session_state:
    st.session_state.current_search = None
if 'current_results' not in st.session_state:
    st.session_state.current_results = None

# Main header
st.markdown("<h1 class='main-header'>Interview Preparation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Your AI-Powered Guide to Technical Interviews</p>", unsafe_allow_html=True)

# Load data function
@st.cache_data
def load_data(db_path):
    try:
        with open(db_path, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['Title'] = df['Title'].astype(str)
        df['Topics'] = df['Topics'].astype(str)
        df['SourceFolder'] = df['SourceFolder'].astype(str)
        df['search_text'] = df['SourceFolder'] + " " + df['Title'] + " " + df['Topics']
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

# Initialize models
def initialize_models(df, embedding_model_name, api_key):
    with st.spinner("Initializing search models..."):
        # TF-IDF Vectorizer for BM25
        vectorizer = TfidfVectorizer()
        bm25_matrix = vectorizer.fit_transform(df['search_text'])

        # Sentence transformer for embeddings
        model = SentenceTransformer(embedding_model_name)
        embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=False)
        
        # Build FAISS index
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel(gemini_model)
        
        return vectorizer, bm25_matrix, model, embeddings, index, gemini

# Hybrid search function
def hybrid_search(query, top_k=10):
    if not query:
        return None

    # BM25 part
    tfidf_q = st.session_state.vectorizer.transform([query])
    bm25_scores = cosine_similarity(tfidf_q, st.session_state.bm25_matrix).flatten()

    # Embedding similarity part
    embed_q = st.session_state.model.encode([query])
    _, embed_indices = st.session_state.index.search(np.array(embed_q), top_k)
    embed_indices = embed_indices.flatten()

    # Combine and sort results by average score
    scores = [(i, bm25_scores[i]) for i in embed_indices]
    scores.sort(key=lambda x: x[1], reverse=True)

    return st.session_state.df.iloc[[i[0] for i in scores[:top_k]]]

# Ask Gemini function
def ask_gemini(company, results_df):
    if results_df is None or results_df.empty:
        return "No data available for analysis."

    context = "\n".join(
        [f"Title: {row['Title']}, Topics: {row['Topics']}, Difficulty: {row['Difficulty']}"
         for _, row in results_df.iterrows()]
    )

    prompt = f"""
You are an expert technical interview coach.
I am preparing for interviews at {company}.
Based on the following data of interview questions from {company} or similar companies, provide an analysis that includes:

1. Key topics to focus on, ranked by importance
2. Breakdown of difficulty levels (percentage of Easy/Medium/Hard questions)
3. Top 5 most important question titles with brief explanations
4. 3 personalized preparation tips for this company's interviews
5. Recommended study plan (1-week timeline)

Data:
{context}

Format your response with clear headings for each section.
"""
    try:
        response = st.session_state.gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error from Gemini API: {e}")
        return "Unable to generate analysis at this time. Please try again later."

# Create company-specific visualizations
def create_company_visualizations(results_df, company_name):
    if results_df is None or results_df.empty:
        return None, None
        
    # Difficulty distribution for the company
    difficulty_counts = results_df['Difficulty'].value_counts().reset_index()
    difficulty_counts.columns = ['Difficulty', 'Count']
    
    color_map = {'EASY': '#4CAF50', 'MEDIUM': '#FF9800', 'HARD': '#F44336'}
    colors = [color_map.get(d, '#9E9E9E') for d in difficulty_counts['Difficulty']]
    
    fig1 = px.pie(
        difficulty_counts, 
        values='Count', 
        names='Difficulty',
        title=f'{company_name}: Questions by Difficulty',
        color_discrete_sequence=colors
    )
    fig1.update_layout(height=350)
    
    # Topic distribution for the company
    all_topics = []
    for topics_str in results_df['Topics']:
        topics_list = topics_str.split(', ')
        all_topics.extend(topics_list)

    topic_counts = pd.Series(all_topics).value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    topic_counts = topic_counts.head(10)  # Top 10 topics
    
    fig2 = px.bar(
        topic_counts,
        x='Topic',
        y='Count',
        title=f'{company_name}: Most Common Interview Topics',
        color='Count',
        color_continuous_scale='Viridis'
    )
    fig2.update_layout(height=350, xaxis_tickangle=-45)
    
    return fig1, fig2

# Load data if path is provided
try:
    df = load_data(db_path)
    if df is not None and st.session_state.df is None:
        st.session_state.df = df
        
        # Initialize models
        st.session_state.vectorizer, st.session_state.bm25_matrix, st.session_state.model, \
        st.session_state.embeddings, st.session_state.index, st.session_state.gemini = initialize_models(
            df, embedding_model, api_key
        )
        
        st.success("Database loaded successfully!")
    elif df is not None:
        st.session_state.df = df
        
except Exception as e:
    st.error(f"Error loading database: {e}")

# Create dashboard components if data is loaded
if st.session_state.df is not None:
    # Main search interface
    st.markdown("<div class='section-header'>🔍 Interview Question Search</div>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<div class='search-container'>", unsafe_allow_html=True)
        # Search box
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input("Search by company name, topic, or question title", key="search_query")
        with search_col2:
            search_button = st.button("Search", type="primary")
        
        top_k = st.slider("Number of results", min_value=5, max_value=50, value=10)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Execute search
    if search_button and search_query:
        st.session_state.current_search = search_query
        with st.spinner("Searching..."):
            results = hybrid_search(search_query, top_k=top_k)
            st.session_state.current_results = results
            
            if results is not None and not results.empty:
                st.success(f"Found {len(results)} relevant questions for '{search_query}'")
                
                # Add to search history if it's a company name (likely an entity)
                if len(search_query.split()) <= 2:  # Simple heuristic for company names
                    if search_query not in [item['query'] for item in st.session_state.search_history]:
                        st.session_state.search_history.append({
                            'query': search_query,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'results_count': len(results)
                        })
                
                # Company-specific visualizations
                st.markdown("<div class='section-header'>📊 Company Analysis</div>", unsafe_allow_html=True)
                fig1, fig2 = create_company_visualizations(results, search_query)
                
                if fig1 and fig2:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Display search results
                st.markdown("<div class='section-header'>📝 Search Results</div>", unsafe_allow_html=True)
                for i, (_, row) in enumerate(results.iterrows()):
                    with st.expander(f"{i+1}. {row['Title']} ({row['SourceFolder']})"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**Topics:** {row['Topics']}")
                        
                        with col2:
                            difficulty_class = "difficulty-easy" if row['Difficulty'] == "EASY" else \
                                              "difficulty-medium" if row['Difficulty'] == "MEDIUM" else \
                                              "difficulty-hard"
                            st.markdown(f"**Difficulty:** <span class='{difficulty_class}'>{row['Difficulty']}</span>", unsafe_allow_html=True)
                        
                        with col3:
                            st.write(f"**Frequency:** {row['Frequency']}")
                
                # Generate AI Analysis button with improved styling
                st.markdown("<div style='text-align: center; margin: 20px 0;'>", unsafe_allow_html=True)
                if st.button("✨ Generate AI Interview Insights", key="ai_analysis", help="Get AI-powered analysis of these interview questions"):
                    with st.spinner("Analyzing interview patterns and generating insights..."):
                        analysis = ask_gemini(search_query, results)
                        st.session_state.company_analysis[search_query] = analysis
                        st.markdown(f"<div class='gemini-response'>{analysis}</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("No results found. Try a different search term.")
    
    # Show AI analysis if available
    if st.session_state.current_search in st.session_state.company_analysis:
        st.markdown("<div class='section-header'>🧠 AI Interview Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='gemini-response'>{st.session_state.company_analysis[st.session_state.current_search]}</div>", unsafe_allow_html=True)
    
    # Recent Searches
    if st.session_state.search_history:
        st.markdown("<div class='section-header'>📜 Recent Searches</div>", unsafe_allow_html=True)
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True, 
                   column_config={
                       "query": "Company/Topic",
                       "timestamp": "Search Time",
                       "results_count": "Results Found"
                   })
    
    # Study Tips Section
    st.markdown("<div class='section-header'>💡 Interview Preparation Tips</div>", unsafe_allow_html=True)
    
    with st.expander("View General Interview Tips", expanded=False):
        st.markdown("""
        <div class='tips-container'>
            <h4>🔍 Effective Interview Preparation</h4>
            <ul>
                <li><strong>Understand the concepts:</strong> Don't just memorize solutions; understand the underlying principles.</li>
                <li><strong>Practice regularly:</strong> Set a schedule for daily practice to build muscle memory.</li>
                <li><strong>Mock interviews:</strong> Practice with a friend or use platforms like Pramp or interviewing.io.</li>
                <li><strong>Review your code:</strong> After solving a problem, review your solution for optimization.</li>
                <li><strong>Learn from mistakes:</strong> Keep track of problems you struggled with and revisit them.</li>
            </ul>
            
            <h4>🧠 Problem-Solving Approaches</h4>
            <ul>
                <li><strong>Break it down:</strong> Divide complex problems into smaller, manageable parts.</li>
                <li><strong>Think aloud:</strong> Practice explaining your thought process while solving problems.</li>
                <li><strong>Consider edge cases:</strong> Always think about potential edge cases in your solutions.</li>
                <li><strong>Start with brute force:</strong> Begin with a simple solution, then optimize.</li>
                <li><strong>Time management:</strong> Learn to recognize when to move on from a challenging problem.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>Interview Prep Dashboard | Powered by Streamlit and Google Gemini</p>
    </div>
    """, 
    unsafe_allow_html=True
)