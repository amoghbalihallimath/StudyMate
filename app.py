import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from groq import Groq
from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import urllib.parse

# Load environment variables
load_dotenv()

# Configure Groq
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'current_document' not in st.session_state:
    st.session_state.current_document = None
if 'document_texts' not in st.session_state:
    st.session_state.document_texts = {}
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}
if 'quiz_submitted' not in st.session_state:
    st.session_state.quiz_submitted = False
if 'answer_mode' not in st.session_state:
    st.session_state.answer_mode = 'PDF'

# Initialize models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks of specified size."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # +1 for the space
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for a list of text chunks."""
    return embedding_model.encode(texts, convert_to_numpy=True)

def create_faiss_index(embeddings: np.ndarray):
    """Create a FAISS index from embeddings."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

def search_similar_texts(query: str, index, texts: List[str], k: int = 3) -> List[str]:
    """Search for similar texts using FAISS index."""
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_embedding.astype('float32'), k)
    return [texts[i] for i in I[0] if i < len(texts)]

def web_search(query: str, num_results: int = 3) -> str:
    """Perform a simple web search and return results."""
    try:
        # Simple DuckDuckGo search (no API key required)
        search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            results = []
            
            # Extract search results
            for result in soup.find_all('a', class_='result__a')[:num_results]:
                title = result.get_text(strip=True)
                if title:
                    results.append(title)
            
            return "\n".join(results) if results else "No search results found."
        else:
            return "Unable to perform web search at the moment."
    except Exception as e:
        return f"Web search error: {str(e)}"

def generate_response(prompt: str, context: str = "", use_web: bool = False) -> str:
    """Generate a response using Groq API."""
    try:
        if use_web:
            # Get web search results
            web_results = web_search(prompt)
            full_prompt = f"Based on the following web search results, please answer the question:\n\nWeb Results: {web_results}\n\nQuestion: {prompt}\n\nAnswer:"
        else:
            full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:" if context else prompt
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def generate_quiz(text: str, num_questions: int = 5) -> List[Dict[str, Any]]:
    """Generate quiz questions from the text."""
    prompt = f"""Generate {num_questions} multiple-choice questions based on the following text.
    For each question, provide 4 options (a, b, c, d) and indicate the correct answer.
    Return the questions in JSON format with the following structure:
    [
        {{
            "question": "...",
            "options": ["...", "...", "...", "..."],
            "correct_answer": "..."
        }}
    ]
    
    Text: {text}"""
    
    try:
        response = generate_response(prompt)
        # Extract JSON from the response
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        json_str = response[start_idx:end_idx]
        return json.loads(json_str)
    except Exception as e:
        st.error(f"Error generating quiz: {str(e)}")
        return []

def main():
    st.set_page_config(
        page_title="PDF Q&A with Groq", 
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling with dark mode support
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #4a90e2 0%, #357abd 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #4a90e2;
        background-color: #bbdefb !important;
        color: #1565c0 !important;
        border: 1px solid #4a90e2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .quiz-container {
        background-color: #e1f5fe !important;
        color: #0277bd !important;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 2px solid #29b6f6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .quiz-container h4 {
        color: #01579b !important;
        font-weight: bold;
    }
    .quiz-container p {
        color: #0277bd !important;
    }
    .correct-answer {
        background-color: #a5d6a7 !important;
        border: 3px solid #4caf50;
        color: #1b5e20 !important;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }
    .wrong-answer {
        background-color: #ef9a9a !important;
        border: 3px solid #f44336;
        color: #b71c1c !important;
        box-shadow: 0 2px 8px rgba(244, 67, 54, 0.3);
    }
    .stButton > button {
        background: linear-gradient(90deg, #4a90e2 0%, #357abd 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    /* Ensure text visibility in all containers */
    .stMarkdown, .stText {
        color: inherit;
    }
    /* Force text color in custom containers */
    .chat-message *, .quiz-container *, .correct-answer *, .wrong-answer * {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<div class="main-header"><h1>📚 PDF Q&A Assistant with Groq</h1><p>Upload PDFs, ask questions, generate summaries, and take quizzes!</p></div>', unsafe_allow_html=True)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("🔧 Settings")
        
        # Answer mode selection
        st.subheader("Answer Mode")
        answer_mode = st.radio(
            "Choose how to answer questions:",
            ["📄 PDF Only", "🌐 Web Search"],
            index=0 if st.session_state.answer_mode == 'PDF' else 1
        )
        st.session_state.answer_mode = 'PDF' if answer_mode == "📄 PDF Only" else 'Web'
        
        if st.session_state.answer_mode == 'Web':
            st.info("💡 In Web mode, answers will be generated using web search results instead of PDF content.")
        
        st.divider()
        
        st.header("📁 Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file is not None:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner("Processing PDF..."):
                    # Extract text
                    text = extract_text_from_pdf(uploaded_file)
                    chunks = get_text_chunks(text)
                    
                    # Get embeddings
                    embeddings = get_embeddings(chunks)
                    
                    # Create FAISS index
                    index = create_faiss_index(embeddings)
                    
                    # Store in session state
                    st.session_state.document_texts[uploaded_file.name] = {
                        'chunks': chunks,
                        'embeddings': embeddings,
                        'index': index
                    }
                    st.session_state.uploaded_files.append(uploaded_file.name)
                    st.session_state.current_document = uploaded_file.name
                    st.success("PDF processed successfully!")
        
        # Show uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📚 Your Documents")
            for i, doc in enumerate(st.session_state.uploaded_files):
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(f"📄 {doc}", key=f"doc_{i}"):
                        st.session_state.current_document = doc
                with col2:
                    if st.button("🗑️", key=f"del_{i}", help="Delete document"):
                        if doc in st.session_state.document_texts:
                            del st.session_state.document_texts[doc]
                        st.session_state.uploaded_files.remove(doc)
                        if st.session_state.current_document == doc:
                            st.session_state.current_document = None
                        st.experimental_rerun()
    
    # Main content area
    if st.session_state.current_document:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(f"📖 Current Document: {st.session_state.current_document}")
        with col2:
            st.info(f"Mode: {st.session_state.answer_mode}")
        
        # Tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📝 Summarize", "❓ Quiz"])
        
        with tab1:  # Chat tab
            st.markdown("### 💬 Ask questions about your document")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                st.session_state.conversation = []
                st.experimental_rerun()
            
            # Display chat messages
            for message in st.session_state.conversation:
                with st.chat_message(message["role"]):
                    st.markdown(f'<div class="chat-message">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Chat input
            if prompt := st.chat_input("Ask a question..."):
                # Add user message to chat
                st.session_state.conversation.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(f'<div class="chat-message">{prompt}</div>', unsafe_allow_html=True)
                
                # Generate response based on mode
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        if st.session_state.answer_mode == 'PDF':
                            # Get relevant context from PDF
                            doc_data = st.session_state.document_texts[st.session_state.current_document]
                            similar_chunks = search_similar_texts(
                                prompt, 
                                doc_data['index'], 
                                doc_data['chunks']
                            )
                            context = "\n\n".join(similar_chunks)
                            response = generate_response(prompt, context, use_web=False)
                        else:
                            # Use web search
                            response = generate_response(prompt, use_web=True)
                        
                        st.markdown(f'<div class="chat-message">{response}</div>', unsafe_allow_html=True)
                        st.session_state.conversation.append({"role": "assistant", "content": response})
        
        with tab2:  # Summarize tab
            st.markdown("### 📝 Document Summary")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                summary_type = st.selectbox(
                    "Summary Type:",
                    ["Brief Summary", "Detailed Summary", "Key Points", "Executive Summary"]
                )
            with col2:
                summary_length = st.slider("Summary Length (words)", 50, 500, 200)
            
            if st.button("📝 Generate Summary", key="summary_btn"):
                doc_data = st.session_state.document_texts[st.session_state.current_document]
                full_text = " ".join(doc_data['chunks'][:10])  # Use first 10 chunks
                
                prompt = f"Please provide a {summary_type.lower()} of approximately {summary_length} words for the following text: {full_text[:4000]}"
                
                with st.spinner("Generating summary..."):
                    summary = generate_response(prompt, use_web=False)
                    st.markdown(f'<div class="chat-message"><h4>{summary_type}</h4>{summary}</div>', unsafe_allow_html=True)
        
        with tab3:  # Quiz tab
            st.markdown("### ❓ Interactive Quiz")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                num_questions = st.slider("Number of questions", 1, 10, 5)
            with col2:
                difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
            with col3:
                quiz_type = st.selectbox("Quiz Type", ["Multiple Choice", "True/False", "Mixed"])
            
            if st.button("🎯 Generate New Quiz", key="quiz_btn"):
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                
                doc_data = st.session_state.document_texts[st.session_state.current_document]
                full_text = " ".join(doc_data['chunks'][:8])  # Use more chunks for better quiz
                
                with st.spinner("Generating quiz questions..."):
                    quiz = generate_quiz(full_text, num_questions)
                    st.session_state.current_quiz = quiz
            
            # Display quiz if available
            if 'current_quiz' in st.session_state and st.session_state.current_quiz:
                quiz = st.session_state.current_quiz
                
                with st.form("quiz_form"):
                    for i, q in enumerate(quiz, 1):
                        st.markdown(f'<div class="quiz-container"><h4>Question {i}</h4><p>{q["question"]}</p></div>', unsafe_allow_html=True)
                        
                        # Create radio buttons for options
                        options = [f"{chr(ord('A') + j)}. {option}" for j, option in enumerate(q['options'])]
                        selected = st.radio(
                            f"Select your answer for Question {i}:",
                            options,
                            key=f"q_{i}",
                            index=None
                        )
                        
                        if selected:
                            st.session_state.quiz_answers[i] = selected[0]  # Store just the letter (A, B, C, D)
                    
                    submitted = st.form_submit_button("📊 Submit Quiz")
                    
                    if submitted:
                        st.session_state.quiz_submitted = True
                        
                        # Calculate score
                        correct_answers = 0
                        total_questions = len(quiz)
                        
                        st.markdown("### 🎯 Quiz Results")
                        
                        for i, q in enumerate(quiz, 1):
                            user_answer = st.session_state.quiz_answers.get(i, "Not answered")
                            correct_answer = q['correct_answer'].upper()
                            
                            if user_answer == correct_answer:
                                correct_answers += 1
                                st.markdown(f'<div class="quiz-container correct-answer"><strong>Question {i}: ✅ Correct!</strong><br>Your answer: {user_answer}<br>Correct answer: {correct_answer}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="quiz-container wrong-answer"><strong>Question {i}: ❌ Incorrect</strong><br>Your answer: {user_answer}<br>Correct answer: {correct_answer}</div>', unsafe_allow_html=True)
                        
                        # Show final score
                        score_percentage = (correct_answers / total_questions) * 100
                        
                        if score_percentage >= 80:
                            emoji = "🏆"
                            message = "Excellent work!"
                        elif score_percentage >= 60:
                            emoji = "👍"
                            message = "Good job!"
                        else:
                            emoji = "📚"
                            message = "Keep studying!"
                        
                        st.markdown(f'<div class="quiz-container" style="text-align: center; font-size: 1.2em;"><strong>{emoji} Final Score: {correct_answers}/{total_questions} ({score_percentage:.1f}%)</strong><br>{message}</div>', unsafe_allow_html=True)
    else:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h2>🚀 Welcome to PDF Q&A Assistant!</h2>
            <p style="font-size: 1.2em; color: #666;">Upload a PDF document to get started with:</p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin: 2rem 0;">
                <div style="text-align: center;">
                    <div style="font-size: 2em;">💬</div>
                    <strong>Interactive Chat</strong><br>
                    Ask questions about your document
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2em;">📝</div>
                    <strong>Smart Summaries</strong><br>
                    Generate concise summaries
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 2em;">❓</div>
                    <strong>Interactive Quizzes</strong><br>
                    Test your knowledge
                </div>
            </div>
            <p style="color: #888;">👈 Use the sidebar to upload your first PDF document</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
