import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from groq import Groq
from database import Database
import os
from datetime import datetime, timedelta
import requests
import PyPDF2
from docx import Document
import pandas as pd
import time
from chromadb_operations import ChromaOperations
import uuid
from image_operations import ImageOperations
from llama32_vision import VisionOperations
import re
import urllib.parse
from typing import Tuple
from urllib.parse import urlparse, parse_qs
import sqlite3
from groq import Groq
import base64
from PIL import Image
import cv2
from pathlib import Path

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize database
db = Database()

# Initialize ChromaDB operations
chroma_ops = ChromaOperations(host="127.0.0.1", port=8000)

def init_session_state():
    """Initialize session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'page' not in st.session_state:
        st.session_state.page = 'login'
    if 'api_choice' not in st.session_state:
        st.session_state.api_choice = 'OpenAI'
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    if 'ollama_models' not in st.session_state:
        st.session_state.ollama_models = []
    if 'selected_ollama_model' not in st.session_state:
        st.session_state.selected_ollama_model = None
    if 'groq_models' not in st.session_state:
        st.session_state.groq_models = []
    if 'selected_groq_model' not in st.session_state:
        st.session_state.selected_groq_model = None
    if 'current_file_content' not in st.session_state:
        st.session_state.current_file_content = None
    if 'chat_session_id' not in st.session_state:
        st.session_state.chat_session_id = None
    if 'session_files' not in st.session_state:
        st.session_state.session_files = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = '0'
    if 'image_mode' not in st.session_state:
        st.session_state.image_mode = False
    if 'vision_mode' not in st.session_state:
        st.session_state.vision_mode = None        
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = None
    if 'vision_processor' not in st.session_state:
        st.session_state.vision_processor = None
    if 'session_images' not in st.session_state:
        st.session_state.session_images = []
    if 'session_media' not in st.session_state:
        st.session_state.session_media = []
    if 'image_uploader_key' not in st.session_state:
        st.session_state.image_uploader_key = str(time.time())
    if 'chat_mode' not in st.session_state:
        st.session_state.chat_mode = 'regular'
    if 'vision_uploader_key' not in st.session_state:
        st.session_state.vision_uploader_key = 0
    if 'vision_processor' not in st.session_state:
        st.session_state.vision_processor = None

    
def get_groq_models():
    """Get list of available Groq models"""
    # Groq's available models (as of 2024)
    return [
        "llama-3.3-70b-versatile",
        "llama-3.2-90b-vision-preview",
        "mixtral-8x7b-32768"
    ]

def process_documents_for_chromadb(content, filename):
    """Process document content for ChromaDB storage with improved chunking"""
    # Split content into smaller chunks
    MAX_CHUNK_SIZE = 1024
    words = content.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > MAX_CHUNK_SIZE:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Remove empty chunks and strip whitespace
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    # Prepare metadata for each chunk
    metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
    
    # Generate unique IDs for each chunk
    ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
    
    return chunks, metadatas, ids

def handle_image_upload(uploaded_image, user_id, session_id):
    """Handle image file upload with duplicate check"""
    try:
        if not st.session_state.image_processor:
            st.session_state.image_processor = ImageOperations()
        
        # Check if image already exists in current session
        existing_images = db.get_session_images(session_id)
        if existing_images:
            for _, filename, _, _, _ in existing_images:
                if filename == uploaded_image.name:
                    st.info(f"Image '{uploaded_image.name}' already exists in this session.")
                    return True
        
        # Save image file if it's new
        file_path, file_type = st.session_state.image_processor.save_image(
            uploaded_image, 
            session_id,
            user_id
        )
        
        if not file_path or not file_type:
            st.error("Failed to save image file")
            return False
        
        # Save to database
        image_id = db.save_uploaded_image(
            user_id,
            session_id,
            uploaded_image.name,
            file_path,
            file_type
        )
        
        if image_id:
            st.success(f"Successfully uploaded image: {uploaded_image.name}")
            return True
        return False
        
    except Exception as e:
        st.error(f"Error uploading image: {str(e)}")
        return False

def process_image_query(image_path, query):
    """Process image query using Qwen2-VL model"""
    try:
        if not st.session_state.image_processor:
            st.session_state.image_processor = ImageOperations()
            
        response = st.session_state.image_processor.process_image(image_path, query)
        return response
        
    except Exception as e:
        st.error(f"Error processing image query: {str(e)}")
        return None



def process_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def process_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error processing DOCX: {str(e)}")
        return None

def process_csv(file):
    """Process CSV file and return content as string"""
    try:
        df = pd.read_csv(file)
        return df.to_string()
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

def process_txt(file):
    """Process text file"""
    try:
        text = file.getvalue().decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error processing TXT: {str(e)}")
        return None

def save_uploaded_file(uploaded_file, user_id, session_id):
    """Save uploaded file and return file path"""
    try:
        file_type = uploaded_file.type.split('/')[-1]
        file_path = os.path.join('uploads', f"{session_id}_{uploaded_file.name}")
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        # Save file information to database
        file_id = db.save_uploaded_file(user_id, session_id, uploaded_file.name, file_path, file_type)
        return file_path, file_id
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None, None

def get_ollama_models():
    """Get list of available OLLAMA models"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json()
            return [model['name'] for model in models['models']]
        return []
    except Exception as e:
        st.error(f"Failed to connect to OLLAMA server: {str(e)}")
        return []




def login_page():
    """Display login page with fixed form buttons"""
    st.title("Multi-Modal RAG Chat Application Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        auth_type = st.radio("Login Method", ["Local", "Domain (AD)"])
        
        # Login Form
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submitted = st.form_submit_button("Login")
        
        # Handle login submission
        if login_submitted:
            if not username or not password:
                st.error("Please enter both username and password")
            else:
                user_id, message = db.verify_user(
                    username, 
                    password,
                    'ad' if auth_type == "Domain (AD)" else 'local'
                )
                
                if user_id:
                    st.success("Login successful!")
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    load_conversation_history()
                    st.rerun()
                else:
                    st.error(message)
                    if "verify" in message.lower():
                        # Show verification help outside the form
                        st.info("""
                        If you haven't received the verification email:
                        1. Check your spam folder
                        2. Use the resend verification option below
                        """)
                        # Resend verification form
                        with st.form("resend_verification_form"):
                            st.write("Resend Verification Email")
                            resend_submitted = st.form_submit_button("Resend")
                            if resend_submitted:
                                with st.spinner("Sending verification email..."):
                                    if db.resend_verification_email(username):
                                        st.success("Verification email sent! Please check your inbox.")
                                    else:
                                        st.error("Failed to send verification email. Please try again later.")
    
    with tab2:
        if auth_type == "Domain (AD)":
            st.warning("Registration not available for domain accounts")
        else:
            # Registration Form
            with st.form("register_form", clear_on_submit=True):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                email = st.text_input("Email")
                register_submitted = st.form_submit_button("Register")
            
            # Handle registration submission
            if register_submitted:
                if not all([new_username, new_password, confirm_password, email]):
                    st.error("Please fill in all fields")
                elif new_password != confirm_password:
                    st.error("Passwords do not match")
                elif not validate_email(email):
                    st.error("Please enter a valid email address")
                else:
                    with st.spinner("Sending Verification Email..."):
                        user_id, message = db.create_user(new_username, new_password, email)
                        if user_id:
                            st.success(message)
                            st.info("""
                            ✉️ Please check your email for the verification link.
                            
                            Note: 
                            - The verification link will expire in 24 hours
                            - Check your spam folder if you don't see the email
                            - You cannot log in until your email is verified
                            """)
                        else:
                            st.error(message)

def verify_email_page():
    """Handle email verification with fixed form buttons"""
    st.title("Email Verification")
    
    token = st.query_params.get("token", "")
    
    if not token:
        st.error("❌ Invalid verification link")
        st.info("Please use the link sent to your email")
        
        # Return to login form
        with st.form("return_login_form"):
            if st.form_submit_button("Return to Login"):
                st.query_params.clear()
                st.rerun()
        return
    
    with st.spinner("Verifying your email..."):
        db = Database()
        success, message = db.verify_email(token)
    
    if success:
        st.success("✅ " + message)
        st.balloons()
        st.info("You can now log in to your account")
        
        with st.form("goto_login_form"):
            if st.form_submit_button("Go to Login"):
                # Clear query parameters
                st.query_params.clear()
                # Reset the page state to login
                st.session_state.page = "login"
                # Clear any existing user session data
                for key in list(st.session_state.keys()):
                    if key != "page":  # Preserve the page state
                        del st.session_state[key]
                st.rerun()
    else:
        st.error("❌ " + message)
        if "expired" in message.lower():
            st.info("Need a new verification link?")
            
            # Resend verification form
            with st.form("resend_verification_form"):
                username = st.text_input("Enter your username")
                resend_submitted = st.form_submit_button("Resend Verification Email")
                
                if resend_submitted and username:
                    with st.spinner("Sending verification email..."):
                        if db.resend_verification_email(username):
                            st.success("New verification email sent!")
                        else:
                            st.error("Failed to send verification email")

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def get_verification_url(token: str) -> str:
    """Generate verification URL"""
    base_url = os.getenv('VERIFICATION_URL', 'http://192.168.6.111:8501')
    # Instead of /verify, use query parameter 'page=verify'
    return f"{base_url}/?page=verify&token={urllib.parse.quote(token)}"

def handle_registration(username: str, password: str, email: str) -> Tuple[bool, str]:
    """Handle user registration and email verification"""
    try:
        if not all([username, password, email]):
            return False, "Please fill in all fields"
            
        if not validate_email(email):
            return False, "Please enter a valid email address"
            
        # Create user and send verification email
        user_id, message = db.create_user(username, password, email)
        
        if not user_id:
            return False, message
            
        # Log registration event
        logger.info(f"New user registered: {username} ({email})")
        
        return True, "Registration successful! Please check your email to verify your account."
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return False, f"Registration failed: {str(e)}"

def api_settings():
    """Display API settings page"""
    st.subheader("API Settings")
    
    # Display current API keys from environment
    openai_key = os.getenv('OPENAI_API_KEY', '')
    anthropic_key = os.getenv('ANTHROPIC_API_KEY', '')
    
    st.info("""API keys are now configured through environment variables.
                To set your API keys:
                1. Create a .env file in your project directory
                2. Add your API keys in the following format:
                    OPENAI_API_KEY=your_key_here
                    ANTHROPIC_API_KEY=your_key_here
                3. Restart the application for changes to take effect""")
    # Display masked versions of current keys if they exist
    if openai_key:
        st.text("OpenAI API Key: " + "*" * len(openai_key))
    if anthropic_key:
        st.text("Anthropic API Key: " + "*" * len(anthropic_key))

def format_chat_history(chat_history):
    """Format chat history into a readable string"""
    formatted_history = []
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted_history.append(f"{role}: {msg['content']}")
    return "\n\n".join(formatted_history)

def get_ai_response(message, api_choice):
    """Get response from selected AI API with ChromaDB integration, source display, and chat history"""
    collection_name = db.get_collection_name(st.session_state.chat_session_id)
    context = ""
    sources = []
    
    # Get relevant document context if available
    if collection_name:
        results = chroma_ops.query_collection(collection_name, message)
        if results and results['documents']:
            for doc, meta, distance in zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            ):
                source = meta.get('source', 'Unknown')
                chunk = meta.get('chunk', 0)
                chunk_key = f"{source}-chunk-{chunk}"
                
                if 'document_chunks' not in st.session_state:
                    st.session_state.document_chunks = {}
                st.session_state.document_chunks[chunk_key] = doc
                
                sources.append({
                    'source': source,
                    'chunk': chunk,
                    'chunk_key': chunk_key,
                    'distance': distance,
                    'chunk_content': doc
                })
            
            sources.sort(key=lambda x: x['distance'])    
            context = "\n".join([s['chunk_content'] for s in sources])

    # Prepare chat history context
    chat_history = []
    # Get last 5 message pairs from current session for context
    recent_messages = st.session_state.messages[-10:] if len(st.session_state.messages) > 0 else []
    for msg in recent_messages:
        chat_history.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Build the complete prompt with context and chat history
    if context:
        prompt = f"""Previous conversation:
{format_chat_history(chat_history)}

Context from uploaded documents:
{context}

Current user question:
{message}

Please provide a response based on the conversation history and context above. Include specific references to source documents when possible."""
    else:
        prompt = f"""Previous conversation:
{format_chat_history(chat_history)}

Current user question:
{message}

Please provide a response based on the conversation history."""

    if api_choice == "Groq":
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            return "Please set up your Groq API key in the environment variables.", None
        
        client = Groq(api_key=api_key)
        try:
            print("Groq.....")
            print(f"Groq : {st.session_state.selected_groq_model}")
            #input("按任意鍵繼續...")
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            # Add chat history
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            # Add current message
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=st.session_state.selected_groq_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                top_p=1,
                stream=False
            )
            return str(response.choices[0].message.content), sources
        except Exception as e:
            return f"Error with Groq API: {str(e)}", None


    if api_choice == "OpenAI":
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            return "Please set up your OpenAI API key in the environment variables.", None
        
        client = OpenAI(api_key=api_key)
        try:
            print("OpenAI.....")
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            # Add chat history
            for msg in chat_history:
                messages.append({"role": msg["role"], "content": msg["content"]})
            # Add current message
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=messages
            )
            return str(response.choices[0].message.content), sources
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}", None
    
    elif api_choice == "Anthropic":
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            return "Please set up your Anthropic API key in the environment variables.", None
        
        client = Anthropic(api_key=api_key)
        try:
            print("Anthropic.....")
            # Format messages for Anthropic API
            formatted_messages = []
            for msg in chat_history:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system="You are a helpful AI assistant.",
                messages=[
                    *formatted_messages,
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return str(response.content[0].text), sources
        except Exception as e:
            return f"Error with Anthropic API: {str(e)}", None
    
    else:  # OLLAMA
        if not st.session_state.selected_ollama_model:
            return "Please select an OLLAMA model first.", None
        
        try:
            print("OLLAMA.....")
            # For OLLAMA, we'll format the chat history and context into a single prompt
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    "model": st.session_state.selected_ollama_model,
                    "prompt": prompt,
                    "stream": False
                })
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data.get('response', 'No response received'), sources
            return f"Error: Failed to get response from OLLAMA (Status code: {response.status_code})", None
        except Exception as e:
            return f"Error: Failed to connect to OLLAMA server. Make sure it's running. ({str(e)})", None


def load_conversation_history():
    """Load conversation history from database"""
    try:
        st.session_state.conversation_history = []
        
        conversations = db.get_user_conversations_with_sources(st.session_state.user_id)
        if conversations:
            current_session = None
            current_conversation = []
            
            for conv in conversations:
                session_id, api_source, user_msg, assistant_msg, timestamp, session_created_at, sources = conv
                
                if current_session != session_id:
                    if current_conversation:
                        st.session_state.conversation_history.append({
                            'session_id': current_session,
                            'created_at': session_created_at,
                            'messages': current_conversation.copy()
                        })
                    current_session = session_id
                    current_conversation = []
                
                try:
                    # 解析來源並確保內容被包含
                    parsed_sources = []
                    if sources:
                        import json
                        sources_list = sources.split(',')
                        for source_str in sources_list:
                            source = json.loads(source_str)
                            chunk_key = f"{source['source']}-chunk-{source['chunk']}"
                            
                            # 保存來源文件內容到 session state
                            if 'content' in source:
                                if 'document_chunks' not in st.session_state:
                                    st.session_state.document_chunks = {}
                                st.session_state.document_chunks[chunk_key] = source['content']
                            
                            parsed_sources.append(source)
                except Exception as e:
                    print(f"Error parsing sources: {e}")
                    parsed_sources = []
                
                current_conversation.extend([
                    {
                        "role": "user",
                        "content": str(user_msg),
                        "api": str(api_source)
                    },
                    {
                        "role": "assistant",
                        "content": str(assistant_msg),
                        "api": str(api_source),
                        "sources": parsed_sources
                    }
                ])
            
            if current_conversation:
                st.session_state.conversation_history.append({
                    'session_id': current_session,
                    'created_at': session_created_at,
                    'messages': current_conversation
                })
                
    except Exception as e:
        st.error(f"Error loading conversation history: {str(e)}")
        st.session_state.conversation_history = []

def display_conversation_history():
    """Display conversation history in the sidebar"""
    st.sidebar.title("Conversation History")
    
    if not st.session_state.conversation_history:
        st.sidebar.write("No conversation history yet.")
        return
    
    for conv_group in st.session_state.conversation_history:
         # Find first user message in the conversation
        first_user_msg = next(
            (msg["content"][:12] for msg in conv_group['messages'] if msg["role"] == "user"),
            "Chat"  # Default text if no user message found
        )

        col1, col2 = st.sidebar.columns([10,1])
        with col1:
            if st.button(f"{first_user_msg}...", key=f"hist_{conv_group['session_id']}"):
                # Reset session state
                st.session_state.messages = []
                st.session_state.chat_session_id = conv_group['session_id']
                st.session_state.api_choice = conv_group['messages'][0]['api']
                st.session_state.document_chunks = {}  # Reset document chunks cache
                
                # Load conversation messages
                for msg in conv_group['messages']:
                    if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                        # Process each source's content
                        for source in msg["sources"]:
                            if 'content' in source:
                                chunk_key = f"{source['source']}-chunk-{source['chunk']}"
                                st.session_state.document_chunks[chunk_key] = source['content']
                                
                    st.session_state.messages.append(msg)
                
                # 載入其他會話相關資料
                st.session_state.session_files = db.get_session_files(conv_group['session_id'])
                st.session_state.session_images = db.get_session_images(conv_group['session_id'])
                st.session_state.session_media = db.get_session_images(conv_group['session_id'])
                
                #if st.session_state.session_images:
                #    st.session_state.image_mode = True
                #    if not st.session_state.image_processor:
                #        st.session_state.image_processor = ImageOperations()
                
                if st.session_state.session_media:
                    st.session_state.vision_mode = True
                    if not st.session_state.vision_processor:
                        st.session_state.vision_processor = VisionOperations()


                st.rerun()

def start_new_chat():
    """Start a new chat session"""
    st.session_state.messages = []
    st.session_state.current_file_content = None
    st.session_state.session_files = []
    st.session_state.session_images = []  # Reset images
    st.session_state.image_mode = False   # Disable image mode
    st.session_state.chat_session_id = db.create_chat_session(
        st.session_state.user_id,
        st.session_state.api_choice
    )
    st.rerun()  

def handle_multiple_files_upload(uploaded_files, user_id, session_id):
    """Handle multiple files upload with improved ChromaDB integration"""
    try:
        if len(uploaded_files) > 5:
            st.error("You can only upload up to 5 files at once.")
            return False

        collection_name = db.get_collection_name(session_id)
        if not collection_name:
            collection_name = chroma_ops.generate_collection_name()
            db.save_collection_name(session_id, collection_name)

        all_content = []
        processed_files = []
        failed_files = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing file {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            try:
                file_path, file_id = save_uploaded_file(uploaded_file, user_id, session_id)
                if not file_path or not file_id:
                    failed_files.append((uploaded_file.name, "Failed to save file"))
                    continue
                    
                content = None
                if uploaded_file.type == 'application/pdf':
                    content = process_pdf(uploaded_file)
                elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    content = process_docx(uploaded_file)
                elif uploaded_file.type == 'text/csv':
                    content = process_csv(uploaded_file)
                elif uploaded_file.type == 'text/plain':
                    content = process_txt(uploaded_file)
                else:
                    st.warning(f"Unsupported file type for {uploaded_file.name}")
                    continue

                if not content:
                    failed_files.append((uploaded_file.name, "Failed to extract content"))
                    continue

                # Process content into chunks with IDs
                chunk_data = chroma_ops.process_documents_for_chromadb(
                    content, 
                    uploaded_file.name,
                    session_id
                )
                
                # Process chunks in smaller batches
                BATCH_SIZE = 10
                for i in range(0, len(chunk_data), BATCH_SIZE):
                    batch = chunk_data[i:i + BATCH_SIZE]
                    
                    # Extract components for ChromaDB
                    texts = [chunk['content'] for chunk in batch]
                    metadatas = [chunk['metadata'] for chunk in batch]
                    ids = [chunk['id'] for chunk in batch]
                    
                    # Add to ChromaDB
                    success = chroma_ops.add_documents(
                        collection_name=collection_name,
                        texts=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    
                    if success:
                        # Save chunk information to database
                        for chunk in batch:
                            db.save_chunk_info(
                                chunk_id=chunk['id'],
                                chunk_name=chunk['name'],
                                file_name=uploaded_file.name,
                                session_id=session_id,
                                user_id=user_id
                            )
                    else:
                        failed_files.append((uploaded_file.name, f"Failed to process chunk batch {i//BATCH_SIZE + 1}"))
                        continue
                
                processed_files.append(uploaded_file.name)
                
            except Exception as e:
                failed_files.append((uploaded_file.name, f"Processing error: {str(e)}"))
                continue
                
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        progress_bar.empty()
        status_text.empty()
        
        if processed_files:
            st.success(f"Successfully processed {len(processed_files)} files")
            st.session_state.session_files = db.get_session_files(session_id)
            
        if failed_files:
            st.error("Failed to process the following files:")
            for name, error in failed_files:
                st.error(f"- {name}: {error}")
        
        return len(processed_files) > 0
        
    except Exception as e:
        st.error(f"Error during files upload: {str(e)}")
        return False

def display_document_sources(sources):
    """Display document sources with content"""
    if sources:
        st.markdown("#### Referenced Documents")
        for source in sources:
            expander_key = f"source_{source['source']}_{source['chunk']}"
            
            col1, col2 = st.columns([4, 1])
            with col1:
                with st.expander(f"📄 {source['source']} (Chunk {source['chunk']})"):
                    chunk_key = f"{source['source']}-chunk-{source['chunk']}"
                    
                    # Get content from various sources
                    content = (
                        source.get('content') or               # 直接從 source 對象
                        source.get('chunk_content') or         # 或者從 chunk_content 字段
                        st.session_state.document_chunks.get(chunk_key)  # 或者從 session state
                    )
                    
                    if content:
                        st.code(content, language="text")
                    else:
                        st.write("Content not available")
            
            with col2:
                if 'distance' in source:
                    try:
                        similarity = 1 - float(source['distance'])
                        similarity_pct = f"{similarity * 100:.2f}%"
                        st.text(similarity_pct)
                    except (ValueError, TypeError):
                        st.text("N/A")

def handle_file_removal(file_id, session_id):
    """Handle complete file removal including physical file and all associated data"""
    try:
        # Get collection name
        collection_name = db.get_collection_name(session_id)
        if not collection_name:
            st.error("Could not find associated ChromaDB collection")
            return False
            
        # Get file chunks before removal
        chunk_ids, filename = db.get_file_chunks(file_id)
        
        # Remove file and get file path
        file_path, filename, session_id = db.remove_file(file_id)
        if not file_path:
            st.error("Could not find file information")
            return False
            
        # Remove physical file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            st.error(f"Error removing physical file: {e}")
            
        # Remove from ChromaDB
        if chunk_ids:
            success = chroma_ops.remove_documents(collection_name, chunk_ids)
            if not success:
                st.warning("Some chunks might not have been removed from ChromaDB")
                
        # Update session files
        st.session_state.session_files = db.get_session_files(session_id)
        
        st.success(f"Successfully removed file: {filename}")
        return True
        
    except Exception as e:
        st.error(f"Error during file removal: {e}")
        return False
    
def start_new_chat(collection_name=None):
    """Start a new chat session with optional collection name"""
    st.session_state.messages = []
    st.session_state.current_file_content = None
    st.session_state.session_files = []
    st.session_state.session_images = []
    st.session_state.session_media = []
    st.session_state.image_mode = False
    st.session_state.vision_mode = False
    
    # Create new chat session
    st.session_state.chat_session_id = db.create_chat_session(
        st.session_state.user_id,
        st.session_state.api_choice
    )
    
    # If collection_name is provided, save it to the session
    if collection_name:
        db.save_collection_name(st.session_state.chat_session_id, collection_name)
        st.session_state.using_existing_collection = True
    else:
        st.session_state.using_existing_collection = False
        
    st.rerun()

def handle_media_upload(uploaded_file, user_id, session_id):
    """Handle media file upload with duplicate check"""
    try:
        # Check for duplicate
        existing_media = st.session_state.vision_processor.get_session_media(session_id)
        if existing_media:
            for _, filename, _, _, _ in existing_media:
                if filename == uploaded_file.name:
                    st.info(f"File '{uploaded_file.name}' already exists in this session.")
                    return True
        
        # Save new file
        file_path, file_type = st.session_state.vision_processor.save_media(
            uploaded_file,
            session_id,
            user_id
        )
        
        if not file_path or not file_type:
            st.error("Failed to save media file")
            return False

        # Save to database
        image_id = db.save_uploaded_image(
            user_id,
            session_id,
            uploaded_file.name,
            file_path,
            file_type
        )

        st.success(f"Successfully uploaded: {uploaded_file.name}")
        return True
        
    except Exception as e:
        st.error(f"Error uploading media: {str(e)}")
        return False

def chat_interface():
    """Display main chat interface"""
    st.title("Multi-Modal RAG Chat Application")
    
# Add to existing chat_interface function
    with st.sidebar:
        #st.session_state.image_mode = st.checkbox("Enable Image Processing Mode")
        st.session_state.vision_mode = st.checkbox("Enable Llama Vision 90b")
        
    #if 'vision_processor' not in st.session_state:
    st.session_state.vision_processor = VisionOperations()
        
    if st.session_state.vision_mode:
        st.subheader("Vision Chat Mode")
        st.markdown("Upload images (.jpg/.png) or videos (.mp4) for analysis")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose file",
            type=['jpg', 'jpeg', 'png', 'mp4'],
            key=st.session_state.image_uploader_key
        )
        
        if uploaded_file:
            if handle_media_upload(
                uploaded_file,
                st.session_state.user_id,
                st.session_state.chat_session_id
            ):
                st.session_state.session_media = st.session_state.vision_processor.get_session_media(
                    st.session_state.chat_session_id
                )
                # Reset the file uploader
                st.session_state.image_uploader_key = str(time.time())
                st.rerun()
        
        # Display uploaded media
        if hasattr(st.session_state, 'session_media') and st.session_state.session_media:
            st.subheader("Media in Current Session")
            
            for media_id, filename, file_path, file_type, timestamp in st.session_state.session_media:
                if os.path.exists(file_path):
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        if file_type in ['jpg', 'jpeg', 'png']:
                            st.image(file_path, caption=filename, width=200)
                        elif file_type == 'mp4':
                            st.video(file_path)
                            
                    with col2:
                        if st.button("Remove", key=f"remove_media_{media_id}"):
                            if st.session_state.vision_processor.remove_media(file_path):
                                st.session_state.session_media = st.session_state.vision_processor.get_session_media(
                                    st.session_state.chat_session_id
                                )
                                st.rerun()
        
        # Chat input for media processing
        if prompt := st.chat_input("Ask about the media or images..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process media with query
            with st.chat_message("assistant"):
                responses = []
                for _, _, file_path, file_type, _ in st.session_state.session_media:
                    response = st.session_state.vision_processor.process_media(file_path, prompt)
                    if response:
                        responses.append(response)
                
                if responses:
                    combined_response = "\n\n".join(responses)
                    st.markdown(combined_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": combined_response
                    })
                    
                    # Save conversation
                    updated_vision_conversations = db.save_conversation(
                        st.session_state.chat_session_id,
                        st.session_state.user_id,
                        "llama3.2-vision",
                        prompt,
                        combined_response
                    )
                    if updated_vision_conversations:
                        load_conversation_history()
                        st.rerun()

    if st.session_state.image_mode:
        st.subheader("Image Processing Mode")
        st.markdown("Upload images (.jpg or .png) for visual analysis or text translation")
    
        uploaded_image = st.file_uploader(
            "Choose image",
            type=['jpg', 'jpeg', 'png'],
            key=st.session_state.image_uploader_key
        )
    
        if uploaded_image:
            if handle_image_upload(
                uploaded_image,
                st.session_state.user_id,
                st.session_state.chat_session_id
            ):
                # Update session images list
                st.session_state.session_images = db.get_session_images(
                    st.session_state.chat_session_id
                )
                # Reset the file uploader
                st.session_state.image_uploader_key = str(time.time())
                st.rerun()
            
        # Display all images in the current session
        if hasattr(st.session_state, 'session_images') and st.session_state.session_images:
            st.subheader("Images in This Session")
            for image_id, filename, file_path, file_type, _ in st.session_state.session_images:
                if os.path.exists(file_path):  # Only show images that still exist
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        try:
                            st.image(file_path, caption=filename, width=200)
                        except Exception as e:
                            st.error(f"Error displaying image {filename}: {str(e)}")
                    with col2:
                        if st.button("Remove", key=f"remove_img_{image_id}"):
                            file_path = db.remove_image(image_id)
                            if file_path and st.session_state.image_processor:
                                st.session_state.image_processor.remove_image(file_path)
                                st.session_state.session_images = db.get_session_images(
                                    st.session_state.chat_session_id
                                )
                                st.rerun()
        
        # Chat input for image processing
        if prompt := st.chat_input("Ask about the images..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Process each image with the query
            with st.chat_message("assistant"):
                responses = []
                for _, _, file_path, _, _ in st.session_state.session_images:
                    response = process_image_query(file_path, prompt)
                    if response:
                        responses.append(response)
                
                if responses:
                    combined_response = "\n\n".join(responses)
                    st.markdown(combined_response)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": combined_response
                    })
                    
                    # Save conversation
                    updated_jpg_conversations = db.save_conversation(
                        st.session_state.chat_session_id,
                        st.session_state.user_id,
                        "Qwen2-VL",
                        prompt,
                        combined_response
                    )
                    if updated_jpg_conversations:
                        load_conversation_history()
                        st.rerun()


    # Create new chat session if none exists
    if st.session_state.chat_session_id is None:
        st.session_state.chat_session_id = db.create_chat_session(
            st.session_state.user_id,
            st.session_state.api_choice
        )
    
    # Sidebar
    with st.sidebar:
        st.write(f"Welcome, {st.session_state.username}!")

        if st.button("New Chat"):
            start_new_chat()
        
        st.session_state.api_choice = st.radio(
            "Select API", 
            ["OpenAI", "Anthropic", "Groq", "OLLAMA"]
        )
        # OLLAMA model selection
        if st.session_state.api_choice == "OLLAMA":
            st.session_state.ollama_models = get_ollama_models()
            if st.session_state.ollama_models:
                st.session_state.selected_ollama_model = st.selectbox(
                    "Select OLLAMA Model",
                    st.session_state.ollama_models
                )
            else:
                st.error("No OLLAMA models found. Make sure OLLAMA server is running.")
        
        elif st.session_state.api_choice == "Groq":
            st.session_state.groq_models = get_groq_models()
            if st.session_state.groq_models:
                st.session_state.selected_groq_model = st.selectbox(
                    "Select Groq Model",
                    st.session_state.groq_models,
                    help="Choose a Groq model for chat completion"
                )
                
                # Display model descriptions
                if st.session_state.selected_groq_model == "llama-3.3-70b-versatile":
                    st.info("llama-3.3-70b-versatile")
                elif st.session_state.selected_groq_model == "llama-3.2-90b-vision-preview":
                    st.info("llama-3.2-90b-vision-preview")
                elif st.session_state.selected_groq_model == "mixtral-8x7b-32768":
                    st.info("mixtral-8x7b-32768")
            else:
                st.error("No Groq models available")

        st.markdown("### ChromaDB Collections")
        collections = chroma_ops.list_collections()
        if collections:
            # Initialize selected_collection in session state if not exists
            if 'selected_collection' not in st.session_state:
                st.session_state.selected_collection = None
            
            # Create selection box for collections
            collection_names = [f"{c['name']} ({c['count']} docs)" for c in collections]
            selected_index = st.selectbox(
                "Select Collection",
                range(len(collection_names)),
                format_func=lambda x: collection_names[x],
                key="collection_selector"
            )
            
            # Update selected collection
            if selected_index is not None:
                selected_collection = collections[selected_index]['name']
                
                # Display collection details
                with st.expander("Collection Details", expanded=True):
                    st.write(f"**Name:** {collections[selected_index]['name']}")
                    st.write(f"**Documents:** {collections[selected_index]['count']}")
                    if collections[selected_index]['metadata']:
                        st.write("**Metadata:**")
                        st.json(collections[selected_index]['metadata'])
                    
                # Add button to start new chat with selected collection
                if st.button("Use Selected Collection"):
                    start_new_chat(selected_collection)
        else:
            st.info("No collections available")

        # Add password change button
        #if st.button("Change Password"):
        #    st.session_state.page = "password_change"
        #    st.rerun()
        
        #if st.button("API Settings"):
        #    st.session_state.show_settings = True
        #    st.session_state.show_history = False
    
        # Always display conversation history in sidebar
        if st.button("View History"):
            st.session_state.show_history = True
            st.session_state.show_settings = False
            
        display_conversation_history()

        # Add profile update button
        if st.button("Update Profile"):
            st.session_state.page = "profile_update"
            st.rerun()
            
        if st.button("Logout"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()    
    # Show API settings if requested
    #if st.session_state.show_settings:
    #    api_settings()
    #    if st.button("Back to Chat"):
    #        st.session_state.show_settings = False
    #        st.rerun()
    #   return
    
    # Only show file upload section if not using existing collection
    if not st.session_state.get('using_existing_collection', False):
        # Display current session's uploaded files
        if st.session_state.session_files:
            st.subheader("Uploaded Files in This Session")
            for file_id, filename, _, _, timestamp in st.session_state.session_files:
                st.text(f"📎 {filename}")

        # File upload section
        st.markdown("### Upload Files")
        st.markdown("You can upload up to 5 files (PDF, DOCX, CSV, TXT)")
    
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'csv', 'txt'],
            accept_multiple_files=True,
            key=st.session_state.file_uploader_key
        )
    
        if uploaded_files:
            if len(uploaded_files) > 5:
                st.error("Please select no more than 5 files.")
            else:
                success = handle_multiple_files_upload(
                    uploaded_files,
                    st.session_state.user_id,
                    st.session_state.chat_session_id
                )
                if success:
                    st.info("You can now ask questions about the uploaded files' content!")
                    # Reset file uploader
                    st.session_state.file_uploader_key = str(time.time())
                    st.rerun()  # Changed from experimental_rerun()
    
        # Display current session's uploaded files
        if st.session_state.session_files:
            st.subheader("Files in Current Session")
            for file_id, filename, _, _, timestamp in st.session_state.session_files:
                col1, col2 = st.columns([5,1])
                with col1:
                    st.text(f"📎 {filename}")
                with col2:
                    if st.button("Remove", key=f"remove_{file_id}"):
                        if handle_file_removal(file_id, st.session_state.chat_session_id):
                            st.rerun()
    
    
   # 修改聊天消息顯示部分
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(str(message["content"]))
            
            # 如果是助手回覆且有來源文檔，顯示來源
            if message["role"] == "assistant" and "sources" in message and message["sources"]:
                st.markdown("---")
                display_document_sources(message["sources"])
    
    # 修改聊天輸入處理部分
    if prompt := st.chat_input("What would you like to ask?"):
        prompt = str(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            response_container = st.container()
            
            with response_container:
                response, sources = get_ai_response(prompt, st.session_state.api_choice)
                st.markdown(response)
                
                #if sources:
                st.markdown("---")
                display_document_sources(sources)
                    
                # 確保來源內容被正確保存
                for source in sources:
                    chunk_key = f"{source['source']}-chunk-{source['chunk']}"
                    if chunk_key in st.session_state.document_chunks:
                        source['chunk_content'] = st.session_state.document_chunks[chunk_key]
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
        
        # Save conversation with complete source information
        updated_conversations = db.save_conversation_with_sources(
            st.session_state.chat_session_id,
            st.session_state.user_id,
            str(st.session_state.api_choice),
            prompt,
            response,
            sources
        )
        
        if updated_conversations:
            load_conversation_history()
            st.rerun()
        
        # Save conversation to database and update conversation history
        updated_conversations = db.save_conversation(
            st.session_state.chat_session_id,
            st.session_state.user_id,
            str(st.session_state.api_choice),
            prompt,
            response
        )
        
        # Update conversation history in session state
        if updated_conversations:
            # Reconstruct conversation history to match the expected format
            current_session_history = {
                'session_id': st.session_state.chat_session_id,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'messages': []
            }
            
            for conv in updated_conversations:
                current_session_history['messages'].extend([
                    {"role": "user", "content": conv[2], "api": str(st.session_state.api_choice)},
                    {"role": "assistant", "content": conv[3], "api": str(st.session_state.api_choice)}
                ])
            
            # Update or add to conversation history
            existing_session = next((
                (idx, session) for idx, session in enumerate(st.session_state.conversation_history) 
                if session['session_id'] == st.session_state.chat_session_id
            ), None)
            
            if existing_session:
                # Update existing session
                st.session_state.conversation_history[existing_session[0]] = current_session_history
            else:
                # Add new session
                st.session_state.conversation_history.append(current_session_history)

def password_change_page():
    """Display password change interface"""
    st.title("Change Password")
    
    with st.form("password_change_form", clear_on_submit=True):
        current_password = st.text_input(
            "Current Password", 
            type="password"
        )
        new_password = st.text_input(
            "New Password", 
            type="password"
        )
        confirm_password = st.text_input(
            "Confirm New Password", 
            type="password"
        )
        
        submitted = st.form_submit_button("Change Password")
        
        if submitted:
            if not all([current_password, new_password, confirm_password]):
                st.error("Please fill in all fields")
            elif new_password != confirm_password:
                st.error("New passwords do not match")
            elif len(new_password) < 6:
                st.error("New password must be at least 6 characters long")
            elif current_password == new_password:
                st.error("New password must be different from current password")
            else:
                success, message = db.change_password(
                    st.session_state.user_id,
                    current_password,
                    new_password
                )
                
                if success:
                    st.success(message)
                    st.info("Please log in again with your new password")
                    # Log out user
                    for key in st.session_state.keys():
                        del st.session_state[key]
                    time.sleep(2)  # Give user time to read the message
                    st.rerun()
                else:
                    st.error(message)
    
    with st.container():
        if st.button("Back to Chat"):
            st.session_state.page = "chat"
            st.rerun()

def profile_update_page():
    """Display profile update interface with improved validation"""
    st.title("Update Profile")
    
    try:
        # Get current email using the database instance
        conn = sqlite3.connect(db.db_name)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT email, is_verified FROM users WHERE user_id = ?",
            (st.session_state.user_id,)
        )
        result = cursor.fetchone()
        current_email, is_verified = result if result else (None, None)
        conn.close()
        
        # Show current status
        st.info(
            f"""
            **Current Email Status**  
            Email: {current_email or 'Not set'}  
            Status: {'✅ Verified' if is_verified else '⚠️ Not verified'}
            """
        )
        
        with st.form("profile_update_form", clear_on_submit=False):  # Changed to false to preserve input on error
            current_password = st.text_input(
                "Current Password (required)", 
                type="password",
                help="Enter your current password to make any changes"
            )
            
            tab1, tab2 = st.tabs(["Change Password", "Change Email"])
            
            with tab1:
                new_password = st.text_input(
                    "New Password", 
                    type="password",
                    help="Minimum 6 characters"
                )
                confirm_password = st.text_input(
                    "Confirm New Password", 
                    type="password"
                )
                
            with tab2:
                new_email = st.text_input(
                    "New Email",
                    value=current_email,
                    help="Enter new email address if you want to change it"
                )
            
            st.write("")
            submitted = st.form_submit_button("Update Profile", use_container_width=True)
            
            if submitted:
                # Reset validation flag
                is_valid = True
                
                # Validate current password
                if not current_password:
                    st.error("⚠️ Current password is required")
                    is_valid = False
                
                # Validate password change if attempted
                if new_password or confirm_password:
                    if new_password != confirm_password:
                        st.error("⚠️ New passwords do not match")
                        is_valid = False
                    elif len(new_password) < 6:
                        st.error("⚠️ New password must be at least 6 characters long")
                        is_valid = False
                    elif current_password == new_password:
                        st.error("⚠️ New password must be different from current password")
                        is_valid = False
                
                # Validate email change if attempted
                if new_email and new_email != current_email:
                    if not validate_email(new_email):
                        st.error("⚠️ Please enter a valid email address")
                        is_valid = False
                
                # Validate that at least one change was requested
                if not new_password and new_email == current_email:
                    st.warning("⚠️ No changes were requested")
                    is_valid = False
                
                # Only proceed if all validations pass
                if is_valid:
                    success, message = db.update_profile(
                        st.session_state.user_id,
                        current_password,
                        new_password if new_password else None,
                        new_email if new_email != current_email else None
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        
                        # If password was changed, log out user
                        if new_password:
                            st.info("🔄 Password changed. Please log in again with your new password")
                            time.sleep(2)
                            for key in st.session_state.keys():
                                del st.session_state[key]
                            st.rerun()
                        elif new_email != current_email:
                            st.info(
                                """
                                📧 Email address updated. Please check your inbox for verification.
                                You can continue using the application while verification is pending.
                                """
                            )
                            time.sleep(2)
                            st.session_state.page = "chat"
                            st.rerun()
                    else:
                        st.error(f"⚠️ {message}")
        
        # Back button
        st.write("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Back to Chat", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in profile_update_page: {str(e)}")


def main():
    """Main application with error handling"""
    try:
        init_session_state()
        
        # Get current page from query parameters
        query_params = st.query_params
        current_page = query_params.get("page", "")
        
        # Update session state page if specified in query params
        if current_page:
            st.session_state.page = current_page
        
        # Route to appropriate page
        if st.session_state.user_id is not None:
            if st.session_state.page == "profile_update":
                profile_update_page()
            else:
                chat_interface()
        elif st.session_state.page == 'verify':
            verify_email_page()
        else:
            login_page()
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    clear = lambda: os.system('cls')
    clear()
    main()
