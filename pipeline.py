import os
import json
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pickle
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple


# ✅ Force HuggingFace to use custom cache directory
os.environ["TRANSFORMERS_CACHE"] = r"D:\HF_MODELS\transformers"

# Configuration
VECTOR_STORE_DIR = "vectorstore/"
PDF_DIR = "data/"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# Initialize Sentence Transformer for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
EMBEDDING_DIM = 384  # Dimension of the embeddings

# Initialize FAISS index
index_path = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
metadata_path = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

# Load or create FAISS index
try:
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        documents = pickle.load(f)
except:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    documents = []

# Simple response template
def format_response(results, query):
    """Format the response from retrieved documents."""
    if not results or not results[0]:
        return "No relevant information found in the documents."
    
    response = "📚 Relevant information:\n"
    for doc in results:
        response += f"\n---\n{doc}"
    return response

def check_poppler_installation(poppler_path: str = None) -> bool:
    """Check if poppler is properly installed and accessible."""
    try:
        import subprocess
        import platform
        
        # Determine the command based on the operating system
        if platform.system() == 'Windows':
            cmd = ['where' if poppler_path is None else os.path.join(poppler_path, 'pdftocairo.exe'), 'pdftocairo']
        else:
            cmd = ['which', 'pdftocairo']
            
        # Run the command
        result = subprocess.run(cmd, 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE,
                              text=True)
        
        if result.returncode == 0:
            print(f"✅ Found poppler at: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Poppler not found. Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ Error checking poppler installation: {e}")
        return False

def extract_images_from_pdf(pdf_path: str, output_dir: str = "data/images") -> List[Dict[str, Any]]:
    """Extract images from a PDF and save them to the output directory."""
    try:
        print(f"\n{'='*50}")
        print(f"Processing PDF: {pdf_path}")
        print(f"Output directory: {os.path.abspath(output_dir)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if input file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Common poppler paths for different operating systems
        poppler_paths = [
            r"D:\New folder (3)\poppler-windows\Library\bin",  # User's custom path
            r"D:\New folder (3)\poppler-windows\bin",           # Alternative bin location
            None  # Try without path last
        ]
        
        # Debug: Print all paths that will be checked
        print("\nChecking for poppler in the following locations:")
        for i, path in enumerate(poppler_paths):
            exists = "✅" if path and os.path.exists(path) else "❌"
            print(f"{i+1}. {exists} {path or 'System PATH'}")
        
        images = None
        last_error = None
        
        # Try each poppler path until one works
        for poppler_path in poppler_paths:
            try:
                print(f"\nTrying poppler path: {poppler_path or 'System PATH'}")
                
                if poppler_path:
                    # Check if the path exists
                    if not os.path.exists(poppler_path):
                        print(f"Path does not exist: {poppler_path}")
                        continue
                        
                    # Check if required files exist
                    required_files = ['pdftocairo.exe', 'pdftoppm.exe']
                    missing_files = [f for f in required_files 
                                   if not os.path.exists(os.path.join(poppler_path, f))]
                    
                    if missing_files:
                        print(f"Missing required files in {poppler_path}: {', '.join(missing_files)}")
                        continue
                    
                    # Try to use this path
                    images = convert_from_path(pdf_path, poppler_path=poppler_path)
                else:
                    # Try system PATH
                    if check_poppler_installation():
                        images = convert_from_path(pdf_path)
                
                if images is not None:
                    print(f"✅ Successfully loaded PDF with {len(images)} pages")
                    break
                    
            except Exception as e:
                last_error = e
                print(f"❌ Error with path {poppler_path or 'System PATH'}: {str(e)}")
                continue
                
        if images is None:
            error_msg = """
            =========================================================================
            ❌ PDF to image conversion failed. Poppler is required but not found.
            
            Please install poppler:
            
            Windows:
              1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
              2. Extract to C:\\poppler
              3. Add C:\\poppler\\Library\\bin to your system PATH
              
            OR
            
              1. Download and extract poppler to D:\\New folder (3)\\poppler-windows
              2. Make sure the 'bin' folder contains pdftocairo.exe and other binaries
            
            After installation, restart your IDE/terminal and try again.
            =========================================================================
            """
            print(error_msg)
            raise Exception(f"PDF to image conversion failed. Last error: {last_error}")
        
        # Process the extracted images
        extracted_images = []
        print(f"\nProcessing {len(images)} pages...")
        
        for i, image in enumerate(images):
            filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            
            print(f"Saving page {i+1} to {filepath}")
            image.save(filepath, "PNG")
            
            try:
                print(f"  Running OCR on page {i+1}...")
                text = pytesseract.image_to_string(image)
                print(f"  Extracted {len(text)} characters of text")
            except Exception as e:
                print(f"  ❌ Error extracting text from image: {e}")
                text = ""
                
            extracted_images.append({
                'path': filepath,
                'page': i + 1,
                'text': text.strip(),
                'dimensions': image.size
            })
        
        print(f"\n✅ Successfully processed {len(extracted_images)} pages")
        print("="*50 + "\n")
        return extracted_images
        
    except Exception as e:
        print(f"\n❌ Error in extract_images_from_pdf: {str(e)}")
        print("="*50 + "\n")
        return []

def process_pdf(pdf_path: str, chat_id: str = None) -> Tuple[bool, str]:
    """
    Process a PDF file, extract text and images, and add to the vector store.
    
    Args:
        pdf_path: Path to the PDF file
        chat_id: Optional chat ID to create a separate directory for each user
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    global index, documents
    
    try:
        # Create user-specific directories if chat_id is provided
        if chat_id:
            user_vector_dir = os.path.join(VECTOR_STORE_DIR, str(chat_id))
            user_pdf_dir = os.path.join(PDF_DIR, str(chat_id))
            os.makedirs(user_vector_dir, exist_ok=True)
            os.makedirs(user_pdf_dir, exist_ok=True)
            
            # Update paths for this user
            current_index_path = os.path.join(user_vector_dir, "faiss_index.bin")
            current_metadata_path = os.path.join(user_vector_dir, "metadata.pkl")
        else:
            current_index_path = index_path
            current_metadata_path = metadata_path
        
        # Read PDF text content
        pdf_reader = PdfReader(pdf_path)
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text() + "\n"
        
        # Extract images to user-specific directory if chat_id is provided
        output_dir = os.path.join(PDF_DIR, str(chat_id), 'images') if chat_id else 'data/images'
        images = extract_images_from_pdf(pdf_path, output_dir=output_dir)
        
        # Generate embedding for the text content
        embedding = model.encode([text_content])
        
        # Add to FAISS index
        document_id = len(documents)
        index.add(embedding)
        
        # Store document with metadata
        doc_data = {
            'text': text_content,
            'images': images,
            'pdf_path': pdf_path,
            'chat_id': chat_id,
            'timestamp': os.path.getmtime(pdf_path)
        }
        documents.append(doc_data)
        
        # Save the updated index and metadata
        faiss.write_index(index, current_index_path)
        with open(current_metadata_path, 'wb') as f:
            pickle.dump(documents, f)
            
        return True, f"Processed {len(images)} pages with {sum(1 for img in images if img.get('text', ''))} images containing text"
        
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"

def search_youtube_videos(query: str, max_results: int = 3, api_key: str = None) -> list:
    """
    Search for YouTube videos using the YouTube Data API v3 with improved relevance.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (max 10)
        api_key: Your YouTube Data API v3 key
        
    Returns:
        List of video information dictionaries with title, url, and thumbnail
    """
    try:
        if not api_key:
            print("⚠️ YouTube API key not provided. Please set GOOGLE_API_KEY in your environment variables.")
            return []
            
        from googleapiclient.discovery import build
        
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # Enhanced search parameters for better relevance
        request = youtube.search().list(
            q=query,
            part='snippet',
            maxResults=min(max_results, 10),  # Keep it small for better quality
            type='video',
            order='relevance',  # Most relevant first
            relevanceLanguage='en',
            safeSearch='moderate',
            videoDuration='medium',  # Prefer standard length videos (4-20 min)
            videoDefinition='high',  # Prefer HD videos
            videoEmbeddable='true'   # Ensure videos can be embedded
        )
        
        response = request.execute()
        
        # Get video details for better information
        if response.get('items'):
            video_ids = [item['id']['videoId'] for item in response['items']]
            
            # Get video details for duration and view count
            video_request = youtube.videos().list(
                part='contentDetails,statistics',
                id=','.join(video_ids)
            )
            video_details = video_request.execute()
            
            # Create a mapping of video_id to details
            details_map = {item['id']: item for item in video_details.get('items', [])}
        else:
            details_map = {}
        
        # Format the results with enhanced information
        results = []
        for item in response.get('items', []):
            try:
                video_id = item['id']['videoId']
                details = details_map.get(video_id, {})
                
                # Get the highest resolution thumbnail available
                thumbnails = item['snippet']['thumbnails']
                thumbnail_url = (
                    thumbnails.get('maxres', {}) or 
                    thumbnails.get('high', {}) or 
                    thumbnails.get('medium', {}) or 
                    thumbnails.get('default', {})
                ).get('url', '')
                
                # Format duration (e.g., PT5M30S -> 5:30)
                duration = 'N/A'
                if 'contentDetails' in details:
                    duration_iso = details['contentDetails'].get('duration', '')
                    # Convert ISO 8601 duration to readable format
                    import re
                    match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration_iso)
                    if match:
                        hours, minutes, seconds = match.groups()
                        duration_parts = []
                        if hours:
                            duration_parts.append(f"{int(hours[:-1]):02d}")
                        if minutes:
                            duration_parts.append(f"{int(minutes[:-1]):02d}")
                        if seconds:
                            duration_parts.append(f"{int(seconds[:-1]):02d}")
                        duration = ':'.join(duration_parts)
                
                # Get view count if available
                view_count = int(details.get('statistics', {}).get('viewCount', 0))
                view_count_str = f"{view_count:,} views" if view_count > 0 else ""
                
                results.append({
                    'title': item['snippet']['title'],
                    'channel': item['snippet']['channelTitle'],
                    'url': f"https://youtube.com/watch?v={video_id}",
                    'thumbnail': thumbnail_url,
                    'duration': duration,
                    'views': view_count_str,
                    'description': item['snippet'].get('description', '')[:150] + '...'  # Shorter description
                })
                
            except Exception as e:
                print(f"⚠️ Error processing video result: {e}")
                continue
            
        return results
        
    except Exception as e:
        print(f"❌ Error searching YouTube: {e}")
        import traceback
        traceback.print_exc()
        return []


def generate_answer_with_gemini(query: str, context: str, google_api_key: str = None) -> str:
    """
    Generate an answer to a query based on the provided context using Google's Gemini.
    
    Args:
        query: The user's question
        context: The relevant context from PDFs to base the answer on
        google_api_key: Your Google API key for Gemini
        
    Returns:
        str: Generated answer or error message
    """
    try:
        import google.generativeai as genai
        
        if not google_api_key:
            return "Error: Google API key is required for this feature."
            
        try:
            # Configure the API key with the correct API version
            genai.configure(api_key=google_api_key)
            
            # List available models to debug
            # available_models = [m.name for m in genai.list_models()]
            # print(f"Available models: {available_models}")
            
            # Use the specified model
            model_name = 'gemini-1.5-flash'  # Using the specified model
            
            # Initialize the model with safety settings
            model = genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                },
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )
            
            # Create the prompt
            prompt = f"""
            You are an expert Teacher and can answer any question related to the given context. You also give a detailed answer trailored to provide a clear understanding of the concept and 
            you also provide a step by step solution to the question if needed.
            Context: {context}
            
            Question: {query}
            
            Answer:"""
            
            # Generate the response with error handling
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt)
            
            # Extract the text from the response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts'):
                return ' '.join(part.text for part in response.parts if hasattr(part, 'text'))
            elif hasattr(response, 'candidates') and response.candidates:
                return response.candidates[0].content.parts[0].text
            else:
                return "Error: Could not extract response from Gemini API"
                
        except Exception as e:
            # Try a different model if the first one fails
            try:
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(f"Question: {query}\n\nContext: {context}\n\nAnswer:")
                if hasattr(response, 'text'):
                    return response.text
                return str(response)
            except Exception as e2:
                return f"Error generating answer (tried multiple models): {str(e2)}"
        
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def get_relevant_images(query: str, doc_index: int, threshold: float = 0.3) -> List[str]:
    """Find images relevant to the query in a document."""
    try:
        # Get document data
        doc_data = documents[doc_index]
        if not isinstance(doc_data, dict) or 'images' not in doc_data or not doc_data['images']:
            return []
            
        # Encode query
        query_embedding = model.encode([query])
        
        # Find relevant images based on their extracted text
        relevant_images = []
        for img in doc_data['images']:
            if not img.get('text'):
                continue
                
            # Encode image text and compare with query
            img_embedding = model.encode([img['text']])
            similarity = util.pytorch_cos_sim(query_embedding, img_embedding).item()
            
            if similarity >= threshold:
                relevant_images.append({
                    'path': img['path'],
                    'page': img['page'],
                    'similarity': similarity
                })
        
        # Sort by relevance and return top 3 image paths
        relevant_images.sort(key=lambda x: x['similarity'], reverse=True)
        return [img['path'] for img in relevant_images[:3]]
        
    except Exception as e:
        print(f"Error finding relevant images: {e}")
        return []

def process_query(
    query: str, 
    k: int = 3, 
    use_gemini: bool = False, 
    google_api_key: str = None,
    include_youtube: bool = True,
    chat_id: str = None
) -> Tuple[str, List[str], List[Dict[str, str]]]:
    """
    Process a query and return the most relevant document chunks or a generated answer.
    
    Args:
        query: The user's question
        k: Number of relevant chunks to retrieve
        use_gemini: Whether to use Google's Gemini for answer generation
        google_api_key: Required if use_gemini is True
        include_youtube: Whether to include YouTube video results
        chat_id: Optional chat ID to filter documents by user
        
    Returns:
        Tuple of (response_text, relevant_images, youtube_videos)
    """
    try:
        global index, documents, index_path, metadata_path
        
        # Load the appropriate index based on chat_id
        if chat_id:
            user_vector_dir = os.path.join(VECTOR_STORE_DIR, str(chat_id))
            current_index_path = os.path.join(user_vector_dir, "faiss_index.bin")
            current_metadata_path = os.path.join(user_vector_dir, "metadata.pkl")
            
            if os.path.exists(current_index_path) and os.path.exists(current_metadata_path):
                index = faiss.read_index(current_index_path)
                with open(current_metadata_path, 'rb') as f:
                    documents = pickle.load(f)
        
        # Generate query embedding
        query_embedding = model.encode([query])
        
        # Search in FAISS index
        D, I = index.search(query_embedding, k)
        
        # Get the most relevant documents and their indices
        results = []
        result_indices = []
        for idx in I[0]:
            if 0 <= idx < len(documents):
                doc = documents[idx]
                # Skip documents that don't belong to this chat (if chat_id is provided)
                if chat_id and isinstance(doc, dict) and doc.get('chat_id') != chat_id:
                    continue
                    
                if isinstance(doc, dict):
                    results.append(doc.get('text', ''))
                else:
                    results.append(str(doc))
                result_indices.append(idx)
        
        if not results:
            return "No relevant information found in your documents. Try uploading some PDFs first!", [], []
        
        # Get relevant images from the top document
        relevant_images = []
        youtube_videos = []
        
        if result_indices and isinstance(documents[result_indices[0]], dict):
            relevant_images = get_relevant_images(query, result_indices[0])
        
        # Combine all relevant context
        context = "\n\n".join(results)
        
        # Get YouTube videos if requested and API key is available
        if include_youtube and google_api_key and any(keyword in query.lower() for keyword in ['video', 'youtube', 'watch', 'tutorial']):
            youtube_videos = search_youtube_videos(query, max_results=3, api_key=google_api_key)
        
        # Use Gemini for answer generation if requested
        if use_gemini and google_api_key:
            answer = generate_answer_with_gemini(query, context, google_api_key)
        else:
            # Fall back to simple formatting if Gemini is not used
            answer = format_response(results, query)
        
        # Add YouTube section to answer if videos were found
        if youtube_videos:
            answer += "\n\n📺 **Relevant Videos:**\n"
            for i, video in enumerate(youtube_videos, 1):
                answer += f"{i}. [{video['title']}]({video['url']}) - {video['channel']}\n"
        
        return answer, relevant_images, youtube_videos
        
    except Exception as e:
        return f"❌ Error processing your query: {str(e)}", [], []