import os
import time
import torch
from PIL import Image
import logging
from typing import Optional, Tuple
from pathlib import Path
import sqlite3
import json
import cv2
import base64
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

class VisionOperations:
    def __init__(self, base_path: str = "uploads", groq_api_key: Optional[str] = None):
        self.base_path = base_path
        self.media_dir = os.path.join(base_path, "images")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        # Get Groq API key from environment variables
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            self.logger.warning("GROQ_API_KEY not found in environment variables")
        self.groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
        self._setup_directories()
        self._init_db()

    def _setup_directories(self):
        os.makedirs(self.media_dir, exist_ok=True)

    def _init_db(self):
        """Initialize media database"""
        conn = sqlite3.connect('chat_app.db')
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS media_files (
            media_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
        ''')
        conn.commit()
        conn.close()

    def validate_file(self, file) -> bool:
        """Validate uploaded file type and content"""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.mp4'}
        file_ext = Path(file.name).suffix.lower()
        
        if file_ext not in valid_extensions:
            self.logger.warning(f"Invalid file extension: {file_ext}")
            return False
            
        try:
            if file_ext in {'.jpg', '.jpeg', '.png'}:
                img = Image.open(file)
                img.verify()
            elif file_ext == '.mp4':
                # Read first frame to verify video
                temp_path = os.path.join(self.media_dir, "temp.mp4")
                with open(temp_path, 'wb') as f:
                    f.write(file.getvalue())
                cap = cv2.VideoCapture(temp_path)
                ret = cap.read()
                cap.release()
                os.remove(temp_path)
                if not ret:
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Invalid file content: {str(e)}")
            return False

    def save_media(self, file, session_id: int, user_id: int) -> Tuple[Optional[str], Optional[str]]:
        """Save uploaded media file"""
        try:
            if not self.validate_file(file):
                raise ValueError("Invalid file")
                
            file_ext = Path(file.name).suffix.lower()
            timestamp = int(time.time())
            filename = f"{session_id}_{user_id}_{timestamp}{file_ext}"
            file_path = os.path.join(self.media_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(file.getvalue())

            # Save to database
            conn = sqlite3.connect('chat_app.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO media_files (user_id, session_id, filename, file_path, file_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, session_id, file.name, file_path, file_ext.lstrip('.')))
            conn.commit()
            conn.close()
                
            self.logger.info(f"Media saved successfully: {file_path}")
            return file_path, file_ext.lstrip('.')
            
        except Exception as e:
            self.logger.error(f"Error saving media: {str(e)}")
            return None, None

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process_media(self, file_path: str, query: str) -> Optional[str]:
        """Process media with Groq LLaMA 3.2 vision model"""
        try:
            if not self.groq_client:
                raise ValueError("Groq client not initialized. Please provide API key.")

            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in {'.jpg', '.jpeg', '.png'}:
                return self._process_image(file_path, query)
            elif file_ext == '.mp4':
                return self._process_video(file_path, query)
                
        except Exception as e:
            self.logger.error(f"Error processing media: {str(e)}")
            return f"Error processing media: {str(e)}"

    def _process_image(self, image_path: str, query: str) -> Optional[str]:
        """Process image with Groq LLaMA 3.2 vision"""
        try:
            # Encode image to base64
            base64_image = self._encode_image(image_path)
            
            # Create chat completion with Groq
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                model="llama-3.2-90b-vision-preview"
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            return f"Error processing image: {str(e)}"

    def _process_video(self, video_path: str, query: str) -> Optional[str]:
        """Process video with Groq LLaMA 3.2 vision by extracting key frames"""
        try:
            cap = cv2.VideoCapture(video_path)
            responses = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 30 == 0:  # Process every 30th frame
                    # Save frame temporarily
                    frame_path = os.path.join(self.media_dir, f"temp_frame_{frame_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Process frame
                    response = self._process_image(frame_path, query)
                    if response:
                        responses.append(response)
                    
                    # Clean up
                    os.remove(frame_path)
            
            cap.release()
            
            if responses:
                return "\n\nKey observations:\n" + "\n".join(responses)
            return "No relevant content found in video"
            
        except Exception as e:
            return f"Error processing video: {str(e)}"

    def remove_media(self, file_path: str) -> bool:
        """Remove media file and database entry"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                
                conn = sqlite3.connect('chat_app.db')
                cursor = conn.cursor()
                cursor.execute('DELETE FROM media_files WHERE file_path = ?', (file_path,))
                conn.commit()
                conn.close()
                
                self.logger.info(f"Media removed successfully: {file_path}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing media: {str(e)}")
            return False

    def get_session_media(self, session_id: int) -> list:
        """Get all media files for a session"""
        try:
            conn = sqlite3.connect('chat_app.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT media_id, filename, file_path, file_type, upload_time
                FROM media_files
                WHERE session_id = ?
                ORDER BY upload_time DESC
            ''', (session_id,))
            results = cursor.fetchall()
            conn.close()
            return results
        except Exception as e:
            self.logger.error(f"Error fetching session media: {str(e)}")
            return []