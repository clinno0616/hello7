#pip install qwen-vl-utils
#pip install -U git+https://github.com/huggingface/transformers
#pip install torch==2.1.0 
#pip install accelerate
#pip install optimum
#pip install auto-gptq

import os
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import logging
from typing import Tuple, Optional
from pathlib import Path
import time
import json

class ImageOperations:
    def __init__(self, base_path: str = "uploads"):
        self.base_path = base_path
        self.image_dir = os.path.join(base_path, "images")
        self.model_dir = os.path.join(base_path, "models")
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def _download_model(self):
        """Download and setup Qwen2-VL model"""
        try:
            model_name = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"
            local_model_path = os.path.join(self.model_dir, "qwen2-vl-gptq")
            
            if not os.path.exists(local_model_path):
                self.logger.info("Downloading Qwen2-VL model...")
                
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    min_pixels=256*28*28,  # Optimize for memory usage
                    max_pixels=1280*28*28
                )
                
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                
                # Save model locally
                self.model.save_pretrained(local_model_path)
                self.processor.save_pretrained(local_model_path)
                
                self.logger.info("Model downloaded successfully")
            else:
                self.logger.info("Loading model from local storage...")
                self.processor = AutoProcessor.from_pretrained(
                    local_model_path,
                    min_pixels=256*28*28,
                    max_pixels=1280*28*28
                )
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    local_model_path,
                    torch_dtype="auto",
                    device_map="auto"
                )
                
        except Exception as e:
            self.logger.error(f"Error setting up model: {str(e)}")
            raise
    
    def ensure_model_loaded(self):
        """Ensure model is loaded before use"""
        if self.model is None or self.processor is None:
            self._download_model()
            
    def validate_image(self, image_file) -> bool:
        """Validate image file type and content"""
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = Path(image_file.name).suffix.lower()
        
        if file_ext not in valid_extensions:
            self.logger.warning(f"Invalid file extension: {file_ext}")
            return False
            
        try:
            img = Image.open(image_file)
            img.verify()
            return True
        except Exception as e:
            self.logger.error(f"Invalid image content: {str(e)}")
            return False
            
    def save_image(self, image_file, session_id: int, user_id: int) -> Tuple[Optional[str], Optional[str]]:
        """Save uploaded image file"""
        try:
            if not self.validate_image(image_file):
                raise ValueError("Invalid image file")
                
            file_ext = Path(image_file.name).suffix.lower()
            timestamp = int(time.time())
            filename = f"{session_id}_{user_id}_{timestamp}{file_ext}"
            file_path = os.path.join(self.image_dir, filename)
            
            with open(file_path, 'wb') as f:
                f.write(image_file.getbuffer())
                
            self.logger.info(f"Image saved successfully: {file_path}")
            return file_path, file_ext.lstrip('.')
            
        except Exception as e:
            self.logger.error(f"Error saving image: {str(e)}")
            return None, None

    def process_vision_info(self, messages):
        """Process vision information from messages"""
        image_inputs = []
        video_inputs = []  # We don't handle videos in this implementation
        
        for message in messages:
            if message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "image":
                        if isinstance(content["image"], (str, Path)):
                            image = Image.open(content["image"])
                            image_inputs.append(image)
        
        return image_inputs, video_inputs
            
    def process_image(self, image_path: str, query: str) -> Optional[str]:
        """Process image with Qwen2-VL model"""
        try:
            self.ensure_model_loaded()
            
            try:
                # Load image
                image = Image.open(image_path)
                
                # Prepare messages in the required format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image  # Pass PIL Image object directly
                            },
                            {
                                "type": "text", 
                                "text": query
                            }
                        ],
                    }
                ]
                
                # Prepare input for the model
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                # Process inputs directly with the image object
                inputs = self.processor(
                    text=[text],
                    images=[image],  # Pass image directly
                    return_tensors="pt",
                    padding=True
                )
                
                # Move inputs to device
                inputs = inputs.to(self.device)
                
                # Generate response
                with torch.inference_mode():
                    try:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.8,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id
                        )
                        
                        # Get the full response first
                        full_response = self.processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                        
                        # Clean up the response
                        # Remove input text if present
                        if text in full_response:
                            response = full_response[len(text):].strip()
                        else:
                            response = full_response.strip()
                        
                        return response if response else "No response generated"
                        
                    except Exception as e:
                        self.logger.error(f"Error during generation: {str(e)}")
                        return f"Error generating response: {str(e)}"

            except torch.cuda.OutOfMemoryError:
                self.logger.warning("GPU memory full, trying with reduced parameters...")
                
                # Resize image
                image = image.resize((512, 512))
                
                # Try again with reduced parameters
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image
                            },
                            {
                                "type": "text", 
                                "text": query
                            }
                        ],
                    }
                ]
                
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                
                inputs = self.processor(
                    text=[text],
                    images=[image],
                    return_tensors="pt",
                    padding=True
                )
                
                inputs = inputs.to(self.device)
                
                with torch.inference_mode():
                    try:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=256,  # Reduced tokens
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.8,
                            pad_token_id=self.processor.tokenizer.pad_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id
                        )
                        
                        full_response = self.processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                        
                        if text in full_response:
                            response = full_response[len(text):].strip()
                        else:
                            response = full_response.strip()
                        
                        return response if response else "No response generated"
                        
                    except Exception as e:
                        self.logger.error(f"Error during generation with reduced parameters: {str(e)}")
                        return f"Error generating response: {str(e)}"
                    
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            self.logger.exception("Full traceback:")
            return f"Error processing image: {str(e)}"
            
    def remove_image(self, file_path: str) -> bool:
        """Remove image file from storage"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info(f"Image removed successfully: {file_path}")
                return True
            else:
                self.logger.warning(f"Image file not found: {file_path}")
                return False
        except Exception as e:
            self.logger.error(f"Error removing image: {str(e)}")
            return False
            
    def cleanup_old_images(self, days: int = 7):
        """Clean up images older than specified days"""
        try:
            current_time = time.time()
            for filename in os.listdir(self.image_dir):
                file_path = os.path.join(self.image_dir, filename)
                if os.path.getmtime(file_path) < current_time - (days * 86400):
                    os.remove(file_path)
                    self.logger.info(f"Removed old image: {filename}")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")