#pip install tenacity
import chromadb
from chromadb.config import Settings
import string
import random
import os
from chromadb import Settings, Client
import chromadb.utils.embedding_functions as embedding_functions
import uuid
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import List, Dict, Any, Optional
import requests

class ChromaOperations:
    def __init__(self, host="127.0.0.1", port=8000):
        # Configure custom session with timeout
        self.session = requests.Session()
        self.session.request = lambda method, url, **kwargs: \
            requests.Session.request(self.session, method, url, timeout=60, **kwargs)

        """Initialize ChromaDB client in HTTP mode"""
        self.client = chromadb.HttpClient(
            host=host,
            port=port
        )
        # Initialize OpenAI embeddings function
        # Note: This will use the OpenAI API key from environment variables

        self.embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
                api_key="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",
                model_name="BAAI/bge-m3"
            )
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_collection_name(self) -> str:
        """Generate a 12-character collection name using lowercase letters and numbers"""
        characters = string.ascii_lowercase + string.digits
        return ''.join(random.choice(characters) for _ in range(12))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def create_collection(self, collection_name: str):
        """Create a new collection with retry mechanism"""
        try:
            return self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Optimize for similarity search
            )
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def get_or_create_collection(self, collection_name: str):
        """Get existing collection or create new one with retry mechanism"""
        try:
            try:
                collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                return collection
            except Exception:
                return self.create_collection(collection_name)
        except Exception as e:
            self.logger.error(f"Error getting or creating collection: {e}")
            raise

    def add_documents_batch(self, collection, texts: List[str], metadatas: List[Dict], 
                          ids: List[str], batch_size: int = 3) -> bool:
        """Add documents in smaller batches with improved error handling"""
        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size] if metadatas else None
                batch_ids = ids[i:i + batch_size]
                
                self.logger.info(f"Processing batch {i//batch_size + 1} of {len(texts)//batch_size + 1}")
                
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        collection.add(
                            documents=batch_texts,
                            metadatas=batch_metadatas if metadatas else None,
                            ids=batch_ids
                        )
                        self.logger.info(f"Successfully added batch {i//batch_size + 1}")
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            self.logger.error(f"Failed to add batch after {max_retries} attempts: {e}")
                            return False
                        self.logger.warning(f"Retry {retry_count} for batch {i//batch_size + 1}")
                        time.sleep(2 ** retry_count)  # Exponential backoff
                
                # Add a longer delay between batches
                time.sleep(1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return False

    def add_documents(self, collection_name: str, texts: List[str], 
                     metadatas: Optional[List[Dict]] = None, 
                     ids: Optional[List[str]] = None) -> bool:
        """Add documents with improved error handling and smaller batch processing"""
        try:
            # Input validation
            if not texts:
                self.logger.error("No texts provided")
                return False
                
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]
            
            # Get or create collection
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return False
            
            # Process in smaller batches
            return self.add_documents_batch(
                collection=collection,
                texts=texts,
                metadatas=metadatas,
                ids=ids,
                batch_size=3  # Reduced batch size
            )
            
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            return False

    def rerank_results(self, query: str, results: dict, top_k: int = 3) -> dict:
        """Rerank search results using BAAI/bge-reranker-v2-m3"""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Initialize reranker
            tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
            model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
            
            if torch.cuda.is_available():
                model = model.cuda()
            model.eval()

            # Prepare pairs for reranking
            pairs = []
            original_indices = []
            for idx, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                pairs.append([query, doc])
                original_indices.append(idx)

            # Tokenize and compute scores
            with torch.no_grad():
                inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                scores = model(**inputs).logits.squeeze()
                scores = scores.cpu().numpy() if torch.cuda.is_available() else scores.numpy()

            # Sort results by reranking scores
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            # Reorder results
            reranked_results = {
                'documents': [[results['documents'][0][i] for i in sorted_indices]],
                'metadatas': [[results['metadatas'][0][i] for i in sorted_indices]],
                'distances': [[results['distances'][0][i] for i in sorted_indices]]
            }
            
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Error in reranking: {e}")
            return results  # Return original results if reranking fails

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def query_collection(self, collection_name: str, query_text: str, n_results: int = 3) -> Optional[Dict[str, Any]]:
        """Query the collection with reranking"""
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return None
            
            # Initial retrieval with more results for reranking
            initial_results = collection.query(
                query_texts=[query_text],
                n_results=min(10, n_results * 3),  # Retrieve more candidates for reranking
                include=['documents', 'metadatas', 'distances']
            )
            
            # Apply reranking
            reranked_results = self.rerank_results(query_text, initial_results, top_k=n_results)
            return reranked_results
            
        except Exception as e:
            self.logger.error(f"Error querying collection: {e}")
            return None

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return False

    def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics about a collection"""
        try:
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return None
            
            return {
                "count": collection.count(),
                "name": collection.name,
                "metadata": collection.metadata
            }
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return None
        
    def generate_chunk_id(self) -> str:
        """Generate a 12-character unique chunk ID"""
        characters = string.ascii_lowercase + string.digits
        return ''.join(random.choice(characters) for _ in range(12))

    def process_documents_for_chromadb(self, content, filename, session_id):
        """Process document content for ChromaDB storage with chunk IDs"""
        MAX_CHUNK_SIZE = 500
        words = content.split()
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_data = []
        
        chunk_counter = 0
        for word in words:
            current_size += len(word) + 1
            if current_size > MAX_CHUNK_SIZE:
                chunk_id = self.generate_chunk_id()
                chunk_name = f"chunk_{chunk_counter}"
                chunks.append(" ".join(current_chunk))
                chunk_data.append({
                    'id': chunk_id,
                    'name': chunk_name,
                    'content': " ".join(current_chunk),
                    'metadata': {
                        'source': filename,
                        'chunk': chunk_counter,
                        'chunk_id': chunk_id
                    }
                })
                current_chunk = [word]
                current_size = len(word)
                chunk_counter += 1
            else:
                current_chunk.append(word)
        
        if current_chunk:
            chunk_id = self.generate_chunk_id()
            chunk_name = f"chunk_{chunk_counter}"
            chunks.append(" ".join(current_chunk))
            chunk_data.append({
                'id': chunk_id,
                'name': chunk_name,
                'content': " ".join(current_chunk),
                'metadata': {
                    'source': filename,
                    'chunk': chunk_counter,
                    'chunk_id': chunk_id
                }
            })
        
        return chunk_data
    
    def remove_documents(self, collection_name: str, chunk_ids: list) -> bool:
        """Remove documents from ChromaDB collection"""
        try:
            if not chunk_ids:
                return True
                
            collection = self.get_or_create_collection(collection_name)
            if not collection:
                return False
            
            # Delete in batches to prevent timeout
            BATCH_SIZE = 10
            for i in range(0, len(chunk_ids), BATCH_SIZE):
                batch_ids = chunk_ids[i:i + BATCH_SIZE]
                retry_count = 0
                while retry_count < 3:
                    try:
                        collection.delete(
                            ids=batch_ids
                        )
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == 3:
                            self.logger.error(f"Failed to delete batch after 3 attempts: {e}")
                            return False
                        time.sleep(2 ** retry_count)
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing documents from ChromaDB: {e}")
            return False

    def list_collections(self) -> List[Dict[str, Any]]:
        """Get list of all collections with their stats"""
        try:
            collections = []
            collection_names = self.client.list_collections()
            
            for col in collection_names:
                try:
                    stats = self.get_collection_stats(col.name)
                    if stats:
                        collections.append({
                            'name': col.name,
                            'count': stats['count'],
                            'metadata': stats.get('metadata', {})
                        })
                except Exception as e:
                    self.logger.error(f"Error getting stats for collection {col.name}: {e}")
                    continue
                    
            return collections
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []        