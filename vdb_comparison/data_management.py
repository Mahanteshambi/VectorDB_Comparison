"""
Data Management Module
Handles vector generation, ground truth computation, and database indexing.
"""

import json
import time
import logging
from typing import Tuple, List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Database clients
import psycopg2
from psycopg2.extras import RealDictCursor
import weaviate
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class VectorDataManager:
    """Manages vector data generation and ground truth computation"""
    
    def __init__(self, 
                 model_name: str = "intfloat/e5-large-v2",
                 vector_dim: int = 1024,
                 use_ms_marco: bool = True):
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.use_ms_marco = use_ms_marco
        self.model = None
        self.embeddings = None
        self.query_embeddings = None
        self.ground_truth = None
        self.passage_texts = []  # Store passage texts for evaluation
        self.query_texts = []    # Store query texts for evaluation
        
    def load_model(self):
        """Load the sentence transformer model"""
        logger.info(f"Loading model: {self.model_name}")
        try:
            # Force CPU usage for stability
            self.model = SentenceTransformer(self.model_name, device='cpu')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            logger.info("Falling back to random embeddings for testing...")
            self.model = None
    
    def generate_embeddings(self, num_vectors: int, num_queries: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate embeddings for indexing and querying"""
        if self.model is None:
            # Fallback to random embeddings for testing
            logger.info("Using random embeddings for testing...")
            np.random.seed(42)
            embeddings = np.random.randn(num_vectors, self.vector_dim).astype('float32')
            texts = [f"synthetic_text_{i}" for i in range(num_vectors)]
        else:
            if self.use_ms_marco:
                logger.info(f"Generating {num_vectors} embeddings from MS MARCO dataset...")
                texts = self._load_ms_marco_data(num_vectors)
            else:
                logger.info(f"Generating {num_vectors} embeddings from synthetic data...")
                texts = self._generate_synthetic_data(num_vectors)
            
            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        
        # Normalize for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Split into index and query sets
        split_idx = int(0.8 * num_vectors)
        index_embeddings = embeddings[:split_idx]
        
        # Store texts for evaluation
        self.passage_texts = texts[:split_idx]  # Store passage texts
        
        # For MS MARCO, we need to extract actual query texts, not use passage texts as queries
        if self.use_ms_marco:
            self.query_texts = self._extract_query_texts(num_queries)
            # Generate embeddings for the actual query texts
            if self.model is not None:
                query_embeddings = self.model.encode(self.query_texts, show_progress_bar=True, batch_size=32)
                query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
            else:
                # Fallback to random embeddings
                query_embeddings = np.random.randn(num_queries, self.vector_dim).astype('float32')
        else:
            self.query_texts = texts[split_idx:split_idx + num_queries]  # Store query texts
            query_embeddings = embeddings[split_idx:split_idx + num_queries]
        
        self.embeddings = index_embeddings
        self.query_embeddings = query_embeddings
        
        logger.info(f"Generated {len(index_embeddings)} index vectors and {len(query_embeddings)} query vectors")
        
        return index_embeddings, query_embeddings
    
    def _load_ms_marco_data(self, num_vectors: int) -> List[str]:
        """Load MS MARCO dataset for benchmarking"""
        try:
            from datasets import load_dataset
            import time
            logger.info("Loading MS MARCO dataset...")
            
            # Load only a subset of MS MARCO to avoid memory issues
            # Use streaming to load data more efficiently
            dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
            
            # Extract passages efficiently
            texts = []
            from tqdm import tqdm
            
            start_time = time.time()
            timeout = 300  # 5 minute timeout
            
            for i, example in enumerate(tqdm(dataset, desc="Loading MS MARCO", total=num_vectors)):
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout reached after {timeout}s, using {len(texts)} passages")
                    break
                    
                if i >= num_vectors:
                    break
                
                # Debug: print the first few examples to understand structure
                if i < 3:
                    logger.info(f"Example {i}: {list(example.keys())}")
                    logger.info(f"Example {i} content: {example}")
                
                # Extract passages from MS MARCO structure
                passages = example.get('passages', {})
                if isinstance(passages, dict) and 'passage_text' in passages:
                    # MS MARCO has passages['passage_text'] as a list
                    passage_texts = passages['passage_text']
                    if isinstance(passage_texts, list):
                        for passage_text in passage_texts:
                            if passage_text and len(str(passage_text).strip()) > 10:
                                texts.append(str(passage_text).strip())
                                if len(texts) >= num_vectors:
                                    break
                    elif passage_texts and len(str(passage_texts).strip()) > 10:
                        texts.append(str(passage_texts).strip())
                else:
                    # Fallback: try different possible field names for the text content
                    passage = None
                    for field in ['passage', 'text', 'content', 'body', 'document']:
                        if field in example and example[field]:
                            passage = example[field]
                            break
                    
                    if passage and len(str(passage).strip()) > 10:  # Filter out very short passages
                        texts.append(str(passage).strip())
                
                if len(texts) >= num_vectors:
                    break
            
            # If we don't have enough data, cycle through the dataset again (with timeout)
            if len(texts) < num_vectors and time.time() - start_time < timeout:
                logger.info(f"Only found {len(texts)} passages, cycling through dataset...")
                dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
                for example in dataset:
                    if time.time() - start_time > timeout:
                        break
                    if len(texts) >= num_vectors:
                        break
                    passage = example.get('passage', '')
                    if passage and len(passage.strip()) > 10:
                        texts.append(passage.strip())
            
            # Truncate to exact number requested
            texts = texts[:num_vectors]
            
            if len(texts) == 0:
                logger.warning("No valid passages found in MS MARCO dataset, falling back to synthetic data")
                return self._generate_synthetic_data(num_vectors)
            
            logger.info(f"Loaded {len(texts)} passages from MS MARCO dataset")
            return texts
            
        except ImportError:
            logger.warning("datasets library not found. Installing it...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
            
            # Retry after installation
            from datasets import load_dataset
            return self._load_ms_marco_data(num_vectors)
            
        except Exception as e:
            logger.warning(f"Failed to load MS MARCO dataset: {e}")
            logger.info("Falling back to synthetic data...")
            return self._generate_synthetic_data(num_vectors)
    
    def _generate_synthetic_data(self, num_vectors: int) -> List[str]:
        """Generate synthetic text data as fallback"""
        logger.info("Generating synthetic text data...")
        
        # More realistic synthetic data
        topics = [
            "artificial intelligence", "machine learning", "data science", "computer vision",
            "natural language processing", "deep learning", "neural networks", "robotics",
            "automation", "technology", "software engineering", "algorithms", "statistics",
            "mathematics", "physics", "chemistry", "biology", "medicine", "healthcare",
            "finance", "economics", "business", "marketing", "education", "research"
        ]
        
        actions = [
            "analyzing", "processing", "implementing", "developing", "creating", "building",
            "designing", "optimizing", "improving", "enhancing", "studying", "investigating",
            "exploring", "discovering", "understanding", "learning", "teaching", "training"
        ]
        
        objects = [
            "data", "models", "algorithms", "systems", "applications", "solutions",
            "techniques", "methods", "approaches", "strategies", "frameworks", "tools",
            "platforms", "services", "products", "technologies", "concepts", "theories"
        ]
        
        np.random.seed(42)  # For reproducibility
        
        texts = []
        for i in range(num_vectors):
            topic = np.random.choice(topics)
            action = np.random.choice(actions)
            obj = np.random.choice(objects)
            
            # Create more varied and realistic text
            text = f"This document discusses {topic} and its applications in {action} {obj}. "
            text += f"The research focuses on advanced {topic} techniques for {action} complex {obj}. "
            text += f"Recent developments in {topic} have shown significant improvements in {action} {obj}. "
            text += f"The methodology involves {action} large-scale {obj} using state-of-the-art {topic} approaches."
            
            texts.append(text)
        
        return texts
    
    def _extract_query_texts(self, num_queries: int) -> List[str]:
        """Extract actual query texts from MS MARCO dataset"""
        try:
            from datasets import load_dataset
            logger.info("Extracting query texts from MS MARCO dataset...")
            
            dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
            query_texts = []
            
            for example in dataset:
                if len(query_texts) >= num_queries:
                    break
                    
                query_text = example.get('query', '').strip()
                if query_text and len(query_text) > 5:  # Filter out very short queries
                    query_texts.append(query_text)
            
            logger.info(f"Extracted {len(query_texts)} query texts from MS MARCO")
            return query_texts
            
        except Exception as e:
            logger.error(f"Failed to extract query texts: {e}")
            # Fallback to synthetic queries
            return [f"synthetic_query_{i}" for i in range(num_queries)]
    
    def compute_ground_truth(self, k: int = 10) -> np.ndarray:
        """Compute ground truth using numpy (FAISS alternative)"""
        logger.info("Computing ground truth with numpy...")
        
        try:
            # Use numpy for ground truth computation
            return self._compute_ground_truth_numpy(k)
            
        except Exception as e:
            logger.error(f"Error in compute_ground_truth: {e}")
            return None
    
    def _compute_ground_truth_numpy(self, k: int = 10) -> np.ndarray:
        """Compute ground truth using numpy operations"""
        logger.info("Computing ground truth with numpy operations...")
        
        # Ensure data is float32
        embeddings = self.embeddings.astype('float32')
        queries = self.query_embeddings.astype('float32')
        
        # Compute cosine similarity matrix
        # For normalized vectors, cosine similarity = dot product
        similarity_matrix = np.dot(queries, embeddings.T)
        logger.info(f"Computed similarity matrix shape: {similarity_matrix.shape}")
        
        # Get top-k indices for each query
        ground_truth = np.argsort(similarity_matrix, axis=1)[:, -k:][:, ::-1]
        logger.info(f"Computed ground truth shape: {ground_truth.shape}")
        
        self.ground_truth = ground_truth
        logger.info(f"Computed ground truth for {len(self.query_embeddings)} queries")
        
        return ground_truth


class DatabaseIndexer:
    """Handles indexing vectors into different databases"""
    
    def __init__(self):
        self.postgres_conn = None
        self.weaviate_client = None
        self.qdrant_client = None
        
    def connect_databases(self):
        """Establish connections to all databases"""
        logger.info("Connecting to databases...")
        
        # PostgreSQL connection
        try:
            self.postgres_conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="vectordb",
                user="postgres",
                password="postgres"
            )
            logger.info("Connected to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
        
        # Weaviate connection
        try:
            self.weaviate_client = weaviate.Client("http://localhost:8080")
            logger.info("Connected to Weaviate")
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
        
        # Qdrant connection
        try:
            self.qdrant_client = QdrantClient(host="localhost", port=6333)
            logger.info("Connected to Qdrant")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise
    
    def index_postgres(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Index vectors in PostgreSQL with pgvector"""
        logger.info("Indexing vectors in PostgreSQL...")
        
        cursor = self.postgres_conn.cursor()
        
        # Clear existing data
        cursor.execute("TRUNCATE TABLE vectors")
        
        # Insert vectors
        start_time = time.time()
        for i, embedding in enumerate(embeddings):
            cursor.execute(
                "INSERT INTO vectors (vector_id, embedding, metadata) VALUES (%s, %s, %s)",
                (i, embedding.tolist(), json.dumps({"id": i}))
            )
        
        self.postgres_conn.commit()
        insert_time = time.time() - start_time
        
        # Build index
        logger.info("Building PostgreSQL index...")
        start_time = time.time()
        cursor.execute("REINDEX INDEX vectors_embedding_idx")
        self.postgres_conn.commit()
        index_time = time.time() - start_time
        
        cursor.close()
        
        return {
            "insert_time": insert_time,
            "index_time": index_time,
            "total_time": insert_time + index_time
        }
    
    def index_weaviate(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Index vectors in Weaviate"""
        logger.info("Indexing vectors in Weaviate...")
        
        # Clear existing schema
        try:
            self.weaviate_client.schema.delete_all()
        except:
            pass
        
        # Create schema
        schema = {
            "classes": [{
                "class": "Vector",
                "description": "A vector class",
                "vectorizer": "none",
                "properties": [{
                    "name": "vector_id",
                    "dataType": ["int"],
                    "description": "Vector ID"
                }, {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Text content"
                }],
                "vectorIndexConfig": {
                    "distance": "cosine",
                    "ef": 100,
                    "efConstruction": 200,
                    "maxConnections": 32
                }
            }]
        }
        
        self.weaviate_client.schema.create(schema)
        
        # Insert vectors
        start_time = time.time()
        batch_size = 100
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            
            for j, embedding in enumerate(batch):
                self.weaviate_client.data_object.create(
                    data_object={
                        "vector_id": i + j,
                        "text": f"Document {i + j}"
                    },
                    class_name="Vector",
                    vector=embedding.tolist()
                )
        
        insert_time = time.time() - start_time
        
        # Weaviate builds index automatically
        index_time = 0.0
        
        return {
            "insert_time": insert_time,
            "index_time": index_time,
            "total_time": insert_time + index_time
        }
    
    def index_qdrant(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Index vectors in Qdrant"""
        logger.info("Indexing vectors in Qdrant...")
        
        collection_name = "vectors"
        
        # Delete existing collection
        try:
            self.qdrant_client.delete_collection(collection_name)
        except:
            pass
        
        # Create collection
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embeddings.shape[1],
                distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(
                m=32,
                ef_construct=200,
                full_scan_threshold=10000
            )
        )
        
        # Insert vectors
        start_time = time.time()
        batch_size = 100
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size]
            points = []
            
            for j, embedding in enumerate(batch):
                points.append(models.PointStruct(
                    id=i + j,
                    vector=embedding.tolist(),
                    payload={"vector_id": i + j, "metadata": {"id": i + j}}
                ))
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
        
        insert_time = time.time() - start_time
        
        # Qdrant builds index automatically
        index_time = 0.0
        
        return {
            "insert_time": insert_time,
            "index_time": index_time,
            "total_time": insert_time + index_time
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage from Docker containers"""
        import subprocess
        
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", 
                 "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}"],
                capture_output=True, text=True
            )
            
            stats = {}
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        container = parts[0]
                        mem_usage = parts[1].replace('MiB', '').replace('GiB', '000')
                        stats[container] = float(mem_usage)
            
            return stats
        except Exception as e:
            logger.warning(f"Could not get Docker stats: {e}")
            return {}
    
    def close_connections(self):
        """Close all database connections"""
        if self.postgres_conn:
            self.postgres_conn.close()
        # Weaviate and Qdrant clients don't need explicit closing
