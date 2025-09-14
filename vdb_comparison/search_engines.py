"""
Search Engines Module
Handles query execution across different vector databases.
"""

import time
import logging
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# Database clients
import psycopg2
from psycopg2.extras import RealDictCursor
import weaviate
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Base class for vector search engines"""
    
    def __init__(self, name: str):
        self.name = name
        self.connection = None
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Perform a single search query"""
        raise NotImplementedError
    
    def batch_search(self, query_vectors: np.ndarray, k: int = 10, 
                    num_workers: int = 1) -> Tuple[List[float], List[List[int]]]:
        """Perform batch search with timing"""
        latencies = []
        results = []
        
        if num_workers == 1:
            # Single-threaded
            for query_vector in tqdm(query_vectors, desc=f"{self.name} queries"):
                start_time = time.time()
                result = self.search(query_vector, k)
                latencies.append(time.time() - start_time)
                results.append(result)
        else:
            # Multi-threaded
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.search, query_vector, k) 
                          for query_vector in query_vectors]
                
                for future in tqdm(as_completed(futures), total=len(futures), 
                                 desc=f"{self.name} queries"):
                    start_time = time.time()
                    result = future.result()
                    latencies.append(time.time() - start_time)
                    results.append(result)
        
        return latencies, results


class PostgresSearchEngine(VectorSearchEngine):
    """PostgreSQL with pgvector search engine"""
    
    def __init__(self, connection):
        super().__init__("postgres")
        self.connection = connection
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Search using PostgreSQL with pgvector"""
        try:
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT vector_id, embedding <=> %s::vector as distance FROM vectors ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_vector.tolist(), query_vector.tolist(), k)
            )
            results = cursor.fetchall()
            cursor.close()
            return [r['vector_id'] for r in results]
        except Exception as e:
            # If there's a transaction error, rollback and retry
            self.connection.rollback()
            cursor = self.connection.cursor(cursor_factory=RealDictCursor)
            cursor.execute(
                "SELECT vector_id, embedding <=> %s::vector as distance FROM vectors ORDER BY embedding <=> %s::vector LIMIT %s",
                (query_vector.tolist(), query_vector.tolist(), k)
            )
            results = cursor.fetchall()
            cursor.close()
            return [r['vector_id'] for r in results]


class WeaviateSearchEngine(VectorSearchEngine):
    """Weaviate search engine"""
    
    def __init__(self, client):
        super().__init__("weaviate")
        self.client = client
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Search using Weaviate"""
        result = self.client.query.get(
            "Vector", ["vector_id"]
        ).with_near_vector({
            "vector": query_vector.tolist()
        }).with_limit(k).do()
        
        return [obj['vector_id'] for obj in result['data']['Get']['Vector']]


class QdrantSearchEngine(VectorSearchEngine):
    """Qdrant search engine"""
    
    def __init__(self, client):
        super().__init__("qdrant")
        self.client = client
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[int]:
        """Search using Qdrant"""
        result = self.client.search(
            collection_name="vectors",
            query_vector=query_vector.tolist(),
            limit=k,
            with_payload=True
        )
        return [point.id for point in result]


class SearchEngineManager:
    """Manages multiple search engines"""
    
    def __init__(self):
        self.engines = {}
        self.postgres_conn = None
        self.weaviate_client = None
        self.qdrant_client = None
    
    def connect_databases(self):
        """Establish connections to all databases"""
        logger.info("Connecting to search databases...")
        
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
    
    def initialize_engines(self):
        """Initialize all search engines"""
        self.engines = {
            "postgres": PostgresSearchEngine(self.postgres_conn),
            "weaviate": WeaviateSearchEngine(self.weaviate_client),
            "qdrant": QdrantSearchEngine(self.qdrant_client)
        }
        logger.info(f"Initialized {len(self.engines)} search engines")
    
    def benchmark_engine(self, engine_name: str, query_vectors: np.ndarray, 
                        k: int = 10, num_workers: int = 1) -> Tuple[List[float], List[List[int]]]:
        """Benchmark a specific search engine"""
        if engine_name not in self.engines:
            raise ValueError(f"Unknown engine: {engine_name}")
        
        engine = self.engines[engine_name]
        logger.info(f"Benchmarking {engine_name} with {num_workers} workers...")
        
        return engine.batch_search(query_vectors, k, num_workers)
    
    def benchmark_all_engines(self, query_vectors: np.ndarray, k: int = 10, 
                             num_workers: int = 1) -> Dict[str, Tuple[List[float], List[List[int]]]]:
        """Benchmark all search engines"""
        results = {}
        
        for engine_name in self.engines:
            try:
                latencies, search_results = self.benchmark_engine(
                    engine_name, query_vectors, k, num_workers
                )
                results[engine_name] = (latencies, search_results)
            except Exception as e:
                logger.error(f"Failed to benchmark {engine_name}: {e}")
                results[engine_name] = ([], [])
        
        return results
    
    def close_connections(self):
        """Close all database connections"""
        if self.postgres_conn:
            self.postgres_conn.close()
        # Weaviate and Qdrant clients don't need explicit closing
