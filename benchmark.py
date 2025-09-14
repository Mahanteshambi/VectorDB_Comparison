#!/usr/bin/env python3
"""
Vector Database Benchmark Suite
Compares Postgres+pgvector, Weaviate, and Qdrant on vector similarity search.
"""

import logging
import argparse
from typing import List, Dict, Tuple, Any

import numpy as np

# Import our modular components
from vdb_comparison.data_management import VectorDataManager, DatabaseIndexer
from vdb_comparison.search_engines import SearchEngineManager
from vdb_comparison.metrics import MetricsCalculator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorDatabaseBenchmark:
    """Main benchmark class for comparing vector databases"""
    
    def __init__(self, 
                 num_vectors: int = 100000,
                 num_queries: int = 1000,
                 vector_dim: int = 1024,
                 model_name: str = "intfloat/e5-large-v2",
                 use_ms_marco: bool = True):
        self.num_vectors = num_vectors
        self.num_queries = num_queries
        self.vector_dim = vector_dim
        self.model_name = model_name
        self.use_ms_marco = use_ms_marco
        
        # Initialize modular components
        self.data_manager = VectorDataManager(model_name, vector_dim, use_ms_marco)
        self.indexer = DatabaseIndexer()
        self.search_manager = SearchEngineManager()
        self.metrics_calculator = MetricsCalculator()
        
    def run_benchmark(self):
        """Run the complete benchmark suite"""
        logger.info("Starting vector database benchmark...")
        
        # Load model and generate embeddings
        self.data_manager.load_model()
        embeddings, query_embeddings = self.data_manager.generate_embeddings(
            self.num_vectors, self.num_queries
        )
        
        # Compute ground truth using FAISS
        logger.info("Computing ground truth with FAISS...")
        ground_truth = self.data_manager.compute_ground_truth(k=10)
        
        # Connect to databases and index data
        self.indexer.connect_databases()
        
        # Index data in all databases
        databases = ["postgres", "weaviate", "qdrant"]
        indexing_results = {}
        
        for db in databases:
            logger.info(f"Indexing data in {db}...")
            if db == "postgres":
                result = self.indexer.index_postgres(embeddings)
            elif db == "weaviate":
                result = self.indexer.index_weaviate(embeddings)
            elif db == "qdrant":
                result = self.indexer.index_qdrant(embeddings)
            
            indexing_results[db] = result
            
            # Record indexing metrics
            metrics = self.metrics_calculator.calculate_indexing_metrics(
                db, result["insert_time"], result["index_time"]
            )
            self.metrics_calculator.add_results(metrics)
        
        # Get memory usage
        memory_stats = self.indexer.get_memory_usage()
        for db_name, mem_usage in memory_stats.items():
            if 'postgres' in db_name.lower():
                db = "postgres"
            elif 'weaviate' in db_name.lower():
                db = "weaviate"
            elif 'qdrant' in db_name.lower():
                db = "qdrant"
            else:
                continue
                
            memory_metrics = self.metrics_calculator.calculate_indexing_metrics(
                db, 0, 0, mem_usage
            )
            self.metrics_calculator.add_results(memory_metrics)
        
        # Initialize search engines
        self.search_manager.connect_databases()
        self.search_manager.initialize_engines()
        
        # Benchmark queries with different worker counts
        worker_counts = [1, 4, 8, 16]
        
        for workers in worker_counts:
            logger.info(f"Running benchmarks with {workers} workers...")
            
            # Benchmark all engines
            search_results = self.search_manager.benchmark_all_engines(
                query_embeddings, k=10, num_workers=workers
            )
            
            # Calculate performance metrics for each engine
            for engine_name, (latencies, search_results_list) in search_results.items():
                if latencies:  # Only if we have results
                    perf_metrics = self.metrics_calculator.calculate_performance_metrics(
                        engine_name, latencies, workers
                    )
                    self.metrics_calculator.add_results(perf_metrics)
        
        # Calculate quality metrics (using single-threaded results)
        logger.info("Computing quality metrics...")
        single_thread_results = self.search_manager.benchmark_all_engines(
            query_embeddings, k=10, num_workers=1
        )
        
        # Skip quality metrics if no ground truth
        if ground_truth is not None:
            for engine_name, (_, search_results_list) in single_thread_results.items():
                if search_results_list:  # Only if we have results
                    quality_metrics = self.metrics_calculator.calculate_quality_metrics(
                        engine_name, search_results_list, ground_truth
                    )
                    self.metrics_calculator.add_results(quality_metrics)
        else:
            logger.info("Skipping quality metrics computation (no ground truth available)")
        
        # Save results and print summary
        self.metrics_calculator.save_results()
        self.metrics_calculator.print_summary()
        
        # Close connections
        self.indexer.close_connections()
        self.search_manager.close_connections()
        
        logger.info("Benchmark completed!")

def main():
    parser = argparse.ArgumentParser(description="Vector Database Benchmark")
    parser.add_argument("--vectors", type=int, default=100000, help="Number of vectors to index")
    parser.add_argument("--queries", type=int, default=1000, help="Number of query vectors")
    parser.add_argument("--model", type=str, default="intfloat/e5-large-v2", help="Sentence transformer model")
    parser.add_argument("--use-ms-marco", action="store_true", default=True, 
                       help="Use MS MARCO dataset (default: True)")
    parser.add_argument("--use-synthetic", action="store_true", 
                       help="Use synthetic data instead of MS MARCO")
    parser.add_argument("--max-marco-samples", type=int, default=10000,
                       help="Maximum number of MS MARCO samples to load (default: 10000)")
    
    args = parser.parse_args()
    
    # Determine data source
    use_ms_marco = args.use_ms_marco and not args.use_synthetic
    
    # Limit MS MARCO samples if using MS MARCO
    if use_ms_marco:
        args.vectors = min(args.vectors, args.max_marco_samples)
        logger.info(f"Limited to {args.vectors} vectors for MS MARCO dataset")
    
    # Run benchmark
    benchmark = VectorDatabaseBenchmark(
        num_vectors=args.vectors,
        num_queries=args.queries,
        model_name=args.model,
        use_ms_marco=use_ms_marco
    )
    
    benchmark.run_benchmark()

if __name__ == "__main__":
    main()
