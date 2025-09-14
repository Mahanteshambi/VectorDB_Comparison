#!/usr/bin/env python3
"""
Evaluation module for vector database benchmarking with MS MARCO dataset.
Provides ground truth extraction and evaluation metrics.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class MSMarcoEvaluator:
    """Handles MS MARCO dataset evaluation with ground truth extraction and metrics."""
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        """Initialize the evaluator with a sentence transformer model."""
        self.model_name = model_name
        self.model = None
        self.ground_truth = {}
        self.query_embeddings = None
        self.passage_embeddings = None
        self.passage_id_to_text = {}
        self.query_id_to_text = {}
        
    def load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device='cpu')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def extract_ground_truth(self, num_queries: int = 1000, num_passages: int = 100000) -> Dict[str, List[str]]:
        """
        Extract ground truth from MS MARCO dataset.
        
        Args:
            num_queries: Number of queries to extract
            num_passages: Number of passages to extract
            
        Returns:
            Dictionary mapping query text to list of relevant passage texts
        """
        logger.info(f"Extracting ground truth from MS MARCO dataset...")
        logger.info(f"Target: {num_queries} queries, {num_passages} passages")
        
        try:
            # Load MS MARCO dataset
            dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
            
            ground_truth = {}
            passage_texts = []
            passage_ids = []
            query_texts = []
            query_ids = []
            
            # Collect data from dataset
            collected_queries = 0
            collected_passages = 0
            
            logger.info("Processing MS MARCO dataset...")
            for example in tqdm(dataset, desc="Processing MS MARCO"):
                if collected_queries >= num_queries and collected_passages >= num_passages:
                    break
                
                # Extract query
                if collected_queries < num_queries:
                    query_text = example.get('query', '').strip()
                    query_id = example.get('query_id', f'q_{collected_queries}')
                    
                    if query_text and len(query_text) > 5:  # Filter out very short queries
                        query_texts.append(query_text)
                        query_ids.append(query_id)
                        self.query_id_to_text[query_id] = query_text
                        collected_queries += 1
                
                # Extract passages
                passages = example.get('passages', {})
                if isinstance(passages, dict) and 'passage_text' in passages:
                    passage_texts_list = passages['passage_text']
                    is_selected_list = passages.get('is_selected', [])
                    
                    for i, passage_text in enumerate(passage_texts_list):
                        if collected_passages >= num_passages:
                            break
                            
                        if passage_text and len(passage_text.strip()) > 10:  # Filter short passages
                            passage_id = f'p_{collected_passages}'
                            passage_texts.append(passage_text.strip())
                            passage_ids.append(passage_id)
                            self.passage_id_to_text[passage_id] = passage_text.strip()
                            collected_passages += 1
                
                # Build ground truth for current query
                if collected_queries > 0 and collected_passages > 0:
                    current_query = query_texts[-1]
                    current_query_id = query_ids[-1]
                    
                    # Find relevant passages for this query
                    relevant_passages = []
                    if isinstance(passages, dict) and 'passage_text' in passages:
                        passage_texts_list = passages['passage_text']
                        is_selected_list = passages.get('is_selected', [])
                        
                        for i, (passage_text, is_selected) in enumerate(zip(passage_texts_list, is_selected_list)):
                            if is_selected == 1 and passage_text and len(passage_text.strip()) > 10:
                                relevant_passages.append(passage_text.strip())
                    
                    if relevant_passages:
                        ground_truth[current_query] = relevant_passages
            
            # Truncate to exact numbers requested
            query_texts = query_texts[:num_queries]
            passage_texts = passage_texts[:num_passages]
            
            logger.info(f"Extracted {len(query_texts)} queries and {len(passage_texts)} passages")
            logger.info(f"Found ground truth for {len(ground_truth)} queries")
            
            self.ground_truth = ground_truth
            
            return ground_truth
            
        except Exception as e:
            logger.error(f"Failed to extract ground truth: {e}")
            raise
    
    def generate_embeddings(self, query_texts: List[str], passage_texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate embeddings for queries and passages.
        
        Args:
            query_texts: List of query texts
            passage_texts: List of passage texts
            
        Returns:
            Tuple of (query_embeddings, passage_embeddings)
        """
        if self.model is None:
            self.load_model()
        
        logger.info("Generating embeddings...")
        
        # Generate query embeddings
        query_embeddings = self.model.encode(query_texts, show_progress_bar=True, batch_size=32)
        query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Generate passage embeddings
        passage_embeddings = self.model.encode(passage_texts, show_progress_bar=True, batch_size=32)
        passage_embeddings = passage_embeddings / np.linalg.norm(passage_embeddings, axis=1, keepdims=True)
        
        self.query_embeddings = query_embeddings
        self.passage_embeddings = passage_embeddings
        
        logger.info(f"Generated embeddings: {query_embeddings.shape[0]} queries, {passage_embeddings.shape[0]} passages")
        
        return query_embeddings, passage_embeddings
    
    def recall_at_k(self, results: Dict[str, List[str]], ground_truth: Dict[str, List[str]], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            results: Dictionary mapping query to list of retrieved passages
            ground_truth: Dictionary mapping query to list of relevant passages
            k: Number of top results to consider
            
        Returns:
            Recall@K score
        """
        if not results or not ground_truth:
            return 0.0
        
        total_recall = 0.0
        valid_queries = 0
        
        for query, retrieved in results.items():
            if query not in ground_truth:
                continue
                
            relevant = set(ground_truth[query])
            if not relevant:
                continue
            
            # Get top-k retrieved passages
            top_k_retrieved = set(retrieved[:k])
            
            # Calculate recall for this query
            if relevant:
                recall = len(relevant.intersection(top_k_retrieved)) / len(relevant)
                total_recall += recall
                valid_queries += 1
        
        return total_recall / valid_queries if valid_queries > 0 else 0.0
    
    def mean_reciprocal_rank(self, results: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            results: Dictionary mapping query to list of retrieved passages
            ground_truth: Dictionary mapping query to list of relevant passages
            
        Returns:
            MRR score
        """
        if not results or not ground_truth:
            return 0.0
        
        total_reciprocal_rank = 0.0
        valid_queries = 0
        
        for query, retrieved in results.items():
            if query not in ground_truth:
                continue
                
            relevant = set(ground_truth[query])
            if not relevant:
                continue
            
            # Find rank of first relevant document
            for rank, passage in enumerate(retrieved, 1):
                if passage in relevant:
                    total_reciprocal_rank += 1.0 / rank
                    break
            else:
                # No relevant document found
                total_reciprocal_rank += 0.0
            
            valid_queries += 1
        
        return total_reciprocal_rank / valid_queries if valid_queries > 0 else 0.0
    
    def evaluate_retrieval_results(self, results: Dict[str, List[str]], 
                                 ground_truth: Optional[Dict[str, List[str]]] = None) -> Dict[str, float]:
        """
        Evaluate retrieval results against ground truth.
        
        Args:
            results: Dictionary mapping query to list of retrieved passages
            ground_truth: Ground truth dictionary (uses self.ground_truth if None)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if ground_truth is None:
            ground_truth = self.ground_truth
        
        if not ground_truth:
            logger.warning("No ground truth available for evaluation")
            return {}
        
        metrics = {}
        
        # Calculate Recall@K for different K values
        for k in [1, 5, 10, 20]:
            recall = self.recall_at_k(results, ground_truth, k)
            metrics[f'recall@{k}'] = recall
            logger.info(f"Recall@{k}: {recall:.4f}")
        
        # Calculate MRR
        mrr = self.mean_reciprocal_rank(results, ground_truth)
        metrics['mrr'] = mrr
        logger.info(f"MRR: {mrr:.4f}")
        
        return metrics


def extract_ground_truth(dataset, num_queries: int = 1000) -> Dict[str, List[str]]:
    """
    Standalone function to extract ground truth from MS MARCO dataset.
    
    Args:
        dataset: MS MARCO dataset
        num_queries: Number of queries to extract
        
    Returns:
        Dictionary mapping query text to list of relevant passage texts
    """
    evaluator = MSMarcoEvaluator()
    return evaluator.extract_ground_truth(num_queries)


def recall_at_k(results: Dict[str, List[str]], ground_truth: Dict[str, List[str]], k: int) -> float:
    """
    Standalone function to calculate Recall@K.
    
    Args:
        results: Dictionary mapping query to list of retrieved passages
        ground_truth: Dictionary mapping query to list of relevant passages
        k: Number of top results to consider
        
    Returns:
        Recall@K score
    """
    evaluator = MSMarcoEvaluator()
    return evaluator.recall_at_k(results, ground_truth, k)


def mean_reciprocal_rank(results: Dict[str, List[str]], ground_truth: Dict[str, List[str]]) -> float:
    """
    Standalone function to calculate Mean Reciprocal Rank.
    
    Args:
        results: Dictionary mapping query to list of retrieved passages
        ground_truth: Dictionary mapping query to list of relevant passages
        
    Returns:
        MRR score
    """
    evaluator = MSMarcoEvaluator()
    return evaluator.mean_reciprocal_rank(results, ground_truth)
