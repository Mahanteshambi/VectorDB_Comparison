"""
Metrics Module
Handles performance and quality metric calculations.
"""

import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    database: str
    operation: str
    metric: str
    value: float
    worker_count: int = 1
    query_id: int = None


class PerformanceMetrics:
    """Calculates performance metrics from benchmark data"""
    
    @staticmethod
    def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
        """Calculate latency statistics"""
        if not latencies:
            return {}
        
        latencies_ms = np.array(latencies) * 1000  # Convert to milliseconds
        
        return {
            "p50_ms": np.percentile(latencies_ms, 50),
            "p95_ms": np.percentile(latencies_ms, 95),
            "p99_ms": np.percentile(latencies_ms, 99),
            "mean_ms": np.mean(latencies_ms),
            "std_ms": np.std(latencies_ms),
            "min_ms": np.min(latencies_ms),
            "max_ms": np.max(latencies_ms)
        }
    
    @staticmethod
    def calculate_throughput(latencies: List[float]) -> float:
        """Calculate queries per second"""
        if not latencies:
            return 0.0
        
        total_time = np.sum(latencies)
        return len(latencies) / total_time if total_time > 0 else 0.0
    
    @staticmethod
    def calculate_scaling_efficiency(single_worker_qps: float, 
                                   multi_worker_qps: float, 
                                   worker_count: int) -> float:
        """Calculate scaling efficiency as percentage"""
        if single_worker_qps == 0:
            return 0.0
        
        theoretical_max = single_worker_qps * worker_count
        return (multi_worker_qps / theoretical_max) * 100 if theoretical_max > 0 else 0.0


class QualityMetrics:
    """Calculates quality metrics for search results"""
    
    @staticmethod
    def calculate_recall(predicted: List[List[int]], 
                        ground_truth: np.ndarray, 
                        k: int) -> float:
        """Calculate recall@k"""
        if not predicted or len(predicted) == 0:
            return 0.0
        
        recalls = []
        for pred, gt in zip(predicted, ground_truth):
            if len(pred) > 0 and len(gt) > 0:
                recall = len(set(pred) & set(gt[:k])) / min(len(pred), k)
                recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    @staticmethod
    def calculate_precision(predicted: List[List[int]], 
                           ground_truth: np.ndarray, 
                           k: int) -> float:
        """Calculate precision@k"""
        if not predicted or len(predicted) == 0:
            return 0.0
        
        precisions = []
        for pred, gt in zip(predicted, ground_truth):
            if len(pred) > 0 and len(gt) > 0:
                precision = len(set(pred) & set(gt[:k])) / len(pred)
                precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    @staticmethod
    def calculate_map(predicted: List[List[int]], 
                     ground_truth: np.ndarray, 
                     k: int) -> float:
        """Calculate mAP@k (mean Average Precision)"""
        if not predicted or len(predicted) == 0:
            return 0.0
        
        maps = []
        for pred, gt in zip(predicted, ground_truth):
            if len(pred) > 0 and len(gt) > 0:
                ap = 0.0
                relevant_count = 0
                for i, doc_id in enumerate(pred[:k]):
                    if doc_id in gt[:k]:
                        relevant_count += 1
                        ap += relevant_count / (i + 1)
                if relevant_count > 0:
                    ap /= min(len(pred), k)
                maps.append(ap)
        
        return np.mean(maps) if maps else 0.0
    
    @staticmethod
    def calculate_ndcg(predicted: List[List[int]], 
                      ground_truth: np.ndarray, 
                      k: int) -> float:
        """Calculate NDCG@k (Normalized Discounted Cumulative Gain)"""
        if not predicted or len(predicted) == 0:
            return 0.0
        
        ndcgs = []
        for pred, gt in zip(predicted, ground_truth):
            if len(pred) > 0 and len(gt) > 0:
                # Calculate DCG
                dcg = 0.0
                for i, doc_id in enumerate(pred[:k]):
                    if doc_id in gt[:k]:
                        dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
                # Calculate IDCG (ideal DCG)
                idcg = 0.0
                for i in range(min(len(gt), k)):
                    idcg += 1.0 / np.log2(i + 2)
                
                # Calculate NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0


class MetricsCalculator:
    """Main class for calculating all metrics"""
    
    def __init__(self):
        self.performance_metrics = PerformanceMetrics()
        self.quality_metrics = QualityMetrics()
        self.results = []
    
    def calculate_performance_metrics(self, database: str, latencies: List[float], 
                                    worker_count: int = 1) -> List[BenchmarkResult]:
        """Calculate performance metrics for a database"""
        results = []
        
        # Latency statistics
        latency_stats = self.performance_metrics.calculate_latency_stats(latencies)
        for metric, value in latency_stats.items():
            results.append(BenchmarkResult(
                database=database,
                operation="latency",
                metric=metric,
                value=value,
                worker_count=worker_count
            ))
        
        # Throughput
        qps = self.performance_metrics.calculate_throughput(latencies)
        results.append(BenchmarkResult(
            database=database,
            operation="throughput",
            metric="qps",
            value=qps,
            worker_count=worker_count
        ))
        
        return results
    
    def calculate_quality_metrics(self, database: str, predicted: List[List[int]], 
                                ground_truth: np.ndarray, k_values: List[int] = [1, 5, 10]) -> List[BenchmarkResult]:
        """Calculate quality metrics for a database"""
        results = []
        
        for k in k_values:
            # Recall
            recall = self.quality_metrics.calculate_recall(predicted, ground_truth, k)
            results.append(BenchmarkResult(
                database=database,
                operation="quality",
                metric=f"recall@{k}",
                value=recall
            ))
            
            # Precision
            precision = self.quality_metrics.calculate_precision(predicted, ground_truth, k)
            results.append(BenchmarkResult(
                database=database,
                operation="quality",
                metric=f"precision@{k}",
                value=precision
            ))
            
            # mAP
            map_score = self.quality_metrics.calculate_map(predicted, ground_truth, k)
            results.append(BenchmarkResult(
                database=database,
                operation="quality",
                metric=f"mAP@{k}",
                value=map_score
            ))
            
            # NDCG
            ndcg = self.quality_metrics.calculate_ndcg(predicted, ground_truth, k)
            results.append(BenchmarkResult(
                database=database,
                operation="quality",
                metric=f"NDCG@{k}",
                value=ndcg
            ))
        
        return results
    
    def calculate_indexing_metrics(self, database: str, insert_time: float, 
                                 index_time: float, memory_usage: float = None) -> List[BenchmarkResult]:
        """Calculate indexing metrics for a database"""
        results = []
        
        # Insert time
        results.append(BenchmarkResult(
            database=database,
            operation="index",
            metric="insert_time",
            value=insert_time
        ))
        
        # Index build time
        results.append(BenchmarkResult(
            database=database,
            operation="index",
            metric="index_time",
            value=index_time
        ))
        
        # Total time
        results.append(BenchmarkResult(
            database=database,
            operation="index",
            metric="total_time",
            value=insert_time + index_time
        ))
        
        # Memory usage
        if memory_usage is not None:
            results.append(BenchmarkResult(
                database=database,
                operation="memory",
                metric="usage_mb",
                value=memory_usage
            ))
        
        return results
    
    def add_results(self, results: List[BenchmarkResult]):
        """Add results to the calculator"""
        self.results.extend(results)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all results as a DataFrame"""
        # Handle both dict and object results
        data = []
        for result in self.results:
            if isinstance(result, dict):
                data.append(result)
            else:
                data.append(result.__dict__)
        return pd.DataFrame(data)
    
    def save_results(self, filename: str = "results.csv"):
        """Save results to CSV file"""
        df = self.get_results_dataframe()
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename} ({len(df)} records)")
    
    def print_summary(self):
        """Print a summary of the benchmark results"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        df = self.get_results_dataframe()
        
        for db in df['database'].unique():
            print(f"\n{db.upper()}:")
            db_results = df[df['database'] == db]
            
            # Index build time
            index_time = db_results[(db_results['operation'] == 'index') & 
                                  (db_results['metric'] == 'total_time')]['value'].iloc[0]
            print(f"  Index Build Time: {index_time:.2f}s")
            
            # Memory usage
            memory_data = db_results[(db_results['operation'] == 'memory') & 
                                   (db_results['metric'] == 'usage_mb')]
            if not memory_data.empty:
                memory = memory_data['value'].iloc[0]
                print(f"  Memory Usage: {memory:.1f} MB")
            else:
                print(f"  Memory Usage: N/A")
            
            # Latency (single worker)
            single_worker = db_results[db_results['worker_count'] == 1]
            latency_data = single_worker[single_worker['operation'] == 'latency']
            
            if not latency_data.empty:
                p50_data = latency_data[latency_data['metric'] == 'p50_ms']
                p95_data = latency_data[latency_data['metric'] == 'p95_ms']
                p99_data = latency_data[latency_data['metric'] == 'p99_ms']
                
                if not p50_data.empty and not p95_data.empty and not p99_data.empty:
                    p50 = p50_data['value'].iloc[0]
                    p95 = p95_data['value'].iloc[0]
                    p99 = p99_data['value'].iloc[0]
                    print(f"  Latency (1 worker): P50={p50:.1f}ms, P95={p95:.1f}ms, P99={p99:.1f}ms")
                else:
                    print(f"  Latency (1 worker): N/A")
            else:
                print(f"  Latency (1 worker): N/A")
            
            # Throughput (max workers)
            max_workers = db_results[db_results['worker_count'] == db_results['worker_count'].max()]
            if not max_workers.empty:
                qps = max_workers[(max_workers['operation'] == 'throughput') & 
                                (max_workers['metric'] == 'qps')]['value'].iloc[0]
                print(f"  Max Throughput: {qps:.1f} QPS")
            
            # Quality metrics
            quality = db_results[db_results['operation'] == 'quality']
            if not quality.empty:
                recall_1_data = quality[quality['metric'] == 'recall@1']
                recall_10_data = quality[quality['metric'] == 'recall@10']
                map_10_data = quality[quality['metric'] == 'mAP@10']
                
                if not recall_1_data.empty and not recall_10_data.empty and not map_10_data.empty:
                    recall_1 = recall_1_data['value'].iloc[0]
                    recall_10 = recall_10_data['value'].iloc[0]
                    map_10 = map_10_data['value'].iloc[0]
                    print(f"  Quality: Recall@1={recall_1:.3f}, Recall@10={recall_10:.3f}, mAP@10={map_10:.3f}")
                else:
                    print(f"  Quality: N/A")
            else:
                print(f"  Quality: N/A")
            
            # MS MARCO evaluation metrics
            evaluation = db_results[db_results['operation'] == 'evaluation']
            if not evaluation.empty:
                print(f"  MS MARCO Evaluation:")
                for metric in ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'mrr']:
                    metric_data = evaluation[evaluation['metric'] == metric]
                    if not metric_data.empty:
                        value = metric_data['value'].iloc[0]
                        print(f"    {metric.upper()}: {value:.4f}")
                    else:
                        print(f"    {metric.upper()}: N/A")
            else:
                print(f"  MS MARCO Evaluation: N/A")
