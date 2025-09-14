#!/usr/bin/env python3
"""
Results Visualization Script
Creates comprehensive visualizations of benchmark results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ResultsVisualizer:
    """Creates visualizations from benchmark results"""
    
    def __init__(self, results_file: str = "results.csv"):
        self.results_file = results_file
        self.df = None
        self.load_results()
    
    def load_results(self):
        """Load results from CSV file"""
        try:
            self.df = pd.read_csv(self.results_file)
            logger.info(f"Loaded {len(self.df)} benchmark results")
            logger.info(f"Databases: {self.df['database'].unique()}")
            logger.info(f"Operations: {self.df['operation'].unique()}")
        except FileNotFoundError:
            logger.error(f"Results file {self.results_file} not found. Run benchmark first.")
            raise
    
    def create_summary_table(self, operation: str, metric: str, databases=None):
        """Create a summary table for specific operation and metric"""
        subset = self.df[(self.df['operation'] == operation) & (self.df['metric'] == metric)]
        if databases:
            subset = subset[subset['database'].isin(databases)]
        
        if 'worker_count' in subset.columns:
            pivot = subset.pivot_table(
                index='worker_count', 
                columns='database', 
                values='value', 
                aggfunc='mean'
            ).round(3)
        else:
            pivot = subset.pivot_table(
                index='database', 
                values='value', 
                aggfunc='mean'
            ).round(3)
        
        return pivot
    
    def plot_indexing_metrics(self, save_path: str = None):
        """Plot index build time and memory usage"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Index build time
        index_data = self.df[(self.df['operation'] == 'index') & (self.df['metric'] == 'total_time')]
        sns.barplot(data=index_data, x='database', y='value', ax=ax1)
        ax1.set_title('Index Build Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Database')
        ax1.tick_params(axis='x', rotation=45)
        
        # Memory usage
        memory_data = self.df[(self.df['operation'] == 'memory') & (self.df['metric'] == 'usage_mb')]
        sns.barplot(data=memory_data, x='database', y='value', ax=ax2)
        ax2.set_title('Memory Usage', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_xlabel('Database')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Indexing metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_latency_analysis(self, save_path: str = None):
        """Plot latency comparison across different percentiles"""
        latency_metrics = ['p50_ms', 'p95_ms', 'p99_ms']
        latency_data = self.df[(self.df['operation'] == 'latency') & (self.df['worker_count'] == 1)]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(latency_metrics):
            metric_data = latency_data[latency_data['metric'] == metric]
            sns.barplot(data=metric_data, x='database', y='value', ax=axes[i])
            axes[i].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Latency (ms)')
            axes[i].set_xlabel('Database')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Latency analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_throughput_scaling(self, save_path: str = None):
        """Plot throughput scaling with worker count"""
        throughput_data = self.df[(self.df['operation'] == 'throughput') & (self.df['metric'] == 'qps')]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        databases = self.df['database'].unique()
        for db in databases:
            db_throughput = throughput_data[throughput_data['database'] == db]
            workers = db_throughput['worker_count'].values
            qps = db_throughput['value'].values
            
            ax.plot(workers, qps, marker='o', linewidth=2, markersize=8, label=db)
            ax.scatter(workers, qps, s=100, alpha=0.7)
        
        ax.set_xlabel('Number of Workers', fontsize=12)
        ax.set_ylabel('Queries Per Second (QPS)', fontsize=12)
        ax.set_title('Throughput Scaling with Worker Count', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Throughput scaling plot saved to {save_path}")
        
        plt.show()
    
    def plot_quality_metrics(self, save_path: str = None):
        """Plot quality metrics (recall and mAP)"""
        quality_data = self.df[self.df['operation'] == 'quality']
        
        if quality_data.empty:
            logger.warning("No quality metrics data found - creating placeholder plot")
            # Create an empty plot with a message
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.text(0.5, 0.5, 'No quality metrics data available\n(Ground truth computation was disabled)', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Quality Metrics (Not Available)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Empty quality metrics plot saved to {save_path}")
            
            plt.show()
            return
        
        # Separate recall and mAP data
        recall_data = quality_data[quality_data['metric'].str.startswith('recall')]
        map_data = quality_data[quality_data['metric'].str.startswith('mAP')]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Recall comparison
        if not recall_data.empty:
            recall_pivot = recall_data.pivot_table(
                index='metric', 
                columns='database', 
                values='value'
            )
            
            recall_pivot.plot(kind='bar', ax=ax1, width=0.8)
            ax1.set_title('Recall@K Comparison', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Recall')
            ax1.set_xlabel('K Value')
            ax1.legend(title='Database')
            ax1.tick_params(axis='x', rotation=0)
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'No Recall@K data', ha='center', va='center', fontsize=12)
            ax1.set_title('Recall@K (No Data)', fontsize=14, fontweight='bold')
            ax1.axis('off')
        
        # mAP comparison
        if not map_data.empty:
            map_pivot = map_data.pivot_table(
                index='metric', 
                columns='database', 
                values='value'
            )
            
            map_pivot.plot(kind='bar', ax=ax2, width=0.8)
            ax2.set_title('mAP@K Comparison', fontsize=14, fontweight='bold')
            ax2.set_ylabel('mAP')
            ax2.set_xlabel('K Value')
            ax2.legend(title='Database')
            ax2.tick_params(axis='x', rotation=0)
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No mAP@K data', ha='center', va='center', fontsize=12)
            ax2.set_title('mAP@K (No Data)', fontsize=14, fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Quality metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_latency_vs_quality(self, save_path: str = None):
        """Plot latency vs quality trade-off"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get single-threaded latency and recall@10 data
        latency_p50 = self.df[(self.df['operation'] == 'latency') & 
                             (self.df['metric'] == 'p50_ms') & 
                             (self.df['worker_count'] == 1)]
        recall_10 = self.df[(self.df['operation'] == 'quality') & 
                           (self.df['metric'] == 'recall@10')]
        
        if latency_p50.empty or recall_10.empty:
            logger.warning("No latency or quality data available for latency vs quality plot")
            ax.text(0.5, 0.5, 'No quality metrics data available\n(Ground truth computation was disabled)', 
                   ha='center', va='center', fontsize=14, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Latency vs Quality Trade-off (Not Available)', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Empty latency vs quality plot saved to {save_path}")
            
            plt.show()
            return
        
        databases = self.df['database'].unique()
        for db in databases:
            db_latency_data = latency_p50[latency_p50['database'] == db]
            db_recall_data = recall_10[recall_10['database'] == db]
            
            if not db_latency_data.empty and not db_recall_data.empty:
                db_latency = db_latency_data['value'].iloc[0]
                db_recall = db_recall_data['value'].iloc[0]
                
                ax.scatter(db_latency, db_recall, s=200, alpha=0.7, label=db)
                ax.annotate(db, (db_latency, db_recall), xytext=(5, 5), 
                           textcoords='offset points', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('P50 Latency (ms)', fontsize=12)
        ax.set_ylabel('Recall@10', fontsize=12)
        ax.set_title('Latency vs Quality Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add quadrant lines
        ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='High Quality Threshold')
        ax.axvline(x=10, color='blue', linestyle='--', alpha=0.5, label='Low Latency Threshold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Latency vs quality plot saved to {save_path}")
        
        plt.show()
    
    def create_performance_heatmap(self, save_path: str = None):
        """Create a comprehensive performance heatmap"""
        databases = self.df['database'].unique()
        
        metrics = {
            'Index Time (s)': ('index', 'total_time'),
            'Memory (MB)': ('memory', 'usage_mb'),
            'P50 Latency (ms)': ('latency', 'p50_ms'),
            'P95 Latency (ms)': ('latency', 'p95_ms'),
            'P99 Latency (ms)': ('latency', 'p99_ms'),
            'Max QPS': ('throughput', 'qps'),
            'Recall@1': ('quality', 'recall@1'),
            'Recall@10': ('quality', 'recall@10'),
            'mAP@10': ('quality', 'mAP@10')
        }
        
        matrix_data = []
        for metric_name, (op, met) in metrics.items():
            row = {'Metric': metric_name}
            
            if op == 'throughput':
                # Get max QPS across all worker counts
                data = self.df[(self.df['operation'] == op) & (self.df['metric'] == met)]
                for db in databases:
                    db_data = data[data['database'] == db]
                    row[db] = db_data['value'].max() if not db_data.empty else 0
            elif op == 'latency':
                # Get single-threaded latency
                data = self.df[(self.df['operation'] == op) & 
                              (self.df['metric'] == met) & 
                              (self.df['worker_count'] == 1)]
                for db in databases:
                    db_data = data[data['database'] == db]
                    row[db] = db_data['value'].iloc[0] if not db_data.empty else 0
            else:
                # Get single value per database
                data = self.df[(self.df['operation'] == op) & (self.df['metric'] == met)]
                for db in databases:
                    db_data = data[data['database'] == db]
                    row[db] = db_data['value'].iloc[0] if not db_data.empty else 0
            
            matrix_data.append(row)
        
        performance_matrix = pd.DataFrame(matrix_data)
        
        # Create normalized heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        heatmap_data = performance_matrix.set_index('Metric').T
        
        # For metrics where higher is better, invert the normalization
        higher_better = ['Max QPS', 'Recall@1', 'Recall@10', 'mAP@10']
        for col in heatmap_data.columns:
            if col in higher_better:
                heatmap_data[col] = heatmap_data[col] / heatmap_data[col].max()
            else:
                heatmap_data[col] = 1 - (heatmap_data[col] / heatmap_data[col].max())
        
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5, 
                   fmt='.2f', cbar_kws={'label': 'Normalized Performance'}, ax=ax)
        ax.set_title('Normalized Performance Heatmap\n(Green = Better, Red = Worse)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Database')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance heatmap saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print a comprehensive summary of results"""
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK SUMMARY")
        print("="*80)
        
        # Show basic statistics
        print("\n=== INDEX BUILD TIMES ===")
        print(self.create_summary_table('index', 'total_time'))
        
        print("\n=== MEMORY USAGE (MB) ===")
        print(self.create_summary_table('memory', 'usage_mb'))
        
        print("\n=== SINGLE-THREADED LATENCY (ms) ===")
        latency_1w = self.df[(self.df['operation'] == 'latency') & (self.df['worker_count'] == 1)]
        latency_pivot = latency_1w.pivot_table(
            index='metric', 
            columns='database', 
            values='value'
        ).round(2)
        print(latency_pivot)
        
        print("\n=== THROUGHPUT (QPS) BY WORKER COUNT ===")
        throughput_data = self.df[(self.df['operation'] == 'throughput') & (self.df['metric'] == 'qps')]
        throughput_pivot = throughput_data.pivot_table(
            index='worker_count', 
            columns='database', 
            values='value'
        ).round(1)
        print(throughput_pivot)
        
        print("\n=== QUALITY METRICS ===")
        quality_data = self.df[self.df['operation'] == 'quality']
        recall_data = quality_data[quality_data['metric'].str.startswith('recall')]
        map_data = quality_data[quality_data['metric'].str.startswith('mAP')]
        
        recall_pivot = recall_data.pivot_table(
            index='metric', 
            columns='database', 
            values='value'
        ).round(4)
        print("\nRecall@K:")
        print(recall_pivot)
        
        map_pivot = map_data.pivot_table(
            index='metric', 
            columns='database', 
            values='value'
        ).round(4)
        print("\nmAP@K:")
        print(map_pivot)
    
    def generate_all_plots(self, output_dir: str = "plots"):
        """Generate all plots and save them"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Generating all plots in {output_dir}/")
        
        self.plot_indexing_metrics(f"{output_dir}/indexing_metrics.png")
        self.plot_latency_analysis(f"{output_dir}/latency_analysis.png")
        self.plot_throughput_scaling(f"{output_dir}/throughput_scaling.png")
        self.plot_quality_metrics(f"{output_dir}/quality_metrics.png")
        self.plot_latency_vs_quality(f"{output_dir}/latency_vs_quality.png")
        self.create_performance_heatmap(f"{output_dir}/performance_heatmap.png")
        
        logger.info("All plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument("--results", type=str, default="results.csv", 
                       help="Path to results CSV file")
    parser.add_argument("--output-dir", type=str, default="plots", 
                       help="Directory to save plots")
    parser.add_argument("--show-plots", action="store_true", 
                       help="Display plots interactively")
    parser.add_argument("--summary-only", action="store_true", 
                       help="Only print summary, don't generate plots")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = ResultsVisualizer(args.results)
    
    # Print summary
    visualizer.print_summary()
    
    if not args.summary_only:
        if args.show_plots:
            # Generate, save, and show plots
            logger.info(f"Generating and saving plots to {args.output_dir}/")
            visualizer.generate_all_plots(args.output_dir)
            
            # Also show plots interactively
            logger.info("Displaying plots interactively...")
            visualizer.plot_indexing_metrics()
            visualizer.plot_latency_analysis()
            visualizer.plot_throughput_scaling()
            visualizer.plot_quality_metrics()
            visualizer.plot_latency_vs_quality()
            visualizer.create_performance_heatmap()
        else:
            # Generate and save plots only
            visualizer.generate_all_plots(args.output_dir)


if __name__ == "__main__":
    main()
