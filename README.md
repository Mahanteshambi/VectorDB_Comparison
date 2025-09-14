# Vector Database Comparison Benchmark

A comprehensive benchmark suite comparing **Postgres with pgvector**, **Weaviate**, and **Qdrant** on vector similarity search performance. This project provides a modular, reproducible framework for evaluating vector databases across multiple dimensions including latency, throughput, quality, and resource usage.

## 🚀 Features

- **Modular Architecture**: Clean separation of data management, search engines, and metrics calculation
- **Comprehensive Metrics**: Index build time, memory usage, latency percentiles, throughput scaling, recall, mAP, and NDCG
- **Multi-threaded Testing**: Performance evaluation with 1, 4, 8, and 16 workers
- **Docker Integration**: Easy setup with Docker Compose for all databases
- **Rich Visualizations**: Automated generation of performance charts and heatmaps
- **Reproducible Results**: Pinned versions and deterministic data generation

## 📊 Databases Compared

| Database | Version | Vector Extension | Index Type |
|----------|---------|------------------|------------|
| PostgreSQL | 16 | pgvector | IVFFlat |
| Weaviate | 1.22.4 | Built-in | HNSW |
| Qdrant | 1.7.4 | Built-in | HNSW |

## 🏗️ Project Structure

```
vdb_comparison/
├── benchmark.py              # Main benchmark script
├── data_management.py        # Vector generation and database indexing
├── search_engines.py         # Database search implementations
├── metrics.py               # Performance and quality calculations
├── visualize_results.py     # Results visualization and analysis
├── docker-compose.yml       # Database services configuration
├── init_postgres.sql        # PostgreSQL initialization script
├── requirements.txt         # Python dependencies
├── pyproject.toml          # UV package management
└── README.md               # This file
```

## 🛠️ Setup

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd vdb_comparison
   ```

2. **Install dependencies with UV** (recommended):
   ```bash
   uv sync
   ```

   Or with pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the databases**:
   ```bash
   docker-compose up -d
   ```

   This will start:
   - PostgreSQL with pgvector on port 5432
   - Weaviate on port 8080
   - Qdrant on port 6333

4. **Wait for services to be ready** (about 30-60 seconds):
   ```bash
   docker-compose logs -f
   ```

## 🚀 Usage

### Running the Benchmark

**Basic benchmark** (100k vectors, 1k queries):
```bash
python benchmark.py
```

**Custom parameters**:
```bash
python benchmark.py --vectors 50000 --queries 500 --model intfloat/e5-large-v2
```

**Using MS MARCO dataset (default)**:
```bash
python benchmark.py --vectors 100000 --queries 1000 --use-ms-marco
```

**Using synthetic data**:
```bash
python benchmark.py --vectors 100000 --queries 1000 --use-synthetic
```

**Using UV**:
```bash
uv run benchmark --vectors 200000 --queries 2000
```

### Available Parameters

- `--vectors`: Number of vectors to index (default: 100,000)
- `--queries`: Number of query vectors (default: 1,000)
- `--model`: Sentence transformer model (default: intfloat/e5-large-v2)
- `--use-ms-marco`: Use MS MARCO dataset (default: True)
- `--use-synthetic`: Use synthetic data instead of MS MARCO

### Visualizing Results

**Generate all plots**:
```bash
python visualize_results.py
```

**Show plots interactively**:
```bash
python visualize_results.py --show-plots
```

**Save plots to directory**:
```bash
python visualize_results.py --output-dir my_plots
```

**Print summary only**:
```bash
python visualize_results.py --summary-only
```

**Using UV**:
```bash
uv run visualize --show-plots
```

## 📈 Benchmark Metrics

### Performance Metrics
- **Index Build Time**: Time to insert and index vectors
- **Memory Usage**: RAM consumption during operation
- **Latency**: P50, P95, P99 query response times
- **Throughput**: Queries per second (QPS) with different worker counts
- **Scaling Efficiency**: How well databases scale with multiple workers

### Quality Metrics
- **Recall@K**: Fraction of relevant results found in top-K
- **Precision@K**: Fraction of top-K results that are relevant
- **mAP@K**: Mean Average Precision at K
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

### Test Configuration
- **Dataset**: MS MARCO v1.1 (real-world text passages) or synthetic data
- **Vector Dimension**: 1024 (e5-large-v2 embeddings)
- **Similarity**: Cosine similarity
- **HNSW Parameters**: M=32, ef_construction=200, ef=100
- **Worker Counts**: 1, 4, 8, 16
- **K Values**: 1, 5, 10 for quality metrics

## 📊 Output

The benchmark generates:

1. **results.csv**: Detailed metrics for all databases and configurations
2. **Console Summary**: Key performance indicators printed to terminal
3. **Plots** (if using visualization):
   - Indexing metrics (build time, memory usage)
   - Latency analysis (P50, P95, P99)
   - Throughput scaling with worker count
   - Quality metrics (recall, mAP)
   - Latency vs quality trade-off
   - Performance heatmap

## 🔧 Configuration

### Database Parameters

**PostgreSQL + pgvector**:
- Index type: IVFFlat with 100 lists
- Distance: Cosine similarity
- Connection pool: 200 max connections

**Weaviate**:
- Index type: HNSW
- Distance: Cosine
- Parameters: ef=100, efConstruction=200, maxConnections=32

**Qdrant**:
- Index type: HNSW
- Distance: Cosine
- Parameters: m=32, ef_construct=200

### Customizing the Benchmark

You can modify the benchmark by editing the configuration in `benchmark.py`:

```python
# Change worker counts
worker_counts = [1, 2, 4, 8, 16, 32]

# Change quality metrics K values
k_values = [1, 5, 10, 20, 50]

# Change vector dimensions
vector_dim = 768  # For different models
```

## 🐛 Troubleshooting

### Common Issues

1. **Database connection errors**:
   ```bash
   # Check if services are running
   docker-compose ps
   
   # Check logs
   docker-compose logs postgres
   docker-compose logs weaviate
   docker-compose logs qdrant
   ```

2. **Memory issues with large datasets**:
   - Reduce `--vectors` parameter
   - Increase Docker memory limits
   - Use smaller models

3. **Slow performance**:
   - Ensure databases are fully started before running benchmark
   - Check system resources (CPU, RAM)
   - Consider using SSD storage

### Performance Tips

- Run on a machine with at least 8GB RAM
- Use SSD storage for better I/O performance
- Close other applications during benchmarking
- Run multiple times and average results for stability

## 📝 Example Output

```
BENCHMARK SUMMARY
================================================================================

POSTGRES:
  Index Build Time: 45.23s
  Memory Usage: 1024.5 MB
  Latency (1 worker): P50=12.3ms, P95=25.1ms, P99=45.2ms
  Max Throughput: 156.7 QPS
  Quality: Recall@1=0.987, Recall@10=0.995, mAP@10=0.923

WEAVIATE:
  Index Build Time: 23.45s
  Memory Usage: 2048.2 MB
  Latency (1 worker): P50=8.7ms, P95=18.3ms, P99=32.1ms
  Max Throughput: 234.1 QPS
  Quality: Recall@1=0.992, Recall@10=0.998, mAP@10=0.945

QDRANT:
  Index Build Time: 18.67s
  Memory Usage: 1536.8 MB
  Latency (1 worker): P50=6.2ms, P95=12.8ms, P99=22.4ms
  Max Throughput: 312.5 QPS
  Quality: Recall@1=0.994, Recall@10=0.999, mAP@10=0.951
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [pgvector](https://github.com/pgvector/pgvector) for PostgreSQL vector extension
- [Weaviate](https://weaviate.io/) for the vector database
- [Qdrant](https://qdrant.tech/) for the vector database
- [sentence-transformers](https://www.sbert.net/) for embedding generation
- [FAISS](https://faiss.ai/) for ground truth computation
