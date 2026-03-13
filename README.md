# Fraud-GNN-Shield 🛡️🏦

[![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-red.svg)](https://pytorch-geometric.readthedocs.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)

An enterprise-grade **Graph Neural Network (GNN)** designed specifically for real-time financial fraud detection. Built with **PyTorch Geometric**, this model excels at uncovering hidden rings of illicit activity by analyzing the topological structure of transaction networks.

## 🚀 Why Graph Neural Networks for Banking?
Traditional tabular Machine Learning models (like XGBoost or Random Forests) treat transactions as isolated events. However, money laundering and sophisticated fraud happen in **networks**. 

By representing accounts as *nodes* and transactions as *edges*, `Fraud-GNN-Shield` uses **GraphSAGE** and **Graph Attention Networks (GAT)** to propagate suspicious behaviors across the network, catching bad actors based on "who they transact with", not just "how much they transfer."

## 🧩 Architecture Highlights
- **Spatial Aggregation (GraphSAGE)**: Efficiently samples and aggregates features from local transaction neighborhoods.
- **Attention Mechanisms (GAT)**: Dynamically assigns higher importance to suspicious edges (e.g., rapid, high-volume transfers to newly created accounts).
- **Focal Loss Implementation**: Handles the extreme class imbalance typical in banking datasets (where <0.1% of transactions are fraudulent).

## 🛠 Directory Structure
- `src/model.py`: Core PyTorch Geometric GNN definitions.
- `src/data_loader.py`: (WIP) Scripts for ingesting and structuring Neo4j/PostgreSQL relational data into PyTorch `Data` objects.
- `models/`: Checkpoints for saved GNN weights.

## 💻 Quick Start
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
python src/model.py
```

---
**Maintained by:** [Ravshanjon Ahmadjonov](https://github.com/Symotix) | NLP & Machine Learning Engineer | Garant Bank
