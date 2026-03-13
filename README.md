# RESEARCH FRAMEWORK: Topological Feature Invariance in Financial Fraud Manifolds

[![Research](https://img.shields.io/badge/Research-ML--Scientist-red.svg)](ABSTRACT.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **advanced research platform** for exploring **topological feature invariance** and **adversarial manifold robustness** in the context of financial fraud detection. This framework implements a Graph Neural Network (GNN) architecture designed to detect latent rings of illicit activity through the analysis of account-transaction topologies.

Built for **computational transparency, statistical rigor, and high-precision fraud intelligence**.

---

## 🔬 Core Research Domains

### 1. Topological Feature Extraction
The framework utilizes **GraphSAGE** and **Graph Attention Networks (GAT)** to identify the underlying structural properties of transaction networks. By capturing the relational dynamics between accounts, we can effectively isolate suspicious clusters.

### 2. Adversarial Manifold Robustness
Investigates the robustness of GNN models against **adversarial topology manipulation**. We explore how fraudulent actors attempt to obscure their activities through complex transaction chains and how the GNN maintains its predictive integrity.

### 3. Class Imbalance & Focal Loss Heuristics
Implements an **Adaptive Focal Loss** mechanism to address the extreme class imbalance $(>0.1\%)$ characteristic of financial fraud datasets. This ensures the model converges toward detecting rare fraudulent events without sacrificing overall precision.

---

## 🏗 System Architecture

- `src/model.py`: Core GNN definitions implementing **Spatial Aggregation** and **Attention-based Pooling**.
- `models/`: Research checkpoints for saved model weights and optimization trajectories.

---

## 🧪 Scientific FAQ

### How is the graph structure constructed?
Accounts are represented as **Nodes** $(v \in \mathcal{V})$ and transactions as **Edges** $(e \in \mathcal{E})$. Each node is initialized with a high-dimensional feature vector $(h_v)$ derived from account activity.

### What is the primary evaluation metric?
We evaluate the **Area Under the Precision-Recall Curve (AUPRC)** and the **Stochastic Recall** across discrete optimization epochs to ensure reliable detection of minority-class fraudulent events.

---

**Lead Researcher:** [Ravshanjon Ahmadjonov](https://github.com/RavshanjonAhmadjonov) | NLP & ML Research Scientist
