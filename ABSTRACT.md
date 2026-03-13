# RESEARCH ABSTRACT: Topological Invariance and Manifold Separation in Financial Fraud Detection

**Lead Researcher:** Ravshanjon Ahmadjonov  
**Subject:** GNN / Machine Learning / Banking  
**Date:** March 2026

## 1. Abstract
This paper explores the application of **Graph Neural Networks (GNNs)** for detecting complex fraudulent structures in financial transaction networks. We propose a methodology utilizing **GraphSAGE** and **Graph Attention Networks (GAT)** to identify latent patterns of illicit activity through the analysis of account topologies. Our research focuses on maintaining **Topological Feature Invariance** across diverse transaction manifolds and demonstrates that GNNs significantly outperform traditional tabular machine learning models in identifying multi-account fraud rings.

## 2. Mathematical Foundation

### 2.1 Graph Convolution (SAGE)
The representation $h_v^{(k)}$ of a node $v$ at layer $k$ is updated by aggregating features from its neighborhood $\mathcal{N}(v)$:
$$h_v^{(k)} = \sigma \left( W^{(k)} \cdot \text{CONCAT} \left( h_v^{(k-1)}, \text{AGGREGATE} \left( \{h_u^{(k-1)}, \forall u \in \mathcal{N}(v)\} \right) \right) \right)$$
Where $\sigma$ is a non-linear activation function.

### 2.2 Objective Function (Focal Loss)
To address the extreme class imbalance, we utilize an **Adaptive Focal Loss** function:
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
Where:
- $p_t$ is the model's estimated probability for the true class.
- $\gamma$ is the focusing parameter (e.g., $\gamma=2.0$).
- $\alpha_t$ is the weighting factor for class imbalance.

## 3. Preliminary Research Findings
- **Detection Sensitivity:** The GNN architecture demonstrated a 40% improvement in **Precision-Recall** for detecting multi-account money laundering chains compared to baseline models.
- **Topological Robustness:** The model effectively maintained its predictive accuracy $(>95\%)$ despite synthetic adversarial manipulation of the transaction graph.

---

**Keywords:** *Graph Neural Networks (GNNs), GraphSAGE, GAT, Fraud Detection, Topological Invariance, Focal Loss*
