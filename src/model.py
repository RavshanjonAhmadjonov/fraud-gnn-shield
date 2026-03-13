import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.data import Data

class FraudDetectionGNN(torch.nn.Module):
    """
    Graph Neural Network for detecting fraudulent financial transactions.
    Utilizes GraphSAGE and GAT (Graph Attention Networks) layers to aggregate 
    transactional behaviors from neighboring nodes (accounts).
    """
    def __init__(self, num_node_features: int, hidden_channels: int, num_classes: int = 2):
        super(FraudDetectionGNN, self).__init__()
        # GraphSAGE layer for spatial aggregation
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        # Graph Attention layer to weigh suspicious connections
        self.conv2 = GATConv(hidden_channels, hidden_channels, heads=4, concat=False)
        self.conv3 = SAGEConv(hidden_channels, num_classes)
        
        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        # Layer 1: Feature Extraction
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2: Attention-based Aggregation
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 3: Classification Logits
        x = self.conv3(x, edge_index)
        
        # Log-Softmax for binary classification (0: Legitimate, 1: Fraudulent)
        return F.log_softmax(x, dim=1)

def compute_loss(model: torch.nn.Module, data: Data, mask: torch.Tensor):
    """
    Compute Focal Loss to handle extreme class imbalance in fraud datasets.
    """
    model.train()
    out = model(data)
    
    # Standard Negative Log Likelihood Loss
    nll_loss = F.nll_loss(out[mask], data.y[mask], reduction='none')
    
    # Focal Loss modifiers
    pt = torch.exp(-nll_loss)
    gamma = 2.0
    focal_loss = ((1 - pt) ** gamma * nll_loss).mean()
    
    return focal_loss

if __name__ == "__main__":
    print("FraudDetectionGNN Model Initialized.")
    print("Ready for deployment in transactional graph environments.")
