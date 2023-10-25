import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm,GATConv


class GCN(nn.Module):
        def __init__(self, num_features, hidden_channels, num_classes):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_features, hidden_channels)
            self.bn1 = BatchNorm(hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.bn2 = BatchNorm(hidden_channels)
            self.conv3 = GCNConv(hidden_channels, num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            # First Convolutional layer
            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # Second Convolutional layer
            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # Third Convolutional layer
            x = self.conv3(x, edge_index)

            return x
        

class GAT(nn.Module):
        def __init__(self, num_features, hidden_channels, num_classes):
            super(GAT, self).__init__()
            self.conv1 = GATConv(num_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
            self.conv3 = GATConv(hidden_channels, num_classes)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index

            # First Attention layer
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # Second Attention layer
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            # Third Attention layer
            x = self.conv3(x, edge_index)

            return x
class EnhancedGCN(nn.Module):
    def __init__(self, num_features=64, hidden_channels=128, num_classes=1):
        super(EnhancedGCN, self).__init__()
        
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        
        self.conv4 = GCNConv(hidden_channels, num_classes)
        
        # Multi-head attention layer (optional)
        # Uncomment the line below if you want to use GAT instead of GCN
        # self.gat_conv = GATConv(hidden_channels, hidden_channels, heads=2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First Convolutional layer
        x1 = self.conv1(x, edge_index)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=0.5, training=self.training)
        
        # Second Convolutional layer
        x2 = self.conv2(x1, edge_index)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=0.5, training=self.training)
        
        # Third Convolutional layer
        x3 = self.conv3(x2, edge_index)
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        x3 = F.dropout(x3, p=0.5, training=self.training)
        
        # Skip Connection
        x3 += x1
        
        # Fourth Convolutional layer
        x4 = self.conv4(x3, edge_index)
        
        # Uncomment the lines below if you want to use GAT
        # x5 = self.gat_conv(x3, edge_index)
        # x5 = x5.mean(dim=1)
        
        return x4  # or return x5 if using GAT