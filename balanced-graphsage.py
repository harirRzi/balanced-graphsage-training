import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, precision_score, roc_curve, auc
from torch.utils.data import random_split, TensorDataset, DataLoader as TorchDataLoader
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper class to apply SMOTE for balancing class distributions in each graph
class BalancingData:
    @staticmethod
    def balancingDataList(data_list):
        updated_data_list = []
        for graph in data_list:
            original_node_labels = graph.y.numpy()
            label_counts = np.bincount(original_node_labels)
            if len(label_counts) < 2:
                updated_data_list.append(graph)
                continue
            count_0, count_1 = label_counts[0], label_counts[1]
            if count_0 > 0 and count_1 > 0:
                min_count = min(count_0, count_1)
                smote = SMOTE(k_neighbors=min(5, min_count - 1))
                try:
                    features_resampled, labels_resampled = smote.fit_resample(graph.x.numpy(), original_node_labels)
                except ValueError:
                    updated_data_list.append(graph)
                    continue
                original_sample_count = graph.x.size(0)
                synthetic_data_indices = np.arange(original_sample_count, len(labels_resampled))
                synthetic_features = features_resampled[synthetic_data_indices]
                synthetic_labels = labels_resampled[synthetic_data_indices]
                updated_x = torch.cat([graph.x, torch.tensor(synthetic_features, dtype=torch.float)], dim=0)
                updated_labels = torch.cat([graph.y, torch.tensor(synthetic_labels, dtype=torch.long)], dim=0)
                updated_data = Data(x=updated_x, edge_index=graph.edge_index, y=updated_labels)
                updated_data_list.append(updated_data)
            else:
                updated_data_list.append(graph)
        return updated_data_list

# GraphSAGE model with one GraphSAGE layer followed by a linear layer
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear(x)
        out = F.log_softmax(x, dim=1)
        label = out.argmax(dim=-1)
        return label, out

# Training loop
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        _, output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

# Evaluation loop
@torch.no_grad()
def test(model, loader, criterion):
    model.eval()
    ys, preds = [], []
    total_loss = 0
    for data in loader:
        data = data.to(device)
        label, output = model(data.x, data.edge_index)
        loss = criterion(output, data.y)
        total_loss += loss.item() * data.num_graphs
        ys.append(data.y.cpu())
        preds.append(label.cpu())

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    f1_binary = f1_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    precision = precision_score(y_true, y_pred, average='binary')
    f1_micro = f1_score(y_true, y_pred, average='micro')

    return total_loss / len(loader.dataset), f1_micro, f1_binary, recall, precision, y_true, y_pred

# Main function to run K-fold cross-validation

def run_cross_validation(data_list, in_channels):
    num_splits = 5
    hidden_channels = 118
    out_channels = 2
    dropout_rate = 0.0087
    learning_rate = 0.0548

    model = GraphSAGE(in_channels, hidden_channels, out_channels, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    kf = KFold(n_splits=num_splits)
    tprs, aucs, all_y_true, all_y_scores = [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    
    for train_idx, test_idx in kf.split(data_list):
        train_data = [data_list[i] for i in train_idx]
        test_data = [data_list[i] for i in test_idx]
        train_data = BalancingData.balancingDataList(train_data)
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=2)
        val_loader = DataLoader(val_data, batch_size=2)
        test_loader = DataLoader(test_data, batch_size=2)

        for epoch in range(1, 101):
            train_loss = train(model, train_loader, optimizer, criterion)
            val_loss, val_f1, _, _, _, _, _ = test(model, val_loader, criterion)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val F1={val_f1:.4f}")

        _, _, _, _, _, y_true, y_pred = test(model, test_loader, criterion)
        all_y_true.append(y_true)
        all_y_scores.append(y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()