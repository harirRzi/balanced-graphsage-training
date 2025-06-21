# balanced-graphsage-training

## Anomaly Detection in Vehicular Networks

This repository contains code for detecting abnormal driving behaviors using a GraphSAGE model.

### Note
The dataset used for this project is not publicly available due to confidentiality.

### Requirements
- Python 3.8+
- PyTorch
- scikit-learn
- torch_geometric
- imbalanced-learn (for SMOTE)

### Usage
To test with your own data, provide CSV files formatted as expected by `data_loader.py`.

- `data_list`: A list of `torch_geometric.data.Data` graph objects.
- `in_channels`: Number of features per node.

### What This Code Does
- Prepares graph data using PyTorch Geometric and balances the dataset using SMOTE.
- Defines a GraphSAGE model with one graph convolution layer and a linear output layer.
- Trains the model using `CrossEntropyLoss` and evaluates performance using F1 score, recall, and precision.
- Performs 5-fold cross-validation, training on 4 folds and testing on the remaining 1.
- Plots an averaged ROC curve to evaluate classification performance across folds.
- The data_list is a collection of graph objects, each representing a snapshot of the vehicular network at a specific time. Each graph contains node features, graph connectivity (edge indices), and node-level ground truth labels. 
