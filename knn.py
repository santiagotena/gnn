# Colab installs
!pip install seaborn
!pip install torch-geometric
!pip install networkx
!pip install ucimlrepo
!pip install xgboost

# Core imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import accuracy_score

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

# XGBoost
import xgboost as xgb

# UCI ML repository import
from ucimlrepo import fetch_ucirepo

# Itertools
import itertools

class DataLoader():
  def __init__(self, parameters, dataset):
    self.parameters = parameters
    self.dataset = dataset
    self.loaded_dataset = fetch_ucirepo(id=dataset['id'])

class DataProcessor():
  def __init__(self, parameters, pipeline_registry, dataset_name):
    self.parameters = parameters
    self.pipeline_registry = pipeline_registry
    self.dataset_name = dataset_name
    self.device = parameters['device']
    self.loaded_dataset = pipeline_registry[dataset_name]['data_loader'].loaded_dataset
    self.X = self.loaded_dataset.data.features
    self.X_numerical_features, self.X_categorical_features = self.split_feature_types()

    if self.X_numerical_features.empty:
      self.X_numeric_scaled = pd.DataFrame()
    else:
      self.X_numeric_scaled = self.scale_numeric()

    if self.X_categorical_features.empty:
      self.X_categorical_encoded = pd.DataFrame()
    else:
      self.X_categorical_encoded = pd.get_dummies(self.X_categorical_features)

    self.X_prepared = pd.concat([self.X_numeric_scaled, self.X_categorical_encoded], axis=1)
    self.x_tensor = torch.tensor(self.X_prepared.values.astype(np.float32), dtype=torch.float).to(self.device)

    self.y = self.loaded_dataset.data.targets
    self.y_encoded = self.encode_target()
    self.num_classes = len(self.y_encoded['target'].unique())
    self.y_tensor = torch.tensor(self.y_encoded.values.ravel(), dtype=torch.long).to(self.device)

  def split_feature_types(self):
    numerical_features = self.X.select_dtypes(include=[np.number])
    categorical_features = self.X.select_dtypes(exclude=[np.number])
    return numerical_features, categorical_features

  def scale_numeric(self):
    scaler = StandardScaler()
    X_numeric_scaled = pd.DataFrame(scaler.fit_transform(self.X_numerical_features), columns=self.X_numerical_features.columns)
    return X_numeric_scaled

  def encode_target(self):
    encoder = LabelEncoder()
    y_encoded = pd.DataFrame(encoder.fit_transform(self.y.values.ravel()), columns=['target'])
    return y_encoded

class KNNGraph():
  def __init__(self, parameters, pipeline_registry, dataset_name):
    self.parameters = parameters
    self.pipeline_registry = pipeline_registry
    self.device = parameters['device']

    self.adj_matrix = kneighbors_graph(pipeline_registry[dataset_name]['data_processor'].X_numeric_scaled,
                                       n_neighbors=parameters['knn_graph']['k'],
                                       mode='connectivity',
                                       include_self=False)
    self.G = nx.from_scipy_sparse_array(self.adj_matrix)
    self.graph_data = from_networkx(self.G).to(self.device)
    self.graph_data.x = pipeline_registry[dataset_name]['data_processor'].x_tensor
    self.graph_data.y = pipeline_registry[dataset_name]['data_processor'].y_tensor

class GNNModel:
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']

        self.data_processor = pipeline_registry[dataset_name]['data_processor']
        self.knn_graph = pipeline_registry[dataset_name]['knn_graph']

        self.graph_data = self.knn_graph.graph_data
        self.num_features = self.data_processor.X_prepared.shape[1]
        self.num_classes = self.data_processor.num_classes

        self.results = {
            'f1_scores': [],
            'accuracy_scores': [],
            'best_hyperparameters': []
        }

        self.run_model()

    def run_model(self):
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.parameters['random_seed'])
        final_f1_scores = []
        final_accuracy_scores = []
        final_hyperparameters = []

        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(self.graph_data.x.cpu(), self.graph_data.y.cpu())):
            print(f"\nRunning fold {fold_idx + 1}/10...")
            train_mask, test_mask = self.create_masks(len(self.graph_data.x), train_idx, test_idx)

            best_model_state = None
            best_val_f1 = -float('inf')
            best_params = None

            # Hyperparameter grid search
            param_grid = self.get_param_grid()
            for params in param_grid:
                hidden_dim, lr = params['hidden_dim'], params['lr']
                model = self.build_gnn_model(hidden_dim)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

                for epoch in range(self.parameters['gnn_model']['epochs']):
                    model.train()
                    optimizer.zero_grad()
                    out = model(self.graph_data.x, self.graph_data.edge_index)
                    loss = F.cross_entropy(out[train_mask], self.graph_data.y[train_mask])
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_out = out[test_mask]
                    val_preds = val_out.argmax(dim=1)
                    val_f1 = f1_score(self.graph_data.y[test_mask].cpu(), val_preds.cpu(), average='weighted')

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_model_state = model.state_dict()
                        best_params = params

            print(f"Fold {fold_idx + 1} - Best Validation F1: {best_val_f1:.4f}")
            print(f"Fold {fold_idx + 1} - Best Hyperparameters: {best_params}")

            # Retrain on best model and test
            model = self.build_gnn_model(best_params['hidden_dim'])
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                test_out = model(self.graph_data.x, self.graph_data.edge_index)
                test_preds = test_out[test_mask].argmax(dim=1)
                test_f1 = f1_score(self.graph_data.y[test_mask].cpu(), test_preds.cpu(), average='weighted')
                test_accuracy = accuracy_score(self.graph_data.y[test_mask].cpu(), test_preds.cpu())

                final_f1_scores.append(test_f1)
                final_accuracy_scores.append(test_accuracy)
                final_hyperparameters.append(best_params)

                print(f"Fold {fold_idx + 1} - Test F1: {test_f1:.4f}, Test Accuracy: {test_accuracy:.4f}")

        self.results['f1_scores'] = final_f1_scores
        self.results['accuracy_scores'] = final_accuracy_scores
        self.results['best_hyperparameters'] = final_hyperparameters

        print("\nFinal Results:")
        print(f"F1 Score - Mean: {np.mean(final_f1_scores):.4f}, Std: {np.std(final_f1_scores):.4f}")
        print(f"Accuracy - Mean: {np.mean(final_accuracy_scores):.4f}, Std: {np.std(final_accuracy_scores):.4f}")

        print("\nBest Hyperparameters for Each Fold:")
        for i, params in enumerate(final_hyperparameters, start=1):
            print(f"Fold {i}: {params}")

        # Determine most frequently selected hyperparameters
        most_common_params = max(set(tuple(d.items()) for d in final_hyperparameters), 
                                 key=lambda x: final_hyperparameters.count(dict(x)))
        print(f"\nMost Frequently Selected Hyperparameters: {dict(most_common_params)}")

    def get_param_grid(self):
        lr_grid = self.parameters['gnn_model']['lr_grid']
        hidden_dim_grid = self.parameters['gnn_model']['hidden_dim_grid']

        param_grid = []
        for lr, hidden_dim in itertools.product(lr_grid, hidden_dim_grid):
            param_grid.append({'lr': lr, 'hidden_dim': hidden_dim})
        return param_grid

    def build_gnn_model(self, hidden_dim):
        class GCN(torch.nn.Module):
            def __init__(self, num_features, hidden_dim, num_classes):
                super(GCN, self).__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, num_classes)

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
                x = self.conv2(x, edge_index)
                return x

        return GCN(self.num_features, hidden_dim, self.num_classes).to(self.device)

    def create_masks(self, num_nodes, train_idx, test_idx):
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        return train_mask.to(self.device), test_mask.to(self.device)

def build_parameters():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  random_seed = 42

  datasets = [
    					{
                'name': 'dry_bean',
								'id': 602,
              },
            ]

  knn_graph = {
     						'k': 5,
              }
  
  gnn_model = {
                'hidden_dim': [16],
                'num_hidden_layers': [2, 3],
                'epochs': 100,
                'lr_grid': [0.01, 0.001],
                'hidden_dim_grid': [16, 32],
                'weight_decay_grid': [0, 5e-4],
              }

  return {
          'device': device,
          'random_seed': random_seed,
          'datasets': datasets,
          'knn_graph': knn_graph,
          'gnn_model': gnn_model,
          }

def build_pipeline_registry(dataset_names):
  pipeline_registry = {}
  for _, dataset_name in enumerate(dataset_names):
    pipeline_registry.setdefault(dataset_name, {})
  return pipeline_registry

def main():
    parameters = build_parameters()
    dataset_names = [dataset['name'] for dataset in parameters['datasets']]
    pipeline_registry = build_pipeline_registry(dataset_names)

    for dataset in parameters['datasets']:
        dataset_name = dataset['name']
        print("--------------------------------")
        print(f"Loading dataset: {dataset_name}")
        print("--------------------------------")
        pipeline_registry[dataset_name]['data_loader'] = DataLoader(parameters=parameters, dataset=dataset)
        pipeline_registry[dataset_name]['data_processor'] = DataProcessor(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name)
        pipeline_registry[dataset_name]['knn_graph'] = KNNGraph(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name)
        pipeline_registry[dataset_name]['gnn_model'] = GNNModel(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name)
        
main()