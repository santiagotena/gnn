# Colab installs
# !pip install torch-geometric
# !pip install networkx
# !pip install ucimlrepo
# !pip install seaborn

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import kneighbors_graph

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

# UCI ML repository import
from ucimlrepo import fetch_ucirepo

# Itertools
import itertools

class Dataset:
  def __init__(self, parameters, dataset):
    self.parameters = parameters
    self.dataset = dataset
    self.device = parameters['device']
    self.loaded_dataset = fetch_ucirepo(id=dataset['id'])
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
    self.x = torch.tensor(self.X_prepared.values.astype(np.float32), dtype=torch.float).to(self.device)

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

class Graph():
  def __init__(self, parameters, pipeline_registry, dataset_name):
    self.parameters = parameters
    self.pipeline_registry = pipeline_registry
    self.device = parameters['device']

    self.adj_matrix = kneighbors_graph(pipeline_registry[dataset_name]['dataset'].X_prepared,
                                       n_neighbors=parameters['graph']['knn']['k'],
                                       mode='connectivity',
                                       include_self=False)
    self.G = nx.from_scipy_sparse_array(self.adj_matrix)
    self.graph_data = from_networkx(self.G)
    self.graph_data.x = self.pipeline_registry[dataset_name]['dataset'].x
    self.graph_data.y = self.pipeline_registry[dataset_name]['dataset'].y_tensor
    self.graph_data = self.graph_data.to(self.device)

class SplitData():
  def __init__(self, parameters, pipeline_registry, dataset_name):
    self.parameters = parameters
    self.pipeline_registry = pipeline_registry
    self.device = parameters['device']
    self.random_seed = parameters['random_seed']

    self.graph_data = pipeline_registry[dataset_name]['graph'].graph_data
    self.row_number = pipeline_registry[dataset_name]['dataset'].X.shape[0]

    self.train_idx, self.temp_idx = train_test_split(range(self.row_number),
                                                     test_size=self.parameters['split_data']['test_size'],
                                                     random_state=self.random_seed)
    self.val_idx, self.test_idx = train_test_split(self.temp_idx,
                                                   test_size=self.parameters['split_data']['val_size'],
                                                   random_state=self.random_seed)

    self.train_mask = torch.tensor(self.train_idx, dtype=torch.long).to(self.device)
    self.val_mask = torch.tensor(self.val_idx, dtype=torch.long).to(self.device)
    self.test_mask = torch.tensor(self.test_idx, dtype=torch.long).to(self.device)

    self.graph_data.train_mask = torch.zeros(self.graph_data.num_nodes, dtype=torch.bool).to(self.device)
    self.graph_data.train_mask[self.train_mask] = True

    self.graph_data.val_mask = torch.zeros(self.graph_data.num_nodes, dtype=torch.bool).to(self.device)
    self.graph_data.val_mask[self.val_mask] = True

    self.graph_data.test_mask = torch.zeros(self.graph_data.num_nodes, dtype=torch.bool).to(self.device)
    self.graph_data.test_mask[self.test_mask] = True

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_hidden_layers):
        super(GCN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(num_node_features, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i in range(self.num_hidden_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class Trainer():
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.device = parameters['device']
        self.graph_data = pipeline_registry[dataset_name]['graph'].graph_data
        self.num_classes = pipeline_registry[dataset_name]['dataset'].num_classes

        self.best_val_acc = 0
        self.best_params = {}

    def train(self, model, optimizer, criterion, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def validate(self, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out[data.val_mask].max(1)[1]
            correct = pred.eq(data.y[data.val_mask]).sum().item()
            accuracy = correct / data.val_mask.sum().item()
        return accuracy

    def test(self, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out[data.test_mask].max(1)[1]
            correct = pred.eq(data.y[data.test_mask]).sum().item()
            accuracy = correct / data.test_mask.sum().item()
        return accuracy

    def run(self):
        print(f"Number of classes: {self.num_classes}")
        
        for hidden_dim, num_hidden_layers, lr, weight_decay in itertools.product(self.parameters['trainer']['architecture']['hidden_dim'],
                                                                              self.parameters['trainer']['architecture']['num_hidden_layers'],
                                                                              self.parameters['trainer']['training']['lr_grid'],
                                                                              self.parameters['trainer']['training']['weight_decay_grid']):
            print(f"Training with")
            print(f"Architecture: hidden_dim={hidden_dim}, num_hidden_layers={num_hidden_layers}")
            print(f"Hyperparameters: lr={lr}, weight_decay={weight_decay}")

            model = GCN(num_node_features=self.graph_data.num_node_features,
                        hidden_dim=hidden_dim,
                        num_classes=self.num_classes,
                        num_hidden_layers=num_hidden_layers
                        ).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(self.parameters['trainer']['training']['epochs']):
                loss = self.train(model, optimizer, criterion, self.graph_data)
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

            val_acc = self.validate(model, self.graph_data)
            print(f"Validation accuracy: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = {'hidden_dim': hidden_dim,
                                    'num_hidden_layers': num_hidden_layers,
                                    'lr': lr,
                                    'weight_decay': weight_decay}
        return self

class Evaluator():
  def __init__(self, parameters, pipeline_registry, dataset_name):
    self.parameters = parameters
    self.pipeline_registry = pipeline_registry
    self.dataset_name = dataset_name
    self.device = parameters['device']

    self.graph_data = pipeline_registry[dataset_name]['graph'].graph_data
    self.best_params = pipeline_registry[dataset_name]['trainer'].best_params
    self.num_classes = pipeline_registry[dataset_name]['dataset'].num_classes

  def run(self):
    best_model = GCN(num_node_features=self.graph_data.num_node_features,
                hidden_dim=self.best_params['hidden_dim'],
                num_classes=self.num_classes,
                num_hidden_layers=self.best_params['num_hidden_layers']
                ).to(self.device)

    optimizer = torch.optim.Adam(best_model.parameters(),
                                 lr=self.best_params['lr'],
                                 weight_decay=self.best_params['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(self.parameters['evaluator']['epochs']):
        loss = self.pipeline_registry[self.dataset_name]['trainer'].train(best_model, optimizer, criterion, self.graph_data)

    test_acc = self.pipeline_registry[self.dataset_name]['trainer'].test(best_model, self.graph_data)
    print(f"Test accuracy: {test_acc:.4f}")

    best_model.eval()
    with torch.no_grad():
        out = best_model(self.graph_data)
        pred = out[self.graph_data.test_mask].max(1)[1]
        y_true = self.graph_data.y[self.graph_data.test_mask]

    print(classification_report(y_true.cpu(), pred.cpu()))

    cm = confusion_matrix(y_true.cpu(), pred.cpu())
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

def build_parameters():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  random_seed = 42

  datasets = [
              {'name': 'abalone',
               'id': 1,},
              # {'name': 'adult',
              #  'id': 2,},
              # {'name': 'dry_bean',
              #  'id': 602,},
              # {'name': 'electrical_grid',
              #  'id': 471,},
              # {'name': 'isolet',
              #  'id': 54,},
              # {'name': 'musk_v2',
              #  'id': 75,},
              # {'name': 'occupancy_detection',
              #  'id': 357,},
              ]

  graph = {
          'knn': {
              'k': 5,
              }
          }

  split_data = {
                'test_size': 0.4,
                'val_size': 0.5,
                }

  trainer = {
            'architecture': {
                'hidden_dim': [16],
                'num_hidden_layers': [2, 3],
                },
            'training': {
                'epochs': 100,
                'lr_grid': [0.01, 0.001],
                'hidden_dim_grid': [16, 32],
                'weight_decay_grid': [0, 5e-4],
                },
              }

  evaluator = {
              'epochs': 200,
              }

  return {
          'device': device,
          'random_seed': random_seed,
          'datasets': datasets,
          'graph': graph,
          'split_data': split_data,
          'trainer': trainer,
          'evaluator': evaluator,
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

    pipeline_registry[dataset_name]['dataset'] = Dataset(parameters=parameters, dataset=dataset)
    pipeline_registry[dataset_name]['graph'] = Graph(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name)
    pipeline_registry[dataset_name]['split_data'] = SplitData(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name)
    pipeline_registry[dataset_name]['trainer'] = Trainer(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()
    pipeline_registry[dataset_name]['evaluator'] = Evaluator(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()

main()