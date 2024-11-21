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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
    self.x = torch.tensor(self.X_prepared.values.astype(np.float32), dtype=torch.float).to(self.device)

    self.y = self.loaded_dataset.data.targets
    self.y_encoded = self.encode_target()
    self.num_classes = len(self.y_encoded['target'].unique())
    self.y_tensor = torch.tensor(self.y_encoded.values.ravel(), dtype=torch.long).to(self.device)

    self.train_idx, self.val_idx, self.test_idx = self.split_data()
    self.train_mask, self.val_mask, self.test_mask = self.create_masks()

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

  def split_data(self):
    random_seed = self.parameters['random_seed']
    test_size = self.parameters['data_splitter']['test_size']
    val_size = self.parameters['data_splitter']['val_size']

    row_number = self.X_prepared.shape[0]
    train_idx, temp_idx = train_test_split(range(row_number), test_size=test_size, random_state=random_seed)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_size, random_state=random_seed)

    return train_idx, val_idx, test_idx

  def create_masks(self):
    row_number = self.X_prepared.shape[0]
    train_mask = torch.zeros(row_number, dtype=torch.bool).to(self.device)
    val_mask = torch.zeros(row_number, dtype=torch.bool).to(self.device)
    test_mask = torch.zeros(row_number, dtype=torch.bool).to(self.device)

    train_mask[self.train_idx] = True
    val_mask[self.val_idx] = True
    test_mask[self.test_idx] = True

    return train_mask, val_mask, test_mask

class Graph():
  def __init__(self, parameters, pipeline_registry, dataset_name):
    self.parameters = parameters
    self.pipeline_registry = pipeline_registry
    self.device = parameters['device']

    self.adj_matrix = kneighbors_graph(pipeline_registry[dataset_name]['data_processor'].X_prepared,
                                       n_neighbors=parameters['graph']['knn']['k'],
                                       mode='connectivity',
                                       include_self=False)
    self.G = nx.from_scipy_sparse_array(self.adj_matrix)
    self.graph_data = from_networkx(self.G)
    self.graph_data.x = self.pipeline_registry[dataset_name]['data_processor'].x
    self.graph_data.y = self.pipeline_registry[dataset_name]['data_processor'].y_tensor
    self.graph_data.train_mask = self.pipeline_registry[dataset_name]['data_processor'].train_mask
    self.graph_data.val_mask = self.pipeline_registry[dataset_name]['data_processor'].val_mask
    self.graph_data.test_mask = self.pipeline_registry[dataset_name]['data_processor'].test_mask
    self.graph_data = self.graph_data.to(self.device)

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

class GraphTrainer():
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.device = parameters['device']
        self.graph_data = pipeline_registry[dataset_name]['graph'].graph_data
        self.num_classes = pipeline_registry[dataset_name]['data_processor'].num_classes

        self.train_mask = pipeline_registry[dataset_name]['data_processor'].train_mask
        self.val_mask = pipeline_registry[dataset_name]['data_processor'].val_mask

        self.best_val_acc = 0
        self.best_params = {}

    def train(self, model, optimizer, criterion, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[self.train_mask], data.y[self.train_mask])
        loss.backward()
        optimizer.step()
        return loss

    def validate(self, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data)
            pred = out[self.val_mask].max(1)[1]
            correct = pred.eq(data.y[self.val_mask]).sum().item()
            accuracy = correct / self.val_mask.sum().item()
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

        for hidden_dim, num_hidden_layers, lr, weight_decay in itertools.product(self.parameters['graph_trainer']['architecture']['hidden_dim'],
                                                                              self.parameters['graph_trainer']['architecture']['num_hidden_layers'],
                                                                              self.parameters['graph_trainer']['training']['lr_grid'],
                                                                              self.parameters['graph_trainer']['training']['weight_decay_grid']):
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

            for epoch in range(self.parameters['graph_trainer']['training']['epochs']):
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

class GraphEvaluator():
  def __init__(self, parameters, pipeline_registry, dataset_name):
    self.parameters = parameters
    self.pipeline_registry = pipeline_registry
    self.dataset_name = dataset_name
    self.device = parameters['device']

    self.graph_data = pipeline_registry[dataset_name]['graph'].graph_data
    self.best_params = pipeline_registry[dataset_name]['graph_trainer'].best_params
    self.num_classes = pipeline_registry[dataset_name]['data_processor'].num_classes

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

    for epoch in range(self.parameters['graph_evaluator']['epochs']):
        loss = self.pipeline_registry[self.dataset_name]['graph_trainer'].train(best_model, optimizer, criterion, self.graph_data)

    test_acc = self.pipeline_registry[self.dataset_name]['graph_trainer'].test(best_model, self.graph_data)
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
    plt.title("GNN Confusion Matrix")
    plt.show()

class XGBoostTrainer:
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']
        
        self.X_prepared = pipeline_registry[dataset_name]['data_processor'].X_prepared
        self.y_encoded = pipeline_registry[dataset_name]['data_processor'].y_encoded
        self.train_idx = pipeline_registry[dataset_name]['data_processor'].train_idx
        self.val_idx = pipeline_registry[dataset_name]['data_processor'].val_idx

        self.X_train = self.X_prepared.iloc[self.train_idx].values
        self.y_train = self.y_encoded.iloc[self.train_idx].values.ravel()
        self.X_val = self.X_prepared.iloc[self.val_idx].values
        self.y_val = self.y_encoded.iloc[self.val_idx].values.ravel()

        self.best_val_acc = 0
        self.best_model = None

    def train(self, model, X_train, y_train, params):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        return model

    def validate(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        return accuracy

    def run(self):
        param_grid = self.parameters['xgboost_trainer']['hyperparameters']
        
        best_accuracy = 0
        best_params = None
        best_model = None
        
        for params in itertools.product(*param_grid.values()):
            current_params = dict(zip(param_grid.keys(), params))
            print(f"Training XGBoost with params: {current_params}")
            
            model = self.train(model=None, X_train=self.X_train, y_train=self.y_train, params=current_params)
            
            val_acc = self.validate(model, self.X_val, self.y_val)
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_params = current_params
                best_model = model
        
        self.best_val_acc = best_accuracy
        self.best_model = best_model
        
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best Hyperparameters: {best_params}")
        
        return best_model


class XGBoostEvaluator:
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']

        self.X_prepared = pipeline_registry[dataset_name]['data_processor'].X_prepared
        self.y_encoded = pipeline_registry[dataset_name]['data_processor'].y_encoded
        self.test_idx = pipeline_registry[dataset_name]['data_processor'].test_idx

        self.X_test = self.X_prepared.iloc[self.test_idx].values
        self.y_test = self.y_encoded.iloc[self.test_idx].values.ravel()

        self.model = pipeline_registry[dataset_name]['xgboost_trainer']

    def run(self):
        """
        Evaluate the best XGBoost model on the test set.
        """
        y_pred = self.model.predict(self.X_test)
        test_accuracy = accuracy_score(self.y_test, y_pred)

        print(f"XGBoost Test Accuracy: {test_accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("XGBoost Confusion Matrix")
        plt.show()

        return test_accuracy
    
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_hidden_layers):
        super(MLP, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layers = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(input_dim, hidden_dim))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(torch.nn.Linear(hidden_dim, num_classes))

    def forward(self, x):
        for i in range(self.num_hidden_layers):
            x = F.relu(self.layers[i](x))
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)

class MLPTrainer():
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']

        self.X_prepared = pipeline_registry[dataset_name]['data_processor'].X_prepared
        self.y_encoded = pipeline_registry[dataset_name]['data_processor'].y_encoded
        self.train_idx = pipeline_registry[dataset_name]['data_processor'].train_idx
        self.val_idx = pipeline_registry[dataset_name]['data_processor'].val_idx

        self.X_train = torch.tensor(self.X_prepared.iloc[self.train_idx].values, dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(self.y_encoded.iloc[self.train_idx].values.ravel(), dtype=torch.long).to(self.device)
        self.X_val = torch.tensor(self.X_prepared.iloc[self.val_idx].values, dtype=torch.float32).to(self.device)
        self.y_val = torch.tensor(self.y_encoded.iloc[self.val_idx].values.ravel(), dtype=torch.long).to(self.device)

        self.best_val_acc = 0
        self.best_model_state = None

    def train(self, model, optimizer, criterion):
        model.train()
        optimizer.zero_grad()
        out = model(self.X_train)
        loss = criterion(out, self.y_train)
        loss.backward()
        optimizer.step()
        return loss

    def validate(self, model, criterion):
        model.eval()
        with torch.no_grad():
            out = model(self.X_val)
            val_loss = criterion(out, self.y_val)
            pred = out.max(1)[1]
            correct = pred.eq(self.y_val).sum().item()
            accuracy = correct / len(self.y_val)
        return val_loss, accuracy

    def run(self):
        input_dim = self.X_prepared.shape[1]
        num_classes = self.y_encoded['target'].nunique()

        for hidden_dim, num_hidden_layers, lr, weight_decay in itertools.product(
            self.parameters['mlp_trainer']['architecture']['hidden_dim'],
            self.parameters['mlp_trainer']['architecture']['num_hidden_layers'],
            self.parameters['mlp_trainer']['training']['lr_grid'],
            self.parameters['mlp_trainer']['training']['weight_decay_grid']
        ):
            print(f"Training MLP with hidden_dim={hidden_dim}, num_hidden_layers={num_hidden_layers}, lr={lr}, weight_decay={weight_decay}")

            model = MLP(input_dim, hidden_dim, num_classes, num_hidden_layers).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = torch.nn.CrossEntropyLoss()

            for epoch in range(self.parameters['mlp_trainer']['training']['epochs']):
                loss = self.train(model, optimizer, criterion)
                if epoch % 10 == 0:
                    val_loss, val_acc = self.validate(model, criterion)
                    print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_acc:.4f}")

                _, val_acc = self.validate(model, criterion)
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_model_state = model.state_dict()

        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        return model

class MLPEvaluator():
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']

        self.X_prepared = pipeline_registry[dataset_name]['data_processor'].X_prepared
        self.y_encoded = pipeline_registry[dataset_name]['data_processor'].y_encoded
        self.test_idx = pipeline_registry[dataset_name]['data_processor'].test_idx

        self.X_test = torch.tensor(self.X_prepared.iloc[self.test_idx].values, dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(self.y_encoded.iloc[self.test_idx].values.ravel(), dtype=torch.long).to(self.device)

        self.model = pipeline_registry[dataset_name]['mlp_trainer']

    def run(self):
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.X_test)
            pred = out.max(1)[1]
            correct = pred.eq(self.y_test).sum().item()
            test_accuracy = correct / len(self.y_test)

        print(f"MLP Test Accuracy: {test_accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(self.y_test.cpu(), pred.cpu()))

        cm = confusion_matrix(self.y_test.cpu(), pred.cpu())
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("MLP Confusion Matrix")
        plt.show()

        return test_accuracy


def build_parameters():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  random_seed = 42

  datasets = [
              # {'name': 'abalone',
              #  'id': 1,},
              # {'name': 'adult',
              #  'id': 2,},
              {'name': 'dry_bean',
               'id': 602,},
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

  data_splitter = {
                  'test_size': 0.4,
                  'val_size': 0.5,
                  }

  graph_trainer = {
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

  graph_evaluator = {
              'epochs': 200,
              }
  
  xgboost_trainer = {
              'hyperparameters': {
                  'learning_rate': [0.1],
                  'max_depth': [5, 7],
                  'n_estimators': [100],
                  # 'subsample': [0.8, 1.0],
                  # 'colsample_bytree': [0.8, 1.0],
                  # 'gamma': [0, 0.1],
                  # 'reg_alpha': [0, 0.1],
                  # 'reg_lambda': [1, 1.5],
              },
              'training': {
                  'epochs': 100,
                  'early_stopping_rounds': 10,
              }
          }
  
  mlp_trainer = {
        'architecture': {
            'hidden_dim': [16, 32],
            'num_hidden_layers': [1, 2],
        },
        'training': {
            'epochs': 100,
            'lr_grid': [0.01, 0.001],
            'weight_decay_grid': [0, 5e-4],
        },
    }

  return {
          'device': device,
          'random_seed': random_seed,
          'datasets': datasets,
          'graph': graph,
          'data_splitter': data_splitter,
          'graph_trainer': graph_trainer,
          'graph_evaluator': graph_evaluator,
          'xgboost_trainer': xgboost_trainer,
          'mlp_trainer': mlp_trainer,
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
        pipeline_registry[dataset_name]['graph'] = Graph(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name)
        pipeline_registry[dataset_name]['graph_trainer'] = GraphTrainer(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()
        pipeline_registry[dataset_name]['graph_evaluator'] = GraphEvaluator(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()
        pipeline_registry[dataset_name]['xgboost_trainer'] = XGBoostTrainer(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()
        pipeline_registry[dataset_name]['xgboost_evaluator'] = XGBoostEvaluator(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()
        pipeline_registry[dataset_name]['mlp_trainer'] = MLPTrainer(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()
        pipeline_registry[dataset_name]['mlp_evaluator'] = MLPEvaluator(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name).run()

main()