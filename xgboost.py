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

class DataSplitter:
    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

    def split(self, X, y):
        return self.kfold.split(X, y)

    def train_test_split(self, X, y, test_size=0.1, stratify=None):
        return train_test_split(X, y, test_size=test_size, random_state=self.random_seed, stratify=stratify)

class XGBoostModel:
    def __init__(self, parameters, pipeline_registry, dataset_name):
        self.parameters = parameters
        self.pipeline_registry = pipeline_registry
        self.dataset_name = dataset_name
        self.device = parameters['device']
        self.X = pipeline_registry[dataset_name]['data_processor'].X_prepared
        self.y = pipeline_registry[dataset_name]['data_processor'].y_encoded['target']
        self.x_tensor = pipeline_registry[dataset_name]['data_processor'].x_tensor
        self.y_tensor = pipeline_registry[dataset_name]['data_processor'].y_tensor

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

        for fold_idx, (train_val_idx, test_idx) in enumerate(kfold.split(self.X, self.y)):
            print(f"\nRunning fold {fold_idx + 1}/10...")
            X_train_val, X_test = self.X.iloc[train_val_idx], self.X.iloc[test_idx]
            y_train_val, y_test = self.y.iloc[train_val_idx], self.y.iloc[test_idx]
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.1, random_state=self.parameters['random_seed'], stratify=y_train_val)

            best_model = None
            best_score = -float('inf')
            best_params = None
            param_grid = self.get_param_grid()

            for params in param_grid:
                model = xgb.XGBClassifier(**params, objective='multi:softmax', 
                                          num_class=self.pipeline_registry[self.dataset_name]['data_processor'].num_classes, 
                                          eval_metric='mlogloss')
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                val_preds = model.predict(X_val)
                val_score = f1_score(y_val, val_preds, average='weighted')

                if val_score > best_score:
                    best_score = val_score
                    best_model = model
                    best_params = params

            print(f"Best validation F1 score for fold {fold_idx + 1}: {best_score:.4f}")
            print(f"Best hyperparameters for fold {fold_idx + 1}: {best_params}")

            fold_f1_scores = []
            fold_accuracy_scores = []
            for retrain_run in range(3):
                print(f"Retraining best model (run {retrain_run + 1}/3)...")
                best_model.fit(X_train_val, y_train_val)
                test_preds = best_model.predict(X_test)
                test_f1 = f1_score(y_test, test_preds, average='weighted')
                test_accuracy = accuracy_score(y_test, test_preds)
                fold_f1_scores.append(test_f1)
                fold_accuracy_scores.append(test_accuracy)
            avg_f1 = np.mean(fold_f1_scores)
            avg_accuracy = np.mean(fold_accuracy_scores)
            print(f"Fold {fold_idx + 1} - Avg F1: {avg_f1:.4f}, Avg Accuracy: {avg_accuracy:.4f}")

            final_f1_scores.append(avg_f1)
            final_accuracy_scores.append(avg_accuracy)
            final_hyperparameters.append(best_params)

        self.results['f1_scores'] = final_f1_scores
        self.results['accuracy_scores'] = final_accuracy_scores
        self.results['best_hyperparameters'] = final_hyperparameters

        print("\nFinal Results:")
        print(f"F1 Score - Mean: {np.mean(final_f1_scores):.4f}, Std: {np.std(final_f1_scores):.4f}")
        print(f"Accuracy - Mean: {np.mean(final_accuracy_scores):.4f}, Std: {np.std(final_accuracy_scores):.4f}")

        print("\nBest Hyperparameters for Each Fold:")
        for i, params in enumerate(final_hyperparameters, start=1):
            print(f"Fold {i}: {params}")

        most_common_params = max(set(tuple(d.items()) for d in final_hyperparameters), 
                                 key=lambda x: final_hyperparameters.count(dict(x)))
        print(f"\nMost Frequently Selected Hyperparameters: {dict(most_common_params)}")

    def get_param_grid(self):
        learning_rates = self.parameters['xgboost_model']['learning_rate']
        max_depths = self.parameters['xgboost_model']['max_depth']
        n_estimators_list = self.parameters['xgboost_model']['n_estimators']

        param_grid = []
        for lr, depth, n_est in itertools.product(learning_rates, max_depths, n_estimators_list):
            param_grid.append({'learning_rate': lr, 'max_depth': depth, 'n_estimators': n_est})
        return param_grid

def build_parameters():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  random_seed = 42

  datasets = [
    					{
                'name': 'dry_bean',
								'id': 602,
              },
            ]

  xgboost_model = {
     								'learning_rate': [0.1],
										'max_depth': [5, 7],
										'n_estimators': [100],
										'epochs': 100,
                  }

  return {
          'device': device,
          'random_seed': random_seed,
          'datasets': datasets,
          'xgboost_model': xgboost_model,
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
        pipeline_registry[dataset_name]['xgboost_model'] = XGBoostModel(parameters=parameters, pipeline_registry=pipeline_registry, dataset_name=dataset_name)
        
main()