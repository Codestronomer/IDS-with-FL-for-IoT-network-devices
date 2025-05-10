import time
import json
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from typing import List, Tuple
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss, precision_score, f1_score, recall_score
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation

df = pd.read_csv(
        '/Users/macbookpro/Downloads/Edge IIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv',
        low_memory=False
    )

# Replace infinite values with NaNs
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaNs
df.dropna(inplace=True)

cols_to_drop = [
    'frame.time', 'ip.src_host', 'ip.dst_host', 'http.file_data', 'http.request.uri.query', 'http.referer',
    'http.request.full_uri', 'dns.qry.name', 'mqtt.topic', 'mqtt.msg', 'mqtt.msg_decoded_as', 'mqtt.protoname',
    'arp.dst.proto_ipv4', 'arp.src.proto_ipv4', 'tcp.options', 'tcp.payload', 'tcp.srcport', 'dns.qry.name.len', 'mqtt.conack.flags'
]

df.drop(columns=cols_to_drop, inplace=True)

# Encode categorical columns 
categorical_cols = ['http.request.method', 'http.request.version', 'http.response', 'mqtt.msgtype',
                    'mqtt.ver', 'mqtt.conflags', 'Attack_type']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Seperate Label
label_col = 'Attack_label'
features = df.drop(columns=[label_col])
labels = df[label_col]

# Ensure all selected columns are numeric
features = features.select_dtypes(include=[np.number])

# Normalize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Reassemble final dataset
processed_df = pd.DataFrame(scaled_features, columns=features.columns)
processed_df[label_col] = labels.reset_index(drop=True)

# Seperate features (X) from target (y)
X = processed_df.drop(columns=['Attack_label', 'Attack_type'])
y = processed_df['Attack_label']

XGB_PARAMS = {
    "max_depth": 5,
    "learning_rate": 0.2,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "gamma": 1,
    "reg_lambda": 1,
    "eval_metric": "logloss",
    "seed": 42
}

class XGBoostClient(NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = self._create_model()
        self.is_model_fit = False
        
    def _create_model(self):
        """Create a new XGBoost model with the predefined parameters"""
        params = XGB_PARAMS.copy()
        
        # Handle class imbalance for each client
        counter = Counter(self.y_train)
        neg = counter.get(0, 1)
        pos = counter.get(1, 1)
        scale_pos_weight = neg / pos if pos != 0 else 1.0
        params["scale_pos_weight"] = scale_pos_weight

        return xgb.XGBClassifier(**params)

    def get_parameters(self, config=None):
        """Return model parameters as feature importances"""
        try:
            # If the model hasn't been fit yet, we need to fit it before getting parameters
            if not self.is_model_fit:
                print(f"Fitting initial model with classes: {np.unique(self.y_train)}")
                self.model.fit(self.X_train, self.y_train)
                self.is_model_fit = True
            
            # Create a simple dictionary of feature importances
            param_dict = {"feature_importances": self.model.feature_importances_.tolist()}
            
            # Convert to bytes for Flower protocol
            return [json.dumps(param_dict).encode("utf-8")]
        except Exception as e:
            print(f"Error in get_parameters: {e}")
            # Return uniform importances
            param_dict = {"feature_importances": [1.0/self.X_train.shape[1]] * self.X_train.shape[1]}
            return [json.dumps(param_dict).encode("utf-8")]

    def set_parameters(self, parameters):
        """Set model parameters - in this approach we're not sharing model parameters"""
        # In this model, we don't actually set parameters from other clients
        # But we make sure the model is initialized and fit
        if not self.is_model_fit:
            print(f"Training local model with {len(self.y_train)} samples")
            self.model.fit(self.X_train, self.y_train)
            self.is_model_fit = True
        return self

    def fit(self, parameters, config):
        """Train the model on local data"""
        print(f"Fitting client with {len(self.y_train)} samples, classes: {np.unique(self.y_train)}")
        
        # Make sure model is initialized and trained
        if not self.is_model_fit:
            self.model.fit(self.X_train, self.y_train)
            self.is_model_fit = True
        else:
            # If already fit, we could optionally do additional training here
            # self.model.fit(self.X_train, self.y_train, xgb_model=self.model)
            pass
            
        # Return current model's parameters
        return self.get_parameters(), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        """Evaluate model"""
        # Ensure model is fit before evaluation
        if not self.is_model_fit:
            print("Fitting model before evaluation")
            self.model.fit(self.X_train, self.y_train)
            self.is_model_fit = True
            
        try:
            # Ensure y_test is in the right format - convert to numpy array if it's a pandas Series
            y_test_np = self.y_test.values if hasattr(self.y_test, 'values') else self.y_test
            
            y_pred = self.model.predict(self.X_test)
            
            # Ensure the predictions are in the right format
            if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                # This is a multi-class format, get the class with highest probability
                y_pred_classes = np.argmax(y_pred, axis=1)
            else:
                # This is already in the right format
                y_pred_classes = y_pred
                
            try:
                y_pred_proba = self.model.predict_proba(self.X_test)
                # Check if we need to convert labels to binary indicators for log_loss
                n_classes = len(np.unique(y_test_np))
                if y_pred_proba.shape[1] != n_classes:
                    print(f"Warning: probability shape {y_pred_proba.shape} doesn't match classes {n_classes}")
                    loss = -np.mean(np.log(np.maximum(y_pred_proba[np.arange(len(y_test_np)), y_test_np], 1e-10)))
                else:
                    loss = log_loss(y_test_np, y_pred_proba)
            except Exception as e:
                print(f"Error calculating loss: {e}")
                loss = float("inf")
            
            accuracy = accuracy_score(y_test_np, y_pred_classes)
            # precision = precision_score(y_test, y_pred, zero_division=0)
            # recall = recall_score(y_test, y_pred, zero_division=0)
            # f1 = f1_score(y_test, y_pred, zero_division=0)

            print(f"Evaluation:\n Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            return loss, len(self.X_test), {"accuracy": float(accuracy)}
        except Exception as e:
            print(f"Error in evaluate: {e}")
            return float("inf"), len(self.X_test), {"accuracy": 0.0}

class FederatedXGBoostStrategy(FedAvg):
    """Strategy for federated XGBoost that doesn't aggregate model parameters"""
    
    def aggregate_fit(self, server_round, results, failures):
        """Don't actually aggregate - just return None to skip parameter distribution"""
        print(f"Round {server_round}: {len(results)} results, {len(failures)} failures")
        
        # For a true federated setup, you would aggregate parameters here
        # But in this implementation each client trains their own model
        # Just return None to signal that clients should use their own parameters
        return None, {}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate accuracy metrics from multiple clients."""
    # Check if we have metrics to average
    if not metrics:
        return {"accuracy": 0.0}
        
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Check if we have any examples
    if sum(examples) == 0:
        return {"accuracy": 0.0}
        
    return {"accuracy": sum(accuracies) / sum(examples)}

NUM_CLIENTS=10
client_datasets = []

# Stratified split to ensure class balance across clients
grouped = {}
unique_labels = y.unique()
for label in unique_labels:
    idx = y[y == label].index
    grouped[label] = idx
    
# Create client partitions with balanced class distribution
client_indices = [[] for _ in range(NUM_CLIENTS)]
for label, indices in grouped.items():
    # Shuffles the indices for this class
    indices = indices.to_numpy()
    np.random.shuffle(indices)
    
    # split indices evenly among clients
    splits = np.array_split(indices, NUM_CLIENTS)
    for i, split in enumerate(splits):
        client_indices[i].extend(split)
        

# Create train/test split for eaach client
for i in range(NUM_CLIENTS):
    indices = client_indices[i]
    X_client = X.iloc[indices]
    y_client = y.iloc[indices]
     
    X_train, X_test, y_train, y_test = train_test_split(
        X_client, y_client, test_size=0.2, stratify=y_client
    )
    client_datasets.append((X_train, y_train, X_test, y_test))
    
print("Client dataset length: ", len(client_datasets))

for i, (X_train, y_train, _, _) in enumerate(client_datasets):
    print(f"Client {i} - Training set class distribution: {y_train.value_counts().to_dict()}")

def server_fn(context: Context) -> ServerAppComponents:
    """Create server components with custom strategy."""
    strategy = FederatedXGBoostStrategy(
        fraction_fit=0.5,        # Use 50% of available clients for training
        fraction_evaluate=0.5,   # Use 50% of available clients for evaluation
        min_fit_clients=2,       # At least 2 clients for training
        min_evaluate_clients=2,  # At least 2 clients for evaluation
        min_available_clients=2, # Wait for at least 2 clients before starting
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=10)
    return ServerAppComponents(strategy=strategy, config=config)

def client_fn(context: Context):
    """Create a flower client representing a single organization."""
    
    # Get the client ID from the node configuration
    partition_id = context.node_config["partition-id"]
    
    X_train, y_train, X_test, y_test = client_datasets[partition_id]
    
    print(f"Client {partition_id} created with {len(X_train)} training samples")
    print(f"Classes in training data: {np.unique(y_train)}")

    return XGBoostClient(X_train, y_train, X_test, y_test).to_client()

server = ServerApp(server_fn=server_fn)
client = ClientApp(client_fn=client_fn)

backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}

# run simulation
run_simulation(
	server_app=server,
	client_app=client,
	num_supernodes=NUM_CLIENTS,
    backend_config=backend_config,
)
