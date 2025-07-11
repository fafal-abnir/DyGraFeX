import networkx as nx
import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
import argparse
import numpy as np
from datetime import datetime
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from models.model import EdgeFraudNet
from models.lightning_modules import LightningEdgeGNN
from pytorch_lightning.callbacks import ModelCheckpoint

drop_cols = ['time_step', 'isFlaggedFraud', 'laundering_type', 'timestamp', 'txId', 'id', 'tx_id', 'alert_id',
             'time', 'date',
             'sender', 'receiver']

graph_features = ['total_amount', 'pmi', 'count', 'exclusivity_amt',
                  'sender_pagerank_None', 'receiver_pagerank_None',
                  'sender_pagerank_total_amount', 'receiver_pagerank_total_amount',
                  'sender_pagerank_count', 'receiver_pagerank_count',
                  'sender_pagerank_pmi', 'receiver_pagerank_pmi',
                  'sender_pagerank_exclusivity_amt', 'receiver_pagerank_exclusivity_amt',
                  'sender_pagerank_None_norm', 'receiver_pagerank_None_norm',
                  'sender_pagerank_total_amount_norm',
                  'receiver_pagerank_total_amount_norm', 'sender_pagerank_count_norm',
                  'receiver_pagerank_count_norm', 'sender_pagerank_pmi_norm',
                  'receiver_pagerank_pmi_norm', 'sender_pagerank_exclusivity_amt_norm',
                  'receiver_pagerank_exclusivity_amt_norm', 'sender_auth_None',
                  'receiver_auth_None', 'sender_hub_None', 'receiver_hub_None',
                  'sender_auth_total_amount', 'receiver_auth_total_amount',
                  'sender_hub_total_amount', 'receiver_hub_total_amount',
                  'sender_auth_count', 'receiver_auth_count', 'sender_hub_count',
                  'receiver_hub_count', 'sender_auth_pmi', 'receiver_auth_pmi',
                  'sender_hub_pmi', 'receiver_hub_pmi', 'sender_auth_exclusivity_amt',
                  'receiver_auth_exclusivity_amt', 'sender_hub_exclusivity_amt',
                  'receiver_hub_exclusivity_amt', 'community_belonging_louvain_1.0',
                  'destination_community_neighbor_count_louvain_1.0',
                  'destination_community_neighbor_count_louvain_1.0_norm',
                  'destination_community_neighbor_weighted_degree_louvain_1.0',
                  'destination_community_neighbor_weighted_degree_louvain_1.0_norm',
                  'community_belonging_leiden_0.01',
                  'destination_community_neighbor_count_leiden_0.01',
                  'destination_community_neighbor_count_leiden_0.01_norm',
                  'destination_community_neighbor_weighted_degree_leiden_0.01',
                  'destination_community_neighbor_weighted_degree_leiden_0.01_norm',
                  'community_belonging_leiden_0.04',
                  'destination_community_neighbor_count_leiden_0.04',
                  'destination_community_neighbor_count_leiden_0.04_norm',
                  'destination_community_neighbor_weighted_degree_leiden_0.04',
                  'destination_community_neighbor_weighted_degree_leiden_0.04_norm',
                  ]

def print_label_counts(data, name):
    labels = data.y
    num_zeros = torch.sum(labels == 0).item()
    num_ones = torch.sum(labels == 1).item()
    print(f"{name} - Label 0: {num_zeros}, Label 1: {num_ones}")

def prepare_pg_graph_data(df, cat_features=None, num_features=None, drop_ratio=0.0, use_graph_features=True):
    df = df.copy()

    # Encode sender/receiver to integer node ids
    node_encoder = LabelEncoder()
    all_nodes = pd.concat([df['sender'], df['receiver']])
    node_encoder.fit(all_nodes)
    df['sender'] = node_encoder.transform(df['sender'])
    df['receiver'] = node_encoder.transform(df['receiver'])

    # Edge index and number of nodes
    all_edges = torch.tensor(df[['sender', 'receiver']].values.T, dtype=torch.long)
    num_nodes = all_edges.max().item() + 1

    # Build NetworkX graph for PageRank and clustering coefficient
    G = nx.DiGraph()
    G.add_edges_from(df[['sender', 'receiver']].values.tolist())

    pagerank = nx.pagerank(G)
    clustering = nx.clustering(G.to_undirected())  # clustering only makes sense on undirected graphs

    # Convert node stats to tensors
    pr = torch.zeros(num_nodes, 1)
    clust = torch.zeros(num_nodes, 1)
    for node, idx in pagerank.items():
        pr[node] = idx
    for node, val in clustering.items():
        clust[node] = val

    # Degree features
    in_deg = degree(all_edges[1], num_nodes=num_nodes).unsqueeze(1)
    out_deg = degree(all_edges[0], num_nodes=num_nodes).unsqueeze(1)
    total_deg = in_deg + out_deg

    # node features
    node_features = torch.cat([in_deg, out_deg, total_deg, pr, clust], dim=1)

    # edge features
    if use_graph_features:
        edge_features = cat_features + num_features
    else:
        edge_features = [col for col in (cat_features + num_features) if col not in graph_features]
    # Split logic
    all_windows = sorted(df['window_index'].unique())
    n = len(all_windows)
    train_windows = set(all_windows[:int(0.7 * n)])
    val_windows = set(all_windows[int(0.7 * n):int(0.8 * n)])
    test_windows = set(all_windows[int(0.8 * n):int(1.0 * n)])

    # df['split'] = 'test'
    df.loc[df['window_index'].isin(train_windows), 'split'] = 'train'
    df.loc[df['window_index'].isin(val_windows), 'split'] = 'val'
    df.loc[df['window_index'].isin(test_windows), 'split'] = 'test'
    if drop_ratio > 0.0:
        train_mask = df['split'] == 'train'
        zero_rows = df[train_mask & (df['label'] == 0)]
        rows_to_drop = zero_rows.sample(frac=drop_ratio, random_state=42)
        print(f"shape of df before under sampling {len(df)}")
        df = df.drop(rows_to_drop.index)
        print(f"shape of df after under sampling {len(df)}")

    feature_df = df[edge_features].copy()
    feature_df.fillna(0, inplace=True)
    cat_features = [col for col in cat_features if col in edge_features]
    num_features = [col for col in num_features if col in edge_features]
    #print(cat_features)
    #print(feature_df.columns)
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', 'passthrough', num_features)
    ])
    features = preprocessor.fit_transform(feature_df).astype(np.float32)
    edge_attr = torch.tensor(features, dtype=torch.float)

    # Labels
    labels = torch.tensor(df['label'].values, dtype=torch.long)

    # Create PyG Data objects
    def create_data(split_name):
        mask = df['split'] == split_name
        edge_index = torch.tensor(df.loc[mask, ['sender', 'receiver']].values.T, dtype=torch.long)
        edge_attr_split = edge_attr[mask.values]
        y_split = labels[mask.values]
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr_split,
            y=y_split
        )
        return data

    train_data = create_data('train')
    val_data = create_data('val')
    test_data = create_data('test')

    return train_data, val_data, test_data


def get_args():
    parser = argparse.ArgumentParser(description="GNN Training Arguments")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate(default:0.01")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers (default: 128)")
    parser.add_argument("--gnn_type", type=str, choices=["GIN", "GAT", "GCN"], default="GCN",
                        help="Type of GNN model: GIN, GAT, or GCN (default: GCN)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--dataset_name", type=str,
                        choices=['AMLSim', 'BitcoinOTC', 'BitcoinAlpha', 'EllipticPP', 'PaySim', 'SAMLSim'],
                        default="AMLSim")
    parser.add_argument("--drop_ratio", type=float, default=0.0, help="drop_ratio of dataset (default:0.0")
    parser.add_argument("--use_graph_features", action="store_true", help="enable using graph features in GNN")
    parser.add_argument("--sliding_window_size", type=int, default=10, help="Window_size of for train test dynamically")
    return parser.parse_args()


def main():
    args = get_args()
    dataset_name = args.dataset_name
    drop_ratio = args.drop_ratio
    gnn_type = args.gnn_type
    epochs = args.epochs
    learning_rate = args.learning_rate
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    dropout = args.dropout
    sliding_window_size = args.sliding_window_size
    use_graph_features = args.use_graph_features
    print(f"\033[94m{dataset_name}\033[0m")
    print(f"\033[91m{gnn_type}\033[0m")
    match dataset_name:
        case 'PaySim':
            df = pd.read_parquet('./data/PaySim/processed/enriched_1_8_1_data.parquet')
        case 'AMLSim':
            df = pd.read_parquet('./data/AMLSim/processed/enriched_7_8_1_data.parquet')
        case 'SAMLSim':
            df = pd.read_parquet('./data/SAMLSim/processed/enriched_7_8_1_data.parquet')
        case 'EllipticPP':
            df = pd.read_parquet('./data/EllipticPP/processed/enriched_1_8_1_data.parquet')
        case 'BitcoinOTC':
            df = pd.read_parquet('./data/BitcoinOTC/processed/enriched_7_56_7_data.parquet')
        case 'BitcoinAlpha':
            df = pd.read_parquet('./data/BitcoinAlpha/processed/enriched_7_56_7_data.parquet')
        case _:
            raise RuntimeError(f"Dataset {dataset_name} is not defined.")

    # potential_drop_cols = potential_drop_cols + ['sender_mean_amt', 'sender_std_amt', 'sender_total_amt', 'sender_txn_count', 'sender_unique_receivers', 'receiver_mean_amt', 'receiver_std_amt', 'receiver_txn_count', 'receiver_unique_senders']
    # drop_cols = [col for col in potential_drop_cols if col in df.columns]
    use_features = [col for col in df.columns if col not in drop_cols]
    categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
    categorical_cols = [col for col in categorical_cols if col not in drop_cols]
    # print("categorical_cols:")
    # print(categorical_cols)
    numerical_cols = [col for col in use_features if col not in categorical_cols]
    # print("numerical_cols:")
    # print(numerical_cols)
    min_window_index = min(df['window_index'])
    max_window_index = max(df['window_index'])
    print(f"\033[93m{vars(args)}\033[0m")
    print(f"Min window_index: {min_window_index}")
    print(f"Max window_index: {max_window_index}")
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    lightning_root_dir = f"experiments/gnn_run/edge_level/sliding"
    for index, start_window_index in enumerate(range(min_window_index, max_window_index - sliding_window_size + 1)):
        print(f"\033[93m processing window: [{start_window_index}-{start_window_index+sliding_window_size})\033[0m")
        df_window = df[(df['window_index']) >= start_window_index & (df['window_index'] < start_window_index + sliding_window_size)]
        train_data, val_data, test_data = prepare_pg_graph_data(df_window, categorical_cols, numerical_cols,
                                                                drop_ratio=drop_ratio,
                                                                use_graph_features=use_graph_features)

        experiments_dir = f"{lightning_root_dir}/{dataset_name}/{gnn_type}/{experiment_datetime}/index_{index}"
        model = EdgeFraudNet(train_data.x.shape[1], edge_attr_dim=train_data.edge_attr.size(1), hidden_size=hidden_size,
                             num_layers=num_layers, dropout=dropout)
        csv_logger = CSVLogger(experiments_dir, version="")
        csv_logger.log_hyperparams(vars(args))
        lightningModule = LightningEdgeGNN(model, learning_rate=learning_rate)
        checkpoint_callback = ModelCheckpoint(
            monitor="val_avg_pr",
            mode="max",
            save_top_k=1,
            save_weights_only=True,
            dirpath=experiments_dir,
            filename="best-checkpoint"
        )
        print(train_data)
        print_label_counts(train_data, "Train")
        print(val_data)
        print_label_counts(val_data, "Validation")
        print(test_data)
        print_label_counts(test_data, "Test")
        # Start training and testing.
        train_loader = DataLoader([train_data], batch_size=1)
        val_loader = DataLoader([val_data], batch_size=1)
        test_loader = DataLoader([test_data], batch_size=1)
        trainer = L.Trainer(
            default_root_dir=experiments_dir,
            accelerator="auto",
            devices="auto",
            enable_progress_bar=True,
            logger=csv_logger,
            log_every_n_steps=5,
            max_epochs=epochs,
            callbacks=[checkpoint_callback]
        )
        print("Starting Training")
        trainer.fit(lightningModule, train_loader, val_loader)
        best_model_path = checkpoint_callback.best_model_path
        print(f"Best model path: {best_model_path}")
        best_model = LightningEdgeGNN.load_from_checkpoint(best_model_path, model=model, learning_rate=learning_rate)
        trainer.test(best_model, test_loader)


if __name__ == '__main__':
    main()
