import torch
from pathlib import Path
import pytorch_lightning as L
from pytorch_lightning.loggers import CSVLogger
import argparse
from datetime import datetime
from torch_geometric.data import Data, DataLoader
from models.gnn_sampled.model import EdgeFraudNetFlexible
from models.gnn_sampled.lightning_modules import LightningEdgeGNN
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def compute_global_random_edge_ap(dataset_list, dataset_name="Train"):
    total_positives = 0
    total_negatives = 0

    for data in dataset_list:
        edge_labels = data.edge_label
        total_positives += (edge_labels == 1).sum().item()
        total_negatives += (edge_labels == 0).sum().item()

    total_edges = total_positives + total_negatives

    if total_edges == 0:
        ap_random = 0.0
    else:
        ap_random = total_positives / total_edges

    print(f"=== Global Edge Statistics {dataset_name} ===")
    print(f"Total positive edges: {total_positives}")
    print(f"Total negative edges: {total_negatives}")
    print(f"Random AP baseline: {ap_random:.4f}")
    print("==============================")


def load_pyg_data_list(data_dir: Path, start: int, end: int):
    data_list = []
    for idx in range(start, end):
        file_path = data_dir / f'data_{idx}.pt'

        if not file_path.exists():
            print(f"[WARNING] {file_path} does not exist. Skipping.")
            continue
        data = torch.load(file_path)
        data_list.append(data)
    return data_list


def get_args():
    parser = argparse.ArgumentParser(description="GNN Training Arguments")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="learning rate(default:0.01")
    parser.add_argument("--hidden_size", type=int, default=128, help="Size of hidden layers (default: 128)")
    parser.add_argument("--gnn_type", type=str,
                        choices=["GIN", "GAT", "GCN", "GINE", "NNConv", "GATv2", "TransformerConv"], default="GCN",
                        help="Type of GNN model: GIN, GAT, or GCN (default: GCN)")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
    parser.add_argument("--dataset_name", type=str,
                        choices=['AMLSim', 'BitcoinOTC', 'BitcoinAlpha', 'EllipticPP', 'PaySim', 'SAMLSim'],
                        default="EllipticPP")
    parser.add_argument("--drop_ratio", type=float, default=0.0, help="drop_ratio of dataset (default:0.0")
    parser.add_argument("--use_graph_features", action="store_true", help="enable using graph features in GNN")
    parser.add_argument("--feature_agg_fun", type=str,
                        choices=['min', 'max', 'mean'],
                        default="mean")
    return parser.parse_args()


def main():
    args = get_args()
    dataset_name = args.dataset_name
    gnn_type = args.gnn_type
    epochs = args.epochs
    learning_rate = args.learning_rate
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    use_graph_features = args.use_graph_features
    dropout = args.dropout
    print(f"\033[94m{dataset_name}\033[0m")
    print(f"\033[91m{gnn_type}\033[0m")
    data_list_dir = None
    match dataset_name:
        case 'PaySim':
            pass
            # data_list_dir = pd.read_parquet('./data/PaySim/processed/enriched_1_8_1_data.parquet')
        case 'AMLSim':
            data_list_dir = Path(f'./data/AMLSim/processed/PyG_list_data_GF_{use_graph_features}_7_8_1')
            train_data_list = load_pyg_data_list(data_list_dir, 9, 21)
            val_data_list = load_pyg_data_list(data_list_dir, 22, 25)
            test_data_list = load_pyg_data_list(data_list_dir, 26, 28)
        case 'SAMLSim':
            data_list_dir = Path(f'./data/SAMLSim/processed/PyG_list_data_GF_{use_graph_features}_7_8_1')
            train_data_list = load_pyg_data_list(data_list_dir, 9, 36)
            val_data_list = load_pyg_data_list(data_list_dir, 37, 40)
            test_data_list = load_pyg_data_list(data_list_dir, 41, 45)
        case 'EllipticPP':
            data_list_dir = Path(f'./data/EllipticPP/processed/PyG_list_data_GF_{use_graph_features}_1_8_1')
            train_data_list = load_pyg_data_list(data_list_dir, 9, 34)
            val_data_list = load_pyg_data_list(data_list_dir, 35, 37)
            test_data_list = load_pyg_data_list(data_list_dir, 38, 42)


        case 'BitcoinOTC':
            pass
            # df = pd.read_parquet('./data/BitcoinOTC/processed/enriched_7_56_7_data.parquet')
        case 'BitcoinAlpha':
            pass
            # df = pd.read_parquet('./data/BitcoinAlpha/processed/enriched_7_56_7_data.parquet')
        case _:
            raise RuntimeError(f"Dataset {dataset_name} is not defined.")
    compute_global_random_edge_ap(train_data_list, "Train")
    compute_global_random_edge_ap(val_data_list, "Val")
    compute_global_random_edge_ap(test_data_list, "Test")
    lightning_root_dir = "experiments/gnn_run/edge_level/list-data/"
    experiment_datetime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    experiments_dir = f"{lightning_root_dir}/{dataset_name}/{gnn_type}/{experiment_datetime}"
    model = EdgeFraudNetFlexible(train_data_list[0].x.shape[1], edge_attr_dim=train_data_list[0].edge_attr.size(1),
                                 hidden_size=hidden_size,
                                 gnn_type=gnn_type, num_layers=num_layers, dropout=dropout)
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
    early_stopping_callback = EarlyStopping(
        monitor='val_au_roc',
        mode='max',
        patience=30
    )
    print(f"\033[93m{vars(args)}\033[0m")
    # Start training and testing.
    train_loader = DataLoader(train_data_list, batch_size=1)
    # cluster_data = ClusterData(train_data, num_parts=100, recursive=False)
    # train_loader = ClusterLoader(cluster_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=1)
    test_loader = DataLoader(test_data_list, batch_size=1)
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
    print(f"Model from best checkpoint")
    trainer.test(best_model, test_loader)


if __name__ == '__main__':
    main()
