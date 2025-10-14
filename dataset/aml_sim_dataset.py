import pandas as pd
import torch
from dataset.temporal_fraud_dataset import TemporalFraudDataset
from utils.community_feature_extraction import enrich_with_edge_features, enrich_with_community_features
from utils.node_feature_extraction import calculate_pagerank, calculate_HIT
from torch_geometric.data import Data


class AMLSimDataset(TemporalFraudDataset):

    def __init__(self, save_dir: str = "data/AMLSim", window_size: int = 7, target_time_step: int = 1,
                 graph_window_size: int = 8,
                 force_reload=False):
        self.window_size = window_size
        self.time_step = time_step
        self.graph_window_size = graph_window_size
        super().__init__(save_dir=save_dir, window_size=window_size, target_time_step=target_time_step,
                         graph_window_size=graph_window_size, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return ("accounts.csv", "transactions.csv")

    def download(self):
        csv_file = self.raw_dir / self.raw_file_names[0]

        if not csv_file.exists():
            raise RuntimeError(
                f"AMLSim must be generated or downloaded manually"
                f"Place the accounts.csv and `transactions.csv` in:{self.raw_dir}")

    def prepare(self, force_reload=False):
        parquet_file_path = self.processed_dir / "data.parquet"
        if parquet_file_path.exists() and not force_reload:
            print(f"Processed file already exists: {parquet_file_path}")
            return

        accounts_file = self.raw_dir / self.raw_file_names[0]
        transactions_file = self.raw_dir / self.raw_file_names[1]

        if not accounts_file.exists() or not transactions_file.exists():
            raise FileNotFoundError("Expected accounts.csv and transactions.csv in raw_dir.")

        df = pd.read_csv(transactions_file)
        df.columns = df.columns.str.lower()
        # Rename for consistency
        df.rename(columns={
            "sender_account_id": "sender",
            "receiver_account_id": "receiver",
            "tx_amount": "amount",
            "is_fraud": "label",
        }, inplace=True)

        # Window index
        min_time = df["timestamp"].min()
        df["window_index"] = ((df["timestamp"] - min_time) // self.window_size).astype(int)

        # Reindex sender and receiver
        all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        df["sender"] = df["sender"].map(id_mapping)
        df["receiver"] = df["receiver"].map(id_mapping)
        df = pd.get_dummies(df, columns=['tx_type'])
        # Save to parquet
        df.to_parquet(parquet_file_path, index=False)
        print(f"Saved processed data to: {parquet_file_path}")

    def create_pyg_data_list(self, with_graph_features=False):
        time_step = self.time_step
        COMMUNITY_ALG = {'louvain': [1.0], 'leiden': [0.01, 0.04]}
        result_graph_feature_df = None
        max_time_step = 28
        data_dir = self.processed_dir / f"PyG_list_data_GF_{with_graph_features}_" \
                                        f"{self.window_size}_{self.graph_window_size}_{self.time_step}/"

        data_dir.mkdir(parents=True, exist_ok=True)
        for time_step in range(1, max_time_step + 1 - self.graph_window_size, time_step):
            print(10 * "=")
            print(f"Time step:{time_step}, graph from time_step:[{time_step},{time_step + self.graph_window_size - 1}]")
            idx = time_step + self.graph_window_size
            file_path = data_dir / f'data_{idx}.pt'
            if file_path.exists():
                print(f"data {idx} already exists. Skipping.")
                continue
            target_df = self.get_window_range(time_step + self.graph_window_size,
                                              time_step + self.graph_window_size + 1)
            if len(target_df) == 0:
                print(f"print no graph in target graph {time_step}")
                break
            # financial graph for feature extraction
            transaction_graph = self.get_graph_window_range(time_step, time_step + self.graph_window_size)

            print("Transaction graph summary(reference graph)")
            print(transaction_graph.summary())
            # target data for enriching with graph features
            # target_time_step_df_wallet = df[
            #     df_wallets_features_classes_combined['Time step'] == time_step + WINDOW_SIZE]

            # enriching dataset
            involved_nodes_target = set(target_df['sender']).union(set(target_df['receiver']))
            node_df = pd.DataFrame({'address': list(involved_nodes_target)})
            # node_df = node_df.drop(columns=['timestamp'])
            # node_df = node_df.drop_duplicates(subset=['address'], keep='first')
            weight_list = [None, "total_amount", "count", "pmi", "exclusivity_amt"]
            if with_graph_features:
                for weight_name in weight_list:
                    pagerank_dict = calculate_pagerank(transaction_graph, weight_name, normalize=False)
                    node_df[f'pagerank_{weight_name}'] = node_df['address'].map(
                        pagerank_dict).fillna(0)
                    normalized_pagerank_dict = calculate_pagerank(transaction_graph, weight_name, normalize=True)
                    node_df[f'pagerank_{weight_name}_norm'] = node_df['address'].map(
                        normalized_pagerank_dict).fillna(0)
                    auth_dict, hub_dict = calculate_HIT(transaction_graph, weight_name)
                    node_df[f'auth_{weight_name}'] = node_df['address'].map(auth_dict).fillna(0)
                    node_df[f'hub_{weight_name}'] = node_df['address'].map(hub_dict).fillna(0)
                print("finished PageRank and HIT features")
            else:
                sender_agg = target_df.groupby("sender").agg({
                    "amount": ["mean", "std", "sum", "count"],
                    "receiver": "nunique"
                })
                sender_agg.columns = ["sender_mean_amt", "sender_std_amt", "sender_total_amt", "sender_txn_count",
                                      "sender_unique_receivers"]
                node_df = node_df.merge(sender_agg, left_on="address", right_on="sender", how="left")

                receiver_agg = target_df.groupby("receiver").agg({
                    "amount": ["mean", "std", "count"],
                    "sender": "nunique"
                })
                receiver_agg.columns = ["receiver_mean_amt", "receiver_std_amt", "receiver_txn_count",
                                        "receiver_unique_senders"]
                node_df = node_df.merge(receiver_agg, left_on="address", right_on="receiver", how="left")
                node_df = node_df.fillna(0)

            node_ids = node_df['address'].values
            node_id_to_idx = {addr: idx for idx, addr in enumerate(node_ids)}
            x = torch.tensor(node_df.drop(['address'], axis=1).values, dtype=torch.float)
            r = target_df
            if with_graph_features:
                r = enrich_with_edge_features(transaction_graph, r)
                r = enrich_with_community_features(transaction_graph, COMMUNITY_ALG, ['unweighted'], r)
            r = r.fillna(0)
            missing_senders = set(r['sender']) - set(node_id_to_idx.keys())
            missing_receivers = set(r['receiver']) - set(node_id_to_idx.keys())
            assert len(missing_senders) == 0, f"Missing senders in node_df: {missing_senders}"
            assert len(missing_receivers) == 0, f"Missing receivers in node_df: {missing_receivers}"
            # Creating edge_index and edge_label_index and labels for data object
            src = r['sender'].map(node_id_to_idx).values
            dst = r['receiver'].map(node_id_to_idx).values
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_feature_columns = [col for col in r.columns if
                                    col not in ['sender', 'receiver', 'label', 'timestamp', 'window_index', 'txId']]
            r[edge_feature_columns] = r[edge_feature_columns].astype(float)
            edge_attr = torch.tensor(r[edge_feature_columns].values, dtype=torch.float)
            labeled_edges = r[r['label'].isin([0, 1])]
            src_label = labeled_edges['sender'].map(node_id_to_idx).values
            dst_label = labeled_edges['receiver'].map(node_id_to_idx).values
            edge_label_index = torch.tensor([src_label, dst_label], dtype=torch.long)
            edge_label = torch.tensor(labeled_edges['label'].values, dtype=torch.long)
            edge_label_attr = torch.tensor(labeled_edges[edge_feature_columns].values, dtype=torch.float)
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_label=edge_label,
                edge_label_index=edge_label_index,
                edge_label_attr=edge_label_attr
            )
            data.node_ids = torch.tensor(node_ids)
            print(data)
            torch.save(data, file_path)
