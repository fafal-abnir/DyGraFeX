import pandas as pd
from tqdm import tqdm
import os
import torch
from dataset.temporal_fraud_dataset import TemporalFraudDataset
from utils.community_feature_extraction import enrich_with_edge_features, enrich_with_community_features
from utils.node_feature_extraction import calculate_pagerank, calculate_HIT
from torch_geometric.data import Data


class EllipticPPDataset(TemporalFraudDataset):
    def __init__(self, save_dir: str = "data/EllipticPP", window_size: int = 1, target_time_step: int = 1,
                 graph_window_size: int = 8,
                 force_reload=False):
        self.window_size = window_size
        self.time_step = time_step
        self.graph_window_size = graph_window_size
        super().__init__(save_dir=save_dir, window_size=window_size, target_time_step=target_time_step,
                         graph_window_size=graph_window_size, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return ["txs_features.csv", "AddrTx_edgelist.csv", "TxAddr_edgelist.csv", "txs_classes.csv",
                "wallets_features.csv"]

    def download(self):
        url = "https://drive.google.com/drive/folders/1MRPXz79Lu_JGLlJ21MDfML44dKN9R08l"
        for csv_file_name in self.raw_file_names:
            csv_file = self.raw_dir / csv_file_name
            if not csv_file.exists():
                raise RuntimeError(
                    f"{csv_file_name}must be downloaded manually from"
                    f"{url} and place the file in: {csv_file}")

    def prepare(self, force_reload=False):
        print("Starting preparing elliptic++ dataset.")
        parquet_transaction_file_path = self.processed_dir / "data.parquet"
        parquet_addr_features_file_path = self.processed_dir / "addr_features.parquet"
        if parquet_transaction_file_path.exists() and not force_reload:
            print(f"Processed file already exists: {parquet_transaction_file_path}")
            return

        txs_features_path = self.raw_dir / self.raw_file_names[0]
        addrTx_edgelist_path = self.raw_dir / self.raw_file_names[1]
        txAddr_edgelist_path = self.raw_dir / self.raw_file_names[2]
        txs_classes_path = self.raw_dir / self.raw_file_names[3]
        addr_features_path = self.raw_dir / self.raw_file_names[4]
        # if not wallets_features_classes_combined_path.exists() or not AddrAddr_edgelist_path.exists():
        #     raise FileNotFoundError(
        #         f"Expected file at {wallets_features_classes_combined_path} or {AddrAddr_edgelist_path}, "
        #         f"but it doesn't exist.")

        txs_features_df = pd.read_csv(txs_features_path)
        txs_features_df.rename(columns={
            "Time step": "timestamp",
            "total_BTC": "amount",
        }, inplace=True)
        addrtx_edgelist_df = pd.read_csv(addrTx_edgelist_path)
        addrtx_edgelist_df.rename(columns={
            "input_address": "sender",
        }, inplace=True)
        txaddr_edgelist_df = pd.read_csv(txAddr_edgelist_path)
        txaddr_edgelist_df.rename(columns={
            "output_address": "receiver",
        }, inplace=True)

        txs_classes_df = pd.read_csv(txs_classes_path)
        txs_classes_df.rename(columns={
            "class": "label"
        }, inplace=True)
        addr_features_df = pd.read_csv(addr_features_path)
        addr_features_df.rename(columns={
            "Time step": "timestamp",
        }, inplace=True)
        txs_classes_df['label'] = txs_classes_df['label'].map({1: 1, 2: 0, 3: 3})
        # txs_classes_df['label'] = txs_classes_df['label'].apply(lambda x: 0 if x >= 2 else 1)
        txs_features_classes_combined_df = pd.merge(txs_features_df, txs_classes_df, how='inner', on='txId')

        max_time_step = max(txs_features_df['timestamp'])
        result_transaction_df = None
        for t in tqdm(range(1, max_time_step + 1 - self.window_size, self.time_step)):
            addr_tx_addr_df = pd.merge(addrtx_edgelist_df, txaddr_edgelist_df, how='inner', on=['txId'])
            txs_features_classes_combined_df_t = txs_features_classes_combined_df[
                txs_features_classes_combined_df['timestamp'] == t]
            df_merged_t = pd.merge(txs_features_classes_combined_df_t, addr_tx_addr_df, how='inner', on='txId')
            df_merged_t["window_index"] = t
            if result_transaction_df is None:
                result_transaction_df = df_merged_t
            else:
                result_transaction_df = pd.concat([result_transaction_df, df_merged_t], ignore_index=True)

        all_ids = pd.unique(result_transaction_df[["sender", "receiver"]].values.ravel())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        addr_features_df['address'] = addr_features_df['address'].map(id_mapping)
        result_transaction_df["sender"] = result_transaction_df["sender"].map(id_mapping)
        result_transaction_df["receiver"] = result_transaction_df["receiver"].map(id_mapping)
        addr_features_df.to_parquet(parquet_addr_features_file_path, index=False)
        result_transaction_df.to_parquet(parquet_transaction_file_path, index=False)
        print(f"Saved processed data with all features to: {parquet_addr_features_file_path} "
              f"and {parquet_transaction_file_path}")

    def create_pyg_data_list(self, with_graph_features=False):
        time_step = self.time_step
        COMMUNITY_ALG = {'louvain': [1.0], 'leiden': [0.01, 0.04]}
        result_graph_feature_df = None
        addr_features_df = pd.read_parquet(self.processed_dir / f"addr_features.parquet")
        max_time_step = 49
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
            # financial graph for feature extraction
            transaction_graph = self.get_graph_window_range(time_step, time_step + self.graph_window_size)
            print("Transaction graph summary(reference graph)")
            print(transaction_graph.summary())
            # target data for enriching with graph features
            # target_time_step_df_wallet = df[
            #     df_wallets_features_classes_combined['Time step'] == time_step + WINDOW_SIZE]
            target_df = self.get_window_range(time_step + self.graph_window_size,
                                              time_step + self.graph_window_size + 1)

            # enriching dataset
            involved_nodes_target = set(target_df['sender']).union(set(target_df['receiver']))
            node_df = addr_features_df[
                addr_features_df['address'].isin(involved_nodes_target)
            ]
            node_df = node_df.drop(columns=['timestamp'])
            node_df = node_df.drop_duplicates(subset=['address'], keep='first')
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
