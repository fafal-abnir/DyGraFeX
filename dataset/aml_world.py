import numpy as np
import pandas as pd
import torch
from typing import Literal
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from snapml import GraphFeaturePreprocessor
from dataset.temporal_fraud_dataset import TemporalFraudDataset
from utils.community_feature_extraction import enrich_with_edge_features, enrich_with_community_features
from utils.node_feature_extraction import calculate_pagerank, calculate_HIT


class AMLWorldDataset(TemporalFraudDataset):
    url = "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml/data"

    def __init__(self, save_dir: str = "data/AMLWorld", window_size: int = 1, target_time_step: int = 1,
                 graph_window_size: int = 2, illicit_level: Literal["HI", "LI"] = "HI",
                 data_size: Literal["Small", "Medium", "Large"] = "Small",
                 force_reload=False):
        save_dir = f"{save_dir}/{illicit_level}-{data_size}"
        self.window_days = window_size
        self.illicit_level = illicit_level
        self.data_size = data_size
        self.csv_file_name = f"{illicit_level}-{data_size}_Trans.csv"
        super().__init__(save_dir=save_dir, window_size=window_size, target_time_step=target_time_step,
                         graph_window_size=graph_window_size, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return ["HI-Large_Trans.csv", "HI-Medium_Trans.csv", "HI-Small_Trans.csv",
                "LI-Large_Trans.csv", "LI-Medium_Trans.csv", "LI-Small_Trans.csv"]

    def download(self):
        csv_file = self.raw_dir / self.csv_file_name

        if not csv_file.exists():
            raise RuntimeError(
                f"{self.csv_file_name} must be downloaded manually"
                f"download from: {self.url}"
                f"Place the accounts.csv and `transactions.csv` in:{self.raw_dir}")

    def prepare(self, force_reload=False):
        parquet_file_path = self.processed_dir / f"data_{self.illicit_level}_{self.data_size}.parquet"
        if parquet_file_path.exists() and not force_reload:
            print(f"Processed file already exists: {parquet_file_path}")
            return

        transaction_file = self.raw_dir / f"{self.illicit_level}-{self.data_size}_Trans.csv"

        if not transaction_file.exists():
            raise FileNotFoundError(f"Expected {self.raw_file_names} in raw_dir.")

        df = pd.read_csv(transaction_file)
        # Rename for consistency
        df.rename(columns={
            "Account": "sender",
            "Account.1": "receiver",
            "Amount Received": "amount",
            "Is Laundering": "label"
        }, inplace=True)
        df.columns = df.columns.str.lower()
        df.columns = df.columns.str.replace(r"\s+", "_", regex=True)
        df['date_time'] = pd.to_datetime((df['timestamp']))
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().any():
            raise ValueError("Found invalid timestamps after parsing. Check input data.")
        df["timestamp"] = (df["timestamp"].view("int64") // 1_000_000_000).astype(np.int64)  # seconds since epoch
        start_ts = df["timestamp"].min()
        df["timestamp"] = (df['timestamp'] - start_ts)
        df = df.sort_values('timestamp')
        # Window index
        window_size = pd.Timedelta(days=self.window_days)
        start_date = df["date_time"].min()
        df["window_index"] = ((df["date_time"] - start_date) // window_size).astype(int)

        # Reindex sender and receiver
        all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        df["sender"] = df["sender"].map(id_mapping)
        df["receiver"] = df["receiver"].map(id_mapping)
        cat_cols = ['from_bank', 'to_bank', 'receiving_currency', "payment_currency", "payment_format"]
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col].astype(str))
        df["edge_id"] = np.arange(1, len(df) + 1, dtype=np.int64)
        ordered = [
            "edge_id",  # 0
            "sender",  # 1
            "receiver",  # 2
            "timestamp",  # 3
            "from_bank",  # 4
            "to_bank",  # 5
            "amount",  # 6
            "receiving_currency",  # 7
            "payment_currency",  # 8
            "payment_format",  # 9
            "label",  # 10 (drop before transform)
            "window_index",  # 11 (not used by GFP)
        ]
        extra = [c for c in df.columns if c not in ordered]
        df = df[ordered + extra]

        # sanity: all numeric
        nn = df.select_dtypes(include=["object"]).columns.tolist()
        if nn:
            raise TypeError(f"Non-numeric columns remain: {nn}")
        # Save to parquet
        df.to_parquet(parquet_file_path, index=False)
        print(f"Saved processed data to: {parquet_file_path}")

    def enrich_with_gpf_features(self, tw_days=1, batch_size=128, num_threads=50):
        def get_num_gf_feats(gf):
            params = gf.get_params()
            feat_num = 0

            # add features names for the graph patterns
            for pattern in ['fan', 'degree', 'scatter-gather', 'temp-cycle', 'lc-cycle']:
                if pattern in params:
                    if params[pattern]:
                        bins = len(params[pattern + '_bins'])
                        if pattern in ['fan', 'degree']:
                            feat_num += 2 * bins
                        else:
                            feat_num += bins

            # add fan, deg, and ratio features
            for k in [0, 1, 2]:
                if k in params["vertex_stats_feats"]:
                    feat_num += 4

            # add avg, sum, min, max, median, var, skew, and kurtosis features
            for k in [3, 4, 5, 6, 7, 8, 9, 10]:
                if k in params["vertex_stats_feats"]:
                    feat_num += 4 * len(params["vertex_stats_cols"])

            return feat_num

        df = self.load_data()
        print("Setting the parameters of the graph feature preprocessor ")
        print("Batch size:", batch_size)
        gp = GraphFeaturePreprocessor()

        gf_params = {
            # Number of software threads to be used
            'num_threads': num_threads,

            # Enable account statistics
            'vertex_stats': True,
            'vertex_stats_cols': [3, 6],

            # Enable graph-pattern-based features
            'fan': False,
            'degree': False,
            'scatter-gather': True,
            'temp-cycle': True,
            'lc-cycle': True,
            'lc-cycle_len': 10,

            # Set time window parameters
            'time_window': tw_days * 24 * 3600,
            'vertex_stats_tw': tw_days * 24 * 3600,
            'scatter-gather_tw': 6 * 3600,
            'temp-cycle_tw': tw_days * 24 * 3600,
            'lc-cycle_tw': tw_days * 24 * 3600,
        }
        gp.set_params(gf_params)
        data_np = df.to_numpy(dtype=np.float64)
        num_rows = np.shape(data_np)[0]
        num_cols = np.shape(data_np)[1] + get_num_gf_feats(gp)

        data_np_gf = np.zeros((num_rows, num_cols))

        for i in tqdm(range(0, num_rows, batch_size)):
            batch_size_true = batch_size
            if i + batch_size > num_rows:
                batch_size_true = num_rows - i

            # Generate graph-based features for a batch of transactions
            Features_in = data_np[i:i + batch_size_true, :].astype('float64')
            Features_out = gp.transform(Features_in)
            data_np_gf[i:i + batch_size_true, :] = Features_out
        num_cols_with_gfp = data_np_gf.shape[1]
        base_cols = [str(c) for c in df.columns]
        num_cols_base =  min(len(base_cols), num_cols_with_gfp)
        cols = base_cols[:num_cols_base] + [f"gfp_{i}" for i in range(num_cols_with_gfp - num_cols_base)]
        parquet_gfp_file_path = self.processed_dir / f"data_{self.illicit_level}_{self.data_size}_gfp.parquet"
        pd.DataFrame(data_np_gf, columns=cols).to_parquet(
            parquet_gfp_file_path, engine="pyarrow", compression="snappy", index=False
        )


    def load_data(self) -> pd.DataFrame:
        file_path = self.processed_dir / f"data_{self.illicit_level}_{self.data_size}.parquet"
        return pd.read_parquet(file_path)

    def create_pyg_data_list(self, with_graph_features=False):
        time_step = self.time_step
        COMMUNITY_ALG = {'louvain': [1.0], 'leiden': [0.01, 0.04]}
        result_graph_feature_df = None
        max_time_step = 45
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
                                    col not in ['sender', 'receiver', 'label', 'timestamp', 'time', 'date',
                                                'window_index', 'tx_id',
                                                'laundering_type']]
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
