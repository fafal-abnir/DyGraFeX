import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Literal

from utils.community_feature_extraction import enrich_with_edge_features, enrich_with_community_features
from utils.node_feature_extraction import enrich_with_pagerank, enrich_with_HIT
from utils.utils import generate_graph


class TemporalFraudDataset(ABC):
    def __init__(self, save_dir: str = "data", window_size: int = 7, time_step: int = 1, graph_window_size: int = 7,
                 normalize_feature_names=True):
        self.save_dir = Path(save_dir)
        self.raw_dir = Path(f"{self.save_dir}/raw_dir")
        self.processed_dir = Path(f"{self.save_dir}/processed")
        self.graph_window_size = graph_window_size
        self.window_size = window_size
        self.time_step = time_step
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.download()
        self.prepare(normalize_feature_names)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple[str, ...]]:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, normalize_feature_names):
        pass

    @abstractmethod
    def download(self):
        """Download the dataset to self.raw_dir"""
        pass

    def load_data(self) -> pd.DataFrame:
        file_path = self.processed_dir / "data.parquet"
        return pd.read_parquet(file_path)

    def get_window_range(self, start_window_index: int, end_window_index: int) -> pd.DataFrame:
        df = self.load_data()
        return df[(df["window_index"] >= start_window_index) & (df["window_index"] < end_window_index)].copy()

    def get_graph_window_range(self, start_window_index: int, end_window_index: int):
        df = self.get_window_range(start_window_index, end_window_index)
        return generate_graph(df, 'sender', 'receiver')

    def enrich_with_graph_features(self):
        time_step = self.time_step
        COMMUNITY_ALG = {'louvain': [1.0], 'leiden': [0.01, 0.04]}
        result_graph_feature_df = None
        df = self.load_data()
        max_time_step = max(df['window_index'])
        OUTPUT_PATH = self.processed_dir / f"g_{self.window_size}_{self.graph_window_size}_{time_step}_data.parquet"
        for time_step in range(1, max_time_step + 1 - self.graph_window_size, time_step):

            print(10 * "=")
            print(f"Time step:{time_step}, graph from time_step:[{time_step},{time_step + self.graph_window_size - 1}]")
            # financial graph for feature extraction
            transaction_graph = self.get_graph_window_range(time_step, time_step + self.graph_window_size)
            # target data for enriching with graph features
            # target_time_step_df_wallet = df[
            #     df_wallets_features_classes_combined['Time step'] == time_step + WINDOW_SIZE]
            target_df = self.get_window_range(time_step + self.graph_window_size,
                                              time_step + self.graph_window_size + 1)

            # enriching dataset
            r = enrich_with_edge_features(transaction_graph, target_df)
            r = enrich_with_pagerank(r, transaction_graph,
                                     weight_list=[None, "total_amount", "count", "pmi", "exclusivity_amt"])
            r = enrich_with_pagerank(r, transaction_graph,
                                     weight_list=[None, "total_amount", "count", "pmi", "exclusivity_amt"],
                                     normalize=True)
            print("finished page rank features")
            r = enrich_with_HIT(r, transaction_graph,
                                weight_list=[None, "total_amount", "count", "pmi", "exclusivity_amt"])
            r = enrich_with_community_features(transaction_graph, COMMUNITY_ALG, ['unweighted'], r)
            r['time_step'] = time_step + self.graph_window_size
            if result_graph_feature_df is None:
                result_graph_feature_df = r
            else:
                result_graph_feature_df = pd.concat([result_graph_feature_df, r], ignore_index=True)

        print("columns names:")
        print(result_graph_feature_df.columns)
        result_graph_feature_df.to_parquet(OUTPUT_PATH, engine='pyarrow', index=False)

    def enrich_non_graph_features(self, df):
        sender_agg = df.groupby("sender").agg({
            "amount": ["mean", "std", "sum", "count"],
            "receiver": "nunique"
        })
        sender_agg.columns = ["sender_mean_amt", "sender_std_amt", "sender_total_amt", "sender_txn_count",
                              "sender_unique_receivers"]
        df = df.merge(sender_agg, on="sender", how="left")

        receiver_agg = df.groupby("receiver").agg({
            "amount": ["mean", "std", "count"],
            "sender": "nunique"
        })
        receiver_agg.columns = ["receiver_mean_amt", "receiver_std_amt", "receiver_txn_count",
                                "receiver_unique_senders"]
        df = df.merge(receiver_agg, on="receiver", how="left")

        return df

    def prepare_training_dataframe_with_graph_features(self, force_reload=True):
        """
        Prepares the training dataset by enriching transaction data with graph-based
        and tabular (non-graph) features, and save it as file in processed directory.
        """
        graph_parquet_path = self.processed_dir / f"g_{self.window_size}_{self.graph_window_size}" \
                                                  f"_{self.time_step}_data.parquet"

        enriched_parquet_path = self.processed_dir / f"enriched_{self.window_size}_{self.graph_window_size}" \
                                                     f"_{self.time_step}_data.parquet"
        if enriched_parquet_path.exists() and not force_reload:
            print(f"{enriched_parquet_path} exist no need to process.")
            return

        graph_df = pd.read_parquet(graph_parquet_path)
        result_enriched_df = None
        time_step = self.time_step
        min_time_step = min(graph_df['window_index'])
        max_time_step = max(graph_df['window_index'])
        for t in range(min_time_step, max_time_step, time_step):

            print(10 * "=")
            print(f"Enriching step:{t}")
            # financial graph for feature extraction
            feature_df = graph_df[graph_df['window_index'] == t]
            enriched_df = self.enrich_non_graph_features(feature_df)

            if result_enriched_df is None:
                result_enriched_df = enriched_df
            else:
                result_enriched_df = pd.concat([result_enriched_df, enriched_df], ignore_index=True)

        print("columns names:")
        print(result_enriched_df.columns)
        result_enriched_df.to_parquet(enriched_parquet_path, engine='pyarrow', index=False)
