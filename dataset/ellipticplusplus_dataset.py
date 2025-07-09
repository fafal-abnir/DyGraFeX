import pandas as pd
from tqdm import tqdm
from dataset.temporal_fraud_dataset import TemporalFraudDataset


class EllipticPPDataset(TemporalFraudDataset):
    def __init__(self, save_dir: str = "data/EllipticPP", window_size: int = 1, time_step: int = 1,
                 graph_window_size: int = 8,
                 force_reload=False):
        self.window_size = window_size
        self.time_step = time_step
        self.graph_window_size = graph_window_size
        super().__init__(save_dir=save_dir, window_size=window_size, time_step=time_step,
                         graph_window_size=graph_window_size, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return ["txs_features.csv", "AddrTx_edgelist.csv", "TxAddr_edgelist.csv", "txs_classes.csv"]

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
        parquet_file_path = self.processed_dir / "data.parquet"
        if parquet_file_path.exists() and not force_reload:
            print(f"Processed file already exists: {parquet_file_path}")
            return

        txs_features_path = self.raw_dir / self.raw_file_names[0]
        addrTx_edgelist_path = self.raw_dir / self.raw_file_names[1]
        txAddr_edgelist_path = self.raw_dir / self.raw_file_names[2]
        txs_classes_path = self.raw_dir / self.raw_file_names[3]
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
        txs_classes_df['label'] = txs_classes_df['label'].apply(lambda x: 0 if x >= 2 else 1)
        txs_features_classes_combined_df = pd.merge(txs_features_df, txs_classes_df, how='inner', on='txId')
        max_time_step = max(txs_features_df['timestamp'])
        result_feature_df = None
        for t in tqdm(range(1, max_time_step + 1 - self.window_size, self.time_step)):
            addr_tx_addr_df = pd.merge(addrtx_edgelist_df, txaddr_edgelist_df, how='inner', on=['txId'])
            txs_features_classes_combined_df_t = txs_features_classes_combined_df[
                txs_features_classes_combined_df['timestamp'] == t]
            df_merged_t = pd.merge(txs_features_classes_combined_df_t, addr_tx_addr_df, how='inner', on='txId')
            df_merged_t["window_index"] = t
            if result_feature_df is None:
                result_feature_df = df_merged_t
            else:
                result_feature_df = pd.concat([result_feature_df, df_merged_t], ignore_index=True)


        all_ids = pd.unique(result_feature_df[["sender", "receiver"]].values.ravel())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        result_feature_df["sender"] = result_feature_df["sender"].map(id_mapping)
        result_feature_df["receiver"] = result_feature_df["receiver"].map(id_mapping)
        result_feature_df.to_parquet(parquet_file_path, index=False)
        print(f"Saved processed data with all features to: {parquet_file_path}")
