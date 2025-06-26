import pandas as pd
from dataset.temporal_fraud_dataset import TemporalFraudDataset


class SAMLSimDataset(TemporalFraudDataset):
    url = "https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml/data"

    def __init__(self, save_dir: str = "data/SAMLSim", window_size: int = 7, time_step: int = 1,
                 graph_window_size: int = 8,
                 normalize_feature_names=True):
        self.window_days = window_size
        super().__init__(save_dir=save_dir, window_size=window_size, time_step=time_step,
                         graph_window_size=graph_window_size, normalize_feature_names=normalize_feature_names)

    @property
    def raw_file_names(self):
        return "SAML-D.csv"

    def download(self):
        csv_file = self.raw_dir / self.raw_file_names

        if not csv_file.exists():
            raise RuntimeError(
                f"SAMLSim must be downloaded manually"
                f"download from: {self.url}"
                f"Place the accounts.csv and `transactions.csv` in:{self.raw_dir}")

    def prepare(self, force_reload=False):
        parquet_file_path = self.processed_dir / "data.parquet"
        if parquet_file_path.exists() and not force_reload:
            print(f"Processed file already exists: {parquet_file_path}")
            return

        transaction_file = self.raw_dir / self.raw_file_names

        if not transaction_file.exists():
            raise FileNotFoundError("Expected SAML-D.csv in raw_dir.")

        df = pd.read_csv(transaction_file)
        df.columns = df.columns.str.lower()
        # Rename for consistency
        df.rename(columns={
            "sender_account": "sender",
            "receiver_account": "receiver",
            "amount": "amount",
            "is_laundering": "label",
        }, inplace=True)

        df['date'] = pd.to_datetime((df['date']))
        df = df.sort_values('date')
        # Window index
        window_size = pd.Timedelta(days=self.window_days)
        start_date = df["date"].min()
        df["window_index"] = ((df["date"] - start_date) // window_size).astype(int)

        # Reindex sender and receiver
        all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        df["sender"] = df["sender"].map(id_mapping)
        df["receiver"] = df["receiver"].map(id_mapping)

        # Save to parquet
        df.to_parquet(parquet_file_path, index=False)
        print(f"Saved processed data to: {parquet_file_path}")
