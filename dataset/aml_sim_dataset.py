import pandas as pd
from dataset.temporal_fraud_dataset import TemporalFraudDataset


class AMLSimDataset(TemporalFraudDataset):

    def __init__(self, save_dir: str = "data/AMLSim", window_days: int = 7):
        self.window_days = window_days
        super().__init__(save_dir=save_dir, window_size=window_days)

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
        df["window_index"] = ((df["timestamp"] - min_time)// self.window_days).astype(int)

        # Reindex sender and receiver
        all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        df["sender"] = df["sender"].map(id_mapping)
        df["receiver"] = df["receiver"].map(id_mapping)

        # Save to parquet
        df.to_parquet(parquet_file_path, index=False)
        print(f"Saved processed data to: {parquet_file_path}")



