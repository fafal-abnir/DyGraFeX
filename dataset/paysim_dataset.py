import pandas as pd
from dataset.temporal_fraud_dataset import TemporalFraudDataset


class PaySimDataset(TemporalFraudDataset):
    def __init__(self, save_dir: str = "data/PaySim", window_days: int = 1):
        self.window_days = window_days
        super().__init__(save_dir=save_dir, window_size=window_days)

    @property
    def raw_file_names(self):
        return "PS_20174392719_1491204439457_log.csv"

    def download(self):
        url = "https://www.kaggle.com/datasets/ntnu-testimon/paysim1"
        csv_file = self.raw_dir / self.raw_file_names

        if not csv_file.exists():
            raise RuntimeError(
                f"PaySim must be downloaded manually from Kaggle:"
                f"{url} and place the file in: {self.raw_dir / self.raw_file_names}")

    def prepare(self, force_reload=False):
        parquet_file_path = self.processed_dir / "data.parquet"
        if parquet_file_path.exists() and not force_reload:
            print(f"Processed file already exists: {parquet_file_path}")
            return

        csv_path = self.raw_dir / self.raw_file_names
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected file at {csv_path}, but it doesn't exist.")

        df = pd.read_csv(csv_path)

        df.rename(columns={
            "nameOrig": "sender",
            "nameDest": "receiver",
            "step": "timestamp",
            "isFraud": "label"
        }, inplace=True)

        steps_per_day = 24
        df["window_index"] = (df["timestamp"] // (steps_per_day * self.window_days)).astype(int)

        all_ids = pd.unique(df[["sender", "receiver"]].values.ravel())
        id_mapping = {old_id: new_id for new_id, old_id in enumerate(all_ids)}
        df["sender"] = df["sender"].map(id_mapping)
        df["receiver"] = df["receiver"].map(id_mapping)
        df.to_parquet(parquet_file_path, index=False)
        print(f"Saved processed data with all features to: {parquet_file_path}")

