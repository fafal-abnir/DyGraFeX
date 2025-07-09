import pandas as pd
import urllib.request
import gzip
import shutil

from dataset.temporal_fraud_dataset import TemporalFraudDataset


class BitcoinOTCDataset(TemporalFraudDataset):

    def __init__(self, save_dir: str = "data/BitcoinOTC", window_size: int = 7, time_step: int = 7,
                 graph_window_size: int = 56,
                 force_reload=False):
        self.window_size = window_size
        self.time_step = time_step
        self.graph_window_size = graph_window_size
        super().__init__(save_dir=save_dir, window_size=window_size, time_step=time_step,
                         graph_window_size=graph_window_size, force_reload=force_reload)

    @property
    def raw_file_names(self):
        return "soc-sign-bitcoinotc.csv"

    def prepare(self, force_reload=False, label_threshold=-5):
        parquet_file_path = self.processed_dir / "data.parquet"
        if parquet_file_path.exists() and not force_reload:
            print(f"Processed file already exists: {parquet_file_path}")
            return

        csv_path = self.raw_dir / self.raw_file_names
        df = pd.read_csv(csv_path, header=None)
        df.columns = ["sender", "receiver", "rating", "timestamp"]

        # Convert timestamp and assign window index
        df["datetime"] = pd.to_datetime(df["timestamp"], unit='s')
        df['amount'] = 1
        df["label"] = (df["rating"] > 0).astype(int)

        min_time = df["datetime"].min()
        df["window_index"] = ((df["datetime"] - min_time).dt.days // self.window_size).astype(int)

        output = df[["sender", "receiver", "timestamp", "amount", "label", "window_index"]].copy()
        output.to_parquet(parquet_file_path, index=False)
        print(f"Saved processed data to: {parquet_file_path}")

    def download(self):
        url = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
        gz_file = self.raw_dir / "soc-sign-bitcoinotc.csv.gz"
        csv_file = self.raw_dir / self.raw_file_names

        if not csv_file.exists():
            print("Downloading Bitcoin OTC dataset...")
            urllib.request.urlretrieve(url, gz_file)
            print("Download complete. Extracting...")
            with gzip.open(gz_file, 'rb') as f_in, open(csv_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_file.unlink()
            print(f"Extracted to: {csv_file}")
        else:
            print(f"Raw file already exists: {csv_file}")
