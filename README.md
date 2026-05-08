
# Graph Feature Engineering for Improving Fraud Detection

This project presents a scalable framework that enhances fraud detection by integrating graph-based features with traditional (default) features using join operations. The current prototype is implemented in Python using pandas for data processing, but the logic can be easily adapted to other frameworks or big data libraries. The framework operates in two stages: an offline phase, where graph features are computed from historical transaction data and stored for later use, and an online phase, where these precomputed features are retrieved and combined with new transaction data for prediction. This design removes the need for real-time graph computation, reduces latency, and supports scalability in large-scale financial systems.





## Getting Started

These instructions will get you the project up and running on your local machine for development and testing purposes.

Prerequisites : 
* Python 3.10 or higher
* [Poetry](https://python-poetry.org/)

### Installing
Clone the repository to your local machine.
To activate the virtual environment and install the required packages, navigate to the project directory and run the following commands:
```
poetry shell
poetry install
```
Abstract dataset API

We provide an abstract class `TemporalFraudDataset` that standardizes data loading, windowing, and feature engineering. The prototype is in Python (pandas), but the logic can be ported to other libraries. To use a new dataset, implement:

- `download(self)`: fetch or place raw files into `self.raw_dir`
- `prepare(self, force_reload)`: read raw data, produce a parquet file at `self.processed_dir/"data.parquet"` with (at minimum) columns:
  - `sender`, `receiver` (IDs)
  - `amount` (numeric)
  - `window_index` (int; time window id)

Once `data.parquet` exists, call:
- `enrich_with_graph_features()`: computes graph features offline over rolling graph windows
- `prepare_training_dataframe_with_graph_features()`: joins graph features with tabular (non-graph) features for training

Integration is done via joins: precomputed graph features are merged with default features on transaction keys/time windows. See `saml_sim_dataset.py` and `ellipticplusplus.py` for concrete examples.

For the AMLWorld datasets, this repository provides an example of how to enrich transaction data with another method:Graph Feature Processing (GFP). The enrichment process follows the methodology introduced in “GFP: Graph Feature Processing for Financial Transaction Networks”[GFP](https://dl.acm.org/doi/10.1145/3677052.3698674). 
This allows us to compare our method with GFP baseline.

For each dataset, the **raw transaction data** is stored in the `[dataset_name]/raw_dir` directory, while the **processed and enriched datasets** are saved in `[dataset_name]/processed`. This structure keeps raw data separate from generated features and allows reproducible, organized data preparation for graph-based fraud detection experiments.

### Model Evaluation

The repository also includes tools to evaluate the impact of graph-based features on different models and datasets:

- **ensemble_model.py** — This script trains ensemble models such as **XGBoost**, **LightGBM**, and **CatBoost** to compare the performance of models **with and without graph features**. It can be used to assess the efficiency of different graph feature engineering strategies across datasets.

- **gnn_run_list.py** — This file evaluates how graph features influence the performance of **Graph Neural Network (GNN)** models.

You can find example commands for running these experiments in the **commands** file.

