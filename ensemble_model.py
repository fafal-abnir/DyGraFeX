import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    average_precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import argparse

graph_features = ['total_amount', 'pmi', 'count', 'exclusivity_amt',
                  'sender_pagerank_None', 'receiver_pagerank_None',
                  'sender_pagerank_total_amount', 'receiver_pagerank_total_amount',
                  'sender_pagerank_count', 'receiver_pagerank_count',
                  'sender_pagerank_pmi', 'receiver_pagerank_pmi',
                  'sender_pagerank_exclusivity_amt', 'receiver_pagerank_exclusivity_amt',
                  'sender_pagerank_None_norm', 'receiver_pagerank_None_norm',
                  'sender_pagerank_total_amount_norm',
                  'receiver_pagerank_total_amount_norm', 'sender_pagerank_count_norm',
                  'receiver_pagerank_count_norm', 'sender_pagerank_pmi_norm',
                  'receiver_pagerank_pmi_norm', 'sender_pagerank_exclusivity_amt_norm',
                  'receiver_pagerank_exclusivity_amt_norm', 'sender_auth_None',
                  'receiver_auth_None', 'sender_hub_None', 'receiver_hub_None',
                  'sender_auth_total_amount', 'receiver_auth_total_amount',
                  'sender_hub_total_amount', 'receiver_hub_total_amount',
                  'sender_auth_count', 'receiver_auth_count', 'sender_hub_count',
                  'receiver_hub_count', 'sender_auth_pmi', 'receiver_auth_pmi',
                  'sender_hub_pmi', 'receiver_hub_pmi', 'sender_auth_exclusivity_amt',
                  'receiver_auth_exclusivity_amt', 'sender_hub_exclusivity_amt',
                  'receiver_hub_exclusivity_amt', 'community_belonging_louvain_1.0',
                  'destination_community_neighbor_count_louvain_1.0',
                  'destination_community_neighbor_count_louvain_1.0_norm',
                  'destination_community_neighbor_weighted_degree_louvain_1.0',
                  'destination_community_neighbor_weighted_degree_louvain_1.0_norm',
                  'community_belonging_leiden_0.01',
                  'destination_community_neighbor_count_leiden_0.01',
                  'destination_community_neighbor_count_leiden_0.01_norm',
                  'destination_community_neighbor_weighted_degree_leiden_0.01',
                  'destination_community_neighbor_weighted_degree_leiden_0.01_norm',
                  'community_belonging_leiden_0.04',
                  'destination_community_neighbor_count_leiden_0.04',
                  'destination_community_neighbor_count_leiden_0.04_norm',
                  'destination_community_neighbor_weighted_degree_leiden_0.04',
                  'destination_community_neighbor_weighted_degree_leiden_0.04_norm',
                  ]


# default_features_PaySim = ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'oldbalanceOrg', 'newbalanceOrig',
#                            'oldbalanceDest', 'newbalanceDest', 'type_PAYMENT', 'type_TRANSFER', 'amount']


def drop_useless_columns(df, verbose=True):
    potential_drop_cols = ['time_step', 'isFlaggedFraud', 'laundering_type', 'timestamp', 'id', 'tx_id', 'alert_id',
                           'time', 'date',
                           ]
    # potential_drop_cols = potential_drop_cols + ['sender_mean_amt', 'sender_std_amt', 'sender_total_amt', 'sender_txn_count', 'sender_unique_receivers', 'receiver_mean_amt', 'receiver_std_amt', 'receiver_txn_count', 'receiver_unique_senders']
    drop_cols = [col for col in potential_drop_cols if col in df.columns]
    use_features = [col for col in df.columns if col not in drop_cols]
    if verbose:
        print(f"Drop cols:{drop_cols}")
        print(f'Used_features:{use_features}')
    return df[use_features]


def split_train_test_by_window(df, window_col='window_index', train_ratio: float = 0.8, remove_leakage=False):
    unique_windows = sorted(df[window_col].unique())
    n_train = int(len(unique_windows) * train_ratio)
    train_windows = set(unique_windows[:n_train])
    train_df = df[df[window_col].isin(train_windows)].copy()
    test_df = df[~df[window_col].isin(train_windows)].copy()
    if remove_leakage:
        train_pairs = set(frozenset((row['sender'], row['receiver'])) for _, row in train_df.iterrows())
        test_df = test_df[~test_df.apply(
            lambda row: frozenset((row['sender'], row['receiver'])) in train_pairs, axis=1
        )].copy()

    return train_df, test_df


def encode_categorical_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    # print(categorical_cols)
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    return df_encoded


def train_test(train_df, test_df, model, feature_columns):
    X_train = train_df[feature_columns]
    y_train = train_df['label']
    X_test = test_df[feature_columns]
    y_test = test_df['label']
    if isinstance(model, IsolationForest):
        model.fit(X_train)
        # Anomaly scores: higher means more abnormal
        scores = -model.decision_function(X_test)
        # Threshold using training distribution if needed
        precision, recall_points, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall_points, precision)
        avg_precision = average_precision_score(y_test, scores)
        print(f"IsolationForest -- Average Precision: {avg_precision:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

        avg_precision = average_precision_score(y_test, y_proba)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        precision, recall_points, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall_points, precision)

        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")


def get_model(name, imbalance_ratio):
    if name == "Lightgbm":
        return LGBMClassifier(
            min_gain_to_split=0.0,
            min_data_in_leaf=10,
            n_estimators=200,
            scale_pos_weight=100,  # if imbalanced
            n_jobs=-1,
            verbosity=-1
        )
    elif name == "Xgboost":
        return XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            tree_method='hist',
            scale_pos_weight=imbalance_ratio,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    elif name == "Catboost":
        return CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=8,
            class_weights=[1.0, imbalance_ratio],
            thread_count=-1,
            verbose=True
        )
    elif name == "Randomforest":
        return RandomForestClassifier(
            n_estimators=200,
            class_weight={0: 1.0, 1: imbalance_ratio},
            n_jobs=-1,
            random_state=42
        )
    elif name == "IsolationForest":
        return IsolationForest(
            n_estimators=200,
            contamination='auto',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model: {name}")


def compute_class_weights(y):
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    return n_neg / n_pos


def get_args():
    parser = argparse.ArgumentParser(description="Ensemble method Training Arguments")
    parser.add_argument("--dataset_name", type=str,
                        choices=['AMLSim', 'BitcoinOTC', 'BitcoinAlpha', 'PaySim', 'SAMLSim'], default="AMLSim")
    parser.add_argument("--model", required=True,
                        choices=["Lightgbm", "Xgboost", "Catboost", "Randomforest", "IsolationForest"])
    parser.add_argument("--remove_leakage", action="store_true", help="remove duplicate pairs from train_test")
    return parser.parse_args()


def main():
    args = get_args()
    dataset_name = args.dataset_name
    model_name = args.model
    remove_leakage = args.remove_leakage
    print(f"\033[94m{dataset_name}\033[0m")
    print(f"\033[91m{model_name}\033[0m")
    match dataset_name:
        case 'PaySim':
            df = pd.read_parquet('./data/PaySim/processed/enriched_1_8_1_data.parquet')
        case 'AMLSim':
            df = pd.read_parquet('./data/AMLSim/processed/enriched_7_8_1_data.parquet')
        case 'SAMLSim':
            df = pd.read_parquet('./data/SAMLSim/processed/enriched_7_8_1_data.parquet')
        case 'BitcoinOTC':
            df = pd.read_parquet('./data/BitcoinOTC/processed/enriched_7_56_7_data.parquet')
        case 'BitcoinAlpha':
            df = pd.read_parquet('./data/BitcoinAlpha/processed/enriched_7_56_7_data.parquet')
        case _:
            RuntimeError(f"Dataset {dataset_name} is not defined.")

    df = drop_useless_columns(df)
    df = encode_categorical_columns(df)
    df = df.fillna(0)
    # print(list(df.columns))
    default_features = [col for col in df.columns if
                        col not in graph_features and col != 'window_index' and col != 'label']

    y = df["label"]
    imbalance_ratio = compute_class_weights(y)

    print(f"\033[94mDefault features:{len(list(default_features))-2}\033[0m")
    # print(list(default_features))
    print(f"\033[94mGraph features:{len(list(graph_features))-2}\033[0m")
    # print(list(graph_features))
    train_df, test_df = split_train_test_by_window(df,remove_leakage=remove_leakage)
    print(f"Number of train rows:{len(train_df)}")
    print(f"Number of test rows:{len(test_df)}")
    print(f"\033[92mTrain test with default features\033[0m")
    model = get_model(model_name, imbalance_ratio)
    train_test(train_df, test_df, model, default_features)
    print(f"\033[92mTrain test with also graph features\033[0m")
    model = get_model(model_name, imbalance_ratio)
    train_test(train_df, test_df, model, default_features + graph_features)


if __name__ == "__main__":
    main()
