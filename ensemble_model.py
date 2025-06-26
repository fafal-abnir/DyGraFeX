import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
    potential_drop_cols = ['time_step','isFlaggedFraud', 'laundering_type', 'timestamp', 'id', 'tx_id', 'alert_id', 'time', 'date',
                           'sender', 'receiver']
    potential_drop_cols = potential_drop_cols + ['sender_mean_amt', 'sender_std_amt', 'sender_total_amt', 'sender_txn_count', 'sender_unique_receivers', 'receiver_mean_amt', 'receiver_std_amt', 'receiver_txn_count', 'receiver_unique_senders']
    drop_cols = [col for col in potential_drop_cols if col in df.columns]
    use_features = [col for col in df.columns if col not in drop_cols]
    if verbose:
        print(f"Drop cols:{drop_cols}")
        print(f'Used_features:{use_features}')
    return df[use_features]


def split_train_test_by_window(df, window_col='window_index', train_ratio: float = 0.8):
    unique_windows = sorted(df[window_col].unique())
    n_train = int(len(unique_windows) * train_ratio)
    train_windows = set(unique_windows[:n_train])
    train_df = df[df[window_col].isin(train_windows)].copy()
    test_df = df[~df[window_col].isin(train_windows)].copy()
    return train_df, test_df


def encode_categorical_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(categorical_cols)
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    return df_encoded


def train_test(train_df, test_df, model, feature_columns):
    X_train = train_df[feature_columns]
    y_train = train_df['label']
    X_test = test_df[feature_columns]
    y_test = test_df['label']
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


def get_args():
    parser = argparse.ArgumentParser(description="Ensemble method Training Arguments")
    parser.add_argument("--dataset_name", type=str,
                        choices=['AMLSim', 'BitcoinOTC', 'BitcoinAlpha', 'PaySim', 'SAMLSim'], default="AMLSim")
    return parser.parse_args()


def main():
    args = get_args()
    dataset_name = args.dataset_name
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
    print(list(df.columns))
    default_features = [col for col in df.columns if
                        col not in graph_features and col != 'window_index' and col != 'label']
    print(f"Default features:")
    print(list(default_features))
    print(f"Graph features:")
    print(graph_features)
    train_df, test_df = split_train_test_by_window(df)
    print("train test with default features")
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    train_test(train_df, test_df, rf_model, default_features)
    print("train test with also graph features")
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    train_test(train_df, test_df, rf_model, default_features + graph_features)


if __name__ == "__main__":
    main()

# print("Paysim")
# df = pd.read_parquet('./data/PaySim/processed/enriched_1_8_1_data.parquet')
# df = drop_useless_columns(df)
# df = encode_categorical_columns(df)
# df = df.fillna(0)
#
# print("AMLSim")
# df = pd.read_parquet('./data/AMLSim/processed/enriched_7_8_1_data.parquet')
# df = drop_useless_columns(df)
# df = encode_categorical_columns(df)
# df = df.fillna(0)
# print("SAMLSim")
# df = pd.read_parquet('./data/SAMLSim/processed/enriched_7_8_1_data.parquet')
# df = drop_useless_columns(df)
# df = encode_categorical_columns(df)
# df = df.fillna(0)
# print(list(df.columns))
# train_df, test_df = split_train_test_by_window(df)
# print(df.columns)
# print("train test with default features")
# rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
# train_test(train_df, test_df, rf_model, default_features)
# print("train test with also graph features")
# train_test(train_df, test_df, rf_model, default_features + graph_features)
