import igraph as ig
import pandas as pd
import numpy as np


def generate_graph(input_df: pd.DataFrame, src_column: str, dest_column: str) -> ig.Graph:
    transaction_count = input_df.shape[0]
    out_degrees = input_df[src_column].value_counts().to_dict()
    in_degrees = input_df[dest_column].value_counts().to_dict()
    df_grouped = input_df.groupby([src_column, dest_column]).agg(
        count=('amount', 'count'),
        total_amount=('amount', 'sum')
    ).reset_index()
    df_grouped['src_in_degree'] = df_grouped[src_column].map(in_degrees)
    df_grouped['src_out_degree'] = df_grouped[src_column].map(out_degrees)
    df_grouped['dest_in_degree'] = df_grouped[dest_column].map(in_degrees)
    df_grouped['dest_out_degree'] = df_grouped[dest_column].map(out_degrees)
    df_grouped['pmi'] = np.log((df_grouped['count'] * transaction_count) / (
            df_grouped['src_out_degree'] * df_grouped['dest_in_degree']))
    df_grouped['pmi'] = df_grouped['pmi'].replace([np.inf, -np.inf, np.nan], 0)
    df_grouped['exclusivity'] = df_grouped['count'] / (
            df_grouped['src_out_degree'] + df_grouped['dest_in_degree'] - df_grouped['count'])

    edges = list(df_grouped.itertuples(index=False, name=None))
    graph = ig.Graph.TupleList(
        edges, directed=True, vertex_name_attr="name",
        edge_attrs=["count", "total_amount", "pmi", "exclusivity_amt"]
    )
    pmi_weights = [
        max(0, weight) if not np.isnan(weight) else 0
        for weight in graph.es['pmi']
    ]
    graph.es['pmi'] = pmi_weights
    exclusivity_amt_weights = [
        max(0, weight) if not np.isnan(weight) else 0
        for weight in graph.es['exclusivity_amt']
    ]
    graph.es['exclusivity_amt'] = exclusivity_amt_weights
    return graph
