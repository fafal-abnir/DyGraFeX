import pandas as pd
import igraph as ig


def calculate_pagerank(g: ig.Graph, weight_name: str, normalize: bool = False, damping_factor=0.85):
    pagerank = g.pagerank(weights=weight_name)
    nodes = g.vs['name']
    pagerank_dict = dict(zip(nodes, pagerank))
    if normalize:
        # Nodes without incoming degree
        num_nodes = g.vcount()
        dangling_nodes = [v['name'] for v in g.vs if g.degree(v.index, mode="in") == 0]
        dangling_pagerank_sum = sum(pagerank_dict[node] for node in dangling_nodes)
        lower_bound = (1.0 / num_nodes) * (damping_factor + ((1 - damping_factor) * dangling_pagerank_sum))
        pagerank_dict = {key: value / lower_bound for key, value in pagerank_dict.items()}
    return pagerank_dict


def calculate_HIT(g: ig.Graph, weight_name: str):
    authority_scores = g.authority_score(weights=weight_name)
    hub_scores = g.hub_score(weights=weight_name)
    authority_dict = dict(zip(g.vs['name'], authority_scores))
    hub_dict = dict(zip(g.vs['name'], hub_scores))
    return authority_dict, hub_dict


def enrich_with_pagerank(input_df: object, g: ig.Graph, weight_list: list, normalize: bool = False,
                         source_name_column="sender", target_name_column="receiver") -> pd.DataFrame:
    for weight_name in weight_list:
        pagerank_dict = calculate_pagerank(g, weight_name, normalize=normalize)
        if normalize:
            input_df[f'{source_name_column}_pagerank_{weight_name}_norm'] = input_df[source_name_column].map(
                pagerank_dict).fillna(0)
            input_df[f'{target_name_column}_pagerank_{weight_name}_norm'] = input_df[target_name_column].map(
                pagerank_dict).fillna(0)
        else:
            input_df[f'{source_name_column}_pagerank_{weight_name}'] = input_df[source_name_column].map(
                pagerank_dict).fillna(0)
            input_df[f'{target_name_column}_pagerank_{weight_name}'] = input_df[target_name_column].map(
                pagerank_dict).fillna(0)
    return input_df


def enrich_with_HIT(input_df, g: ig.Graph, weight_list: list, source_name_column="sender",
                    target_name_column="receiver") -> pd.DataFrame:
    for weight_name in weight_list:
        auth_dict, hub_dict = calculate_HIT(g, weight_name)
        input_df[f'{source_name_column}_auth_{weight_name}'] = input_df[source_name_column].map(auth_dict).fillna(0)
        input_df[f'{target_name_column}_auth_{weight_name}'] = input_df[target_name_column].map(auth_dict).fillna(0)
        input_df[f'{source_name_column}_hub_{weight_name}'] = input_df[source_name_column].map(hub_dict).fillna(0)
        input_df[f'{target_name_column}_hub_{weight_name}'] = input_df[target_name_column].map(hub_dict).fillna(0)
    return input_df
