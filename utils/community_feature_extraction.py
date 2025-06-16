import pandas as pd
import numpy as np
from igraph import VertexClustering, Graph
from scipy.sparse import lil_array, csr_matrix
import time


def get_community_sizes_from_csr_matrix(csr_matrix):
    columns_sums = np.array(csr_matrix.sum(axis=0))

    if columns_sums.ndim == 1:
        columns_sums = columns_sums.reshape(1, -1)

    count_dict = {i: int(columns_sums[0, i]) for i in range(csr_matrix.shape[1])}

    return count_dict


def destination_community_neighbor_count(i, j, mem_matrix, conn_matrix, id_name_dict, comm_size_dict=None):
    if i in id_name_dict and j in id_name_dict:
        community_index = mem_matrix.getrow(id_name_dict[i]).indices.max()
        result = conn_matrix[id_name_dict[j], community_index]
        if comm_size_dict is None:
            return float(result)
        else:
            return float(result / comm_size_dict[community_index])
    else:
        return None


def destination_community_neighbor_weighted_degree(i, j, mem_matrix, w_matrix, id_name_dict, comm_size_dict=None):
    if i in id_name_dict and j in id_name_dict:
        community_index = mem_matrix.getrow(id_name_dict[i]).indices.max()
        result = w_matrix[id_name_dict[j], community_index]
        if comm_size_dict is None:
            return float(result)
        else:
            return float(result / comm_size_dict[community_index])
    else:
        return None


def are_nodes_in_same_community(i, j, mem_matrix, id_name_dict):
    if i in id_name_dict and j in id_name_dict:
        community_index_i = mem_matrix.getrow(id_name_dict[i]).indices.max()
        community_index_j = mem_matrix.getrow(id_name_dict[j]).indices.max()
        if community_index_i == community_index_j:
            return True
        else:
            return False
    else:
        return False


def node_community_connections(vcl: VertexClustering, g: Graph) -> csr_matrix:
    npm = np.array(vcl.membership)
    c_dist = lil_array((g.vcount(), len(vcl)), dtype=np.int8)
    for e in g.es:
        cl_target = npm[e.target]
        cl_source = npm[e.source]
        c_dist[e.source, cl_target] = c_dist[e.source, cl_target] + 1
        c_dist[e.target, cl_source] = c_dist[e.target, cl_source] + 1
    return csr_matrix(c_dist)


def node_community_weighted_connections(vcl: VertexClustering, g: Graph) -> csr_matrix:
    cl_graph = vcl.cluster_graph(combine_edges=False)
    m = g.ecount()
    npm = np.array(vcl.membership)
    c_dist = lil_array((g.vcount(), len(vcl)), dtype=np.float64)
    for e in g.es:
        cl_target = npm[e.target]
        cl_source = npm[e.source]
        c_dist[e.source, cl_target] = c_dist[e.source, cl_target] + (2 * m) / (
                g.degree(e.source) * cl_graph.vs[cl_target].degree())
        c_dist[e.target, cl_source] = c_dist[e.target, cl_source] + (2 * m) / (
                g.degree(e.target) * cl_graph.vs[cl_source].degree())
    return csr_matrix(c_dist)


def node_community_membership_ohe(vcl: VertexClustering, g: Graph) -> csr_matrix:
    npm = np.array(vcl.membership)
    c_ohe = lil_array((g.vcount(), len(vcl)), dtype=np.int8)
    for v in g.vs:
        c_ohe[v.index, npm[v.index]] = 1
    return csr_matrix(c_ohe)


def enrich_with_edge_features(g: Graph, target_week_df: pd.DataFrame, source_name_column="sender",
                              target_name_column="receiver") -> pd.DataFrame:
    edge_data = pd.DataFrame({
        'source': [g.vs[edge.source]['name'] if g.vs[edge.source]['name'] else edge.source for edge in g.es],
        'target': [g.vs[edge.target]['name'] if g.vs[edge.target]['name'] else edge.target for edge in g.es],
        'total_amount': g.es['total_amount'],
        'pmi': g.es['pmi'],
        'count': g.es['count'],
        'exclusivity_amt': g.es['exclusivity_amt']
    })
    target_week_df = target_week_df.merge(edge_data, left_on=[source_name_column, target_name_column],
                                          right_on=['source', 'target'], how='left')
    target_week_df = target_week_df.drop(columns=['source', 'target'])
    return target_week_df


def enrich_with_community_features(g: Graph, comm_alg: dict, weights_list: list,
                                   target_week_df: pd.DataFrame, source_name_column="sender",
                                   target_name_column="receiver") -> pd.DataFrame:
    g.to_undirected()
    community_functions = {
        'louvain': lambda graph, res, weight: graph.community_multilevel(weights=weight, resolution=res),
        'leiden': lambda graph, res, weight: graph.community_leiden(weights=weight, resolution=res,
                                                                    objective_function="CPM")
    }
    start_time = time.time()
    # weights_list = ['unweighted', 'count', 'total', 'pmi', 'exclusivity_amt']
    for algorithm in comm_alg:  # for each feature asked for from the input
        if algorithm in community_functions:  # we only deal with the feature if it is part of the feature functions;
            for comm_param in comm_alg[
                algorithm]:  # for each resolution value demanded for the community finding algoritm;
                for weight in weights_list:
                    if weight == "unweighted":
                        weight = None
                    print(f"processing feature: {algorithm} with weight:{weight} and resolution: {comm_param}")
                    # Checking existance of features(if so, no need to precess again)
                    if weight == 'pmi':
                        g.es['pmi'] = [max(edge['pmi'], 0) for edge in g.es]
                    print(f"Start community detection:{algorithm} with res:{comm_param} and weight:{weight}")
                    clustering = community_functions[algorithm](g, comm_param, weight)
                    connection_sparse_matrix = node_community_connections(clustering, g)
                    weight_sparse_matrix = node_community_weighted_connections(clustering, g)
                    member_sparse_matrix = node_community_membership_ohe(clustering, g)
                    pandas_df = pd.DataFrame({'node_name': g.vs["name"], "index_in_graph": g.vs.indices})
                    id_dict = dict(zip(pandas_df['node_name'], pandas_df['index_in_graph']))
                    communities_size_dict = get_community_sizes_from_csr_matrix(member_sparse_matrix)
                    #####
                    target_week_df[f'community_belonging_{algorithm}_{comm_param}'] = target_week_df.apply(
                        lambda row: are_nodes_in_same_community(row[source_name_column],
                                                                row[target_name_column], member_sparse_matrix,
                                                                id_dict), axis=1)
                    print(
                        f"Finished reading data from hive and store as pandas dataframe. time: {time.time() - start_time} s")

                    target_week_df[
                        f'destination_community_neighbor_count_{algorithm}_{comm_param}'] = target_week_df.apply(
                        lambda row: destination_community_neighbor_count(row[source_name_column],
                                                                         row[target_name_column],
                                                                         member_sparse_matrix, connection_sparse_matrix,
                                                                         id_dict), axis=1)

                    print(
                        f"Finished adding column destination_community_neighbor_count_{algorithm}. time: {time.time() - start_time} s")

                    target_week_df[
                        f'destination_community_neighbor_count_{algorithm}_{comm_param}_norm'] = target_week_df.apply(
                        lambda row: destination_community_neighbor_count(row[source_name_column],
                                                                         row[target_name_column],
                                                                         member_sparse_matrix, connection_sparse_matrix,
                                                                         id_dict, communities_size_dict), axis=1)

                    print(
                        f"Finished adding column destination_community_neighbor_count_{algorithm}_norm. time: {time.time() - start_time} s")

                    target_week_df[
                        f'destination_community_neighbor_weighted_degree_{algorithm}_{comm_param}'] = target_week_df.apply(
                        lambda row: destination_community_neighbor_weighted_degree(row[source_name_column],
                                                                         row[target_name_column],
                                                                         member_sparse_matrix, weight_sparse_matrix,
                                                                         id_dict), axis=1)

                    print(
                        f"Finished adding column destination_community_neighbor_weighted_degree_{algorithm}. time: {time.time() - start_time} s")

                    target_week_df[
                        f'destination_community_neighbor_weighted_degree_{algorithm}_{comm_param}_norm'] = target_week_df.apply(
                        lambda row: destination_community_neighbor_weighted_degree(row[source_name_column],
                                                                         row[target_name_column],
                                                                         member_sparse_matrix, weight_sparse_matrix,
                                                                         id_dict, communities_size_dict), axis=1)

                    print(
                        f"Finished adding column destination_community_neighbor_weighted_degree_{algorithm}_norm. time: {time.time() - start_time} s")

                    #####



        else:
            print(f"Unknown feature: {algorithm}")
    return target_week_df
