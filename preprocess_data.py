import argparse


def get_args():
    parser = argparse.ArgumentParser(description="PoC features engineering")
    parser.add_argument("--dataset_name", type=str,
                        choices=["BitcoinOTC", "BitcoinAlpha", "AMLSim", "EllipticPP", "PaySim", "DGraphFin"],
                        default="BitcoinOTC")
    parser.add_argument("--force_reload_dataset", action="store_true", help="Force to download the dataset again.")
    parser.add_argument("--graph_window_size", type=str, choices=["day", "week", "month"], default="month",
                        help="the size of graph window size")
    parser.add_argument("--enrich_with_graph_features", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_name = args.dataset_name
    force_reload_dataset = args.force_reload_dataset
    graph_window_size = args.graph_window_size
    enrich_with_graph_features = args.enrich_with_graph_features
    if force_reload_dataset:
        pass
    if enrich_with_graph_features:
        pass

    print('PyCharm')
