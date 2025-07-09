from dataset.saml_sim_dataset import SAMLSimDataset
from dataset.paysim_dataset import PaySimDataset
from dataset.ellipticplusplus_dataset import EllipticPPDataset
from dataset.aml_sim_dataset import AMLSimDataset
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="PoC features engineering")
    parser.add_argument("--dataset_name", type=str,
                        choices=["BitcoinOTC", "BitcoinAlpha", "AMLSim", "EllipticPP", "PaySim", "DGraphFin",
                                 "EllipticPP"],
                        default="EllipticPP")
    parser.add_argument("--force_reload_dataset", action="store_true", help="Force to download the dataset again.")
    # parser.add_argument("--graph_window_size", type=str, choices=["day", "week", "month"], default="month",
    #                     help="the size of graph window size")
    parser.add_argument("--enrich_with_graph_features", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_name = args.dataset_name
    force_reload_dataset = args.force_reload_dataset
    # graph_window_size = args.graph_window_size
    enrich_with_graph_features = args.enrich_with_graph_features
    match dataset_name:
        case "AMLSim":
            temporal_data = AMLSimDataset()
        case "EllipticPP":
            temporal_data = EllipticPPDataset()
        case "SAMLSim":
            temporal_data = SAMLSimDataset()
        case "PaySim":
            temporal_data = PaySimDataset()
        case _:
            raise RuntimeError("Dataset name is not among the considered dataset.")
    if force_reload_dataset:
        pass
    if enrich_with_graph_features:
        pass
    temporal_data.prepare(force_reload=True)
    temporal_data.enrich_with_graph_features()
    temporal_data.prepare_training_dataframe_with_graph_features()

    print('Preporcessing is done.')
