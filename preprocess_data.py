from dataset.saml_sim_dataset import SAMLSimDataset
from dataset.paysim_dataset import PaySimDataset
from dataset.ellipticplusplus_dataset import EllipticPPDataset
from dataset.aml_sim_dataset import AMLSimDataset
from dataset.aml_world import AMLWorldDataset
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="PoC features engineering")
    parser.add_argument("--dataset_name", type=str,
                        choices=['AMLSim', 'EllipticPP', 'PaySim', 'SAMLSim', 'AMLWorld_HI_Small', 'AMLWorld_LI_Small',
                                 'AMLWorld_HI_Medium', 'AMLWorld_LI_Medium'],
                        default="EllipticPP")
    parser.add_argument("--force_reload_dataset", action="store_true", help="Force to download the dataset again.")
    parser.add_argument("--enrich_with_graph_features", action="store_true")
    parser.add_argument("--prepare_py_data_list", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    dataset_name = args.dataset_name
    force_reload_dataset = args.force_reload_dataset
    enrich_with_graph_features = args.enrich_with_graph_features
    prepare_py_data_list = args.prepare_py_data_list
    match dataset_name:
        case "AMLSim":
            temporal_data = AMLSimDataset()
        case "EllipticPP":
            temporal_data = EllipticPPDataset()
        case "SAMLSim":
            temporal_data = SAMLSimDataset()
        case "PaySim":
            temporal_data = PaySimDataset()
        case "AMLWorld_HI_Small":
            temporal_data = AMLWorldDataset(graph_window_size=4, illicit_level="HI", data_size="Small")
        case "AMLWorld_LI_Small":
            temporal_data = AMLWorldDataset(graph_window_size=4, illicit_level="LI", data_size="Small")
        case "AMLWorld_HI_Medium":
            temporal_data = AMLWorldDataset(graph_window_size=7, illicit_level="HI", data_size="Medium")
        case "AMLWorld_LI_Medium":
            temporal_data = AMLWorldDataset(graph_window_size=7, illicit_level="LI", data_size="Medium")
        case _:
            raise RuntimeError("Dataset name is not among the considered dataset.")
    temporal_data.prepare(force_reload=force_reload_dataset)
    if enrich_with_graph_features:
        temporal_data.enrich_with_graph_features()
        temporal_data.prepare_training_dataframe_with_graph_features()
    if prepare_py_data_list:
        temporal_data.create_pyg_data_list(with_graph_features=True)
        temporal_data.create_pyg_data_list(with_graph_features=False)



    print('Preporcessing is done.')
