
from dataset.bitcoin_alpha_dataset import BitcoinAlphaDataset
from dataset.bitcoin_otc_dataset import BitcoinOTCDataset
from dataset.paysim_dataset import PaySimDataset
from dataset.aml_sim_dataset import AMLSimDataset
from dataset.saml_sim_dataset import SAMLSimDataset
d = BitcoinAlphaDataset()
# d.prepare(force_reload=True)
# d.load_data()
# d.enrich_with_graph_features(56,7)
d.prepare_training_dataframe_with_graph_features()


d = BitcoinOTCDataset()
# d.prepare(force_reload=True)
# d.load_data()
# d.enrich_with_graph_features(56,7)
d.prepare_training_dataframe_with_graph_features()


d = PaySimDataset()
# d.prepare(force_reload=False)
# d.load_data()
# d.enrich_with_graph_features(8,1)
d.prepare_training_dataframe_with_graph_features()
#
d = AMLSimDataset()
# d.prepare(force_reload=False)
# d.load_data()
# d.enrich_with_graph_features(8,1)
d.prepare_training_dataframe_with_graph_features()

d = SAMLSimDataset()
# d.prepare(force_reload=False)
# d.load_data()
# d.enrich_with_graph_features()
d.prepare_training_dataframe_with_graph_features()
