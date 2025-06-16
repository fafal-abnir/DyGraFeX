
from dataset.bitcoin_alpha_dataset import BitcoinAlphaDataset
from dataset.bitcoin_otc_dataset import BitcoinOTCDataset
from dataset.paysim_dataset import PaySimDataset
from dataset.aml_sim_dataset import AMLSimDataset

d = BitcoinAlphaDataset()
d.prepare(force_reload=True)
d.load_data()
d.enrich_with_graph_features(56,7)

d = BitcoinOTCDataset()
d.prepare(force_reload=True)
d.load_data()
d.enrich_with_graph_features(56,7)

d = PaySimDataset()
d.prepare(force_reload=False)
d.load_data()
d.enrich_with_graph_features(8,1)

d = AMLSimDataset()
d.prepare(force_reload=False)
d.load_data()
d.enrich_with_graph_features(8,1)
