import os
import sys; sys.path.append('.')
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import MSELoss
from dgl.data.utils import load_graphs

from src.gen import GraphElementNetwork
from src.utils.plot import gif_generator, plot_graph_gen_limit

# Load Data
data = load_graphs('data/gsi.bin')
graph = data[0][0]

node_feature = graph.ndata['feat'] # 49 * 3
edge_feature = graph.edata['feat'] # 259 * 1
node_feature_dim = node_feature.shape[1]
edge_feature_dim = edge_feature.shape[1]

# Init Module
model = GraphElementNetwork(node_in_dim=node_feature_dim, 
                            node_out_dim=1, 
                            edge_in_dim=edge_feature_dim, 
                            edge_out_dim=1
                            )
model.load_state_dict(torch.load('output/gsi/model.pt'))
model.eval()

# for i in tqdm.tqdm(range(49)):
#     graph = data[0][i]
#     plot_graph_gen_limit(graph)
#     plt.savefig(f'output/gsi/truth/{i}.png')
#     plt.clf()

for i in tqdm.tqdm(range(49)):
    graph = data[0][i]
    print(graph.ndata['value'])
    output = model(graph) # 49 * 1
    graph.ndata['value'] = output.detach()
    print(graph.ndata['value'])
    plot_graph_gen_limit(graph)
    plt.savefig(f'output/gsi/pred/{i}.png')
    plt.clf()

