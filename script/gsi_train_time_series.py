import os
import sys; sys.path.append('.')
import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.nn import MSELoss
from dgl.data.utils import load_graphs

from src.gen import GraphElementNetwork
from src.utils.plot import plot_compare_graph

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

model.train()
learning_rate = 1e-5
loss_function = MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoches = 100

overall_loss_list = []
for epoch in tqdm.tqdm(range(epoches)):
    loss_list = []

    graph.ndata['value'] = output.detach()
    plot_compare_graph(next_graph, graph, vmin=0, vmax=1)
    plt.savefig('output/gsi/train.png')

    for i in range(49):
        data = load_graphs('data/gsi.bin')
        graph = data[0][i]
        next_graph = data[0][i+1]
        output = model(graph) # 49 * 1
        loss = loss_function(next_graph.ndata['value'], output)
        opt.zero_grad()
        # loss.backward(retain_graph=True)
        loss.backward()
        opt.step()
        loss_list.append(loss.item())


    overall_loss_list.append(sum(loss_list)/len(loss_list))
    print(sum(loss_list)/len(loss_list))


torch.save(model.state_dict(), 'output/gsi/model.pt')
np.save('output/gsi/loss.npy', np.array(overall_loss_list))
