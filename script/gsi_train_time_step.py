import sys; sys.path.append('../'); sys.path.append('./')
import dgl
import tqdm
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt 

from torch.nn import MSELoss
from dgl.data.utils import load_graphs

from src.gen import GraphElementNetwork
from src.utils.plot import plot_compare_graph


data = load_graphs('data/gsi2.bin')
graph = data[0][0]

node_feature = graph.ndata['feat'] # 92 * 3
edge_feature = graph.edata['feat'] # 480 * 1
node_feature_dim = node_feature.shape[1]
edge_feature_dim = edge_feature.shape[1]

model = GraphElementNetwork(node_in_dim=node_feature_dim, 
                            node_out_dim=1, 
                            edge_in_dim=edge_feature_dim,
                            edge_out_dim=1)

model.train()
learning_rate = 1e-5
loss_function = MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

epoches = 50

for epoch in tqdm.tqdm(range(epoches)):
    loss_list = []
    for i in range(49):
        graph = data[0][i]
        next_graph = data[0][i+1]
        output = model(graph) # 25 * 1
        loss = loss_function(next_graph.ndata['value'], output)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_list.append(loss.item())

    print(sum(loss_list)/len(loss_list))
    output_graph = graph.clone()
    output_graph.ndata['value'] = output.detach()

    # plot_compare_graph(next_graph, output_graph, vmin=0, vmax=1)
    # plt.savefig('output/gsi/train_time_series.png')
            
torch.save(model.state_dict(), 'checkpoint/test.pth')
