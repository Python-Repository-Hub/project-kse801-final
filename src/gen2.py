import dgl
import torch
from dgl.function.message import MessageFunction
from dgl.nn.pytorch.conv.atomicconv import reduce_func
import torch.nn as nn
from torch.nn.modules.linear import Linear
from dgl.nn.pytorch.softmax import edge_softmax


class GraphElementNetwork(nn.Module):

    def __init__(self, 
                node_in_dim: int,
                node_out_dim: int,
                edge_in_dim: int,
                edge_out_dim: int,
                num_neurons: int = 64):
        '''
        Args:
            - node_in_dim: the number of features for one node
            - node_out_dim: since it's hidden feature, can be set to 1
        '''
        super(GraphElementNetwork, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_in_dim, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, node_out_dim),
            nn.ReLU(),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, node_out_dim),
            nn.ReLU(),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(3, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, edge_out_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(edge_in_dim, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, node_out_dim),
            nn.ReLU(),
        )

    def forward(self, g):
        # Encoder
        encode_node_feature = self.encoder(g.ndata['feat']) # g.ndata['feat'].shape = (node number, feature number)

        # Softmax Features
        softmax_node_feature = self.softmax(g, encode_node_feature, g.edata['dist'])
        
        # Message Passing
        g.ndata['h'] = softmax_node_feature
        g.edata['h'] = g.edata['feat'] # I think should be this
        g.apply_edges(func=self.edge_update_func)
        g.pull(g.nodes(), message_func=dgl.function.copy_e('h', 'm'), reduce_func=dgl.function.sum('m', 'agg_m'))
        g.apply_nodes(func=self.node_update_func)

        # Softmax Features
        softmax_node_feature = self.softmax(g, g.ndata['h'], g.edata['h'])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        # Delete temp features
        # Important
        _ = g.ndata.pop('h')
        _ = g.ndata.pop('agg_m')
        _ = g.ndata.pop('sum_m')
        _ = g.edata.pop('h')
        _ = g.edata.pop('w')
        _ = g.edata.pop('wh')

        return decode_node_feature

    def softmax(self, g, node_feature, edge_feature):
        g.ndata['h'] = node_feature
        g.edata['w'] = edge_softmax(g, edge_feature)
        g.apply_edges(func=dgl.function.e_mul_u('w', 'h', 'wh'))
        g.update_all(message_func=dgl.function.copy_e('wh', 'm'), reduce_func=dgl.function.sum('m', 'sum_m'))
        softmax_node_feature = g.ndata['sum_m']
        return softmax_node_feature

    def node_update_func(self, nodes):
        aggregation_message = nodes.data['agg_m']
        node_feature = nodes.data['h']
        node_model_input = torch.cat([aggregation_message, node_feature], dim=-1)
        updated_node_feature = self.node_update(node_model_input)
        return {'h': updated_node_feature}

    def edge_update_func(self, edges):
        src_node_feature = edges.src['h']
        dis_node_feature = edges.dst['h']
        edge_feature = edges.data['h']
        edge_model_input = torch.cat([edge_feature, src_node_feature, dis_node_feature], dim=-1)
        updated_edge_feature = self.edge_update(edge_model_input)
        return {'h': updated_edge_feature}

