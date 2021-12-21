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

        self.encoder_linear1 = nn.Linear(3, 64)
        self.encoder_linear2 = nn.Linear(64, 64)
        self.encoder_linear3 = nn.Linear(64, 8)
        self.encoder_relu1 = nn.ReLU()
        self.encoder_relu2 = nn.ReLU()
        self.encoder_relu3 = nn.ReLU()

        self.decoder_linear1 = nn.Linear(8, 64)
        self.decoder_linear2 = nn.Linear(64, 64)
        self.decoder_linear3 = nn.Linear(64, 1)
        self.decoder_relu1 = nn.ReLU()
        self.decoder_relu2 = nn.ReLU()
        self.decoder_relu3 = nn.ReLU()

        self.node_update_linear1 = nn.Linear(9, 64)
        self.node_update_linear2 = nn.Linear(64, 64)
        self.node_update_linear3 = nn.Linear(64, 8)
        self.node_update_relu1 = nn.ReLU()
        self.node_update_relu2 = nn.ReLU()
        self.node_update_relu3 = nn.ReLU()

        self.edge_update_linear1 = nn.Linear(17, 64)
        self.edge_update_linear2 = nn.Linear(64, 64)
        self.edge_update_linear3 = nn.Linear(64, 1)
        self.edge_update_relu1 = nn.ReLU()
        self.edge_update_relu2 = nn.ReLU()
        self.edge_update_relu3 = nn.ReLU()

        torch.nn.init.xavier_normal_(self.encoder_linear1.weight)
        torch.nn.init.xavier_normal_(self.encoder_linear2.weight)
        torch.nn.init.xavier_normal_(self.encoder_linear3.weight)
        torch.nn.init.constant_(self.encoder_linear1.bias, 0.0)
        torch.nn.init.constant_(self.encoder_linear2.bias, 0.0)
        torch.nn.init.constant_(self.encoder_linear3.bias, 0.0)

        torch.nn.init.xavier_normal_(self.decoder_linear1.weight)
        torch.nn.init.xavier_normal_(self.decoder_linear2.weight)
        torch.nn.init.xavier_normal_(self.decoder_linear3.weight)
        torch.nn.init.constant_(self.decoder_linear1.bias, 0.0)
        torch.nn.init.constant_(self.decoder_linear2.bias, 0.0)
        torch.nn.init.constant_(self.decoder_linear3.bias, 0.0)

        torch.nn.init.xavier_normal_(self.node_update_linear1.weight)
        torch.nn.init.xavier_normal_(self.node_update_linear2.weight)
        torch.nn.init.xavier_normal_(self.node_update_linear3.weight)
        torch.nn.init.constant_(self.node_update_linear1.bias, 0.0)
        torch.nn.init.constant_(self.node_update_linear2.bias, 0.0)
        torch.nn.init.constant_(self.node_update_linear3.bias, 0.0)

        torch.nn.init.xavier_normal_(self.edge_update_linear1.weight)
        torch.nn.init.xavier_normal_(self.edge_update_linear2.weight)
        torch.nn.init.xavier_normal_(self.edge_update_linear3.weight)
        torch.nn.init.constant_(self.edge_update_linear1.bias, 0.0)
        torch.nn.init.constant_(self.edge_update_linear2.bias, 0.0)
        torch.nn.init.constant_(self.edge_update_linear3.bias, 0.0)

        # self.encoder = nn.Sequential(
        #     nn.Linear(3, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 8),
        #     nn.ReLU()
        # )
        # nn.init.xavier_uniform_(self.encoder.weight)
        # nn.init.constant_(self.encoder.bias, 0.0)
        # self.node_update = nn.Sequential(
        #     nn.Linear(9, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 8),
        #     nn.ReLU()
        # )
        # self.edge_update = nn.Sequential(
        #     nn.Linear(17, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.ReLU()
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(8, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1),
        #     nn.ReLU()
        # )

    def forward(self, g):
        # Encoder
        encode_node_feature = self.encoder(g.ndata['feat']) # g.ndata['feat'].shape = (node number, feature number)

        # Softmax Features
        softmax_node_feature = self.softmax(g, encode_node_feature, g.edata['dist'])
        
        # Message Passing
        g.ndata['h'] = softmax_node_feature
        g.edata['h'] = g.edata['feat'] 

        # --- Correct so far ---

        g.apply_edges(func=self.edge_update_func)
        g.pull(g.nodes(), message_func=dgl.function.copy_e('h', 'm'), reduce_func=dgl.function.sum('m', 'agg_m'))
        g.apply_nodes(func=self.node_update_func)

        g.apply_edges(func=self.edge_update_func)
        g.pull(g.nodes(), message_func=dgl.function.copy_e('h', 'm'), reduce_func=dgl.function.sum('m', 'agg_m'))
        g.apply_nodes(func=self.node_update_func)

        # Softmax Features
        softmax_node_feature = self.softmax(g, g.ndata['h'], g.edata['dist'])

        # Decoder
        decode_node_feature = self.decoder(softmax_node_feature)

        # Delete temp features
        # Important
        _ = g.ndata.pop('h')
        # _ = g.ndata.pop('m')
        _ = g.ndata.pop('agg_m')
        _ = g.ndata.pop('sum_m')
        _ = g.edata.pop('h')
        _ = g.edata.pop('w')
        _ = g.edata.pop('wh')

        return decode_node_feature

    def softmax(self, g, node_feature, edge_feature):
        ''' This function is checked.'''
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
        ''' This function is checked.'''
        src_node_feature = edges.src['h']
        dis_node_feature = edges.dst['h']
        edge_feature = edges.data['h']
        edge_model_input = torch.cat([edge_feature, src_node_feature, dis_node_feature], dim=-1)
        updated_edge_feature = self.edge_update(edge_model_input)
        return {'h': updated_edge_feature}

    def encoder(self, x):
        x = self.encoder_linear1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_linear2(x)
        x = self.encoder_relu2(x)
        x = self.encoder_linear3(x)
        x = self.encoder_relu3(x)
        return x

    def decoder(self, x):
        x = self.decoder_linear1(x)
        x = self.decoder_relu1(x)
        x = self.decoder_linear2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_linear3(x)
        x = self.decoder_relu3(x)
        return x

    def node_update(self, x):
        x = self.node_update_linear1(x)
        x = self.node_update_relu1(x)
        x = self.node_update_linear2(x)
        x = self.node_update_relu2(x)
        x = self.node_update_linear3(x)
        x = self.node_update_relu3(x)
        return x

    def edge_update(self, x):
        x = self.edge_update_linear1(x)
        x = self.edge_update_relu1(x)
        x = self.edge_update_linear2(x)
        x = self.edge_update_relu2(x)
        x = self.edge_update_linear3(x)
        x = self.edge_update_relu3(x)
        return x