import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        # N x N adjacency matrix
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        # Cols: source, Rows: target
        adjacency_matrix[edge_index[1], edge_index[0]] = 1
        return adjacency_matrix

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        # use in-degree (the number of incoming edges to this node)
        src, dst = edge_index
        edges = torch.tile(dst, dims=(num_nodes, 1))
        mask = (edges == torch.arange(num_nodes, device=edge_index.device)[:, None])
        # Degree = sum of incoming edges to this node
        degree_vector = mask.float().sum(dim = 1)
        # degree of nodes without incoming edge = 1
        inverted_degree_vector = torch.where(degree_vector > 0, 1 / degree_vector, torch.ones_like(degree_vector))
        inverted_degree_matrix = torch.diag(inverted_degree_vector)
        return inverted_degree_matrix

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        A = self.make_adjacency_matrix(edge_index, x.size(0))
        D_inv = self.make_inverted_degree_matrix(edge_index, x.size(0))
        # Weighted sum of linearly projected incoming neighbors
        neighbors_conv = D_inv @ A @ (x @ self.W.T)
        # linear projection of self
        self_conv = x @ self.B.T
        out = neighbors_conv + self_conv
        return out

class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x:torch.Tensor, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """
        # Sources are neighbor nodes, Destinations are center nodes [In-degree neighborhood perspective]
        num_nodes, num_in_features = x.shape
        src, dst = edge_index
        messages = x[src]
        # Graph conv = sum over in-neighbors divided by in-degree = average over incoming messages 
        aggregated_messages = torch.zeros((num_nodes, num_in_features), 
                                          device=x.device).index_add_(dim=0,
                                                                      index=dst,
                                                                      source=messages)
        # need to normalize sum of messages by number of incoming messages to node (in-degree)
        node_degree = torch.zeros(num_nodes, dtype= torch.long, device=x.device
                                 ).index_add(dim = 0, index = dst, source = torch.ones_like(dst))
        # must take care of nodes with no incoming edges same way as in matrix graph conv
        node_degree = torch.where(node_degree == 0, 1, node_degree)
        return aggregated_messages / node_degree[:, None]

    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        # First term does linear projection on center nodes themselves
        # Second term adds linear projection of neighbors
        # in principle in MPN's each step could be followed by nonlinearity
        x = (x @ self.B.T) + (messages @ self.W.T)
        return x

    def forward(self, x, edge_index):
        message = self.message(x, edge_index)
        x = self.update(x, message)
        return x


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> outgoing unweighted messages with shape [num_nodes + num edges, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer

        Hint: the GAT implementation uses only 1 parameter vector and edge index with self loops
        Hint: It is easier to use/calculate only the numerator of the softmax
              and weight with the denominator at the end.

        Hint: check out torch.Tensor.index_add function
        """
        edge_index, _ = add_self_loops(edge_index)
        num_nodes, in_features = x.size()
        # directed edges incl. self-loops
        sources, destinations = edge_index
        # Node-wise: (N, in_features) @ (out_features, in_features).T
        activations = x @ self.W.T # (N, out_features)
        out_features = activations.size(1)
        # in message-passing - edge-wise organization. 
        # Look at messages FROM neighbors / sources
        messages = activations[sources] # (N + n_edges, out_features)

        # Split attention mechanism in half (center node and its neighbors)
        # a.T @ [Wh_i, Wh_j] = a_1/2 @ Wh_i + a_2/2 @ W_h_j
        # i indicates center nodes (destinations)
        a_in = self.a[:out_features]
        att_self = (activations @ a_in)[destinations]
        # j indicates neighbor nodes (sources)
        a_out = self.a[out_features:]
        att_neighbor = (activations @ a_out)[sources]
        # Attention coefficients e_ij from Velickovic 2018
        # importance of node j's features to node i
        # negative slope 0.2 taken from Velickovic 2018
        attention_inputs = F.leaky_relu(att_self + att_neighbor, negative_slope=0.2) # (N + n_edges)

        # softmax numerator - on edge-level
        edge_weights_numerator = torch.exp(attention_inputs) # (N + n_edges)
        # weighted messages incoming to all nodes across network (need to separate into individual nodes again)
        weighted_messages = edge_weights_numerator[:, None] * messages # (N + n_edges) * (N + n_edges, out_features)

        # softmax denominator - on node-level (sum of all inputs per node)
        softmax_denominator = torch.zeros(num_nodes, device=x.device).index_add_(dim = 0, 
                                                                                 index = destinations, 
                                                                                 source=edge_weights_numerator)
        # Node-level normalized by softmax denominator (add all attention-weighted messages incoming to each node)
        aggregated_messages = (1/softmax_denominator)[:, None] * torch.zeros((num_nodes, out_features) , device=x.device
                                                                    ).index_add_(dim = 0, 
                                                                                 index = destinations,
                                                                                 source=weighted_messages)
        if debug:
            return aggregated_messages, {'edge_weights': edge_weights_numerator, 
                                         'softmax_weights': softmax_denominator,
                                         'messages': messages}
        else:
            return aggregated_messages

