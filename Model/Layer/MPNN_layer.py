import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import dgl
import dgl.function as fn
from tensorflow.python.framework.ops import convert_to_tensor, device

# This code was modified from DGL public examples

# code from https://github.com/dmlc/dgl/blob/582f71a173b687325dd4c8138f2634a75475ab6f/python/dgl/nn/tensorflow/conv/sgconv.p
# model from https://arxiv.org/pdf/1704.01212.pdf

# MPNN framework that can generalize GNN, GCN
# do not allow predefine arguements
class MPNN(layers.Layer):
    
    def __init__(self,
                _setup_dict
                 ):
        super(MPNN, self).__init__()

        
        message_dim = _setup_dict['message_dim']
        hidden_dim = _setup_dict['hidden_dim']                  # output feat dim
        layer_num = _setup_dict['layer_num']                    # number of 'hop'
        use_bias = _setup_dict['use_bias']                      # use_bias turn on or off during training
        activation = _setup_dict['activation']                  # activation function slot
        activation_alpha = _setup_dict.get('activation_alpha')      #
        include_edge_feat = _setup_dict['include_edge_feat']    # check if edge info should be included or not
        repeat_msgANDhidden_layer  = _setup_dict['repeat_msgANDhidden_layer']
        


        self.layer_num = layer_num
        self.include_edge_feat = include_edge_feat
        self.repeat_msgANDhidden_layer = repeat_msgANDhidden_layer
        if activation.lower() == "leakyrelu":
            activation = layers.LeakyReLU(alpha=activation_alpha)

        self.current_layer = 0
        self.dense_message_list = []
        self.dense_hidden_list = []

        # generate a list of dense layer if not repeat messenger_layer and hidden_layer
        if not repeat_msgANDhidden_layer:
            for _ in range(self.layer_num):
                self.dense_message_list.append( layers.Dense(message_dim, use_bias=use_bias,activation=activation) )
                self.dense_hidden_list.append( layers.Dense(hidden_dim, use_bias=use_bias,activation=activation) )
        # only one dense layer if not repeat messenger_layer and hidden_layer
        else:
            self.dense_message_list = [layers.Dense(message_dim, use_bias=use_bias,activation=activation)]
            self.dense_hidden_list = [layers.Dense(hidden_dim, use_bias=use_bias,activation=activation)]
        

    def message_func(self,edges): 
            
        if self.include_edge_feat == True:
            m_input= tf.concat([edges.src['h_n'], edges.dst['h_n'], edges.data['h_e']], -1)
        else:
            m_input= tf.concat([edges.src['h_n'], edges.dst['h_n']], -1)
        #print(m_input)
        m_out = self.dense_message_list[self.current_layer](m_input) 
        #print(self.dense_message_list[self.current_layer].get_weights())
        #print(m_out)
        #exit()

        return {'m':m_out}  

    def update_func(self,nodes):    

        h_input = tf.concat( [nodes.data['m_sum'] , nodes.data['h_n']], -1)
    
        #print(h_input)
        h_out = self.dense_hidden_list[self.current_layer](h_input)
        #print(h_out)
        #exit()
        
        # only update current_layer if you want to repeat using the same dense layer
        if self.repeat_msgANDhidden_layer == False:
            self.current_layer = self.current_layer+1

        return {'h_n':h_out}
        
    def call(self, graph_list):
        
        
        # update multiple molecules at once as a single "graph"
        graph = graph_list
        '''import networkx as nx
        import matplotlib.pyplot as plt
        isolated_nodes = tf.cast( tf.squeeze(tf.where((graph.in_degrees() == 0) & (graph.out_degrees() == 0))) , tf.int32)
        graph = dgl.remove_nodes(graph,isolated_nodes)
        nx_graph = graph.to_networkx().to_undirected()
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(nx_graph, prog="circo")
        nx.draw(nx_graph, pos, with_labels=True, node_color=[[.7, .7, .7]])
        plt.show()
        exit()'''

        with graph.local_scope():

            for _ in range(self.layer_num):
                graph.update_all(   
                                    self.message_func,      
                                    fn.sum('m', 'm_sum'),   # same as dgl.function.u_add_v('hu', 'hv', 'he') 
                                    self.update_func
                                )
            
            # get length of graph by 
            nodes_label_list,_ = tf.unique(graph.ndata['id'])
            nodes_label_len = nodes_label_list.shape[0]

            # if number of atoms of graphs are the same:
            split_pattern = [nodes_label_len]*(int(graph.num_nodes()/nodes_label_len))

            # reset self.current_layer
            self.current_layer = 0
            
            #print('graph',graph.device)
            return  tf.squeeze(   # to remove unnecessary dimensions e.g. [ [0],[0] ] -> [0,0]
                        tf.convert_to_tensor(
                            tf.split(                               #split according to individual graph sizes
                                graph.ndata['h_n'],                    
                                num_or_size_splits=split_pattern      
                            )
                        )
                    )
        #______________________________________________________________________________________________
            
