import os
os.environ['DGLBACKEND'] = 'tensorflow'
import numpy as np
import networkx as nx
from tqdm import tqdm
import dgl
from dgl.nn.tensorflow import *
from gat_data_change.open_code.load_data import load_adj_nodeFeature_label


def graph_constructed_common(brain_list, data_threshold, nodal, TotalNum=125):
    '''
    introduction: This function constructs the graph data for the adjacency matrix produced by the percentage threshold.
    :return: cell
    '''
    graph_cell = np.array(())
    for i in tqdm(brain_list):
        graph_dicts = np.array(())
        for j in range(TotalNum):
            adj_ = data_threshold[j, i, :]
            nodal_ = nodal[j, :]
            nodal_ = tf.convert_to_tensor(nodal_)
            graph = nx.from_numpy_matrix(adj_)
            dgl_graph = dgl.from_networkx(graph)

            dgl_graph.ndata['x'] = nodal_
            dgl_graph = dgl.add_self_loop(dgl_graph)
            graph_dicts = np.append(graph_dicts,dgl_graph)
        graph_cell = np.append(graph_cell,graph_dicts)
    return graph_cell.reshape(10,TotalNum)

def graph_constructed_knn(knn_graph_numpy,nodal,TotalNum=125):
    '''
    introduction: This function constructs the graph data for the adjacency matrix produced by the KNN.
    :return: knn_graph
    '''
    graph_dicts_knn = []
    for j in range(TotalNum):
        knn_every = knn_graph_numpy[j].to('/gpu:0')
        nodal_ = nodal[j, :]
        nodal_ = tf.convert_to_tensor(nodal_)
        knn_every.ndata['x'] = nodal_.gpu()
        graph_dicts_knn.append(knn_every)
    return graph_dicts_knn




def get_samples(graphs, labels, way):
    graph = list(graphs[:, way])
    label = list(labels)
    samples = list(zip(graph, label))
    return samples



def create_graph_generator_train_all_path(graphs, batch_size, infinite=False, shuffle=False):

    while True:
        dataset = tf.data.Dataset.range(len(graphs[0]))
        if shuffle:  
            dataset = dataset.shuffle(40) 
        dataset = dataset.batch(batch_size)

        for batch_graph_index in dataset:
            batch_graph_list_knn_src = [graphs[0][i] for i in batch_graph_index]  
            batch_graph_list_knn_pc = [graphs[1][j] for j in batch_graph_index]
            batch_graph_list_pt_src = [graphs[2][k] for k in batch_graph_index]
            batch_graph_list_pt_pc = [graphs[3][l] for l in batch_graph_index]

            _graphs_knn_src, _labels1 = map(list, zip(*batch_graph_list_knn_src))
            _graphs_knn_pc, _labels2 = map(list, zip(*batch_graph_list_knn_pc)) 
            _graphs_pt_src, _labels3 = map(list, zip(*batch_graph_list_pt_src))
            _graphs_pt_pc, _labels4 = map(list, zip(*batch_graph_list_pt_pc))

            batch_graph_knn_src = dgl.batch(_graphs_knn_src)
            batch_graph_knn_pc = dgl.batch(_graphs_knn_pc)
            batch_graph_pt_src = dgl.batch(_graphs_pt_src)
            batch_graph_pt_pc = dgl.batch(_graphs_pt_pc)
            yield batch_graph_knn_src, batch_graph_knn_pc, batch_graph_pt_src, batch_graph_pt_pc, _labels1

        if not infinite:
            break


def load_graphs_construction():
    data_pc, data_spearman, data_threshold_spearman, knn_graph_numpy_pc, knn_graph_numpy_spearman, data_threshold_pc, label_all = load_adj_nodeFeature_label()

    graph_cell_pc = graph_constructed_common(list(range(10)),data_threshold_pc,data_pc)
    graph_cell_spearman = graph_constructed_common(list(range(10)),data_threshold_spearman,data_spearman)
    graph_dict_knn_pc = graph_constructed_knn(knn_graph_numpy_pc,data_pc)
    graph_dict_knn_spearman = graph_constructed_knn(knn_graph_numpy_spearman,data_spearman)
    print('-----------------------------------graph have constructed!!!!!!----------------------------------------------')
    return graph_dict_knn_spearman, graph_dict_knn_pc, graph_cell_spearman, graph_cell_pc, label_all


