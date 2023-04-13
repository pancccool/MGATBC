import os
os.environ['PYTHONHASHSEED'] = str(0)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import dgl
import torch
from gat_data_change.open_code.load_data import load_data
device = torch.device('cuda:0')


n_lambda = 11
subjects = 125
ROIs = 120
K = 5  # {5:50}
dataFile_PC = r'data\BrainNetSet_HC_SZ_PC.mat'
dataFile_SRC = r'data\BrainNetSet_HC_SZ_SRC.mat'
src, label = load_data(dataFile_SRC,subjects, ROIs, n_lambda)
pc, _ = load_data(dataFile_PC,subjects, ROIs, n_lambda)
_single_pc = pc[0]
_single_src = src[0]


# pc
all_knn_graph = []
_knn_all_graphs = np.zeros((subjects,ROIs,ROIs))
for iii in range(subjects):
    nodal__ = _single_pc[iii, :]
    nodal__ = torch.tensor(nodal__)
    knn_pc_ = dgl.knn_graph(nodal__, K)
    all_knn_graph.append(knn_pc_)
dgl.save_graphs('data\knn_PC_%d.bin' % (K), all_knn_graph)


# SRC
all_knn_graph_spearman = []
_knn_all_graphs_spearman = np.zeros((subjects,ROIs,ROIs))
for i in range(subjects):
    nodal_ = _single_src[i, :]
    nodal_ = torch.tensor(nodal_)
    knn_spearman_ = dgl.knn_graph(nodal_, K)
    all_knn_graph_spearman.append(knn_spearman_)
dgl.save_graphs('data\knn_SRC_%d.bin' % (K), all_knn_graph_spearman)
