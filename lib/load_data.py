import os
os.environ['DGLBACKEND'] = 'tensorflow'
import scipy.io as sio
import numpy as np
import dgl
from sklearn.utils import shuffle

nor_num =
pat_num =
def load_data(dataFile, n_subjects, n_regions, n_lambda, nor_num = nor_num,pat_num = pat_num):
    '''
    :param dataFile: location of Sparsity brain network.
    :param n_subjects:  all subjects numbers
    :param n_regions:  all brain regions
    :param n_lambda:   Sparsity parameter
    :return:  all data
    '''
    data = sio.loadmat(dataFile)
    BrainNetSet = data["BrainNetSet"]
    tupleBrainNetSet = ()
    for i in range(n_lambda):
        a = np.array(list(BrainNetSet[i])).reshape(n_subjects, n_regions, n_regions)
        tupleBrainNetSet += (a,)
    nor = np.zeros((nor_num,),dtype = int)
    pat = np.ones((pat_num,),dtype = int)
    label = np.concatenate((nor,pat))
    arrayBrainNetSet = np.array((tupleBrainNetSet))

    return arrayBrainNetSet, label

def threshold(data,n_subjects, n_regions, n_lambda):
    '''
        return:  Binarized matrix
    '''
    threshold_final = np.zeros((n_lambda, n_subjects, n_regions, n_regions))
    for i in range(n_lambda):
        threshold = np.zeros((n_subjects, n_regions, n_regions))
        for j in range(n_subjects):
            sub_wsr = data[i][j]
            _single_threshold = (sub_wsr != 0).astype(int)
            threshold[j, :] = _single_threshold
        threshold_final[i, :] = threshold
    return threshold_final

def transform(threshaold_matrix,n_subjects, n_regions, n_lambda):
    '''
        return: transformed matrix
    '''
    threshold_ten = threshaold_matrix[1:, :]
    threshold_change = np.zeros((n_subjects,n_lambda-1, n_regions, n_regions))
    for ii in range(n_subjects):
        b = threshold_ten[:, ii, :]
        threshold_change[ii, :] = b
    return threshold_change

def load_knn_adj(datafile_knn):
    knn_tuple = dgl.load_graphs(datafile_knn)
    knn_list = knn_tuple[0]
    knn_numpy = np.array(knn_list)
    return knn_numpy

def load_adj_nodeFeature_label(n_subjects = 125,n_regions = 120, n_lambda = 11):
    dataFile_PC = r'data\BrainNetSet_HC_SZ_PC.mat'
    dataFile_SRC = r'data\BrainNetSet_HC_SZ_SRC.mat'
    dataFile_KNN_SRC = r'data\KNN_SRC_5.bin'
    dataFile_KNN_PC = r'data\KNN_PC_5.bin'
    src, label = load_data(dataFile_SRC,n_subjects, n_regions, n_lambda)
    pc, label = load_data(dataFile_PC,n_subjects, n_regions, n_lambda)
    nodeFeature_pc = pc[0]
    nodeFeature_src = src[0]
    threshold_pc = threshold(pc,n_subjects, n_regions, n_lambda)
    threshold_src = threshold(src,n_subjects, n_regions, n_lambda)
    threshold_pc_change = transform(threshold_pc,n_subjects, n_regions, n_lambda)
    threshold_src_change = transform(threshold_src,n_subjects, n_regions, n_lambda)
    knn_numpy_pc = load_knn_adj(dataFile_KNN_PC)
    knn_numpy_src = load_knn_adj(dataFile_KNN_SRC)
    data_pc, data_spearman, data_threshold_spearman, knn_graph_numpy_pc, knn_graph_numpy_spearman, data_threshold_pc, label_all = \
        shuffle(nodeFeature_pc, nodeFeature_src, threshold_src_change, knn_numpy_pc, knn_numpy_src,
                threshold_pc_change, label, random_state=0)
    return data_pc, data_spearman,data_threshold_spearman,knn_graph_numpy_pc,knn_graph_numpy_spearman,data_threshold_pc,label_all

