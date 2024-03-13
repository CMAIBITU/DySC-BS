'''
直接使用 Pretrain类, 调用get_rep_from_cache 方法
'''

import numpy as np
from torch.autograd import Variable
import torch
from Models.hinets.model_ehigcn import EHIgcn2
# from kl.torch_data.sur.utils import get_k_folder_idx_train_val_test_by_k, read_all_data
import torch.nn
import sklearn.metrics as metrics
import time
import glob
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random

def cal_pcc_dummy(d):
    return cal_pcc(d['thr'], d['pcc'])

def cal_pcc(thr, pcc):
    len_pcc = pcc.shape[0]
    len_node = pcc.shape[1]
    corr_matrix = []
    for bb in range(len_pcc):
        for i in range(len_node):
            pcc[bb][i][i] = 0
        corr_mat = np.arctanh(pcc[bb])
        corr_matrix.append(corr_mat)
        
    pcc_array = np.array(corr_matrix)
    pcc_array = np.nan_to_num(pcc_array, nan=0, posinf=0.8, neginf=0.8)
    pcc_array[ np.where( np.abs(pcc_array) < thr)] = 0
    print('pcc_array::::', pcc_array.shape)

    corr_p = np.maximum(pcc_array, 0)
    corr_n = 0 - np.minimum(pcc_array, 0)
    pcc_array = [corr_p, corr_n]
    pcc_array = np.array(pcc_array)
    pcc_array = np.transpose(pcc_array, (1, 0, 2, 3))

    return pcc_array


def load_corr_and_labels(atlas='aal',dataset='abide' , no0=True, seed=0, use_float16=False, check=True):
    if dataset == 'abide':
        check_str = 'checked' if check else 'unchecked'
        path = f'/data3/surrogate/abide/{check_str}/{atlas}/tall'
        if no0:
            path += '_no0'
    elif dataset == 'adhd':
        path = f'/data3/surrogate/adhd200/{atlas}/combine/.npz'
    # print('load_corr_and_labels', path)    
    subs = read_all_data(path, shuffle_seed=seed)
    data_array = [sub['sfc_pcc'] for sub in subs]
    data_array = np.stack(data_array)
    if use_float16:
        data_array.astype(np.float16)
    labels = [sub['label'] for sub in subs]
    
    labels = np.stack(labels)
    return data_array, labels

def load_graph_kernel(len_pcc, name, dataset='abide', no0=True, check=True):
    final_graph = np.ones((len_pcc, len_pcc))
    suffix = '_no0' if no0 else ''
    check_str = '' if check else 'uncheck_'
    path = f'data/{check_str}{dataset}_graph_kernel_'+name + suffix + '_kl.txt'
    # print('load_graph_kernel', path)
    with open(path, 'r') as f:
        count = 0
        for line in f:
            line.strip('\n')
            line = line.split()
            for columns in range(len_pcc):
                final_graph[count][columns] = float(line[columns])
            count += 1

    adj = np.zeros((len_pcc, len_pcc))
    for i in range(len_pcc):
        for j in range(i + 1, len_pcc):
            adj[i][j] = 1
            adj[j][i] = 1

    print('final graph:', final_graph.shape)
    final_graph = np.abs(final_graph)

    return final_graph

# def get_all_node_adj(sub_count=871, name='aal', device='cuda'):
#    node_adj_data = load_graph_kernel(sub_count, name)
#    return  Variable(torch.FloatTensor(node_adj_data), requires_grad=False).to(device)

# def get_train_val_rep(model, ):
def get_node_adj(train_nodes, train_val_nodes, train_test_nodes, node_adj, device='cuda', train_val_only=False):
    if train_val_only: 
        train_val_adj = torch.zeros(len(train_val_nodes), len(train_val_nodes)).to(device)
        print(len(train_val_nodes) - len(train_nodes))
        for i in range(len(train_val_nodes)):
            for j in range(len(train_val_nodes) - len(train_nodes), len(train_val_nodes)):
                train_val_adj[i][j] = node_adj[train_val_nodes[i]][train_val_nodes[j]]
                train_val_adj[j][i] = train_val_adj[i][j]
        return None, train_val_adj, None
    train_adj = torch.zeros(len(train_nodes), len(train_nodes)).to(device)
    for i in range(len(train_nodes)):
        for j in range(i + 1, len(train_nodes)):
            train_adj[i][j] = node_adj[train_nodes[i]][train_nodes[j]]
            train_adj[j][i] = train_adj[i][j]

    train_val_adj = torch.zeros(len(train_val_nodes), len(train_val_nodes)).to(device)
    print(len(train_val_nodes) - len(train_nodes))
    for i in range(len(train_val_nodes)):
        for j in range(len(train_val_nodes) - len(train_nodes), len(train_val_nodes)):
            train_val_adj[i][j] = node_adj[train_val_nodes[i]][train_val_nodes[j]]
            train_val_adj[j][i] = train_val_adj[i][j]

    train_test_adj = torch.zeros(len(train_test_nodes), len(train_test_nodes)).to(device)
    for i in range(len(train_test_nodes)):
        for j in range(len(train_test_nodes) - len(train_nodes), len(train_test_nodes)):
            train_test_adj[i][j] = node_adj[train_test_nodes[i]][train_test_nodes[j]]
            train_test_adj[j][i] = train_test_adj[i][j]

    return train_adj, train_val_adj, train_test_adj

class Pretrain():
    def __init__(self, device='cuda', name='aal', dataset='abide', 
                 data_root='/data3/surrogate/abide/checked/aal/tall_tr2',
                 n_splits=5, n_splits2=8, no_val=False, shuffle_before_split=False, shuffle_seed=0,
                 use_float16=False, check=True):
        self.name = name
        self.dataset = dataset
        pcc, labels = load_corr_and_labels(name, dataset=dataset, check=check)
        self.labels = labels
        self.n_splits = n_splits
        self.n_splits2 = n_splits2
        self.shuffle_before_split = shuffle_before_split
        self.shuffle_seed = shuffle_seed
        self.use_float16 = use_float16
        self.check = check
        
        # if self.shuffle_before_split:
        #         import random
        #         random.Random(self.shuffle_seed).shuffle(idx)
        
        thrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.05, 0.15, 0.25, 0.35, 0.45]
        args = [{'thr': thr, 'pcc':pcc} for i, thr in  enumerate(thrs)]
        from multiprocessing import Pool
        with Pool(10) as p:
            results = p.map(cal_pcc_dummy, args)
            
        graph_adj1 = results[0]
        graph_adj2 = results[1]
        graph_adj3 = results[2]
        graph_adj4 = results[3]
        graph_adj5 = results[4]
        graph_adj6 = results[5]
        graph_adj7 = results[6]
        graph_adj8 = results[7]
        graph_adj9 = results[8]
        graph_adj10 = results[9]

        node_adj_data = load_graph_kernel(pcc.shape[0], self.name, dataset=dataset, check=check)
        self.brain_region_num= pcc.shape[1]
        
        to_tensor = torch.HalfTensor if use_float16 else torch.FloatTensor

        self.graph_adj1 = Variable(to_tensor(graph_adj1), requires_grad=False).to(device)
        self.graph_adj2 = Variable(to_tensor(graph_adj2), requires_grad=False).to(device)
        self.graph_adj3 = Variable(to_tensor(graph_adj3), requires_grad=False).to(device)
        self.graph_adj4 = Variable(to_tensor(graph_adj4), requires_grad=False).to(device)
        self.graph_adj5 = Variable(to_tensor(graph_adj5), requires_grad=False).to(device)
        self.graph_adj6 = Variable(to_tensor(graph_adj6), requires_grad=False).to(device)
        self.graph_adj7 = Variable(to_tensor(graph_adj7), requires_grad=False).to(device)
        self.graph_adj8 = Variable(to_tensor(graph_adj8), requires_grad=False).to(device)
        self.graph_adj9 = Variable(to_tensor(graph_adj9), requires_grad=False).to(device)
        self.graph_adj10 = Variable(to_tensor(graph_adj10), requires_grad=False).to(device)

        self.node_adj_data = Variable(torch.FloatTensor(node_adj_data), requires_grad=False).to(device)
        self.models = {}
        self.data = {}
        self.sid2population = {}
        
        if shuffle_before_split:
            subs = read_all_data(data_root, shuffle_seed=shuffle_seed)
        else:
            subs = read_all_data(data_root)
        
        if self.dataset == 'abide':
            self.sids = [ sub['sid'].item() for sub in subs ] # 个人口图对应，即 self.node_adj_data
        else:
            self.sids = [ str(sub['sid'].item()) + '_' + str(sub['sess'].item()) for sub in subs ]
        self.reps_cache = {}
        self.pred_cache = {}
        # self.labels = [sub['label'] for sub in subs]
        # self.sid2idx = { sub['sid']: i for i, sub in enumerate(subs) }
        # print(self.sids)
        # exit()
        self.no_val = no_val
    
    def __get_folder_k_model__(self, k):
        if k not in self.models or self.models[k] is None:
            model = EHIgcn2(in_dim=5, hidden_dim=5, graph_adj1=self.graph_adj1, graph_adj2=self.graph_adj2, graph_adj3=self.graph_adj3,
                        graph_adj4=self.graph_adj4, graph_adj5=self.graph_adj5, graph_adj6=self.graph_adj6, graph_adj7=self.graph_adj7,
                        graph_adj8=self.graph_adj8, graph_adj9=self.graph_adj9, graph_adj10=self.graph_adj10, num=k, thr1='1', thr2='2',
                        thr3='3', thr4='4', thr5='5', thr6='05', thr7='15', thr8='25', thr9='35', thr10='45',
                        brain_region_num=self.brain_region_num, name=self.name, dataset=self.dataset, n_splits=self.n_splits)
            # path = '/root/kl2/code/tmp/TE-HI-GCN/model/ehigcn/ASD_' + self.name + '/ehigcn_' + str(k+1) + '.pt'
            # if self.name != 'abide':
            #     path = f'/root/kl2/code/tmp/TE-HI-GCN/model/ehigcn/{self.dataset}_' + self.name + '/ehigcn_' + str(k+1) + '.pt'
            # path = f'model/ehigcn/{self.n_splits}/{self.dataset}_{self.name}'
            check_str = 'checked' if self.check else 'unchecked'
            path = f'data/{self.dataset}_{self.name}_base' + '/ehigcn_' + str(k+1) + '.pt'
            # print(path)
            model.load_state_dict(torch.load(path))
            self.models[k] = model
        return self.models[k]
    
    def __sid2population_graph_idx__(self, population_idx):
        # population_idx 数组[idx0, idx1, idx2...], idx0表示在self.sids中的位置。
        sid2graph = {self.sids[population_idx[i]]: i for i in range(len(population_idx))}
        return sid2graph
            
    def __get_folder_k_data__(self, k):
        """_summary_
            注意, 数据的顺序是敏感的，按照 torch_data.sur.utils.read_all_data的顺序
            k_folder的数据也是按照torch_data.sur.utils
        Args:
            k (_type_): folder k,
            mode (str, optional): _description_. Defaults to 'train'.
        """    
        if k not in self.data or self.data[k] is None:
            idx = np.arange(len(self.labels))
            seed = None if self.shuffle_before_split else 0
            train_idx, val_idx, test_idx = get_k_folder_idx_train_val_test_by_k(
                idx=idx, labels=self.labels, k=k, 
                n_splits=self.n_splits,
                n_splits2=self.n_splits2,
                shuffle=not self.shuffle_before_split, 
                random_seed=seed)
            
            
            train_val = np.append(val_idx, train_idx) #按源码写法，实际上是 val在前 train在后
            train_test = np.append(test_idx, train_idx)
            train_pop = self.__sid2population_graph_idx__(train_idx)
            train_val_pop = self.__sid2population_graph_idx__(train_val)
            train_test_pop = self.__sid2population_graph_idx__(train_test)
            
            train_node_adj, train_val_node_adj, train_test_node_adj = get_node_adj(train_idx, train_val, train_test,
                                                                                self.node_adj_data,
                                                                                train_val_only=self.use_float16)
            self.data[k] = { 'node_idx':[train_idx, train_val, train_test] ,'adj':[train_node_adj, train_val_node_adj, train_test_node_adj]}
            self.sid2population[k] = [train_pop, train_val_pop, train_test_pop] # 给 reps_cache 用的
            
            # print('te', test_idx)
            # exit()
            # if self.no_val:
            #     # train_val_idx = train_idx + val_idx
            #     train_val_idx = np.append(val_idx, train_idx) 
            #     train_val_pop = self.__sid2population_graph_idx__(train_val_idx)
                
            #     # print([int(key) for key in train_val_pop])
            #     # print(train_val_idx.tolist())
            #     # print(k)
            #     # exit()
                
            #     # val_pop = self.__sid2population_graph_idx__(val_idx)
            #     test_pop = self.__sid2population_graph_idx__(test_idx)
            #     # train_val = np.append(val_idx, train_idx) #按源码写法，实际上是 val在前 train在后
            #     all_idx = np.append(test_idx, train_val_idx)
            #     _, train_val_node_adj, all_node_adj = get_node_adj(train_val_idx, train_val_idx, all_idx,
            #                                                                         self.node_adj_data)
            #     self.data[k] = { 'node_idx':(train_val_idx, train_val_idx, all_idx) ,'adj':(train_val_node_adj, train_val_node_adj, all_node_adj)}
            #     self.sid2population[k] = (train_val_pop, train_val_pop, test_pop)
            # else:
            #     train_pop = self.__sid2population_graph_idx__(train_idx)
            #     val_pop = self.__sid2population_graph_idx__(val_idx)
            #     test_pop = self.__sid2population_graph_idx__(test_idx)
            #     train_val = np.append(val_idx, train_idx) #按源码写法，实际上是 val在前 train在后
            #     train_test = np.append(test_idx, train_idx)
            #     train_node_adj, train_val_node_adj, train_test_node_adj = get_node_adj(train_idx, train_val, train_test,
            #                                                                         self.node_adj_data)
            #     self.data[k] = { 'node_idx':(train_idx, train_val, train_test) ,'adj':(train_node_adj, train_val_node_adj, train_test_node_adj)}
            #     self.sid2population[k] = (train_pop, val_pop, test_pop) # 给 reps_cache 用的
            
        return self.data[k] # { 'node_idx':(train_idx, train_val, train_test) ,'adj':(train_node_adj, train_val_node_adj, train_test_node_adj)}
        
    def free_folder_k_model_and_data(self, k):
        del self.models[k]
        del self.data[k]
        del self.sid2population[k]
        
    def free_all(self):
        print('TEHIGCN PRETRAIN========== free all cache ===========')
        # for k in self.models:
        #     self.models[k] = self.models[k].cpu()
        self.models.clear()    
        # torch.cuda.empty_cache()
        
        # for k in self.data:
        #     # data = self.data[k]
        #     for i in range(3):
        #         if isinstance(self.data[k]['adj'][i], torch.Tensor):
        #             self.data[k]['adj'][i] = self.data[k]['adj'][i].cpu()
        #     # for key in data:
        #     #     if isinstance(data[key], torch.Tensor):
        #     #         data[key] = data[key].cpu()
        self.data.clear() # 没有释放
        # torch.cuda.empty_cache()
                    
        self.sid2population.clear()
        
        # for k in self.pred_cache:
        #     cache = self.pred_cache[k]
        #     for i in range(3):
        #         if isinstance(cache[i], torch.Tensor):
        #             self.pred_cache[k][i] = cache[i].cpu()
        self.pred_cache.clear()# 没有释放
        # torch.cuda.empty_cache()
        # for k in self.reps_cache:
        #     cache = self.reps_cache[k]
        #     for i in range(3):
        #         if isinstance(cache[i], torch.Tensor):
        #             self.reps_cache[k][i] = cache[i].cpu()
        self.reps_cache.clear()
        # time.sleep(5)
        torch.cuda.empty_cache()
        
    def free_exclude_rep_cache(self):
        print('TEHIGCN PRETRAIN========== free pretrain model ===========')
        # for m in self.models:
        #     m = None
        self.models.clear()
        self.data.clear()
        self.sid2population.clear()
        self.pred_cache.clear()
        # self.reps_cache.clear()
        time.sleep(5)
        torch.cuda.empty_cache()
        
    def get_rep_from_cache(self, folder_k, sids, dataset=0, return_all=False):
        """_summary_

        Args:
            folder_k (_type_): _description_
            sids (_type_): _description_
            dataset (int, optional): 0: train, 1: val, 2: test
        """
        if folder_k not in self.reps_cache or self.reps_cache[folder_k] is None:  
            data = self.__get_folder_k_data__(folder_k)
            node_idx = data['node_idx']
            adj = data['adj']
            
            model = self.__get_folder_k_model__(folder_k)
            
            # reps_train = []
            # pred_train = []
            # reps_train_val = []
            # pred_val = []
            # reps_train_test = []
            # pred_test = []
            
            if self.use_float16:
                
                # reps_train = None
                # pred_train = None
                # reps_train_test = None
                # pred_test = None              
                train_len = len(self.sid2population[folder_k][0])
                reps, pred_val, _ = model(node_idx[1], adj[1])
                val_len = len(self.sid2population[folder_k][1]) - train_len
                # reps_val = reps[:val_len]
                reps_train_val = reps
                
                self.reps_cache[folder_k] = [[], reps_train_val, []]
                self.pred_cache[folder_k] = [[], pred_val, []]
                
            else:
                reps_train, pred_train, _ = model(node_idx[0], adj[0])
                train_len = len(self.sid2population[folder_k][0])
                
                reps, pred_val, _ = model(node_idx[1], adj[1])
                val_len = len(self.sid2population[folder_k][1]) - train_len
                reps_val = reps[:val_len]
                reps_train_val = reps
                
                reps, pred_test, _ = model(node_idx[2], adj[2])
                test_len = len(self.sid2population[folder_k][2]) - train_len
                reps_test = reps[:test_len]
                reps_train_test = reps
                
                pred_test = pred_test[:test_len]
                # test_idx = node_idx[2][:test_len]
                # ground_true = self.labels[test_idx]
                # acc = metrics.accuracy_score(ground_true, pred_test.cpu().data.numpy().argmax(axis=1))
                # print(f'TE-HI-GCN in folder {folder_k}, test acc:{acc}')
                # self.reps_cache[folder_k] = (reps_train, reps_val, reps_test)
                
                # ground_true = self.labels[node_idx[dataset]][sids]
                # pred = pred_val[]
                # graph_idx = [ self.sid2population[folder_k][dataset][sid] for sid in sids ]
                # ground_true = np.array(self.labels)[node_idx[dataset]][graph_idx]
                # pred = pred_val[graph_idx]
                # acc = metrics.accuracy_score(ground_true, pred.cpu().data.numpy().argmax(axis=1))
                # print(f'TE-HI-GCN in folder {folder_k}, train acc:{acc}')
            
                self.reps_cache[folder_k] = (reps_train, reps_train_val, reps_train_test)
                self.pred_cache[folder_k] = (pred_train, pred_val, pred_test)
        try:
            # for sid in sids:
            #     if sid not in self.sid2population[folder_k][dataset]:
            #         print(sid, ' not in')    
            graph_idx = [ self.sid2population[folder_k][dataset][sid] for sid in sids ]
        except:
            # print('eeeeeeeeee,[ self.sid2population[folder_k][dataset][sid] for sid in sids ] wrong', folder_k, dataset)
            # print(sids)
            return None, None, None
            # exit()
        # print(len(graph_idx))
        
        # graph_idx = [ self.sid2population[folder_k][dataset][sid] for sid in sids ]
        data = self.__get_folder_k_data__(folder_k)
        node_idx = data['node_idx']
        ground_true = np.array(self.labels)[node_idx[dataset]][graph_idx]
        pred = self.pred_cache[folder_k][dataset][graph_idx]
        # acc = metrics.accuracy_score(ground_true, pred.cpu().data.numpy().argmax(axis=1))
        # print(f'TE-HI-GCN in folder {folder_k}, train acc:{acc}')
        # if acc < 0.7:
        #     # print(f'TE-HI-GCN in folder {folder_k}, train acc:{acc}')
        #     return None, None
        
        if return_all:
            return self.reps_cache[folder_k][dataset], graph_idx
        reps = self.reps_cache[folder_k][dataset][graph_idx]
        
        ground_true = torch.Tensor(ground_true).to(pred.device)
        right_mask = (ground_true == pred.argmax(axis=1))
        
        return reps, graph_idx, right_mask
    
    def train(self, folder_k, sids):
        data = self.__get_folder_k_data__(folder_k)
        node_idx = data['node_idx']
        adj = data['adj']
        model = self.__get_folder_k_model__(folder_k)  
        reps_train, pred, cluster_loss = model(node_idx[0], adj[0])
        graph_idx = [ self.sid2population[folder_k][0][sid] for sid in sids ]
        return reps_train[graph_idx], pred[graph_idx], cluster_loss

    # def test(self, folder_k, sids):
    #     pass
        
    
    def new_torch_model(self, train_model=False):
        class TE_HI_GCN_Model_Proxy(torch.nn.Module):
            def __init__(self, pretrain: Pretrain, train_model=False):
                super().__init__()
                self.pretrain = pretrain
                self.train_model = train_model
                
            def forward(self, folder_k, sids, dataset=0, big_neg_size=False):
                if self.train_model:
                    return self.train(folder_k, sids)
                else:
                    return self.pretrain.get_rep_from_cache(folder_k, sids, dataset, return_all=big_neg_size)
                
        return TE_HI_GCN_Model_Proxy(pretrain=self, train_model=train_model)
    
    # def get_folder_k_train_rep(self, k, indices, cache=False):
    #     pass
    
    
def k_folder_idx_train_val_test(idx, labels, n_splits=10, n_splits2=10, random_seed=0, shuffle=True):
    """
        不要修改！！！
    Args:
        idx (_type_): np array, 可以是sid, 可以是index
        labels (_type_): np array
        n_splits (int, optional): _description_. Defaults to 10.
        random_seed (int, optional): _description_. Defaults to 0.
    """
    if n_splits2 is None:
        n_splits2 = n_splits
    if shuffle == False:
        random_seed = None
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
    for train_val_idx, test_idx in kf.split(idx, labels):
        test = idx[test_idx]
        train_val = idx[train_val_idx]
        train_val_labels = labels[train_val_idx]
        kf2 = StratifiedKFold(n_splits=n_splits2, shuffle=shuffle, random_state=random_seed)
        for train_idx, val_idx in kf2.split(train_val, train_val_labels):
            train = train_val[train_idx]
            val = train_val[val_idx]
            # break
        yield train, val, test
        
def get_k_folder_idx_train_val_test_by_k(idx, labels, k, n_splits=10, n_splits2=None, random_seed=0, shuffle=True):
    assert k < n_splits
    for kk, (train, val, test) in enumerate(k_folder_idx_train_val_test(idx, labels, n_splits=n_splits, n_splits2=n_splits2, random_seed=random_seed, shuffle=shuffle)):
        if kk == k:
            return train, val, test

def read_all_data(path='/data3/surrogate/abide/checked/aal/tall_tr2', shuffle_seed=None, return_files=False):
    """
        不要修改

    Args:
        path (str, optional): _description_. Defaults to '/data3/surrogate/abide/checked/aal/tall_tr2'.

    Returns:
        _type_: _description_
    """
    if path.endswith('.npz') is False:
        path = os.path.join(path, '*.npz')
    files = glob.glob(path)
    files.sort()
    subs = [np.load(f) for f in files]
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(subs)
        random.Random(shuffle_seed).shuffle(files)
    if return_files:
        return subs, files
    return subs