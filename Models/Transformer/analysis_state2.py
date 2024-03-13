import os
from typing import List, Union

import kl
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import torch
import utils
from kl import diag
from permute.core import two_sample
import tqdm
from utils import calculateMetric
import glob
import matplotlib.pyplot as plt
from nilearn import plotting
import pandas as pd
import seaborn as sns


def difference_test(x, y, reps=10000, stat='mean', alternative='two-sided', seed=20):
    # 返回两个数，t表示x，y的均值差异多大，p表示显著水平， p < 0.05 就可以了
    p, t = two_sample(x, y, reps=reps, stat=stat, alternative=alternative, seed=seed)
    return p, t

def cluster2state(data:Union[np.ndarray, torch.Tensor], state_count=10,):
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    labels = utils.k_means(data, state_count)
    return labels

def cal_occur_count(state: np.ndarray, state_count: int):
    """state的出现次数

    Args:
        state (np.ndarray): shape: (N, L)
        state_count (int): _description_
    return: (N, state_count) 返回每个sub，每个state的出现次数
    """
    ot = np.zeros((state.shape[0], state_count), dtype=int)
    for sub in range(state.shape[0]):
        last = -1
        for t in range(state.shape[1]):
            state_now = state[sub, t]
            if last != state_now:
                ot[sub, state_now] += 1
                last = state_now
    return ot

def cal_switching_frequency(state: np.ndarray):
    """_summary_

    Args:
        state (np.ndarray): shape: (N, L)
        return: (N,) 返回每个sub的状态切换频率
    """
    
    freq = np.zeros((state.shape[0]), dtype=float)
    for sub in range(state.shape[0]):
        last = -1
        for t in range(state.shape[1]):
            if last != state[sub, t]:
                freq[sub] += 1
                last = state[sub, t]
        freq[sub] = freq[sub] / state.shape[1]
    
    return freq

def state_total_time(state, state_count):
    """
    每个subject的每个state出现的总时长 
    Args:
        state (_type_): shape: (N, L)
        state_count
    return: (N, state_count) 
    """
    total = np.empty((state.shape[0], state_count), dtype=np.float32)
    # 每个 state的总时间
    for s in range(state_count):
        total[:, s] = np.count_nonzero(state == s, axis=-1)
    return total

def cal_dwell_time(state, state_count):
    """
    平均dwell time, 
    Args:
        state (_type_): shape: (N, L) 或者array, 因为时长不相等
        state_count
    return: (N, state_count) 每个元素代表这个state的平均dwell time
    """
    if isinstance(state, np.ndarray):
        dwell = state_total_time(state, state_count)
        ot = cal_occur_count(state, state_count) + 1e-10
        
        dwell /= ot
        return dwell
    else:
        pass
     

def cal_dwell_time_group(state, state_count):
    """
    平均dwell time, 
    Args:
        state (_type_): shape: (N, L)
        state_count
    return: (N, state_count) 每个元素代表这个state的平均dwell time
    """
    dwell = state_total_time(state, state_count)
    ot = cal_occur_count(state, state_count)
    
    dwell = np.sum(dwell, axis=0)
    ot = np.sum(ot, axis=0) + 1e-10
    
    dwell /= ot
    return dwell

def transition_to_matrix(state, state_count):
    """计算转移矩阵， 返回 trans_matrix
    trans_matrix[f, t] 就是从state f 转移到state t的次数。

    Args:
        state (_type_): shape: (N, L)
        state_count
    """
    from_state = state[..., :-1]
    to_state = state[..., 1:]
    trans_matrix = np.zeros((state_count, state_count), np.float32)
    for f in range(state_count):
        for t in range(state_count):
            # np.sum()
            c = np.count_nonzero((to_state == t) & (from_state == f))
            # c = np.sum(to_state[from_state == f] == t)
            trans_matrix[f, t] = c
    
    trans_matrix_total = trans_matrix / np.sum(trans_matrix)
    col = np.sum(trans_matrix, axis=1, keepdims=True)
    trans_matrix_to_total = trans_matrix / col
            
    return trans_matrix_total, trans_matrix_to_total

def transition_to_matrix_per_sub(state, state_count):
    """计算转移矩阵， 返回 trans_matrix
    trans_matrix[f, t] 就是从state f 转移到state t的次数。

    Args:
        state (_type_): shape: (N, L)
        state_count
    """
    from_state = state[..., :-1]
    to_state = state[..., 1:]
    trans_matrix = np.zeros((state.shape[0] ,state_count, state_count), np.float32)
    for f in range(state_count):
        for t in range(state_count):
            # np.sum()
            c = np.count_nonzero((to_state == t) & (from_state == f), axis=-1)
            # c = np.sum(to_state[from_state == f] == t)
            trans_matrix[: ,f, t] = c
    
    # trans_matrix /= np.sum(trans_matrix)
            
    return trans_matrix

def communities(trans_matrix):
    """_summary_
    查看更多信息：https://networkx.guide/algorithms/community-detection/
    
    Args:
        trans_matrix (_type_): _description_
    """
    G = nx.DiGraph(trans_matrix)
    # https://github.com/taynaud/python-louvain, 
    # https://python-louvain.readthedocs.io/en/latest/index.html
    # partition = community_louvain.best_partition(G)
    # print(trans_matrix)
    # print(G)
    partition = nx_comm.louvain_communities(G, seed=123)
    modularity = nx_comm.modularity(G, partition)
    # modularity = community_louvain.modularity(partition, G)
    print(partition, modularity)
    
    return G, partition, modularity

def cal_occur_probability(state, state_count):
    ot = state_total_time(state, state_count)
    return ot 

'''
单state组间差异：
1. dwell time
2. FO:fractional occupancy 占用时间比例
3. 切换频率SF
多state 组间差异
1. 状态分布差异
2. FO 相关性
3. 转移矩阵
4. 社区划分
'''

class Analysis(object):
    def __init__(self, subs, num_cluster, state_name='rep_state', result_dir='/root/kl2/code/tmp/BolTkl/Analysis/kl'):
        self.subs = subs
        self.result_dir = os.path.join(result_dir,f'cluster{num_cluster}')
        if os.path.exists(self.result_dir) is False:
            os.makedirs(self.result_dir)
        self.num_cluster = num_cluster
        self.state_name = state_name
        self.group_info = None
        
    def save(self, name, data=None):
        if data is None:
            data = self.subs
        kl.data.save_pickle(data, os.path.join(self.result_dir, name))
        
    def analize_per_sub(self):
        print('analize_per_sub...')
        for sub in self.subs:
            state = sub[self.state_name][None,:]
            # 平均驻留时间
            dwell_time = cal_dwell_time(state, self.num_cluster).squeeze()
            sub['dwell time'] = dwell_time
            # 转移频率
            switch_freq = cal_switching_frequency(state)
            sub['switching frequency'] = switch_freq
            # 出现概率
            fo = cal_occur_probability(state, self.num_cluster).squeeze()
            sub['FO'] = fo
            
            trans_matrix = transition_to_matrix_per_sub(state, self.num_cluster).squeeze()
            
            sub['transition matrix'] = trans_matrix
            
        self.save(f'result_analize_per_sub.pkl')
    
    def draw_state_by_group(self, path, group):
        if not os.path.exists(path):
            os.makedirs(path)
        for sub in group:
            kl.diag.plot_state(sub[self.state_name], num_cluster=self.num_cluster)
            kl.diag.show(file=os.path.join(path, str(sub['sid']) + '_' + str(sub['label']) + '_' + str(sub['answer'])+'.jpg'))
    
    def analize_by_group(self, groups):
        print('analize_by_group...')
        result = []
        for group in groups:
            fc_cluster = np.zeros((self.num_cluster, *self.subs[0]['dfc'].shape[1:]))
            fc_count = np.zeros(self.num_cluster)
            trans_matrix = np.zeros((self.num_cluster, self.num_cluster))
            dwell_time = []
            swith_freq = []
            FO = []
            fo_count = 0
            for sub in group:
                for fc, state in zip(sub['dfc'], sub[self.state_name]):
                    fc_count[state] += 1
                    fc_cluster[state] = fc_cluster[state] + fc
                if 'transition matrix' not in sub:
                    print('..........')
                trans_matrix += sub['transition matrix']
                dwell_time.append(sub['dwell time'])
                swith_freq.append(sub['switching frequency'])
                fo_count += len(sub[self.state_name])
                FO.append(sub['FO'])
                
            fc_cluster = fc_cluster / fc_count[:,None,None]
            trans_matrix = trans_matrix / np.sum(trans_matrix)
            G, partition, modularity = communities(trans_matrix)
            dwell_time = np.stack(dwell_time, axis=0)
            swith_freq = np.array(swith_freq).squeeze()
            FO = np.stack(FO, axis=0) / fo_count
            
            result.append({
                'fc cluster': fc_cluster, # 
                'transition matrix': trans_matrix, # num_cluster, num_cluster
                'partition': partition,
                'dwell time': dwell_time, # N, num_cluster
                'switching frequency': swith_freq, # N
                'FO': FO # N, num_cluster
            })
        
        self.group_info = result

        self.save(data={'subs': self.subs, 'group info': self.group_info}, 
                  name=f'result_analize_by_group.pkl')
    
    def compare_by_group(self):
        print('compare_by_group...')
        # if self.group_info is None:
        #     self.analize_by_group()
        g1 = self.group_info[0]
        g2 = self.group_info[1]
        # 空间分析
        fig, axs = plt.subplots(2, self.num_cluster, figsize=(10*self.num_cluster, 10), sharex=True, sharey=True)
        for i, (fc1, fc2) in enumerate(zip(g1['fc cluster'], g2['fc cluster'])):
            plotting.plot_matrix(mat=fc1, colorbar=False, axes=axs[0][i])
            plotting.plot_matrix(mat=fc2, colorbar=False, axes=axs[1][i])
            axs[1][i].set_title(str(i))
        
        plt.savefig(os.path.join(self.result_dir,'state_fc.jpg'))
        plt.cla()
        plt.close()
        # 时间分析
        # dwell time
        print('dwell time difference:')
        data = {}
        for i in range(self.num_cluster):
            p, t = difference_test(g1['dwell time'][i], g2['dwell time'][i])
            dw1 = np.mean(g1['dwell time'][i])
            dw2 = np.mean(g2['dwell time'][i])
            print(f'state {i}, diff: {t}, p={p}, g1:{dw1} g2:{dw2}')
            data[f'g1_{i}'] = g1['dwell time'][i]
            data[f'g2_{i}'] = g2['dwell time'][i]
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df)
        plt.savefig(os.path.join(self.result_dir, 'dwell_time.jpg'))
        plt.cla()
        plt.close()
        
        '''
         统计检验均值无差异, 不代表dwell time这些两组就完全一样, 可能连分布都不一样！！！再看看
         譬如箱线图看，就已经很不一样了（state dfc）
        '''
        # 转移频率
        print('switch frequency difference:')
        p, t = difference_test(g1['switching frequency'], g2['switching frequency'])
        print(f'state {i}, diff: {t}, p={p}')
        # df = pd.DataFrame({'g1': g1['switching frequency'], 'g2': g2['switching frequency']})
        # sns.boxplot(data=df)
        # plt.savefig(os.path.join(self.result_dir, 'switch_freq.jpg'))
        
        # FO
        print('FO difference:')
        data = {}
        for i in range(self.num_cluster):
            p, t = difference_test(g1['FO'][i], g2['FO'][i])
            fo1 = np.mean(g1['FO'][i])
            fo2 = np.mean(g2['FO'][i])
            print(f'state {i}, diff: {t}, p={p}, g1:{fo1} g2:{fo2}')
            data[f'g1_{i}'] = g1['FO'][i]
            data[f'g2_{i}'] = g2['FO'][i]
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df)
        plt.savefig(os.path.join(self.result_dir, 'FO.jpg'))
        plt.cla()
        plt.close()
        
        # 转移矩阵：
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
        plotting.plot_matrix(mat=g1['transition matrix'], axes=axs[0])
        plotting.plot_matrix(mat=g2['transition matrix'], axes=axs[1])
        plt.savefig(os.path.join(self.result_dir, 'transition matrix.jpg'))
        plt.cla()
        plt.close()
        # axs[0][0].set_title(str(i))        
                
    
    def run(self, rerun=False):
        if not os.path.exists(os.path.join(self.result_dir, 'result_analize_per_sub.pkl')) or rerun:
            self.analize_per_sub()
        else:
            self.subs = kl.data.load_pickle(os.path.join(self.result_dir, 'result_analize_per_sub.pkl'))
        
        groups = [
                [sub for sub in self.subs if sub['label'] == 0], # 疾病
                [sub for sub in self.subs if sub['label'] != 0]  # 健康
            ]
            
        if not os.path.exists(os.path.join(self.result_dir, 'result_analize_by_group.pkl')) or rerun:
            
            self.analize_by_group(groups=groups)
        else:
            data = kl.data.load_pickle(os.path.join(self.result_dir, 'result_analize_by_group.pkl'))
            self.subs = data['subs']
            self.group_info = data['group info']
        
        print('draw state by group')
        self.draw_state_by_group(group=groups[0], path=os.path.join(self.result_dir, 'state_trace', 'autism')) 
        self.draw_state_by_group(group=groups[1], path=os.path.join(self.result_dir, 'state_trace', 'control'))   
        
        self.compare_by_group()

def cluster(subs, use_rep=False, num_clusters=7, filter_sign=False):
    from sklearn.cluster import KMeans
    
    # data = []
    if use_rep:
        if filter_sign:
            data = np.concatenate([sub['ts_rep'][sub['significant_idx']] for sub in subs], axis=0) 
        else:
            data = np.concatenate([sub['ts_rep'] for sub in subs], axis=0) # (39741, 240)
    else:
        if filter_sign:
            data = np.concatenate([sub['dfc'][sub['significant_idx']] for sub in subs], axis=0)
        else:
            data = np.concatenate([sub['dfc'] for sub in subs], axis=0) # (39741, 200, 200)
        data = np.stack([a[np.triu_indices(a.shape[0], k=1)] for a in data]) # (39741, 19900)
    
    # print(data.shape)
    # exit()
    print('ready to k-means')
    model = KMeans(n_clusters=num_clusters, random_state=17).fit(data)
    print('finish k-means')
    labels = model.labels_
    sc, _, _ = kl.stuff.cluster_inner_metrics(data, labels)
    print('sc:', sc)
    key = 'state_rep' if use_rep else 'state_dfc'
    for sub in subs:
        if use_rep:
            data = sub['ts_rep']
        else:
            data = np.stack([a[np.triu_indices(a.shape[0], k=1)] for a in sub['dfc']])
        state = model.predict(data)
        sub[key] = state
    return sc, model.cluster_centers_
        
def test_model(model, dataset, fold):
    
    dataLoader = dataset.getFold(fold, train=False)

    preds = []
    probs = []
    groundTruths = []
    losses = []        

    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

        xTest = data["timeseries"]
        yTest = data["label"]
        sids = data['subjId']
        sids = [str(s) for s in sids.tolist()]

        # NOTE: xTrain and yTrain are still on "cpu" at this point

        _, test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest,sids=sids, train=False, save=True)
        
        torch.cuda.empty_cache()

        preds.append(test_preds)
        probs.append(test_probs)
        groundTruths.append(yTest)
        losses.append(test_loss)

    preds = torch.cat(preds, dim=0).numpy()
    probs = torch.cat(probs, dim=0).numpy()
    groundTruths = torch.cat(groundTruths, dim=0).numpy()
    loss = torch.tensor(losses).numpy().mean()          

    metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
    # print("\n \n Test metrics : {}".format(metrics))                
    
    return preds, probs, groundTruths, loss, metrics

def time_attention_threshold(time_attention, std_cutoff_factor=1.0, consecutive_timepoint=3):
    # [source, target]
    target_attention = time_attention.mean(axis=0)
    threshold = target_attention.mean() + std_cutoff_factor * target_attention.std()
    significant_timepoints = (target_attention>threshold).nonzero()[0]
    # if not len(significant_timepoints) == 0:
    #     significant_timepoints = [cluster[len(cluster)//2] for cluster in np.split(significant_timepoints, np.where(np.diff(significant_timepoints) > 1+consecutive_timepoint)[0]+1)]
    return significant_timepoints

def settle_data(atlas='cc200', cs=True, num_cluster=7):

    # from Dataset.dataset import getDataset
    # from Dataset.datasetDetails import datasetDetailsDict
    # from Models.Transformer.hyperparams import getHyper_transformer
    # datasetDetails = datasetDetailsDict[atlas]
    # dataset = getDataset(datasetDetails)
    # hyperParams = getHyper_transformer(datasetDetails['atlas'])
    # from Models.Transformer.model import Model
    # model = Model(hyperParams, datasetDetails)
    # folders = [2, 6, 4, 3, 5, 7]
    # cs_str = 'vs' if cs else 'no_vs'
    # for k in folders: # k_folder
    #     # dataLoader = dataset.getFold(k, train=True)
    #     state_dict = torch.load(f'./ckpt/stagin/{atlas}/{cs_str}/{k}.cpkt')['state_dict']
    #     model.model.load_state_dict(state_dict)
    #     preds, probs, groundTruths, loss, metrics = test_model(model, dataset, k)
    
    from .stagin.bold import process_dynamic_fc
    from kl.augment import Standardization
    stand = Standardization(dim=0)
    # cache_vs_dir = '/root/kl2/code/tmp/BolTkl/cache/stagin/tmp/vs/cc200_698'
    # cache_ori_dir = '/root/kl2/code/tmp/BolTkl/cache/stagin/ori/cc200_643'
    cache_ori_dir = '/root/kl2/code/tmp/BolTkl/cache/stagin/tmp/ori/cc200'
    # dfc_arr = []
    subs = []
    sparsity = 50
    ff = []
    for i, file in enumerate(glob.glob(os.path.join(cache_ori_dir, '*.npz'))):
        # if i < 100:
        #     continue
        data = np.load(file)
                
        time_attention = data['time_attention'][:,:-1,:-1] # 因为最后一个是cls
        
        plotting.plot_matrix(mat=time_attention[1], colorbar=True)
        # plt.savefig('/root/kl2/code/tmp/BolTkl/Analysis/kl/atten.jpg')
        # exit()
        
        
        # time_attention = np.mean(time_attention, axis=0)
        # m = np.mean(time_attention, axis=1)
        # # range_ = np.max(time_attention, axis=1) - np.min(time_attention, axis=1)
        # range_ = np.percentile(time_attention, q=90, axis=1) - np.percentile(time_attention, q=10, axis=1)
        # f =  range_ / m # 变化程度
        # ff.append(np.mean(f))
        # # print(max(f), sum(f) / len(f), data['answer'].item())
        # continue
        # time_attention = data['time_attention']
        
        time_attention = np.mean(time_attention, axis=0)
        node_attention = data['node_attention']
        # rep = data['rep'] # 
        # print(type(rep), rep.shape)# np, 240
        # exit()
        ts_rep = data['tp_rep'].squeeze()[:-1]
        # ts_rep = rearrange(ts_rep, 'l c k -> l (c k)')
        ts_rep = np.mean(ts_rep, axis=-1)
        answer = data['answer']
        ground_true = data['ground_true']
        data_file = os.path.basename(file)
        data = np.load(f'/data3/surrogate/abide/checked/cc200/tall_no0/{data_file}')
        ts = stand(data['ts'])
        ts = ts[None,:,:]
        # ts = rearrange(ts, 'n l c -> n c l')
        # print(ts.shape)
        ts_torch = torch.Tensor(ts)
        dfc, sampling_points = process_dynamic_fc(ts_torch, window_size=8, window_stride=4)
        dfc.squeeze_()
        dfc = np.stack([(a > np.percentile(a, 100-sparsity)).astype(np.float32) for a in dfc.numpy().copy()])
        # print(data['sid'], time_attention.shape, ts_rep.shape, dfc.shape)
        '''
            严重问题: time_attention 都是差不多的数值！！！
            那就是说, 模型实际上在求均值！！！
        '''
        significant_idx = time_attention_threshold(time_attention, std_cutoff_factor=-1) 
        # # significant_idx2 = time_attention_threshold(time_attention, std_cutoff_factor=0.5)
        # significant_idx3 = time_attention_threshold(time_attention, std_cutoff_factor=0)
        # print(significant_idx1, significant_idx2, significant_idx3)
        # exit()
        significant_mask = np.zeros((dfc.shape[0])).astype(bool)
        significant_mask[significant_idx] = True
        # print(significant_idx)
        # exit()
        subs.append(
            {
                'sid': data['sid'].item(), # str
                'time_attention': time_attention, # np
                'dfc': dfc, # np
                'ts_rep': ts_rep,
                'answer': answer.item(), # bool
                'label': ground_true.item(), # int
                'significant_idx': significant_idx, # list
                'significant_mask': significant_mask, # np (L)
            }
        )
        # print(subs)
        # exit()
        # break
    
    # print(ff, sum(ff) / len(ff))   
    sc_rep, cluster_center_rep = cluster(subs, num_clusters=num_cluster, use_rep=True, filter_sign=False) # 7：sc0.133
    sc_dfc, cluster_center_dfc = cluster(subs, num_clusters=num_cluster, use_rep=False, filter_sign=False)#7, sc0.0013 filter不会提升sc
    kl.data.save_pickle(subs, f'/root/kl2/code/tmp/BolTkl/Analysis/kl/data_cluster{num_cluster}.pkl')
    # kl.data.save_pickle(
    #     {
    #         'subs': subs,
    #         'sc_rep':sc_rep,
    #         'sc_dfc': sc_dfc,
    #         'cluster_center_rep': cluster_center_rep,
    #         'cluster_center_dfc': cluster_center_dfc
    #     }, 
    #     f'/root/kl2/code/tmp/BolTkl/Analysis/kl/data_cluster{num_cluster}.pkl')
        
def anaylize(dir='/root/kl2/code/tmp/BolTkl/Analysis/kl/data_cluster8.pkl', num_cluster=8):
    subs = kl.data.load_pickle(dir)
    '''
    [
        {
            'sid' # str
            'time_attention':, # np
            'dfc':, # np
            'ts_rep': np 
            'answer':  # bool
            'label':  # int
            'significant_idx':  # list
            'significant_mask':  # np (L)
            'state_rep': np (L)
            'state_dfc': np (L)
        }
    ]
    '''
    for sub in subs:
        sub['state_dfc_filter'] = sub['state_dfc'][sub['significant_idx']]
    ana = Analysis(subs=subs, 
                   num_cluster=num_cluster,
                #    state_name='state_dfc_filter',
                    state_name='state_rep',
                   result_dir='/root/kl2/code/tmp/BolTkl/Analysis/kl/rep',
                )
    ana.run(rerun=False)

def main():
    # settle_data(num_cluster=8)
    anaylize()

if __name__ == '__main__':
    main()
    


    
    