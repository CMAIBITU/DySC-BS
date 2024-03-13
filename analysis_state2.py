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
from scipy.stats import ttest_ind_from_stats


def difference_test(x, y, reps=10000, stat='mean', alternative='two-sided', seed=20):
    # 返回两个数，t表示x，y的均值差异多大，p表示显著水平， p < 0.05 即显著差异
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
    return ot / state.shape[1]

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
    def __init__(self, subs, num_cluster, state_name='rep_state', result_dir='/root/kl2/code/tmp/BolTkl/Analysis/kl',
                 save=False):
        self.subs = subs
        self.result_dir = os.path.join(result_dir,f'cluster{num_cluster}')
        if os.path.exists(self.result_dir) is False:
            os.makedirs(self.result_dir)
        self.num_cluster = num_cluster
        self.state_name = state_name
        self.group_info = None
        self.need_save = save
        
    def save(self, name, data=None):
        if data is None:
            data = self.subs
        if os.path.exists(os.path.join(self.result_dir, name)):
            os.remove(os.path.join(self.result_dir, name))
        kl.data.save_pickle(data, os.path.join(self.result_dir, name))
        
    def analize_per_sub(self):
        print('analize_per_sub...')
        all_reps = []
        all_labels = []
        for sub in self.subs:
            state = sub[self.state_name]
            all_reps.append(sub['cls'])
            all_labels.append(state)
            state = state[None,:]
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
            
            sub['transition matrix'] = trans_matrix / np.sum(trans_matrix)
        if self.need_save:
            self.save(f'result_analize_per_sub.pkl')
            
        all_reps = np.concatenate(all_reps, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        diag.cluster_visual_tsne(all_reps, all_labels, file=os.path.join(self.result_dir, 'tsne.jpg'))
        
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
            # fc_cluster = np.zeros((self.num_cluster, *self.subs[0]['dfc'].shape[1:]))
            # fc_count = np.zeros(self.num_cluster)
            # trans_matrix = np.zeros((self.num_cluster, self.num_cluster))
            trans_matrix = []
            dwell_time = []
            swith_freq = []
            FO = []
            fo_count = 0
            for sub in group:
                # for fc, state in zip(sub['dfc'], sub[self.state_name]):
                #     fc_count[state] += 1
                #     fc_cluster[state] = fc_cluster[state] + fc
                if 'transition matrix' not in sub:
                    print('..........')
                trans_matrix.append(sub['transition matrix'])
                dwell_time.append(sub['dwell time'])
                swith_freq.append(sub['switching frequency'])
                fo_count += len(sub[self.state_name])
                FO.append(sub['FO'])
                
            # fc_cluster = fc_cluster / fc_count[:,None,None]
            
            trans_matrix = np.stack(trans_matrix, axis=0)
            G, partition, modularity = communities(np.mean(trans_matrix, axis=0))
            dwell_time = np.stack(dwell_time, axis=0)
            swith_freq = np.array(swith_freq).squeeze()
            FO = np.stack(FO, axis=0)
            
            result.append({
                # 'fc cluster': fc_cluster, # 
                'transition matrix': trans_matrix, # num_cluster, num_cluster
                'partition': partition,
                'dwell time': dwell_time, # N, num_cluster
                'switching frequency': swith_freq, # N
                'FO': FO # N, num_cluster
            })
        
        self.group_info = result

        if self.need_save:
            self.save(data={'subs': self.subs, 'group info': self.group_info}, 
                    name=f'result_analize_by_group.pkl')
           
    def single_state_stat_diff_by_group(self, g1, g2, name):
        # g1 shape:(N1) g2 shape (N2): 
        m1, m2 = np.mean(g1), np.mean(g2)
        print(f'============={name}==============')
        print('mean:', m1, m2)
        # 统计量比较： mean, 在看看用scipy的
        p, t = difference_test(g1, g2)
        if p < 0.05:
            print(f'mean is significant not equal p={p}, t={t}')
        elif p > 0.1:
            print(f'not significant p={p}, t={t}')
        else:
            print(f'emmm...., p={p}, t={t}')
        
        
        # 画分布图
        diag.hist2(data1=g1, label1='autism', data2=g2, label2='control', file=os.path.join(self.result_dir, f'hist_{name}.jpg'))
        
        # 画箱线图
        
        # 求 JS 散度，但是数据长度不等，再看看，感觉这个有前途的！！！
        # 但是先不做这个，
        return m1, m2, p, t
            
    def classify_by_JSD_score(g0, g1, test, groundTrue):
        pass
        
    
    def compare_by_group(self):
        print('compare_by_group...')
        # if self.group_info is None:
        #     self.analize_by_group()
        diff_info = {}
        g1 = self.group_info[0]
        g2 = self.group_info[1]
        
        
        # 转移矩阵：
        print('transition matrix difference:')
        tm1 = g1['transition matrix']
        tm2 = g2['transition matrix']
        p_tm = np.zeros((self.num_cluster, self.num_cluster))
        t_tm = np.zeros_like(p_tm)
        for i in range(self.num_cluster):
            for j in range(self.num_cluster):
                p_tm[i, j], t_tm[i, j] = difference_test(tm1[:,i,j], tm2[:,i,j])
        
        print('diff p:')
        print(p_tm)
        diff_info['transition matrix'] = [tm1, tm2, p_tm, t_tm]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)
        tm1 = np.mean(tm1, axis=0)
        tm2 = np.mean(tm2, axis=0)
        
        plotting.plot_matrix(mat=tm1, axes=axs[0])
        plotting.plot_matrix(mat=tm2, axes=axs[1])
        plt.savefig(os.path.join(self.result_dir, 'transition matrix.jpg'))
        plt.cla()
        plt.close()
        
        # 单状态
        # 时间分析
        # dwell time
        print('dwell time difference:')

        dwell_diff = []
        for i in range(self.num_cluster):
            g1_, g2_ = g1['dwell time'][:, i], g2['dwell time'][:, i]
            m1, m2, p, t = self.single_state_stat_diff_by_group(g1_, g2_, name=f'dwell time {i}')
            dwell_diff.append([m1, m2, p, t])
        
        diff_info['dwell time'] = dwell_diff
        
        # 转移频率
        print('switch frequency difference:')
        p, t = difference_test(g1['switching frequency'], g2['switching frequency'])
        print(np.mean(g1['switching frequency']), np.mean(g2['switching frequency']), f'diff: {t}, p={p}')
        
        diff_info['switching frequency'] = [np.mean(g1['switching frequency']), np.mean(g2['switching frequency']), p, t]
        
        # FO
        print('FO difference:')
        FO_diff = []
        for i in range(self.num_cluster):
            g1_, g2_ = g1['FO'][:, i], g2['FO'][:, i]
            m1, m2, p, t = self.single_state_stat_diff_by_group(g1_, g2_, name=f'FO {i}')
            FO_diff.append([m1, m2, p, t])
        
        diff_info['FO'] = FO_diff
        
        if self.need_save:
            self.save(data=diff_info, name='compare_by_group.pkl')        
    
                    
    
    def run(self, rerun=False):
        if not os.path.exists(os.path.join(self.result_dir, 'result_analize_per_sub.pkl')) or rerun:
            self.analize_per_sub()
        else:
            self.subs = kl.data.load_pickle(os.path.join(self.result_dir, 'result_analize_per_sub.pkl'))
        
        groups = [
                [sub for sub in self.subs if sub['label'] == 0], # ASD
                [sub for sub in self.subs if sub['label'] != 0]  # HC
            ]
            
        if not os.path.exists(os.path.join(self.result_dir, 'result_analize_by_group.pkl')) or rerun:
            
            self.analize_by_group(groups=groups)
        else:
            data = kl.data.load_pickle(os.path.join(self.result_dir, 'result_analize_by_group.pkl'))
            self.subs = data['subs']
            self.group_info = data['group info']
        
        # print('draw state by group')
        # draw_count=50
        # self.draw_state_by_group(group=groups[0][:draw_count], path=os.path.join(self.result_dir, 'state_trace', 'autism')) 
        # self.draw_state_by_group(group=groups[1][:draw_count], path=os.path.join(self.result_dir, 'state_trace', 'control'))   
        
        self.compare_by_group()

def cluster(subs, key_name='', state_name='', num_clusters=7, right_only=True):
    from sklearn.cluster import KMeans
    if right_only:
        data = np.concatenate([sub[key_name] for sub in subs if sub['answer'] is True], axis=0)
    else:
        data = np.concatenate([sub[key_name] for sub in subs], axis=0)
    print('ready to k-means')
    model = KMeans(n_clusters=num_clusters, random_state=17).fit(data)
    print('finish k-means')
    labels = model.labels_
    sc, _, _ = kl.stuff.cluster_inner_metrics(data, labels)
    print('sc:', sc)
    for sub in subs:
        data = sub[key_name]
        state = model.predict(data)
        sub[state_name] = state
    return sc, model.cluster_centers_
        
def settle_data_bolt(atlas='cc200', num_cluster=7, 
                     cache_dir='/root/kl2/code/tmp/BolTkl/cache/bolt/tmp/ori/cc200_norm',
                     save_path='/root/kl2/code/tmp/BolTkl/Analysis/kl/bolt/ori/data_cluster7.pkl',
                     cluster_right_only=True, save=True):
    # cache_vs_dir = '/root/kl2/code/tmp/BolTkl/cache/bolt/tmp/vs/cc200'
    # # cache_ori_dir = '/root/kl2/code/tmp/BolTkl/cache/stagin/ori/cc200_643'
    # cache_ori_dir = '/root/kl2/code/tmp/BolTkl/cache/bolt/tmp/ori/cc200_norm'
    subs = []
    from Models.Transformer.stagin.bold import process_dynamic_fc
    # from .stagin.bold import process_dynamic_fc
    from kl.augment import Standardization
    stand = Standardization(dim=0)
    for i, file in enumerate(glob.glob(os.path.join(cache_dir, '*.npz'))):
        # if i < 100:
        #     continue
        data = np.load(file)
        
        data_file = os.path.basename(file)
        
        # rep_tp = data['cls_layers'][-1] # L,N
        
        # print(np.sum(np.square(rep_tp), axis=-1))
        
        # # d = rep_tp[1:] - rep_tp[:-1]
        # # print(np.mean(np.abs(d / rep_tp[:-1]), axis=-1))
        # exit()
        
        # data_ts = np.load(f'/data3/surrogate/abide/checked/cc200/tall_no0/{data_file}')
        # ts = stand(data_ts['ts'])
        # ts = ts[None,:,:]
        # # print(ts.shape)
        # ts_torch = torch.Tensor(ts)
        # dfc, sampling_points = process_dynamic_fc(ts_torch, window_size=8, window_stride=4)
        # dfc.squeeze_()
        # dfc = dfc.numpy()
        # np.
        subs.append(
            {
                'sid': data['sid'].item(), # str
                # 'cls_layers': data['cls_layers'],
                'answer': data['answer'].item(), # bool
                'label': data['ground_true'].item(), # int
                'cls': data['cls'].squeeze()
                # 'dfc': dfc,
                # **{
                #     f'cls_l_{i}': l for i, l in enumerate(data['cls_layers'])
                # }
            }
        )

    
    # for i in range(4):
    #     sc_rep, cluster_center_rep = cluster(subs, num_clusters=num_cluster, key_name=f'cls_l_{i}', state_name=f'state_{i}', filter_sign=False)
    
    # i=3
    # sc_rep, cluster_center_rep = cluster(subs, num_clusters=num_cluster, key_name=f'cls_l_{i}', state_name=f'state_{i}',)
    sc_rep, cluster_center_rep = cluster(subs, num_clusters=num_cluster, key_name=f'cls', state_name=f'state',)

    if save:
        kl.data.save_pickle(subs, save_path)
    return subs

def settle_data_bolt_folder(atlas='cc200', num_cluster=7, 
                     cache_dir='/root/kl2/code/tmp/BolTkl/cache/bolt/tmp/ori/cc200_norm',
                     save_path='/root/kl2/code/tmp/BolTkl/Analysis/kl/bolt/ori/data_cluster7.pkl',
                     cluster_right_only=True, save=True,
                     folder=0):
    subs = []
    files = glob.glob(os.path.join(cache_dir, 'train', str(folder), '*.npz')) \
        + glob.glob(os.path.join(cache_dir, 'test', str(folder), '*.npz'))
    # print(len(glob.glob(os.path.join(cache_dir, 'test', str(folder), '*.npz'))))
    # exit()
    for i, file in enumerate(files):
        data = np.load(file)
        cls = data['cls'].squeeze()
        subs.append(
            {
                'sid': data['sid'].item(), # str
                'answer': data['answer'].item(), # bool
                'label': data['ground_true'].item(), # int
                'cls': data['cls'].squeeze()
            }
        )

    sc_rep, cluster_center_rep = cluster(subs, num_clusters=num_cluster, key_name=f'cls', state_name=f'state',right_only=cluster_right_only)

    if save:
        kl.data.save_pickle(subs, save_path)
    return subs

def anaylize(subs=None, dir='/root/kl2/code/tmp/BolTkl/Analysis/kl/data_cluster8.pkl', num_cluster=8,state_name='state_0',
             result_dir='/root/kl2/code/tmp/BolTkl/Analysis/kl/rep',
             save=False, ans_right_only=False):
    if subs is None:
        subs = kl.data.load_pickle(dir)
    '''
    [
        {
            'sid' # str
            'answer':  # bool
            'label':  # int
            'dfc': L,C,C
            'state_0': np (L)
            'state_1': np (L)
            'state_2': np (L)
            'state_3': np (L)
        }
    ]
    '''
    
    if ans_right_only:
        subs = [ sub for sub in subs if sub['answer'] is True]

    ana = Analysis(subs=subs, 
                   num_cluster=num_cluster,
                   state_name=state_name,
                   result_dir=result_dir,
                   save=save
                )
    ana.run(rerun=True)

def main():
    num_cluster = 8
    cluster_right_only = True
    state_save = True
    ana_save = True
    subs = None
    answer_right_only=False
    folder = 2
    
    name = 'release1'
    model_cache_dir = f'/root/kl2/code/tmp/BolTkl/cache/bolt/tmp/{name}/vs/cc200'
    subs_result_path = f'/root/kl2/code/tmp/BolTkl/Analysis/kl/bolt/vs/{name}/{folder}/data_cluster{num_cluster}.pkl'
    state_name = 'state'
    result_dir = f'/root/kl2/code/tmp/BolTkl/Analysis/kl/bolt/vs/{name}/{folder}/{state_name}_filter_{answer_right_only}/'
    
    
    subs = settle_data_bolt_folder(num_cluster=num_cluster,
                     cache_dir=model_cache_dir,
                     save_path=subs_result_path,
                     cluster_right_only=cluster_right_only,
                     save=state_save,
                     folder=folder)
    
    anaylize(subs=subs,
             dir=subs_result_path,
             num_cluster=num_cluster,
             state_name=state_name,
             result_dir=result_dir,
             save=ana_save,
             ans_right_only=answer_right_only)
   
if __name__ == '__main__':
    main()
    


    
    