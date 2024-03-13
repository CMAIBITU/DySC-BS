import glob
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import random

PATH_ABIDE_TR2 = '/data3/surrogate/abide/checked/aal/tall_tr2'
PATH_ADHD_TR2 = '/data3/surrogate/adhd200/aal/combine'

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

# def k_folder_

# def k_folder_train_val_test(data, labels, n_splits=10, random_seed=0):
#     """_summary_

#     Args:
#         data (_type_): 数组
#         labels (_type_): 数组
#         n_splits (int, optional): _description_. Defaults to 10.
#         random_seed (int, optional): _description_. Defaults to 0.
#     """
#     kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
#     for train_val_idx, test_idx in kf.split(data, labels):
#         test = data[]

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
        
def sids2files(sids, dataset_name='abide', atlas='aal'):
    root = '/data3/surrogate'
    if dataset_name == 'abide':
        data_dir = os.path.join(root, dataset_name, 'checked', atlas, 'tall_tr2')
        return [os.path.join(data_dir, str(s) + '.npz') for s in sids]
    else:
        raise NotImplementedError()

# def get_k_folder_files_train_val_test_by_k(idx, labels, k, n_splits=10, random_seed=0): 
#     train, val, test = get_k_folder_idx_train_val_test_by_k()
        
def k_folder_abide_train_val_test(path='/data3/surrogate/abide/checked/aal/tall_tr2', return_path=False,
                                  n_splits=10, n_splits2=10, random_seed=0, shuffle_before_split=False):
    """
    不要修改！！！

    Args:
        path (str, optional): _description_. Defaults to '/data3/surrogate/abide/checked/aal/tall_tr2'.

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """
    # files = glob.glob(os.path.join(path, '*.npz'))
    # files.sort()
    # subs = [np.load(f) for f in files]
    # labels = [sub['label'] for sub in subs]
    # labels = np.stack(labels)
    
    shuffle_seed = random_seed if shuffle_before_split else None
    subs, files = read_all_data(path=path, shuffle_seed=shuffle_seed, return_files=True)
    labels = [sub['label'] for sub in subs]
    labels = np.stack(labels)
    idx = np.arange(len(labels))
    
    shuffle = False if shuffle_before_split else True
    
    # rr = lambda x: sum(labels[x]) / len(x)
    # cc = lambda x: sum(labels[x])
    def get_subs(idx):
        return [ subs[i] for i in idx]
    def get_files(idx):
        return [ files[i] for i in idx]
    for train_idx, val_idx, test_idx in k_folder_idx_train_val_test(idx=idx, labels= labels, 
                                                                    n_splits=n_splits, n_splits2=n_splits2, 
                                                                    random_seed=random_seed, shuffle=shuffle):
        # print(rr(train_idx), rr(val_idx), rr(test_idx))
        # print(cc(val_idx) - cc(test_idx))
        if return_path:
            yield get_files(train_idx), get_files(val_idx), get_files(test_idx)
        else:
            yield get_subs(train_idx), get_subs(val_idx), get_subs(test_idx)
            
def get_k_folder_abide_train_val_test_by_k(k, path='/data3/surrogate/abide/checked/aal/tall_tr2', return_path=False,
                                           n_splits=10, n_splits2=10, random_seed=0, shuffle_before_split=False):
    for kk, (train, val, test) in enumerate(k_folder_abide_train_val_test(path=path, return_path=return_path, 
                                                                          n_splits=n_splits, n_splits2=n_splits2, random_seed=random_seed, shuffle_before_split=shuffle_before_split)):
        if kk == k:
            return train, val, test


        
if __name__ == '__main__':
#    for train, val, test in  k_folder_abide_train_val_test():
#        print(len(val))
    for k in range(1):
        subs = read_all_data()
        labels = [sub['label'] for sub in subs]
        labels = np.stack(labels)
        idx = np.arange(len(labels))
        train_idx, val_idx, test_idx = get_k_folder_idx_train_val_test_by_k(idx=idx, labels=labels, k=k)
        train = [ subs[i] for i in train_idx]
        train_sids = [s['sid'].item() for s in train ]
        train, val, test = get_k_folder_abide_train_val_test_by_k(k=k)
        train_sids2 = [s['sid'].item() for s in train]
        
        train_sids.sort()
        train_sids2.sort()
        print(train_sids2 == train_sids)
        print(train_sids)