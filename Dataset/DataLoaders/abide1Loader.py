import torch
import numpy as np
import os
import sys
from .utils import read_all_data
import glob
import Dataset.Prep.data_path as data_path


datadir = "/root/kl2/code/tmp/BolTkl/Dataset/Data"


def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (N, T)
    """


    # remove subjects with dead rois
    # return True
    
    if(np.sum(np.sum(np.abs(roiSignal), axis=1) == 0) > 0):
        return False

    return True    

def abide1Loader(atlas, targetTask, sort=True, check=False, resample=True):
    print('abide1Loader', atlas, ', sort:', sort, ', check:', check, ', resample:', resample)
    """
        x : (#subjects, N)
        /data3/surrogate/abide/checked/cc200/tall_tr2
    """
    # check_str = 'checked' if check else 'unchecked'
    # data_root = f'/data3/surrogate/abide/{check_str}/{atlas}/tall_no0'
    # if resample:
    #     data_root = f'/data3/surrogate/abide/{check_str}/{atlas}/tall_tr2_no0'
    data_root = data_path.get_filte_0_folder(resample, atlas)
    subs = read_all_data(path=data_root)
    
    x=[]
    y=[]
    sids=[]
    for sub in subs:
        if 'ts' in sub:
            x.append(sub['ts_stand'].T)
        else:
            x.append(sub['ts_stand'].T)
        y.append(int(sub['label'].item()))
        sids.append(int(sub['sid'].item()))
        
    return x, y, sids, [],[],[]
    
    dataset = torch.load(datadir + "/dataset_abide_{}.save".format(atlas))
    if sort:
        dataset = sorted(dataset, key=lambda x: x['pheno']['subjectId'])
    x = []
    y = []
    subjectIds = []

    x0 = []
    y0 = []
    subjectIds0 = []
    
    for data in dataset:
        
        if(targetTask == "disease"):
            label = int(data["pheno"]["disease"]) - 1 # 0 for autism 1 for control
        # data["roiTimeseries"].shape: (L, C)
        if(healthCheckOnRoiSignal(data["roiTimeseries"].T)):
        # if check_signal_roi_health(data["roiTimeseries"]):
            x.append(data["roiTimeseries"].T)
            y.append(label)
            subjectIds.append(int(data["pheno"]["subjectId"]))
        else:
            # print('jjjjjjjjj',data['pheno']['subjectId'])
            x0.append(data["roiTimeseries"].T)
            y0.append(label)
            subjectIds0.append(int(data["pheno"]["subjectId"]))
            # exit()
    print(len(x))
    return x, y, subjectIds, x0, y0, subjectIds0


if __name__ == '__main__':
    x, y, sids, x0, y0, sids0 = abide1Loader('aal', 'disease', sort=False)
    print(sids)
    # x, y, sids, x0, y0, sids0 = abide1Loader('aal-tr2', 'disease')
    # print('============================')
    # tx, ty, tsids, tx0, ty0, tsids0 = abide1Loader('aal-tr2-0', 'disease')
    # # target = '51578'
    # target = 51578
    # for ts0, sid0 in zip(tx0, tsids0):
    #     if sid0 == target:
    #         break
    # for ts, sid in zip(x, sids):
    #     if sid == target:
    #         break
    # ts = ts.T
    # ts0 = ts0.T
    # for c in range(116):
    #     if np.all(ts0[:, c] == 0):
    #         print(c) # 101, 106
    
    # print(ts[:, 101])
    
    # print('==============')
    # print(ts[:, 106])
    
    # ts[:, 101] = 0
    # ts[:, 106] = 0
    
    # print(np.all(ts == ts0))
    # print(ts[:, 0] - ts0[:, 0])