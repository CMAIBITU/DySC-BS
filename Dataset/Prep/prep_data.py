import numpy as np
import glob
import os
import math
import csv
import preprocess_data as Reader
import data_path
from kl import diag, data
from kl.data import resample_tr2

ABIDE_name_map2={
    'YALE': 'YALE', 
    'CMU': 'CMU', 
    'KKI': 'KKI', 
    'OLIN': 'OLIN', 
    'OHSU': 'OHSU', 
    'UM_1': 'UM', 
    'SBL': 'SBL', 
    'STANFORD': 'STANFORD', 
    'UCLA_1': 'UCLA', 
    'UCLA_2': 'UCLA', 
    'MAX_MUN': 'MAX_MUN', 
    'NYU': 'NYU', 
    'UM_2': 'UM', 
    'CALTECH': 'CALTECH', 
    'SDSU': 'SDSU', 
    'USM': 'USM', 
    'PITT': 'PITT', 
    'LEUVEN_2': 'LEUVEN', 
    'TRINITY': 'TRINITY', 
    'LEUVEN_1': 'LEUVEN',
    }
ABIDE_params = {
    'Caltech':  {'tr': 2000, 'te': 30, 'count': 37, 'site': 0},
    'CMU':      {'tr': 2000, 'te': 30, 'count': 27, 'site': 1},
    'KKI':      {'tr': 2500, 'te': 30, 'count': 48, 'site': 2},
    'Leuven':   {'tr': 1667, 'te': 33, 'count': 63, 'site': 3},
    'MaxMun':   {'tr': 3000, 'te': 30, 'count': 52, 'site': 4},
    'Max_Mun':   {'tr': 3000, 'te': 30, 'count': 52, 'site': 4},
    'MAX_MUN':   {'tr': 3000, 'te': 30, 'count': 52, 'site': 4},
    'NYU':      {'tr': 2000, 'te': 15, 'count': 175,'site': 5},
    'OHSU':     {'tr': 2500, 'te': 30, 'count': 26, 'site': 6},
    'Olin':     {'tr': 1500, 'te': 27, 'count': 34, 'site': 7},
    'Pitt':     {'tr': 1500, 'te': 25, 'count': 56, 'site': 8},
    'SBL':      {'tr': 2200, 'te': 30, 'count': 30, 'site': 9},
    'SDSU':     {'tr': 2000, 'te': 30, 'count': 36, 'site': 10},
    'Stanford': {'tr': 2000, 'te': 30, 'count': 39, 'site': 11},
    'Trinity':  {'tr': 2000, 'te': 28, 'count': 47, 'site': 12},
    'UCLA':     {'tr': 3000, 'te': 28, 'count': 98, 'site': 13},
    'UM':       {'tr': 2000, 'te': 30, 'count': 140, 'site': 14},
    'USM':      {'tr': 2000, 'te': 28, 'count': 71, 'site': 15},
    'Yale':     {'tr': 2000, 'te': 25, 'count': 56, 'site': 16},
    'YALE':     {'tr': 2000, 'te': 25, 'count': 56, 'site': 16}
}

def get_site_tr_te(site_name):
    ori_site_name = site_name
    if site_name not in ABIDE_params:
        site_name = site_name[0] + site_name[1:].lower()

    if site_name not in ABIDE_params:
        print(ori_site_name, site_name)

    assert site_name in ABIDE_params
    return ABIDE_params[site_name]

class DataInfo(object):
    
    aal={'name': 'aal', 'atlas': 'aal'}
    aal90={'name': 'aal90', 'atlas': 'aal'}
    tt={'name': 'tt', 'atlas': 'tt'}
    cc200={'name': 'cc200', 'atlas': 'cc200'}
    cc400={'name': 'cc400', 'atlas': 'cc400'}
    sch400={'name': 'sch400', 'atlas': 'sch400'}
    
    def __init__(self, 
                #  atlax=[aal, tt, cc200],
                 atlax=[cc200],
                #  save_data_root='/data3/surrogate/abide/checked/', 
                 resample=False, 
                 len_threshold=1,
                 checked=True):
        """_summary_

        Args:
            atlax (list, optional): which atlas need to preprocess
            save_data_root (str, optional): where to save the data
            resample (bool, optional): For the ABIDE dataset, different sites have different TR values. If resample=True, we will perform interpolation to make TR = 2. 
            len_threshold (int, optional): the ROI-series length less then len_threshold will be ignored.
            checked (bool, optional):  it's same with paramater quality_checked in nilearn.fetch_abide_pcp. Defaults to True.
        """
        self.save_data_root = save_data_root
        self.resample=resample
        self.len_threshold = len_threshold
        self.atlax = atlax
        self.checked = checked
        
    def get_save_dataset_abide_params(self):
        for atlas in self.atlax:
            yield  {'path_subject': data_path.get_processed_folder(self.resample, atlas['name'], self.len_threshold),
                    'atlas': atlas['atlas'],
                    'resample': self.resample,
                    'len_threshold': self.len_threshold,
                    'checked': self.checked}


def get_stand(x):
    x_std = np.std(x, axis=0, keepdims=True)
    x_mean = np.mean(x, axis=0, keepdims=True)
    x = (x - x_mean) / x_std
    x = np.nan_to_num(x, 0)
    return x

def get_stand_all(x):
    x_std = np.std(x)
    x_mean = np.mean(x)
    x = (x - x_mean) / x_std
    x = np.nan_to_num(x, 0)
    return x

def get_stand_timestep(x):
    x_std = np.std(x, axis=-1, keepdims=True)
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x = (x - x_mean) / x_std
    x = np.nan_to_num(x, 0)
    return x

def get_norm(x):
    leida = x
    leida_max = np.max(leida)
    leida_min = np.min(leida)
    leida_norm = leida - leida_min
    leida_norm /= (leida_max - leida_min)
    return leida_norm  

def save_dataset_abide(path_subject, atlas='cc200', resample=False, len_threshold=1, checked=True):
    Reader.set_data_folder_defalut(global_signal_regression=True, checked=checked)
    subject_IDs = Reader.get_ids(check=checked, atlas=atlas) #str
    all_time_series = Reader.get_timeseries(subject_IDs, atlas)
    print('count: ', len(subject_IDs))

    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    # ages   = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')# 没空值
    # sexs   = Reader.get_subject_score(subject_IDs, score='SEX') # 没空值
    # eyes   = Reader.get_subject_score(subject_IDs, score='EYE_STATUS_AT_SCAN') # 无空值
    # # hands  = Reader.get_subject_score(subject_IDs, score='HANDEDNESS_CATEGORY') # 很多空值
    # # FIQs = Reader.get_subject_score(subject_IDs, score='FIQ')# 有空值
    sites  = Reader.get_subject_score(subject_IDs, score='SITE_ID') # 没空值
    if not os.path.exists(path_subject):
        # os.mkdir(path_subject)
        os.makedirs(path_subject)
    
    for i, sid in enumerate(subject_IDs):
        time_series = all_time_series[i]# timepoints * ROIs
        if resample:
            site = sites[sid]
            site = ABIDE_name_map2[site]
            tr = get_site_tr_te(site)['tr']
            # time_series_ori = time_series
            time_series = resample_tr2(time_series, tr, order=2, new_tr=2000)
        if len(time_series) < len_threshold:
            continue

        site = sites[sid]
        # site = abide_params.ABIDE_site_id[abide_params.ABIDE_name_map2[site]]
        # # age = round(float(ages[sid]))
        # age = float(ages[sid])
        # gender = int(sexs[sid]) - 1
        # eye = int(eyes[sid]) - 1
        # hand = hands[sid]
        # continue
        if 'aal90' in path_subject:
            time_series = time_series[:, :90]
        
        # print(time_series.shape)
        # leida, phase, dFC = cal_leida(time_series[None, :, :])
        # dFC = np.squeeze(dFC, 0)
        # # print(dFC.shape)
        # dFC = data.get_upper_data(dFC)
        # # print(dFC.shape)
        # # exit()
        # leida = np.squeeze(leida, axis=0)
        # leida = change_leida_sign_with_more_negative(leida)
        # phase = np.squeeze(phase, axis=0)
        label = int(labels[sid]) - 1
        # label = int(label)        
        time_series = time_series.astype(np.float32)
        time_series_stand = get_stand(time_series)
        time_series_stand_all = get_stand_all(time_series)
        time_series_stand_tp = get_stand_timestep(time_series)
        time_series_norm = get_norm(time_series)
        
        # time_series[np.where(time_series == 0 )] += 1e-5
        # leida = leida.astype(np.float32)
        # leida_max = np.max(leida)
        # leida_min = np.min(leida)
        # leida_norm = leida - leida_min
        # leida_norm /= (leida_max - leida_min)
        # leida_std = np.std(leida)
        # leida_mean = np.mean(leida)
        # leida_stand = (leida - leida_mean) / leida_std + 1e-4
        # rest = ts['rest']
        # ts_corr = Reader.compute_pearson_ts(time_series)
        
        # t_sim = data.cos_sim_2d(leida)
        # np.fill_diagonal(t_sim, -1)
        # indics = np.argmax(t_sim, axis=-1)
        # sample_ts = time_series_stand.astype(np.float16)
        # sfc_pcc = Reader.compute_pearson_connectivity(time_series_stand)

        
        np.savez(os.path.join(path_subject, f'{sid}.npz'), **{
            'sid': sid, 
            'ts': time_series,
            'ts_stand': time_series_stand,
            'time_series_stand_all': time_series_stand_all,
            'time_series_stand_tp': time_series_stand_tp,
            # 'ts_norm': time_series_norm,
            # 'leida': leida,
            # 'leida_norm': leida_norm,
            # 'leida_stand': leida_stand, 
            # 'phase': phase, 
            # 'sfc_pcc': sfc_pcc,
            'label': label,
            # 't_most_sim': indics,
            # 'site': site,
            # 'age': age,
            # 'gender': gender,
            # 'eye': eye
            # 'ts_corr': ts_corr,
            # 'dFC': dFC
        })
        # dFC = dFC.astype(np.float16)
        # corr_path = os.path.join(path_subject,'corr')
        # if os.path.exists(corr_path) is False:
        #     os.makedirs(corr_path)
            
        # np.savez(os.path.join(corr_path, f'corr_{sid}.npz'), **{
        #     'sid': sid, 
        #     # 'ts': time_series,
        #     # 'ts_stand': time_series_stand,
        #     # 'ts_norm': time_series_norm
        #     # 'leida': leida,
        #     # 'leida_norm': leida_norm,
        #     # 'leida_stand': leida_stand, 
        #     # 'phase': phase, 
        #     # 'label': label,
        #     # 'ts_corr': ts_corr,
        #     'dFC': dFC
        # })
        
        print(i, 'done')
        print(f'{atlas} in {path_subject} done')
        

# def cal_ts_sim(path_subject, atlas='cc200', resample=False, len_threshold=1):
#     files = glob.glob(os.path.join(path_subject,'*.npz'))
#     for f in files:
#         leida = np.load(f)['leida']
#         t_sim = data.cos_sim_2d(leida)
#         np.fill_diagona(t_sim, -1)
#         indics = np.argmax(t_sim, axis=-1)
        
    
def check_signal_roi_health(ts):
    # ts L C
    std = np.std(ts, axis=0)
    if np.any(std ==0):
        return False
    return True    

def healthCheckOnRoiSignal(roiSignal):
    """
        roiSignal : (L, C)
    """
    # remove subjects with dead rois
    if(np.sum(np.sum(np.abs(roiSignal), axis=0) == 0) > 0):
        return False

    return True    
    
if __name__ == '__main__':
    datainfo = DataInfo(resample=True, 
                        # save_data_root=data_path.processed_save_data_root,
                        checked=True,)
    for params in datainfo.get_save_dataset_abide_params():
        # print(params)
        save_dataset_abide(**params)