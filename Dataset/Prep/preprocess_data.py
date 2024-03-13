# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implcd ied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import csv
import glob
import os
import re
import warnings

import numpy as np
import numpy.ma as ma
import scipy.io as sio
from nilearn import connectome
from scipy.spatial import distance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, OrdinalEncoder
import math
import pickle
import data_path

warnings.filterwarnings("ignore")

def get_abide_pcp_data_folder(pipeline='cpac', global_signal_regression=True, checked=True):
    path = 'ABIDE_pcp'
    if checked:
        path = os.path.join(path, 'checked')
    path = os.path.join(path, pipeline)
    filt = 'filt_global' if global_signal_regression else 'filt_noglobal'
    path = os.path.join(path, filt)
    return path

# Input data variables

root_folder = data_path.root_folder
# data_checked_folder_path = os.path.join(root_folder, 'ABIDE_pcp/checked/cpac/filt_noglobal')
phenotype = os.path.join(root_folder, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')

def get_data_folder(pipeline='cpac', global_signal_regression=False, checked=False):
    fld = get_abide_pcp_data_folder(pipeline=pipeline, 
                                     global_signal_regression=global_signal_regression,
                                     checked=checked
                                     )
    return os.path.join(root_folder, fld)
    # return data_checked_folder_path if checked else data_folder_path

data_folder_default_path = get_data_folder()
    
def set_data_folder_defalut(pipeline='cpac', global_signal_regression=False, checked=False):
    global data_folder_default_path
    data_folder_default_path = get_data_folder(pipeline=pipeline, 
                                     global_signal_regression=global_signal_regression,
                                     checked=checked)
        
def fetch_filenames(subject_IDs, file_type, atlas):
    """
        subject_list : list of short subject IDs in string format
        file_type    : must be one of the available file types
        filemapping  : resulting file name format
    returns:
        filenames    : list of filetypes (same length as subject_list)
    """

    if 'sch400' == atlas:
        filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_' + atlas: '_rois_' + atlas + '.npy'}
    else:
        filemapping = {'func_preproc': '_func_preproc.nii.gz',
                   'rois_' + atlas: '_rois_' + atlas + '.1D'}
    # The list to be filled
    filenames = []

    # Fill list with requested file paths
    # data_folder = get_data_folder(checked=checked)
    data_folder = data_folder_default_path
    for i in range(len(subject_IDs)):
        os.chdir(data_folder)
        try:
            try:
                os.chdir(data_folder)
                filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
            except Exception as ex:
                # print(ex)
                os.chdir(data_folder + '/' + subject_IDs[i])
                filenames.append(glob.glob('*' + subject_IDs[i] + filemapping[file_type])[0])
        except IndexError:
            filenames.append('N/A')
    return filenames


# Get timeseries arrays for list of subjects
def get_timeseries(subject_list, atlas_name, silence=False):
    """
        subject_list : list of short subject IDs in string format
        atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
    returns:
        time_series  : list of timeseries arrays, each of shape (timepoints x regions)
    """

    timeseries = []
    data_folder = data_folder_default_path
    for i in range(len(subject_list)):
        subject_folder = os.path.join(data_folder, subject_list[i])
        if 'rois_sch400' == atlas_name or 'sch400' == atlas_name:
            ext = '.npy'
        else:
            ext = '.1D'
        ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + ext)]
        fl = os.path.join(subject_folder, ro_file[0])
        if silence != True:
            print("Reading timeseries file %s" % fl)
        if 'rois_sch400' == atlas_name or 'sch400' == atlas_name:
            timeseries.append(np.load(fl))
        else:
            timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries

# def get_timeseries_raw(subject_list, atlas_name, silence=False):
#     """
#         subject_list : list of short subject IDs in string format
#         atlas_name   : the atlas based on which the timeseries are generated e.g. aal, cc200
#     returns:
#         time_series  : list of timeseries arrays, each of shape (timepoints x regions)
#     """

#     timeseries = []
#     data_folder = data_folder_default_path
#     for i in range(len(subject_list)):
#         # subject_folder = os.path.join(data_folder, subject_list[i])
#         file_name = 
#         ro_file = [f for f in os.listdir(subject_folder) if f.endswith('_rois_' + atlas_name + '.1D')]
#         fl = os.path.join(data_folder, ro_file[0])
#         if silence != True:
#             print("Reading timeseries file %s" % fl)
#         timeseries.append(np.loadtxt(fl, skiprows=0))

#     return timeseries

def compute_pearson_connectivity(functional):
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(functional.T))
        return corr
    
def compute_pearson_ts(ts):
    with np.errstate(invalid="ignore"):
        corr = np.nan_to_num(np.corrcoef(ts))
        return corr    
    
def compute_pearson_connectivity_ts1D(ts):
    corr = compute_pearson_ts(ts)
    iu = np.triu_indices(len(corr), 1)
    return corr[iu]

def compute_pearson_connectivity1D(functional):
    with np.errstate(invalid="ignore"):
        corr = compute_pearson_connectivity(functional)
        mask = np.invert(np.tri(corr.shape[0], k=-1, dtype=bool))
        m = ma.masked_where(mask == 1, mask)
        return ma.masked_where(m, corr).compressed()


#  compute connectivity matrices
def subject_connectivity(timeseries, subjects=None, atlas_name='aal', kind='correlation', iter_no='', seed=1234,
                         n_subjects='', save=True, save_path=None):
    """
        timeseries   : timeseries table for subject (timepoints x regions)
        subjects     : subject IDs
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        iter_no      : tangent connectivity iteration number for cross validation evaluation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder
    returns:
        connectivity : connectivity matrix (regions x regions)
    """
    if save and save_path is None:
        save_path = data_folder_default_path
    if kind in ['TPE', 'TE', 'correlation','partial correlation', 'pearson']:
        if kind not in ['TPE', 'TE']:
            if kind == 'pearson':
                # if isinstance(timeseries, list):
                    
                connectivity = compute_pearson_connectivity(timeseries)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind=kind)
                connectivity = conn_measure.fit_transform(timeseries)
        else:
            if kind == 'TPE':
                conn_measure = connectome.ConnectivityMeasure(kind='correlation')
                conn_mat = conn_measure.fit_transform(timeseries)
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(conn_mat)
                connectivity = connectivity_fit.transform(conn_mat)
            else:
                conn_measure = connectome.ConnectivityMeasure(kind='tangent')
                connectivity_fit = conn_measure.fit(timeseries)
                connectivity = connectivity_fit.transform(timeseries)

    if save:
        if kind not in ['TPE', 'TE']:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(save_path, subj_id,
                                            subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
                sio.savemat(subject_file, {'connectivity': connectivity[i]})
            return connectivity
        else:
            for i, subj_id in enumerate(subjects):
                subject_file = os.path.join(save_path, subj_id,
                                            subj_id + '_' + atlas_name + '_' + kind.replace(' ', '_') + '_' + str(
                                                iter_no) + '_' + str(seed) + '_' + validation_ext + str(
                                                n_subjects) + '.mat')
                sio.savemat(subject_file, {'connectivity': connectivity[i]})
            return connectivity_fit
    else:
        return connectivity


# Get the list of subject IDs
def get_ids(num_subjects=None, atlas=None, check=False):
    """
    return:
        subject_IDs    : list of all subject IDs
    """
    file_name = 'subject_IDs.txt'
    data_folder = data_folder_default_path
    # print(data_folder)
    # exit()
    if check and atlas is not None:
        # file_name = f'subject_check_{atlas}_IDs.txt'
        file_name = f'subject_checked_IDs.txt'
    print(os.path.join(data_folder, file_name))
    subject_IDs = np.genfromtxt(os.path.join(data_folder, file_name), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs

def save_idx(atlas):
    path = data_folder_default_path
    # print(path)
    # exit()
    file_idx = f'subject_{atlas}_IDs.txt'
    file_idx = os.path.join(path, file_idx)
    print(file_idx)
    if os.path.exists(file_idx):
        print(file_idx, ' has existed')
        return
    subjects = glob.glob(os.path.join(path, f'*{atlas}.1D'))
    # id_set = set()
    with open(file_idx, 'w') as file:
        import re
        for sub in subjects:
            id_ = re.findall(r"_00(\d+)_", sub)[0]
            # if id_ in id_set:
            #     continue
            # id_set.add(id_)
            file.write(id_ + '\n')
            
def save_site_to_subject(checked=True):
    subject_IDs = get_ids(check=True, atlas='cc200') #
    sites  = get_subject_score(subject_IDs, score='SITE_ID')
    site_subs = {abide_params.ABIDE_name_map2[k]:[] for k in abide_params.ABIDE_name_map2} 
    for i, sid in enumerate(subject_IDs):
        site = sites[sid]
        site = abide_params.ABIDE_name_map2[site]
        site_subs[site].append(sid)
    
    file_name = '/data3/surrogate/abide/checked/site_subs.pkl' if checked else '/data3/surrogate/abide/site_subs.pkl'        
    with open(file_name, 'wb') as file:
        pickle.dump(site_subs, file=file)

def get_site_to_subject(checked=True):
    file_name = '/data3/surrogate/abide/checked/site_subs.pkl' if checked else '/data3/surrogate/abide/site_subs.pkl'        
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    return data
         
# def get_site_id(sids):
#     from .abide_params import ABIDE_name_map, ABIDE_params
#     sites = get_subject_score(sids, 'SITE_ID')
#     for k in sites:
#         s = sites[k]
#         sites[k] = ABIDE_params[ABIDE_name_map[s.lower()]]['site']
#     return sites


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                if score == 'HANDEDNESS_CATEGORY':
                    # if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                    #     scores_dict[row['SUB_ID']] = 'R'
                    # elif row[score] == 'Mixed':
                    #     scores_dict[row['SUB_ID']] = 'Ambi'
                    # elif row[score] == 'L->R':
                    #     scores_dict[row['SUB_ID']] = 'Ambi'
                    # else:
                    #     scores_dict[row['SUB_ID']] = row[score]
                    
                    # kl modify
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = None
                    elif row[score] == 'Mixed':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    elif row[score] == 'L->R':
                        scores_dict[row['SUB_ID']] = 'Ambi'
                    else:
                        scores_dict[row['SUB_ID']] = row[score]
                elif (score == 'FIQ' or score == 'PIQ' or score == 'VIQ'):
                    if (row[score].strip() == '-9999') or (row[score].strip() == ''):
                        scores_dict[row['SUB_ID']] = 100
                    else:
                        scores_dict[row['SUB_ID']] = float(row[score])

                else:
                    scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


# preprocess phenotypes. Categorical -> ordinal representation
def preprocess_phenotypes(pheno_ft, params):
    if params['model'] == 'MIDA':
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2])], remainder='passthrough')
    else:
        ct = ColumnTransformer([("ordinal", OrdinalEncoder(), [0, 1, 2, 3])], remainder='passthrough')

    pheno_ft = ct.fit_transform(pheno_ft)
    pheno_ft = pheno_ft.astype('float32')

    return (pheno_ft)


# create phenotype feature vector to concatenate with fmri feature vectors
def phenotype_ft_vector(pheno_ft, num_subjects, params):
    gender = pheno_ft[:, 0]
    if params['model'] == 'MIDA':
        eye = pheno_ft[:, 0]
        hand = pheno_ft[:, 2]
        age = pheno_ft[:, 3]
        fiq = pheno_ft[:, 4]
    else:
        eye = pheno_ft[:, 2]
        hand = pheno_ft[:, 3]
        age = pheno_ft[:, 4]
        fiq = pheno_ft[:, 5]

    phenotype_ft = np.zeros((num_subjects, 4))
    phenotype_ft_eye = np.zeros((num_subjects, 2))
    phenotype_ft_hand = np.zeros((num_subjects, 3))

    for i in range(num_subjects):
        phenotype_ft[i, int(gender[i])] = 1
        phenotype_ft[i, -2] = age[i]
        phenotype_ft[i, -1] = fiq[i]
        phenotype_ft_eye[i, int(eye[i])] = 1
        phenotype_ft_hand[i, int(hand[i])] = 1

    if params['model'] == 'MIDA':
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand], axis=1)
    else:
        phenotype_ft = np.concatenate([phenotype_ft, phenotype_ft_hand, phenotype_ft_eye], axis=1)

    return phenotype_ft


# Load precomputed fMRI connectivity networks
def get_networks(subject_list, kind, iter_no='', seed=1234, n_subjects='', atlas_name="aal",
                 variable='connectivity'):
    """
        subject_list : list of subject IDs
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        atlas_name   : name of the parcellation atlas used
        variable     : variable name in the .mat file that has been used to save the precomputed networks
    return:
        matrix      : feature matrix of connectivity networks (num_subjects x network_size)
    """

    all_networks = []
    data_folder = data_folder_default_path
    for subject in subject_list:
        if len(kind.split()) == 2:
            kind = '_'.join(kind.split())
        fl = os.path.join(data_folder, subject,
                              subject + "_" + atlas_name + "_" + kind.replace(' ', '_') + ".mat")


        matrix = sio.loadmat(fl)[variable]
        all_networks.append(matrix)

    if kind in ['TE', 'TPE']:
        norm_networks = [mat for mat in all_networks]
    else:
        norm_networks = [np.arctanh(mat) for mat in all_networks]

    networks = np.stack(norm_networks)

    return networks

def k_folder_collection(k=10):
    ids = get_ids()
    labels_dict = get_subject_score(ids, score='DX_GROUP')
    group1=[]
    group2=[]
    for sub in ids:
        if labels_dict[sub] == '1':
            group1.append(sub)
        else:
            group2.append(sub)
    
    print(len(group1), len(group2))
    import random
    random.shuffle(group1)
    random.shuffle(group2)
    a = int(len(group1) / k)
    b = int(len(group1) % k)
    g1 = [(i*a, (i+1)*a )for i in range(k)]
    for i in range(b):
        ori = g1[i]
        ori = (ori[0], ori[1] + 1)
        g1[i] = ori
        nxt = g1[i+1]
        nxt = (nxt[0] + 1, nxt[1] + 1)
        g1[i+1] = nxt
    
    a = int(len(group2) / k)
    b = int(len(group2) % k)    
    g2 = [(i*a, (i+1)*a )for i in range(k)]
    for i in range(b):
        ori = g2[i]
        ori = (ori[0], ori[1] + 1)
        g2[i] = ori
        nxt = g2[i+1]
        nxt = (nxt[0] + 1, nxt[1] + 1)
        g2[i+1] = nxt
    # print(g1) 
    # print(g2)
    data = []
    for sg1, sg2 in zip(g1, g2):
        val = group1[sg1[0]: sg1[1]] + group2[sg2[0]: sg2[1]]
        random.shuffle(val)
        train = group1[0: sg1[0]] + group1[sg1[1]: ] + group2[0: sg2[0]] + group2[sg2[1]: ]
        random.shuffle(train)
        data.append({'val': val, 'train': train})
    
    import pickle 
    with open('/data3/surrogate/abide/all_10_folder_sub_idx.pkl', 'wb') as file:
        pickle.dump(data, file)

def cal_sim(a, b, is_num=False):
    if a is None or b is None:
        return 0.2
    if is_num:
        # 高斯核
        return float(math.exp((a - b) ** 2 / -2))
    else: 
        return float(a == b)
    
def check_tr_effect():
    import kl.diag as diag

    # data = get_site_to_subject()
    # PITT 1500, UCLA 3000
    # print('PITT', data['PITT'])
    # print('UCLA', data['UCLA'])
    path = '/data3/surrogate/abide/checked/aal/tall_w1/'
    data1 = np.load(os.path.join(path, '50005.npz')) # PITT
    print(data1['phase'].shape)
    data3 = np.load(os.path.join(path, '51223.npz')) # UCLA
    print(data3['phase'].shape)
    
   
    
    # 因为UCLA tr 3s, PITT 1.5s, 都用100个点，其实UCLA相当于取了300s的数据，而PITT只有150s数据 
    diag.plot(data1['phase'][:100,13], file='tmp/data1_phase.jpg')
    diag.plot(data3['phase'][:50,13], file='tmp/data3_phase.jpg')
    
    path = '/data3/surrogate/abide/checked/aal90/tall_tr2'
    data3 = np.load(os.path.join(path, '50005.npz')) # PITT
    print(data3['phase'].shape)
    data4 = np.load(os.path.join(path, '51223.npz')) # UCLA
    print(data4['phase'].shape)
    diag.plot(data3['phase'][:100,13], file='tmp/data3_phase.jpg')
    diag.plot(data4['phase'][:100,13], file='tmp/data4_phase.jpg')
    
    diag.plot_together(data3['phase'][:,13],pos=(3,1,1), draw_point=False)
    diag.plot_together(data3['phase'][:,33],pos=(3,1,2), draw_point=False)
    diag.plot_together(data4['phase'][:,33],pos=(3,1,3), color='green', draw_point=False)
    diag.show('tmp/kltmp.jpg')
    
    
if __name__ == '__main__':
    # k_folder_collection()
    # with open('/data3/surrogate/abide/all_10_folder_sub_idx.pkl', 'rb') as f: 
    #     data = pickle.load(f)
    # print(data)
    # set_data_folder_defalut(global_signal_regression=True)
    # save_site_to_subject(True)
    # data = get_site_to_subject()
    # print(data)
    check_tr_effect()
    