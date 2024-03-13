from concurrent.futures import process
import os
from random import random
from time import sleep
import nilearn
from nilearn.input_data import NiftiLabelsMasker
import preprocess_data as Reader
import data_path



root_folder = data_path.root_folder
# data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal/')
def fetch_and_process_abide():
    
    download = True
    process = True
    merge = False  
    checked = False
    atlas = 'cc200'
        
    Reader.set_data_folder_defalut(global_signal_regression=True, checked=checked)
    '''
        alff, degree_binarize,
        degree_weighted, dual_regression, eigenvector_binarize,
        eigenvector_weighted, falff, func_mask, func_mean, func_preproc, lfcd,
        reho, rois_aal, rois_cc200, rois_cc400, rois_dosenbach160, rois_ez,
        rois_ho, rois_tt, and vmhc.
    '''
    
    files = ['rois_' + atlas]
    
    
    # ids_aal = Reader.get_ids(atlas='aal', check=True)
    # ids_cc200 = Reader.get_ids(atlas='cc200', check=True)
    # ids_aal = sorted(ids_aal)
    # ids_cc200 = sorted(ids_cc200)
    # print(ids_cc200 == ids_aal)
    # exit()
    if atlas == 'sch400':
        filemapping = {'func_preproc': 'func_preproc.nii.gz',
                   files[0]: files[0] + '.npy'}
    else:
        filemapping = {'func_preproc': 'func_preproc.nii.gz',
                    files[0]: files[0] + '.1D'}

    # Download database files
    if download == True:
        print('fetch_abide start...')
        from nilearn import datasets
        done = False
        while not done:
            try:
                datasets.fetch_abide_pcp(data_dir=root_folder,
                                    derivatives=['rois_' + atlas],
                                    pipeline='cpac',
                                    band_pass_filtering=True,
                                    global_signal_regression=True,
                                    quality_checked=checked)
                done = True
            except Exception as ex:			
                print(ex)
                sleep(5)   
        print('download done---------------------')
        Reader.save_idx(atlas=atlas)
    # exit(1)
    if process:
        
        data_folder = Reader.data_folder_default_path
        import shutil
        subject_IDs = Reader.get_ids(check=checked, atlas=atlas) #changed path to data path
        subject_IDs = subject_IDs.tolist()

        # Create a folder for each subject
        for s, fname in zip(subject_IDs, Reader.fetch_filenames(subject_IDs, files[0], atlas)):
            subject_folder = os.path.join(data_folder, s)
            if not os.path.exists(subject_folder):
                os.mkdir(subject_folder)

            # Get the base filename for each subject
            base = fname.split(files[0])[0]

            # Move each subject file to the subject folder
            for fl in files:
                if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
                    shutil.move(base + filemapping[fl], subject_folder)

        # time_series = Reader.get_timeseries(subject_IDs, atlas)

        # Compute and save connectivity matrices (regions x regions)
        # 问题是：本项目代码对correlation 和 partial correlation 的处理一样
        # Reader.subject_connectivity(time_series, subject_IDs, atlas, 'pearson')
        # Reader.subject_connectivity(time_series, subject_IDs, atlas, 'correlation')
        # Reader.subject_connectivity(time_series, subject_IDs, atlas, 'partial correlation')
    
    if merge:
        # subject的合并为一个矩阵
        subject_IDs = Reader.get_ids(check=True, atlas=atlas)
        networks = Reader.get_networks(subject_IDs, atlas_name=atlas, kind='correlation')
        import scipy.io as sio
        # print(Reader.data_folder_default_path)
        # exit()
        net_file = os.path.join(Reader.data_folder_default_path, f'data_checked_{atlas}.mat')
        
        label = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
        age   = Reader.get_subject_score(subject_IDs, score='AGE_AT_SCAN')
        sex   = Reader.get_subject_score(subject_IDs, score='SEX')
        eye   = Reader.get_subject_score(subject_IDs, score='EYE_STATUS_AT_SCAN')
        hand  = Reader.get_subject_score(subject_IDs, score='HANDEDNESS_CATEGORY')
        site  = Reader.get_subject_score(subject_IDs, score='SITE_ID')
        label = [int(label[k]) - 1 for k in label]
        age = [float(age[k]) for k in age]
        sex = [sex[k] for k in sex]
        eye = [eye[k] for k in eye]
        hand = [hand[k] for k in hand]
        site = [site[k] for k in site] 
        import numpy as np
        l = len(label)
        label = np.array(label)
        
        age_np = np.ones((l,l))
        sex_np = np.ones((l,l))
        eye_np = np.ones((l,l))
        hand_np = np.ones((l,l))
        sex_np = np.ones((l,l))
        site_np = np.ones((l,l))
        for i in range(len(label) - 1):
            for j in range(i + 1, len(label)):
                
                sim = Reader.cal_sim(age[i], age[j], is_num=True)
                age_np[i, j] = sim
                age_np[j, i] = sim
                
                sim = Reader.cal_sim(sex[i], sex[j], is_num=False)
                sex_np[i, j] = sim
                sex_np[j, i] = sim
                
                sim = Reader.cal_sim(eye[i], eye[j], is_num=False)
                eye_np[i, j] = sim
                eye_np[j, i] = sim
                
                sim = Reader.cal_sim(hand[i], hand[j], is_num=False)
                hand_np[i, j] = sim
                hand_np[j, i] = sim
                
                sim = Reader.cal_sim(site[i], site[j], is_num=False)
                site_np[i, j] = sim
                site_np[j, i] = sim
        sio.savemat(net_file, {'correlation': networks, 
                               'label': label, 
                               'sim_age': age_np, 
                               'sim_sex': sex_np,
                               'sim_eye': eye_np,
                               'sim_hand': hand_np,
                               'sim_site': site_np})
    

def prep_atlas(atlas='schaefer7_400'):
    datadir='/data2/abide/func_preproc'
    if(atlas == "schaefer7_400"):
        if(not os.path.exists(datadir + "/Atlasses/{}".format(atlas))):
            atlasInfo = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1, data_dir=datadir + "/Atlasses")            
            atlasImage = nilearn.image.load_img(atlasInfo["maps"])

    return atlasImage

def save_sch400_timeseries():
    atlasImage = prep_atlas()
    import glob
    import numpy as np
    niis = glob.glob('/data2/abide/func_preproc/data/*.nii')
    for nni in niis:
        basename = os.path.basename(nni)
        target_file = basename[:-16] + 'rois_sch400.npy'
        # print(nni[:-16])
        scanImage = nilearn.image.load_img(nni)
        roiTimeseries =  NiftiLabelsMasker(atlasImage).fit_transform(scanImage)
        file = os.path.join('/root/kl2/data/ABIDE_pcp/cpac/filt_global', target_file)
        np.save(file, roiTimeseries)
        print(file, 'done')
        # print(roiTimeseries.shape)
        
def move_sch400_checked():
    Reader.set_data_folder_defalut(global_signal_regression=True, checked=True)
    subject_IDs = Reader.get_ids(check=True, atlas='aal')
    # print(subject_IDs)
    # root = '/root/kl2/data/ABIDE_pcp/cpac/filt_global'
    target_root = '/root/kl2/data/ABIDE_pcp/checked/cpac/filt_global'
    import glob
    count = 0
    for file in glob.glob('/root/kl2/data/ABIDE_pcp/cpac/filt_global/*sch400.npy'):
        start = file.index('_00') + 3
        sid = file[start: start + 5]
        if sid in subject_IDs:
            basename = os.path.basename(file)
            count += 1
            os.system(f'cp {file} {target_root}/{basename}')
            print(basename)
            
    print(count)
    
def move_to_checked(atlas='cc400'):
    Reader.set_data_folder_defalut(global_signal_regression=True, checked=True)
    subject_IDs = Reader.get_ids(check=True, atlas='aal')
    print(subject_IDs)
    # root = '/root/kl2/data/ABIDE_pcp/cpac/filt_global'
    target_root = '/root/kl2/data/ABIDE_pcp/checked/cpac/filt_global/'
    import glob
    count = 0
    
    def move(src_file, sid):
        # src_file = glob.glob(os.path.join(src_root, f'*{atlas}.1D'))[0]
        basename = os.path.basename(src_file)
        print(os.path.join(target_root, sid, basename))
        os.symlink(src_file, os.path.join(target_root, sid, basename))
    
    for sid in subject_IDs:
        src_root = f'/root/kl2/data/ABIDE_pcp/cpac/filt_global/{sid}'
        src_file = glob.glob(os.path.join(src_root, f'*{atlas}.1D'))[0]
        move(src_file, sid)
        
        src_file = glob.glob(os.path.join(src_root, f'*{atlas}_correlation.mat'))[0]
        move(src_file, sid)
        
        src_file = glob.glob(os.path.join(src_root, f'*{atlas}_partial_correlation.mat'))[0]
        move(src_file, sid)
        
        count += 1
    
    # for file in glob.glob('/root/kl2/data/ABIDE_pcp/cpac/filt_global/*sch400.npy'):
    #     start = file.index('_00') + 3
    #     sid = file[start: start + 5]
    #     if sid in subject_IDs:
    #         basename = os.path.basename(file)
    #         count += 1
    #         os.system(f'cp {file} {target_root}/{basename}')
    #         print(basename)
            
    print(count)

if __name__ == '__main__':
    # print('download done---------------------')
    # preprocess_abide()
    fetch_and_process_abide()
    # save_sch400_timeseries()
    # move_sch400_checked()
    # move_to_checked('cc400')
        