root_folder = '/root/kl2/data/'
processed_save_data_root='/data3/surrogate/abide/checked/' # the folder to save path *.npz data after preprocess

import os
def get_processed_folder(resample, atlas, len_threshold=1):
    dir_name = 't' + 'all' if len_threshold==1 else str(len_threshold) + '_w1'
    if resample:
        dir_name += '_tr2'
    return os.path.join(processed_save_data_root, atlas, dir_name)

def get_filte_0_folder(resample, atlas, len_threshold=1):
    """
        filter 0: check method check_signal_roi_health in filter_0.py
        this method return the path where all subjects pass check_signal_roi_health
    Args:
        resample (_type_): _description_
        atlas (_type_): _description_
        len_threshold (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    dir_name = 't' + 'all' if len_threshold==1 else str(len_threshold) + '_w1'
    if resample:
        dir_name += '_tr2'
    dir_name += '_no0'
    return os.path.join(processed_save_data_root, atlas, dir_name)
    