import numpy as np
import os
import glob
import data_path
atlas = 'cc200'

# path = f'/data3/surrogate/abide/checked/{atlas}/tall_tr2_no0'
# '/data3/surrogate/abide/checked/' 
source_path = data_path.get_processed_folder(resample=True, atlas=atlas)
dis_path = data_path.get_filte_0_folder(resample=True, atlas=atlas)

def check_signal_roi_health(ts):
    # ts L C
    std = np.std(ts, axis=0)
    if np.any(std <= 1e-6):
        return False
    return True  

count = 0
# files = glob.glob(f'/data3/surrogate/abide/checked/{atlas}/tall/*.npz')
files = glob.glob(f'{source_path}/*.npz')
if len(files) == 0:
    print(f'no npz file in {data_path.processed_save_data_root}/{atlas}')
for file in files:
    if os.path.exists(dis_path) is False:
        os.makedirs(dis_path)
    data = np.load(file)
    if check_signal_roi_health(data['ts']) is False:
        print(data['sid'])
    else:
        # pass
        count += 1
        name = os.path.basename(file)
        os.system(f'cp {file} {dis_path}/{name}')
print(count)