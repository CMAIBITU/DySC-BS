## data root
All data paths are written in the Dataset/Prep/data_path.py

## download and preprocess abide data
```
cd Dataset/Prep
```
### download
```
python fetch_data.py  \\Modify the fetch_data.py code to download data from other atlases except cc200
```
### preprocess
```
python prep_data.py  \\ Pass in different DataInfo objects to obtain different data preprocessing results.
```
And

```
python filter_0.py \\ Optional. get subjects pass check_signal_roi_health 
```

## run DySC
```
./train.sh
```

##  run DySC-BS
```
./train_bs.sh
```

## analyze
1. In the Models/bs/params_choice.py file, use the  key ·cache_dir_base· to set the folder path for saving dynamic representation.
2. Edit analysis_state2.py File, modify `model_cache_dir`, `subs_result_path`， `result_dir`. Set the model_cache_dir to be the same as `cache_dir_base`.
3. run analysis_state2.py by
```
python analysis_state2.py
``` 

### visualization
to be continue