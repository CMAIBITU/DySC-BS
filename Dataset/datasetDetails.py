
datasetDetailsDict = {

    "abide1_bs" : {
        "datasetName" : "abide1",
        "targetTask" : "disease",
        "nOfClasses" : 2,        
        # "dynamicLength" : 60,
        "dynamicLength" : 90,
        "foldCount" : 10,
        # "atlas" : "schaefer7_400",
        # "atlas": "aal-leida-tr2",
        # 'atlas': 'aal-tr2-0',
        # 'atlas':'cc200',
        'atlas': 'cc200',
        # 'atlas': 'sch400',
        "nOfEpochs" : 40,
        "batchSize" : 64,
        'check': True,
        'save': False,
        'ckpt_dir':'./ckpt/tmp/bolt/',
        'resample': True,  
    },
    
    "abide1" : {
        "datasetName" : "abide1",
        "targetTask" : "disease",
        "nOfClasses" : 2,        
        # "dynamicLength" : 60,
        "dynamicLength" : 60,
        "foldCount" : 10,
        # "atlas" : "schaefer7_400",
        # "atlas": "aal-leida-tr2",
        # 'atlas': 'aal-tr2-0',
        # 'atlas':'cc200',
        'atlas': 'cc200',
        # 'atlas': 'sch400',
        "nOfEpochs" : 20,
        "batchSize" : 64,
        'check': True,
        'save': False,
        'resample': False,
        # 'ckpt_dir':'./ckpt/tmp/bolt/'
          
    },

 }



