consistency_space_dim=200

cc200_ws8 = {
            
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.5,
    "attnDrop" : 0.5,
    "pooling" : "cls", # ["cls", "gmp"]   
    
    "windowSize" : 8, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 1.0/2.0,
    "lambdaCons" :0.5,    
    
    'cs': True, 
    'cs_loss_weight': 0.2, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    'rep_aug': True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': consistency_space_dim,
        'output_dim': consistency_space_dim
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 200,
        'mid_dim': consistency_space_dim,
        'output_dim': consistency_space_dim
    }
}  

cc200_ws14 = {
            
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.5,
    "attnDrop" : 0.5,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 14, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 3.0/7.0,
    "lambdaCons" :0.5,  
      
    'cs': True, 
    'cs_loss_weight': 0.2, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    'rep_aug': True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': consistency_space_dim,
        'output_dim': consistency_space_dim
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 200,
        'mid_dim': consistency_space_dim,
        'output_dim': consistency_space_dim
    }
}

cc200_ws20={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.5,
    "attnDrop" : 0.5,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 20, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 8/20.0,
    "lambdaCons" : 1,
    'rep_fc':{
        'count_of_layers': 0,
        'input_dim': 200,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },  
    'classifier_fc_in_dim': 200,
      
    'cs': False, 
    'cs_loss_weight': 0.5, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    
    'rep_aug': False,# 可能无用,先不用了！
    'aug_fc': True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 200,
        'mid_dim': 200,
        'output_dim': 200
    }
    
}


aal_ws8={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.1,
    "attnDrop" : 0.1,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 8, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 1/2.0,
    "lambdaCons" : 1.0,
    'rep_fc':{
        'count_of_layers': 0,
        'input_dim': 116,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },  
    'classifier_fc_in_dim': 116,
      
    'cs': False, 
    'cs_loss_weight': 0.5, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    
    'rep_aug': True,# 可能无用,先不用了！
    'aug_fc':True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 200,
        'mid_dim': 116,
        'output_dim': 200
    }
    
}

aal_ws14={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.1,
    "attnDrop" : 0.1,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 14, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 6/14.0,
    "lambdaCons" : 1.0,
    'rep_fc':{
        'count_of_layers': 0,
        'input_dim': 116,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },  
    'classifier_fc_in_dim': 116,
      
    'cs': False, 
    'cs_loss_weight': 0.5, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    
    'rep_aug': True,# 可能无用,先不用了！
    'aug_fc':True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 200,
        'mid_dim': 116,
        'output_dim': 200
    }
    
}

aal_ws20={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.1,
    "attnDrop" : 0.1,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 20, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 8/20.0,
    "lambdaCons" : 1.0,
    'rep_fc':{
        'count_of_layers': 2,
        'input_dim': 116,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },  
    'classifier_fc_in_dim': 200,
      
    'cs': True, 
    'cs_loss_weight': 0.5, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    
    'rep_aug': True,# 可能无用,先不用了！
    'aug_fc': True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 200,
        'mid_dim': 116,
        'output_dim': 200
    }
    
}

sch400_ws8={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.5,
    "attnDrop" : 0.5,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 8, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 1.0/2.0,
    "lambdaCons" : 0.5,
    'rep_fc':{
        'count_of_layers': 1,
        'input_dim': 400,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },  
    'classifier_fc_in_dim': 400,
      
    'cs': True, 
    'cs_loss_weight': 0.2, 
    'use_right_mask': True,
    'cs_space_dim':400,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': True, 
    
    'rep_aug': False,# 可能无用,先不用了！
    'aug_fc':False,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 400,
        'mid_dim': consistency_space_dim,
        'output_dim': 400
    }
    
}

sch400_ws12={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.5,
    "attnDrop" : 0.5,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 12, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 5.0/12.0,
    "lambdaCons" : 0.5,
    'rep_fc':{
        'count_of_layers': 1,
        'input_dim': 400,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },  
    'classifier_fc_in_dim': 400,
      
    'cs': True, 
    'cs_loss_weight': 0.2, 
    'use_right_mask': True,
    'cs_space_dim':400,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': True, 
    
    'rep_aug': False,# 可能无用,先不用了！
    'aug_fc':False,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 400,
        'mid_dim': consistency_space_dim,
        'output_dim': 400
    }
    
}

sch400_ws14={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.5,
    "attnDrop" : 0.5,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 14, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 6/14.0,
    "lambdaCons" : 0.5,
    'rep_fc':{
        'count_of_layers': 0,
        'input_dim': 400,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },  
    'classifier_fc_in_dim': 400,
      
    'cs': True, 
    'cs_loss_weight': 0.2, 
    'use_right_mask': True,
    'cs_space_dim':400,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': True, 
    
    'rep_aug': False,# 可能无用,先不用了！
    'aug_fc':False,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 400,
        'mid_dim': consistency_space_dim,
        'output_dim': 400
    }
    
}

sch400_ws16={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.5,
    "attnDrop" : 0.5,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 16, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 7.0/16.0,
    "lambdaCons" : 0.5,
    'rep_fc':{
        'count_of_layers': 1,
        'input_dim': 400,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },  
    'classifier_fc_in_dim': 400,
      
    'cs': True, 
    'cs_loss_weight': 0.2, 
    'use_right_mask': True,
    'cs_space_dim':400,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': True, 
    
    'rep_aug': False,# 可能无用,先不用了！
    'aug_fc':False,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 400,
        'mid_dim': consistency_space_dim,
        'output_dim': 400
    }
    
}

sch400_ws20={
    "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
    "focalRule" : "expand",
    
    "mlpRatio" : 1.0,
    "attentionBias" : True,
    "drop" : 0.1,
    "attnDrop" : 0.1,
    "pooling" : "cls", # ["cls", "gmp"]   

 #===============================================================   
    "windowSize" : 20, # stride = windowSize * shiftCoeff
    "shiftCoeff" : 2.0/5.0,
    "lambdaCons" : 1,
    'rep_fc':{
        'count_of_layers': 2,
        'input_dim': 400,
        'mid_dim': 200, # 不能改
        'output_dim': 400
    },  
    'classifier_fc_in_dim': 400,
      
    'cs': True, 
    'cs_loss_weight': 0.5, 
    'use_right_mask': True,
    'cs_space_dim':400,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': True, 
    
    'rep_aug': True,# 可能无用,先不用了！
    'aug_fc':True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 400
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 400,
        'mid_dim': consistency_space_dim,
        'output_dim': 400
    }
    
}