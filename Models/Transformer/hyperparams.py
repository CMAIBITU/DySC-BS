


from utils import Option
from . import params_choice
# import params_choice

longformer = {
    'model': 'longformer',
    "weightDecay" : 0,
    "lr" : 2e-4,
    'need_scheduler': False,
            # "minLr" : 1e-4,
            # "maxLr" : 2e-3,
            
    'attention_window': [8,16,32,64],
    # 'attention_window': [20,40,80,160],
    # 'attention_window': [12,24,48,96],
    'hidden_size': 240, # fc 输出是这个
    'num_hidden_layers': 4,
    'num_attention_heads': 12,
    'intermediate_size': 1024,
    'dim': 200,
    
    'rep_fc':{
        'count_of_layers': 0,
        'input_dim': 32,
        'mid_dim': 128, # 不能改
        'output_dim': 32
    },  
    'classifier_fc_in_dim': 240,
    
    'cs': True, 
    'cs_loss_weight': 0.5, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    
    'rep_aug': False,#
    'aug_fc': False,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 240,
        'mid_dim': 128,
        'output_dim': 200
    }
}

stagin = {
    'model':'stagin',
    "weightDecay" : 0,
    "lr" : 2e-4,
    "minLr" : 2e-5,
    "maxLr" : 4e-4,
    'need_scheduler': True,
    
    'dim': 200,
    'feature_dim': 200, 
    'hidden_dim': 240,
    'num_heads': 12,
    'num_layers':2,
    'sparsity': 50,
    'dropout': 0.5,
    'cls_token': 'param',
    'readout': 'garo', # sero garo
    # 'reg_lambda': 0.0001,
    'reg_lambda': 1e-5,
    # 'diff_weight': 5, 
    'diff_weight': 0, 
    'dctr_weight': 1e-4,
    'win_size': 8,
    'stride': 4,
    
    'rep_fc':{
        'count_of_layers': 0,
        'input_dim': 240,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },  
    'classifier_fc_in_dim': 240,
    
    'cs': False, 
    'cs_loss_weight': 0.5, 
    'use_right_mask': True,
    'cs_space_dim':200,
    'n_splits':10,      
    'n_splits2':10,
    'use_float16': False, 
    
    'rep_aug': False,# 
    'aug_fc': True,
    'sfc_proj':{
        'count_of_layers': 2,
        'input_dim': 128,
        'mid_dim': 128, # 不能改
        'output_dim': 200
    },
    'ts_proj':{
        'count_of_layers': 2,
        'input_dim': 240,
        'mid_dim': 128,
        'output_dim': 200
    }
}

def getHyper_transformer(atlas=None):
    hyperDict = stagin
    return Option(hyperDict)
