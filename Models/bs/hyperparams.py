


from utils import Option
from . import params_choice
# import params_choice

def getHyper_bolT(atlas=None):
    if atlas == 'aal':
        hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,
            # "lr" : 1e-4,
            # "minLr" : 1e-5,
            # "maxLr" : 2e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            "dim" : 116,

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 8, # stride = windowSize * shiftCoeff
            "shiftCoeff" : 1.0/2.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "expand",
            
            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.5,
            "attnDrop" : 0.5,
            "lambdaCons" :0.5,
            
            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                
            #kl add
            # 'cs': True, 
            # 'cs_loss_weight': 0.2, 
            # 'use_right_mask': True,
            # 'cs_space_dim':200,
            # 'n_splits':10,      
            # 'n_splits2':10,
            # 'use_float16': False,
            'rep_fc':{
                'count_of_layers': 1,
                'input_dim': 128,
                'mid_dim': 128, # 不能改
                'output_dim': 200
            },   
            **params_choice.aal_ws8       

        }
    elif atlas == 'cc200':
        hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,
            # "lr" : 1e-4,
            # "minLr" : 1e-5,
            # "maxLr" : 2e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            "dim" : 200,

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 8, # stride = windowSize * shiftCoeff
            "shiftCoeff" : 3.0/8.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "expand",
            
            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.5,
            "attnDrop" : 0.5,
            "lambdaCons" :0.5,
            
            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                
            #kl add
            'cs': False, 
            'cs_loss_weight': 0.2, 
            'use_right_mask': True,
            'cs_space_dim':200,
            'n_splits':10,      
            'n_splits2':10,
            'use_float16': False,
            'rep_aug': False,
            'rep_fc':{
                'count_of_layers': 1,
                'input_dim': 128,
                'mid_dim': 128, # 不能改
                'output_dim': 200
            }, 
            **params_choice.cc200_ws8_bs
        }
        # hyperDict['brain_state'] = bs
        # if bs:
        #     hyperDict.update(**params_choice.cc200_ws8_bs)
        # else:
        #     hyperDict.update(**params_choice.cc200_ws14)
        
    elif atlas == 'sch400':
        hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,
            # "lr" : 1e-4,
            # "minLr" : 1e-5,
            # "maxLr" : 2e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            "dim" : 400,

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 8, # stride = windowSize * shiftCoeff
            "shiftCoeff" : 2.0/4.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "expand",
            
            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.5,
            "attnDrop" : 0.5,
            "lambdaCons" :0.5,
            
            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                
            #kl add
            'cs': True, 
            'cs_loss_weight': 0.2, 
            'use_right_mask': False,
            'cs_space_dim':200,
            'n_splits':10,      
            'n_splits2':10,
            'use_float16': False,
            'rep_fc':{
                'count_of_layers': 1,
                'input_dim': 128,
                'mid_dim': 128, # 不能改
                'output_dim': 200
            }, 
            **params_choice.sch400_ws20           

        }
    else:
        hyperDict = {

            "weightDecay" : 0,

            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            "dim" : 400,

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 20, # stride = windowSize * shiftCoeff
            "shiftCoeff" : 2.0/5.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "expand",

            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.1,
            "attnDrop" : 0.1,
            "lambdaCons" : 1,

            # extra for ablation study
            "pooling" : "cls", # ["cls", "gmp"]         
                

        }

    return Option(hyperDict)
