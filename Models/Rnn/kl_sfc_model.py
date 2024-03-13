from kl.model.hinets.pretrain import Pretrain as Tehigcn_Pretrain
from torch import nn
import info_nce
import torch
from torch.nn import functional as F
import random
from einops import rearrange

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

def get_proj_header(count_of_layers, input_dim, output_dim, mid_dim=None):
    if count_of_layers == 0:
        return nn.Identity()
    elif count_of_layers == 1:
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # Norm()
        )
    elif count_of_layers == 2:
        if mid_dim is None:
            mid_dim = input_dim
        return nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(mid_dim, output_dim),
            # Norm()
        )
    else:
        raise NotImplementedError()     


class TE_HI_GCN(nn.Module):
    models = None
    
    def test_free_models():
        TE_HI_GCN.models = None # del 无效！
        
    
    def __init__(self, n_splits=5, n_splits2=8, use_right_mask=False, cs_space_dim=200, 
                 name='cc200',atlas_dim=200, use_float16=False, 
                 sfc_proj=None, ts_proj=None, checked=True):
        super().__init__()
        if TE_HI_GCN.models is None:
            check_str = 'checked' if checked else 'unchecked'
            TE_HI_GCN.models = Tehigcn_Pretrain(
               data_root=f'/data3/surrogate/abide/{check_str}/{name}/tall_no0/',
               n_splits=n_splits,
               n_splits2=n_splits2,
               no_val=True,
               shuffle_before_split=True,
               shuffle_seed=0,
               name=name,
               use_float16=use_float16,
               check=checked)
        self.encoder = TE_HI_GCN.models.new_torch_model()
        self.cs_loss = info_nce.InfoNCE()
        self.mse_loss = nn.MSELoss()
        
        self.use_right_mask = use_right_mask
        
        # self.sfc_proj = nn.Sequential(
        #     # nn.Linear(128, 128),
        #     # nn.ReLU(),
        #     nn.Linear(128, cs_space_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(cs_space_dim, cs_space_dim)
            
        # )
        # self.ts_proj = nn.Sequential(
        #     nn.Linear(atlas_dim, cs_space_dim),
        #     # nn.ReLU(),
        #     # nn.Linear(cs_space_dim, cs_space_dim),
        #     # nn.Identity()
        # )
        
        self.sfc_proj = get_proj_header(**sfc_proj)
        self.ts_proj = get_proj_header(**ts_proj)
        
 
        self.to('cuda')
        
        
        
    def free(self):
        if TE_HI_GCN.models is not None:
           TE_HI_GCN.models.free_all()
        torch.cuda.empty_cache()
        
    def free_models(self):
        if TE_HI_GCN.models is not None:
               TE_HI_GCN.models.free_exclude_rep_cache()
        torch.cuda.empty_cache()
        
        
        
    def forward(self, cls_layers, folder_k, sids, big_neg_size=False):
        
        # cls_proj = cls_layers[-1]
        # cls_proj = torch.mean(torch.stack(cls_layers), dim=0)
        # cls_proj = rearrange(cls_proj, 'n c l -> n l c')
        # ts_reps = torch.mean(cls_proj, dim=-1)
        ts_reps = cls_layers
        with torch.no_grad():# 极其重要，不然显存爆炸
            sfc_reps, graph_idx, right_mask = \
            self.encoder(folder_k=folder_k, sids=sids, dataset=1, big_neg_size=big_neg_size)
        # self.free_models()
        if sfc_reps is None:
            print(' ================ return None =============')
            return 0
        # sfc_reps = sfc_reps.detach()
        if self.use_right_mask:
            sfc_reps = sfc_reps[right_mask]
            ts_reps = ts_reps[right_mask]
        
        # return self.mse_loss(sfc_reps, ts_reps)
        
        sfc_reps = self.sfc_proj(sfc_reps)
        ts_reps = self.ts_proj(ts_reps)
       
        
        sfc_reps = F.normalize(sfc_reps, dim=1)
        ts_reps = F.normalize(ts_reps, dim=1)
        return self.cs_loss(ts_reps, sfc_reps), sfc_reps, ts_reps, right_mask
        
        # sfc_reps, graph_idx, right_mask = self.encoder(folder_k=folder_k, sids=sids, dataset=1, big_neg_size=big_neg_size)
        # sfc_reps = sfc_reps.detach()
        
        # cs_loss = 0
        # for cls_proj, cls_proj_fun, weight in zip(cls_layers, self.ts_projs, [0.1, 0.2, 0.3, 0.4]):
        #     cls_proj = rearrange(cls_proj, 'n c l -> n l c')
        #     ts_cls_rep = torch.mean(cls_proj, dim=-1)
        #     sfc_proj = self.sfc_proj(sfc_reps)
        #     ts_proj = cls_proj_fun(ts_cls_rep)
        #     sfc_proj = F.normalize(sfc_proj, dim=1)
        #     ts_proj = F.normalize(ts_proj, dim=1)
        #     cs_loss = cs_loss + self.cs_loss(ts_proj, sfc_proj) * weight
            
        # return cs_loss
        
    def aug(self, ts_reps, sfc_reps):
        # w = torch.normal(mean=0.5, std=0.5, size=(ts_reps.shape[0], 1))
        w = torch.normal(mean=0.5, std=0.15, size=(ts_reps.shape[0], 1))
        w = w.to('cuda')
        return ts_reps * (1 - w) + sfc_reps * w
        # return ts_reps
        # w = torch.rand(size=(ts_reps.shape[0], 1)).to('cuda')
        # return ts_reps * (1 - w) + sfc_reps * w
    
    

        