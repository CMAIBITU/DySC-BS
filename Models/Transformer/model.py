import torch
import numpy as np
from einops import rearrange, repeat
from .kl_sfc_model import TE_HI_GCN
from torch.nn import functional as F
from torch import nn
from .stagin.bold import process_dynamic_fc
from ..base_model import ClassifyModel
import os
from kl import kltorch

def rep_fc_factory(count_of_layers, input_dim, output_dim, mid_dim=None):
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
                nn.Linear(mid_dim, output_dim),
            )
        else:
            raise NotImplementedError()  
            
class TsEncoder_Longformer(ClassifyModel):
    def __init__(self, hyperParams):
        super().__init__(hyperParams)
        from transformers import LongformerModel, LongformerConfig
        l_cfg = LongformerConfig(
            attention_window=hyperParams.attention_window,
            hidden_size=hyperParams.hidden_size,
            num_hidden_layers=hyperParams.num_hidden_layers,
            num_attention_heads=hyperParams.num_attention_heads,
            intermediate_size=hyperParams.intermediate_size,
        )
        self.encoder = LongformerModel(l_cfg)
        self.proj2embeding = nn.Linear(hyperParams.dim, hyperParams.hidden_size)
    
    def forward(self, ts):
        ts = rearrange(ts, 'n c l -> n l c')
        embeding = self.proj2embeding(ts)
        cls_token = torch.ones_like(embeding[:,0:1,:])
        # cls_token = torch.mean(embeding, dim=1, keepdim=True)
        embeding = torch.cat((cls_token, embeding), dim=1)
        attention_mask = torch.ones_like(embeding[:,:,0])
        # attention_mask[:, 0] = 0 # 这会让模型无法学习！
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[:, 0] = 1
        position_ids = torch.arange(0, embeding.shape[1])
        position_ids = repeat(position_ids, 'L -> N L', N=embeding.shape[0])
        position_ids = position_ids.to(embeding.device)

        out = self.encoder(inputs_embeds=embeding,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True
                )
        cls_token_rep = out['last_hidden_state'][:,0]
        # pooler_output = out['pooler_output'] #  (torch.FloatTensor of shape (batch_size, hidden_size)) — Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        return cls_token_rep, 0

class TsEncoder_STAGIN(nn.Module):
    def __init__(self, hyperParams):
        # super().__init__(hyperParams)
        super().__init__()
        from .stagin.model import ModelSTAGIN
        self.model = ModelSTAGIN(
            input_dim=hyperParams.dim,
            hidden_dim=hyperParams.hidden_dim,
            num_classes=2,
            num_heads=hyperParams.num_heads,
            num_layers=hyperParams.num_layers,
            sparsity=hyperParams.sparsity,
            dropout=hyperParams.dropout,
            win_size=hyperParams.win_size,
            cls_token=hyperParams.cls_token,
            readout=hyperParams.readout
        )
        self.win_size = hyperParams.win_size
        self.stride = hyperParams.stride
        self.feature_dim = hyperParams.feature_dim
        self.reg_lambda = hyperParams.reg_lambda

        self.rep_fc = rep_fc_factory(**hyperParams.rep_fc).to('cuda')
        self.fc = torch.nn.Linear(hyperParams.classifier_fc_in_dim, 2).to('cuda')

        
    
    def forward(self, ts):
        # if self.pos_encoder:
        #     ts = self.pos_encoder(ts, tr)
        ts = rearrange(ts, 'n c l -> n l c')
        a, sampling_points = process_dynamic_fc(ts, self.win_size, self.stride)
        sampling_endpoints = [p+self.win_size for p in sampling_points]
        v = repeat(torch.eye(self.feature_dim), 'n1 n2 -> b t n1 n2', t=len(sampling_endpoints), b=ts.shape[0])
        v = v.to(a.device)
        ts = rearrange(ts, 'n l c -> l n c')
        logit, attention, latent, reg_ortho, ts_rep = self.model.forward(v, a, ts, sampling_endpoints)
        reg_ortho_loss = reg_ortho * self.reg_lambda

        ts_cls_rep = torch.mean(latent, dim=1)

        return ts_cls_rep, reg_ortho_loss, logit, attention, ts_rep
    
    # def classify()
        
def get_model(hyperParams):
    if hyperParams.model == 'stagin':
        return TsEncoder_STAGIN(hyperParams)
    else:
        return TsEncoder_Longformer(hyperParams)


class Model():

    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details
        # print(self.hyperParams.dict) 
        # print(self.details.dict)
        '''
        {'weightDecay': 0, 'lr': 0.0002, 'minLr': 2e-05, 'maxLr': 0.0004, 'nOfLayers': 4, 'dim': 200, 'numHeads': 36, 'headDim': 20, 'windowSize': 14, 'shiftCoeff': 0.42857142857142855, 'fringeCoeff': 2, 'focalRule': 'expand', 'mlpRatio': 1.0, 'attentionBias': True, 'drop': 0.5, 'attnDrop': 0.5, 'lambdaCons': 0.5, 'pooling': 'cls', 'cs': True, 'cs_loss_weight': 0.2, 'use_right_mask': False, 'cs_space_dim': 200, 'n_splits': 10, 'n_splits2': 10}
        {'device': 'cuda:0', 'nOfTrains': 771, 'nOfClasses': 2, 'batchSize': 64, 'nOfEpochs': 20}
        '''

        # self.model = BolT(hyperParams, details)
        self.model = get_model(hyperParams)
        self.cs_loss_encoder = None
        need_cs_loss = hyperParams.cs
        
        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)#, weight = classWeights)
        # self.rep_fc = self.rep_fc_factory(**hyperParams.rep_fc).to('cuda')
        # self.fc = torch.nn.Linear(hyperParams.classifier_fc_in_dim, 2).to('cuda')
        # self.rep_fc = self.model.rep_fc
        # self.fc = self.model.fc
        
        # self.fc_aug = self.fc
        # params = list(self.model.parameters()) + list(self.fc.parameters()) + list(self.rep_fc.parameters())

        params = list(self.model.parameters())
        if hasattr(hyperParams, 'aug_fc') and hyperParams.aug_fc:
            self.fc_aug = torch.nn.Linear(hyperParams.cs_space_dim, 2).to('cuda')
            params = params + list(self.fc_aug.parameters())
        else:
            self.fc_aug = self.model.fc
        if need_cs_loss:
            self.cs_loss_encoder = TE_HI_GCN(n_splits=hyperParams.n_splits, 
                                             n_splits2=hyperParams.n_splits2,
                                             use_right_mask=hyperParams.use_right_mask, 
                                             cs_space_dim=hyperParams.cs_space_dim,
                                             atlas_dim=hyperParams.dim,
                                             name=details.atlas,
                                             use_float16=hyperParams.use_float16,
                                             sfc_proj=hyperParams.sfc_proj,
                                             ts_proj=hyperParams.ts_proj,
                                             checked=details.check)
            params = list(params) + list(self.cs_loss_encoder.parameters())
            # params = list(params) + list(self.cs_loss_encoder.sfc_proj.parameters()) + list(self.cs_loss_encoder.ts_proj.parameters())
       
        
        # set optimizer
        self.optimizer = torch.optim.Adam(params, lr = hyperParams.lr, weight_decay = hyperParams.weightDecay)

        # set scheduler
        self.scheduler = None
        if hyperParams.need_scheduler:
            steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))        
            divFactor = hyperParams.maxLr / hyperParams.lr
            finalDivFactor = hyperParams.lr / hyperParams.minLr
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch), div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)

        self.cs_loss_weight = hyperParams.cs_loss_weight
        self.cls_loss_weight = 1
    
    def rep_fc_factory(self, count_of_layers, input_dim, output_dim, mid_dim=None):
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
                nn.Linear(mid_dim, output_dim),
            )
        else:
            raise NotImplementedError()     
                    
    def free(self):
        if self.cs_loss_encoder:
            self.cs_loss_encoder.free()
        self.model = None
        self.criterion = None
        # self.fc = None
        # self.rep_fc = None
        self.optimizer = None
        self.scheduler = None
        torch.cuda.empty_cache()
               
    def step(self, x, y, sids=None, folder_k=0, train=True, epoch=None, save=False):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """

        # PREPARE INPUTS
        
        inputs, y = self.prepareInput(x, y)

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
            # self.rep_fc.train()
        else:
            self.model.eval()
            # self.rep_fc.eval()

        # yHat, cls = self.model(*inputs)
        
        # if self.hyperParams.model == 'stagin':
        #     cls_rep, reg_loss, logit, attention = self.model(*inputs)
        #     # yHat = logit
        #     cls_rep = F.normalize(cls_rep, dim=1)
        #     cls_rep = self.model.rep_fc(cls_rep)
        #     cls_rep = F.normalize(cls_rep, dim=1)
        #     yHat = self.model.fc(cls_rep)
                
        # else:
        #     cls_rep, reg_loss = self.model(*inputs)

        #     cls_rep = F.normalize(cls_rep, dim=1)
        #     cls_rep = self.model.rep_fc(cls_rep)
        #     cls_rep = F.normalize(cls_rep, dim=1)
        #     yHat = self.model.fc(cls_rep)
        
        cls_rep, reg_loss, _ , attention, tp_rep = self.model(*inputs)
        # attention['time-attention'] (N,2,L,L)
        tat = attention['time-attention']
        attention['time-attention'] = tat.detach().cpu()
        time_diff_loss = 0
        for layer in range(tat.shape[1]):
            at = tat[:,layer]
            time_diff_loss = time_diff_loss + torch.mean(torch.square(torch.mean(at,dim=(1,2), keepdim=True) - at))
        cls_rep = F.normalize(cls_rep, dim=1)
        cls_rep = self.model.rep_fc(cls_rep)
        cls_rep = F.normalize(cls_rep, dim=1)
        yHat = self.model.fc(cls_rep)
        dctr_loss = 0
        tp_rep_m = torch.mean(tp_rep[:-1], dim=-1)
        dctr_loss = kltorch.k_means_spectral_relaxation_loss(tp_rep_m.reshape(-1, tp_rep_m.shape[-1]), 8)
        # loss = self.getLoss(yHat, y) + reg_loss + dctr_loss * self.hyperParams.dctr_weight
        # print(time_diff_loss * self.hyperParams.diff_weight)
        loss = self.getLoss(yHat, y) + reg_loss - time_diff_loss * self.hyperParams.diff_weight \
            + dctr_loss * self.hyperParams.dctr_weight
        cs_loss = 0
        if self.cs_loss_encoder and train:
                   
            # cs_loss = self.cs_loss_encoder(cls_layers, folder_k, sids)
            aug_loss = 0
            cs_loss, sfc_reps, ts_reps, right_mask = self.cs_loss_encoder(cls_rep, folder_k, sids)
            if self.hyperParams.rep_aug:    
                aug_cls_reps = self.cs_loss_encoder.aug(ts_reps, sfc_reps)
                aug_cls_reps = F.normalize(aug_cls_reps, dim=1)
                # aug_yHat = self.fc(aug_cls_reps)
                aug_yHat = self.fc_aug(aug_cls_reps)
                if self.hyperParams.use_right_mask:
                    aug_y = y[right_mask]
                else:
                    aug_y = y
                # aug_y = y[right_mask]
                # aug_yHat = aug_yHat[right_mask]
                aug_loss = self.getLoss(aug_yHat, aug_y)
            # loss = cs_loss * self.cs_loss_weight + aug_loss
            loss = loss + cs_loss * self.cs_loss_weight + aug_loss * self.cls_loss_weight
            
        
        preds = yHat.argmax(1)
        probs = yHat.softmax(1)
        y = y.to("cpu")
        if torch.isnan(probs).any():
            print(probs)
            print(yHat)

        if(train):
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if(not isinstance(self.scheduler, type(None))):
                self.scheduler.step()            

        loss = loss.detach().to("cpu")
        cs_loss = 0 if cs_loss ==0 else cs_loss.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")
        
        if self.hyperParams.model == 'stagin' and save:
            
            dir = os.path.join('cache/stagin/tmp',
                                'vs' if self.hyperParams.cs else 'ori', 
                                self.details.atlas)
            if not os.path.exists(dir):
                os.makedirs(dir)
            
            atten_time = attention['time-attention'].detach().cpu().numpy()
            atten_node = attention['node-attention'].detach().cpu().numpy()
            cls_rep_np = cls_rep.detach().cpu().numpy()
            ans = (preds == y).numpy()
            y_np = y.numpy()
            tp_rep = tp_rep.detach().cpu().numpy()
            for at, an, cr, answer, ground_true, sid in zip(atten_time, atten_node, cls_rep_np, ans, y_np, sids):
                np.savez(os.path.join(dir, f'{sid}.npz'),
                            time_attention=at,
                            node_attention=an,
                            rep=cr,
                            answer=answer,
                            ground_true=ground_true,
                            tp_rep=tp_rep) 

       
        cls = None # 有释放，不多
        cls_rep = None
        
        torch.cuda.empty_cache()


        return cs_loss, loss, preds, probs, y
        


    # HELPER FUNCTIONS HERE

    def prepareInput(self, x, y):

        """
            x = (batchSize, N, T)
            y = (batchSize, )

        """
        # to gpu now

        x = x.to(self.details.device)
        y = y.to(self.details.device)


        return (x, ), y

    def getLoss(self, yHat, y):
        cross_entropy_loss = self.criterion(yHat, y)
        return cross_entropy_loss * self.cls_loss_weight

