from Models.bolT import BolT
import torch
import numpy as np
from einops import rearrange
from .sfc_model import TE_HI_GCN
from Models.bs.DTCR import KMeansLossHelperWithEncoder
from torch.nn import functional as F
from torch import nn
import os

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
            

class BolTProxy(nn.Module):
    def __init__(self, hyperParams, details):
        super().__init__()
        self.model = BolT(hyperParams, details)
        self.rep_fc = rep_fc_factory(**hyperParams.rep_fc).to('cuda')
        self.fc = torch.nn.Linear(hyperParams.num_cluster, 2).to('cuda')
        self.proj_cluster = nn.Sequential(
            nn.Linear(hyperParams.classifier_fc_in_dim, hyperParams.num_cluster),
            # nn.Sigmoid(),
            nn.ReLU(),
            nn.Dropout(),
            nn.LayerNorm(hyperParams.num_cluster)
        )
    
    def forward(self, ts):
        return self.model(ts)
    
    def predict4analyse(self, ts, analysis=False):
        # logits, cls, roiSignals_layers, cls_layers = self.forward(ts)
        _, cls, roiSignals_layers, cls_layers = self.model(ts, analysis=analysis)
        cluster_rep_proj = self.proj_cluster(cls)
        yHat = self.fc(torch.mean(cluster_rep_proj, dim=1))
        return yHat, cls
        
    
class BolTDummy(nn.Module):
    def __init__(self, boltEncoder):
        super().__init__()
        self.encoder = boltEncoder
    
    def forward(self, ts):
        yHat, cls, roiSignals_layers, cls_layers = self.encoder(ts)
        return cls.reshape(-1, cls.shape[-1])
        
    
class KmeanLoss(KMeansLossHelperWithEncoder):
    def __init__(self, num_cluster, hyperParams, details, bz):
        super().__init__(num_cluster, bz)
        self.bolt = BolT(hyperParams, details).cuda()
        # self.update_n_epoch = update_n_epoch
        # self.update_epoch_points = [15, 25, 33, 34, 35, 36, 37,38,39,40]
        self.update_epoch_points = hyperParams.update_ep_points
        self.this_epoch = 0
    
    def update_model(self, epoch, new_bolt_model):
        # if epoch % self.update_n_epoch == 0 and epoch > self.this_epoch:
        if epoch in self.update_epoch_points and epoch > self.this_epoch:
            if self.encoder is None:
                self.encoder = BolTDummy(self.bolt)
            self.bolt.load_state_dict(new_bolt_model.state_dict())
            self.this_epoch = epoch



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

        self.model = BolTProxy(hyperParams, details)
        self.cs_loss_encoder = None
        need_cs_loss = hyperParams.cs
        self.shiftSize = int(hyperParams.shiftCoeff * hyperParams.windowSize)
        self.nW = (self.details.dynamicLength - self.hyperParams.windowSize) // self.shiftSize  + 1
        self.brain_state_recoginition = self.hyperParams.brain_state
        # if self.brain_state_recoginition:
        self.kmean_loss = KmeanLoss(hyperParams.num_cluster, hyperParams, details, bz=details.batchSize*self.nW)
        
        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)#, weight = classWeights)
        # self.rep_fc = torch.nn.Linear(hyperParams.dim, hyperParams.cs_space_dim).to('cuda')
        # self.rep_fc = torch.nn.Identity()
        # self.rep_fc = self.rep_fc_factory(**hyperParams.rep_fc).to('cuda')
        # self.fc = torch.nn.Linear(hyperParams.classifier_fc_in_dim, 2).to('cuda')
        self.rep_fc = self.model.rep_fc
        self.fc = self.model.fc
        
        # self.fc_aug = self.fc
        # params = list(self.model.parameters()) + list(self.fc.parameters()) + list(self.rep_fc.parameters())

        # params = list(self.model.parameters()) + list(self.fc.parameters()) + list(self.rep_fc.parameters()) 
        params = list(self.model.parameters()) 
        if hasattr(hyperParams, 'aug_fc') and hyperParams.aug_fc:
            self.fc_aug = torch.nn.Linear(hyperParams.cs_space_dim, 2).to('cuda')
            params = params + list(self.fc_aug.parameters())
        else:
            self.fc_aug = self.fc
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
        steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))        
        
        divFactor = hyperParams.maxLr / hyperParams.lr
        finalDivFactor = hyperParams.lr / hyperParams.minLr
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch), div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)
        # print(self.scheduler.__dict__)
        # print(details.nOfTrains)
        # print('three cal factor:', steps_per_epoch, divFactor, finalDivFactor)

        # exit()
        self.cs_loss_weight = hyperParams.cs_loss_weight
        self.cls_loss_weight = 1
        self.clean_every_step = True
    
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
        
        
    def step(self, x, y, sids=None, folder_k=0, train=True, epoch=None, save=False, save_path=''):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """

        # PREPARE INPUTS
        
        inputs, y = self.prepareInput(x, y)

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
            self.kmean_loss.update_model(epoch=epoch, new_bolt_model=self.model.model)
        else:
            self.model.eval()

        # yHat, cls = self.model(*inputs)
        yHat, cls, roiSignals_layers, cls_layers = self.model(*inputs)
        # yHat:(N,2), cls(N, num_window, 200), roiSignals_layers:array, [N,L,200], cls_layers array,每一个的shape和cls一样
        cls_rep = torch.mean(cls, dim=1)
        
        # cls_rep = F.normalize(cls_rep, dim=1)
        # cls_rep = self.rep_fc(cls)
        # cls_rep = F.normalize(cls_rep, dim=1)
        cluster_rep_proj = self.model.proj_cluster(cls)
        cluster_rep = F.softmax(cluster_rep_proj, dim=-1)
        # cluster_rep = cluster_rep_proj
        
        yHat = self.fc(torch.mean(cluster_rep_proj, dim=1))
        
        loss = self.getLoss(yHat, y, cluster_rep_proj)
        
        k_means_loss = 0
        if train:
            k_means_loss = self.kmean_loss(x=inputs[0] ,h=cluster_rep.reshape(-1, cluster_rep.shape[-1]))
        
        loss = loss + k_means_loss * self.hyperParams.kmeans_loss_weight
        
        cs_loss = 0
        # cs_loss = torch.FloatTensor([0])
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

        if(train):
            # print(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if(not isinstance(self.scheduler, type(None))):
                self.scheduler.step()            

        loss = loss.detach().to("cpu") # 没释放
        cs_loss = 0 if cs_loss ==0 else cs_loss.detach().to("cpu")
        preds = preds.detach().to("cpu")
        probs = probs.detach().to("cpu")

        y = y.to("cpu")
        
        if save:
            # sid,
            dir = os.path.join(self.hyperParams.cache_dir_base, 
                               'vs' if self.hyperParams.cs else 'ori', 
                                self.details.atlas, save_path)
            # dir = os.path.join('cache/bolt/tmp',
            #                     'vs' if self.hyperParams.cs else 'ori', 
            #                     self.details.atlas)
            if not os.path.exists(dir):
                os.makedirs(dir)
            
            tmp = []
            cls_np = cluster_rep.detach().to("cpu").numpy()
            for cls_l in cls_layers:
                tmp.append(cls_l.detach().cpu().numpy())
            cls_layers_np = np.stack(tmp, axis=1)# N, 4, w, c
            ans = (preds == y).numpy()
            y_np = y.numpy()
            for sid, cls_l, answer,ground_true in zip(sids, cls_layers_np, ans, y_np):
                np.savez(os.path.join(dir, f'{sid}.npz'),
                    sid=sid,
                    cls_layers=cls_l,
                    answer=answer,
                    ground_true=ground_true,
                    cls=cls_np,
                )
        
        if self.clean_every_step:
            roiSignals_layers.clear()
            cls_layers.clear()
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

    def getLoss(self, yHat, y, cls=None):
        
        # cls.shape = (batchSize, #windows, featureDim)
        if cls is None:
            clsLoss = 0
        else:
            clsLoss = -torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True))) * self.hyperParams.intra_sub_loss_weight 
            # clsLoss = clsLoss + torch.mean(torch.square(cls.mean(dim=1) - cls.mean(dim=(0, 1))))  * self.hyperParams.inter_subs_loss_weight
            clsLoss = clsLoss + torch.mean(torch.square(cls - cls.mean(dim=(0, 1))))  * self.hyperParams.inter_subs_loss_weight
        # clsLoss = 0
        cross_entropy_loss = self.criterion(yHat, y)

        return cross_entropy_loss * self.cls_loss_weight + clsLoss * self.hyperParams.lambdaCons


