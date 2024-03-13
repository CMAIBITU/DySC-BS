import torch
import numpy as np
from einops import rearrange
from .kl_sfc_model import TE_HI_GCN
from torch.nn import functional as F
from torch import nn

class TsEncoder(nn.Module):
    def __init__(self, hyperParams, details):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = hyperParams.dim,
            hidden_size = hyperParams.hidden_size, # 32 best
            batch_first=True,
            dropout=hyperParams.rnn_dropout,
            num_layers=hyperParams.num_layers,
            bidirectional=True
        )
        self.drop_out = nn.Dropout(hyperParams.cls_dropout)
        self.act = nn.Sigmoid()
    
    def forward(self, ts):
        ts = rearrange(ts, 'n c l -> n l c')
        output, _= self.lstm(ts)
        cls = torch.mean(output, dim=1)
        cls = self.drop_out(cls)
        cls = self.act(cls)
        return cls
        
        

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
        self.model = TsEncoder(hyperParams, details)
        self.cs_loss_encoder = None
        need_cs_loss = hyperParams.cs
        
        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)#, weight = classWeights)
        # self.rep_fc = torch.nn.Linear(hyperParams.dim, hyperParams.cs_space_dim).to('cuda')
        # self.rep_fc = torch.nn.Identity()
        self.rep_fc = self.rep_fc_factory(**hyperParams.rep_fc).to('cuda')
        self.fc = torch.nn.Linear(hyperParams.classifier_fc_in_dim, 2).to('cuda')
        
        # self.fc_aug = self.fc
        # params = list(self.model.parameters()) + list(self.fc.parameters()) + list(self.rep_fc.parameters())

        params = list(self.model.parameters()) + list(self.fc.parameters()) + list(self.rep_fc.parameters()) 
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
        self.scheduler = None
        # steps_per_epoch = int(np.ceil(details.nOfTrains / details.batchSize))        
        # divFactor = hyperParams.maxLr / hyperParams.lr
        # finalDivFactor = hyperParams.lr / hyperParams.minLr
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, hyperParams.maxLr, details.nOfEpochs * (steps_per_epoch), div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)

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
        self.fc = None
        self.rep_fc = None
        self.optimizer = None
        self.scheduler = None
        torch.cuda.empty_cache()
        
        
    def step(self, x, y, sids=None, folder_k=0, train=True, epoch=None):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """

        # PREPARE INPUTS
        
        inputs, y = self.prepareInput(x, y)

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
            self.rep_fc.train()
        else:
            self.model.eval()
            self.rep_fc.eval()

        # yHat, cls = self.model(*inputs)
        cls_rep = self.model(*inputs)

        cls_rep = F.normalize(cls_rep, dim=1)
        cls_rep = self.rep_fc(cls_rep)
        cls_rep = F.normalize(cls_rep, dim=1)
        yHat = self.fc(cls_rep)
        
        loss = self.getLoss(yHat, y)
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

        y = y.to("cpu")
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

