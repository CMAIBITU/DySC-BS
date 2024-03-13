from Models.bolT import BolT
import torch
import numpy as np
from einops import rearrange
from .sfc_model import TE_HI_GCN
from torch.nn import functional as F



class Model():

    def __init__(self, hyperParams, details):

        self.hyperParams = hyperParams
        self.details = details

        self.model = BolT(hyperParams, details)
        self.cs_loss_encoder = None
        need_cs_loss = hyperParams.cs
        
        # load model into gpu
        
        self.model = self.model.to(details.device)

        # set criterion
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)#, weight = classWeights)
        self.fc = torch.nn.Linear(200, 2).to('cuda')
        params = list(self.model.parameters()) + list(self.fc.parameters())
        if need_cs_loss:
            self.cs_loss_encoder = TE_HI_GCN(n_splits=hyperParams.n_splits, 
                                             n_splits2=hyperParams.n_splits2,
                                             use_right_mask=hyperParams.use_right_mask, 
                                             cs_space_dim=hyperParams.cs_space_dim,
                                             name='cc200')
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
        
        
    def free(self):
        if self.cs_loss_encoder:
            self.cs_loss_encoder.free()
        
    def step(self, x, y, sids=None, folder_k=0, train=True, epoch=None):

        """
            x = (batchSize, N, dynamicLength) 
            y = (batchSize, numberOfClasses)

        """
        
        # if epoch:
        #     if epoch > 20:
        #         self.cs_loss_weight = 0.1
        #         self.cls_loss_weight = 1
        #     else:
        #         self.cs_loss_weight = 1
        #         self.cls_loss_weight = 0

        # PREPARE INPUTS
        
        inputs, y = self.prepareInput(x, y)

        # DEFAULT TRAIN ROUTINE
        
        if(train):
            self.model.train()
        else:
            self.model.eval()

        # yHat, cls = self.model(*inputs)
        yHat, cls, roiSignals_layers, cls_layers = self.model(*inputs)
        # cls [32, 6, 116]
        # kl modified
        # cls_mean = cls.mean(dim=1)
        # cls_max = torch.max(cls, dim=1)[0]
        # cls_min = torch.min(cls, dim=1)[0]
        # cls_rep = torch.cat((cls_mean), dim=-1)
        # cls_rep = cls_mean
        
        
        # cls_rep = torch.mean(torch.stack(cls_layers), dim=0)
        # cls_rep = rearrange(cls_rep, 'n c l -> n l c')
        # cls_rep = torch.mean(cls_rep, dim=-1)
        # cls = cls_rep
        cls_rep = torch.mean(cls, dim=1)
        
        
        cls_rep = F.normalize(cls_rep, dim=1)
        yHat = self.fc(cls_rep)
        
        loss = self.getLoss(yHat, y, cls)
        cs_loss = 0
        if self.cs_loss_encoder and train:
                   
            # cs_loss = self.cs_loss_encoder(cls_layers, folder_k, sids)
            cs_loss, sfc_reps, ts_reps, right_mask = self.cs_loss_encoder(cls_rep, folder_k, sids)
            aug_cls_reps = self.cs_loss_encoder.aug(ts_reps, sfc_reps)
            aug_cls_reps = F.normalize(aug_cls_reps, dim=1)
            aug_yHat = self.fc(aug_cls_reps)
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
            clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))
        # clsLoss = 0
        cross_entropy_loss = self.criterion(yHat, y)

        return cross_entropy_loss * self.cls_loss_weight + clsLoss * self.hyperParams.lambdaCons


