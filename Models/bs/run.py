from tqdm import tqdm
import torch
import numpy as np
import random
import os
import sys
import wandb

from datetime import datetime

if(not "utils" in os.getcwd()):
    sys.path.append("../../../")


from utils import Option
from utils import Option, calculateMetric

from Models.bs.model_bs import Model as Model_BS
from Models.model import Model as Model_CLS
from Dataset.dataset import getDataset
import kl
from torchsummary import summary
from kl import torch_log
# from kl_sfc_model import TE_HI_GCN


def train(model, dataset, fold, nOfEpochs, dataset_test=None, group=''):
    # kl.stuff.reproduce() # 加上后性能大降
    dataLoader = dataset.getFold(fold, train=True)
    
    # run = wandb.init(project='Bolt', reinit=True,
    #                      group=group, tags=[f"aal", 'dynamic_len=60'], name= group + '_' + str(fold),
    #                      notes='no data augment, kl RandomCrop, tr=2,  和数据标准化, epoch=20')
    logger = torch_log.TensorBoardLogger('test', group=group, project='bolt', name= str(fold), add_time=True)
    

    
    for epoch in range(nOfEpochs):

            preds = []
            probs = []
            groundTruths = []
            losses = []
            cs_losses = []
            dataset.cur_epoch = epoch
            for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'fold:{fold} epoch:{epoch}')):
                
                xTrain = data["timeseries"] # (batchSize, N, dynamicLength)
                yTrain = data["label"] # (batchSize, )
                sids = data['subjId']
                sids = [str(s) for s in sids.tolist()]
                # sids = [ for sid in sids]
                # NOTE: xTrain and yTrain are still on "cpu" at this point

                train_cs_loss, train_loss, train_preds, train_probs, yTrain = model.step(xTrain, yTrain, sids=sids, folder_k=fold, train=True, epoch=epoch)
                # if need_cs_loss:
                    
                train_cs_loss = train_cs_loss if isinstance(train_cs_loss, (int, float)) else  train_cs_loss.numpy()
                train_loss = train_loss if isinstance(train_loss, (int, float)) else train_loss.numpy()
                train_preds = train_preds.numpy()
                train_probs = train_probs.numpy()
                yTrain = yTrain.numpy()
                
                torch.cuda.empty_cache()

                preds.append(train_preds)
                probs.append(train_probs)
                groundTruths.append(yTrain)
                losses.append(train_loss)
                cs_losses.append(train_cs_loss)

            # preds = torch.cat(preds, dim=0).numpy()
            # probs = torch.cat(probs, dim=0).numpy()
            # groundTruths = torch.cat(groundTruths, dim=0).numpy()
            # losses = torch.tensor(losses).numpy()
            # cs_losses = torch.tensor(cs_losses).numpy()
            preds = np.concatenate(preds, axis=0)
            probs = np.concatenate(probs, axis=0)
            groundTruths = np.concatenate(groundTruths, axis=0)
            losses = np.stack(losses)
            cs_losses = np.stack(cs_losses)

            metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
            print("Train metrics : {}".format(metrics), losses[0], cs_losses[0])
            train_metrics = {'train/' + key: metrics[key] for key in metrics}
            with torch.no_grad():
                _,_,_,_, test_metrics = test(model, dataset_test, fold)  
                test_metrics = { 'val/' + key: test_metrics[key] for key in test_metrics}
            
            # wandb.log({**train_metrics, **test_metrics})                
            logger.log_dict({**train_metrics, **test_metrics}, step=epoch)
            # logger.log_dict({**train_metrics}, step=epoch)
    # wandb.finish()
    # 都是numpy
    return preds, probs, groundTruths, losses



def test(model, dataset, fold, save=False, save_path=''):

    dataLoader = dataset.getFold(fold, train=False)

    preds = []
    probs = []
    groundTruths = []
    losses = []        

    for i, data in enumerate(tqdm(dataLoader, ncols=60, desc=f'Testing fold:{fold}')):

        xTest = data["timeseries"]
        yTest = data["label"]
        sids = data['subjId']
        sids = [str(s) for s in sids.tolist()]

        # NOTE: xTrain and yTrain are still on "cpu" at this point

        _, test_loss, test_preds, test_probs, yTest = model.step(xTest, yTest, sids=sids, train=False, save=save, save_path=save_path)
        
        torch.cuda.empty_cache()

        preds.append(test_preds)
        probs.append(test_probs)
        groundTruths.append(yTest)
        losses.append(test_loss)

    preds = torch.cat(preds, dim=0).numpy()
    probs = torch.cat(probs, dim=0).numpy()
    groundTruths = torch.cat(groundTruths, dim=0).numpy()
    loss = torch.tensor(losses).numpy().mean()          

    metrics = calculateMetric({"predictions":preds, "probs":probs, "labels":groundTruths})
    print("\n \n Test metrics : {}".format(metrics))                
    
    return preds, probs, groundTruths, loss, metrics
    


def run_bolT(hyperParams, datasetDetails, device="cuda", analysis=False):


    # extract datasetDetails

    foldCount = datasetDetails.foldCount
    datasetSeed = datasetDetails.datasetSeed
    nOfEpochs = datasetDetails.nOfEpochs


    dataset = getDataset(datasetDetails)
    import copy
    datasetDetails2 = copy.deepcopy(datasetDetails)
    # datasetDetails2.dynamicLength = None
    # datasetDetails2.batchSize = 1
    dataset_train2 = getDataset(datasetDetails2, swap_train_test=True)
    dataset_test = getDataset(datasetDetails)

    # print(dataset.get_nOfTrains_perFold())
    # exit()
    details = Option({
        "device" : device,
        "nOfTrains" : dataset.get_nOfTrains_perFold(),
        "nOfClasses" : datasetDetails.nOfClasses,
        "batchSize" : datasetDetails.batchSize,
        "nOfEpochs" : nOfEpochs,
        'atlas':datasetDetails.atlas,
        'check':datasetDetails.check,
        'dynamicLength':datasetDetails.dynamicLength,
    })


    results = []
    all_test_metrics = []
    timestamp = kl.stuff.get_readable_time()
    
    def get_Model():
        if hasattr(hyperParams, 'brain_state') and hyperParams.brain_state:
            return Model_BS(hyperParams, details)
        else:
            return Model_CLS(hyperParams, details)
    
    for fold in range(foldCount):

        model = get_Model()
        # summary(model, (32, 60, 116))
        # exit()


        train_preds, train_probs, train_groundTruths, train_loss = train(model, dataset, fold, nOfEpochs, dataset_test, group=timestamp)   
        # train_preds, train_probs, train_groundTruths, train_loss = train(model, dataset, fold, nOfEpochs, None, group=timestamp)   
        test_train_preds, test_train_probs, test_train_groundTruths, test_train_loss, test_train_metrics = test(model, dataset_train2, fold, save=datasetDetails.save, save_path=f'train/{fold}')
        test_preds, test_probs, test_groundTruths, test_loss, test_metrics = test(model, dataset, fold, save=datasetDetails.save, save_path=f'test/{fold}')
        
        torch.cuda.empty_cache()
        
        all_test_metrics.append(test_metrics)
        result = {

            "train" : {
                "labels" : train_groundTruths,
                "predictions" : train_preds,
                "probs" : train_probs,
                "loss" : train_loss
            },

            "test" : {
                "labels" : test_groundTruths,
                "predictions" : test_preds,
                "probs" : test_probs,
                "loss" : test_loss
            }

        }

        results.append(result)
        
        if datasetDetails.save:
            dir = os.path.join(datasetDetails.ckpt_dir, 'vs' if hyperParams.cs else 'ori', datasetDetails.atlas)                 
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(model.model.state_dict(), os.path.join(dir, f'{fold}.ckpt'))


        if(analysis):
            targetSaveDir = "./Analysis/TargetSavedModels/{}/seed_{}/".format(datasetDetails.datasetName, datasetSeed)
            os.makedirs(targetSaveDir, exist_ok=True)
            torch.save(model.model, targetSaveDir + "/model_{}.save".format(fold))
            
        model.free()
        del model
        torch.cuda.empty_cache()

    for i, m in enumerate(all_test_metrics):
        print(i, m)
    return results
