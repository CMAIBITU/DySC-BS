
# parse the arguments

import argparse
import torch
from datetime import datetime


from utils import Option, metricSummer, calculateMetrics, dumpTestResults

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dataset", type=str, default="abide1")
parser.add_argument("-m", "--model", type=str, default="bolT")
parser.add_argument("-a", "--analysis", type=bool, default=False)
# parser.add_argument("-b", "--brain_state", type=bool, default=False)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default="noname")


argv = parser.parse_args()



from Dataset.datasetDetails import datasetDetailsDict

# import model runners

# from Models.SVM.run import run_svm
from Models.run import run_bolT
from Models.bs.run import run_bolT as run_bolT_bs
# from Models.Rnn.run import run_lstm 
# from Models.Transformer.run import run_transformer
# import hyper param fetchers

# from Models.SVM.hyperparams import getHyper_svm
from Models.hyperparams import getHyper_bolT
from Models.bs.hyperparams import getHyper_bolT as getHyper_bolT_bs
# from Models.Rnn.hyperparams import getHyper_lstm
# from Models.Transformer.hyperparams import getHyper_transformer


hyperParamDict = {

        # "svm" : getHyper_svm,
        "bolT" : getHyper_bolT,
        "bolT_bs" : getHyper_bolT_bs,
        # 'lstm': getHyper_lstm,
        # 'transformer': getHyper_transformer,

}

modelDict = {

        # "svm" : run_svm,
        "bolT" : run_bolT,
        'bolT_bs': run_bolT_bs
        # "lstm" : run_lstm,
        # 'transformer': run_transformer,
}


getHyper = hyperParamDict[argv.model]
runModel = modelDict[argv.model]

# print("\nTest model is {}".format(argv.model))
print(argv)


datasetName = argv.dataset
datasetDetails = datasetDetailsDict[datasetName]
# print(datasetDetails['atlas'])
# exit()
hyperParams = getHyper(datasetDetails['atlas'])

print("Dataset details : {}".format(datasetDetails))

# test

if(datasetName == "abide1"):
    # seeds = [0,1,2,3,4]
    seeds = [0]
else:
    seeds = [0]
    
resultss = []

for i, seed in enumerate(seeds):

    # for reproducability
    torch.manual_seed(seed)

    print("Running the model with seed : {}".format(seed))
    if(argv.model == "bolT"):
        results = runModel(hyperParams, Option({**datasetDetails,"datasetSeed":seed}), device="cuda:{}".format(argv.device), analysis=argv.analysis)
    else:
        results = runModel(hyperParams, Option({**datasetDetails,"datasetSeed":seed}), device="cuda:{}".format(argv.device))

    resultss.append(results)
    


metricss = calculateMetrics(resultss) 
meanMetrics_seeds, stdMetrics_seeds, meanMetric_all, stdMetric_all = metricSummer(metricss, "test")


# now dump metrics
dumpTestResults(argv.name, hyperParams, argv.model, datasetName, metricss)

print("\n \ n meanMetrics_all : {}".format(meanMetric_all))
print("stdMetric_all : {}".format(stdMetric_all))

# for m in metricss:
#     print(m['test'])