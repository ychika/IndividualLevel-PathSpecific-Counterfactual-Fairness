import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from util import GermanDataset, AdultDataset, SynthDataset, SynthDataset_r,  SynthMDataset, SynthMDataset_r, SynthUDataset, SynthUDataset_r
from unfairness import *
from train import Optim as Optim
from train import TwoLayerNet, LogisticRegression
from multiprocessing import Pool
from sklearn.metrics import *

import sys

def LoadGermanData(tr_file, te_file, remove_sensitive):
    Trainset = pd.read_csv(tr_file); trainset = Trainset.values
    Testset = pd.read_csv(te_file); testset = Testset.values
    trainset = GermanDataset(trainset) if remove_sensitive is False else GermanDataset(trainset[:, [0,4,5,6,7,8]])
    testset = GermanDataset(testset) if remove_sensitive is False else GermanDataset(testset[:, [0,4,5,6,7,8]]) 
    return trainset, testset

def LoadAdultData(tr_file, te_file, remove_sensitive):
    Trainset = pd.read_csv(tr_file); trainset = Trainset.values
    Testset = pd.read_csv(te_file); testset = Testset.values
    trainset = AdultDataset(trainset) if remove_sensitive is False else AdultDataset(trainset[:, [0,7,8,9]]) 
    testset = AdultDataset(testset) if remove_sensitive is False else AdultDataset(testset[:, [0,7,8,9]])     
    return trainset, testset

def LoadSynthData(tr_file, te_file, remove_sensitive, graph_is_uncertain = False, unobserved_conf=False):
    Trainset = pd.read_csv(tr_file, sep=" "); trainset = Trainset.values
    Testset = pd.read_csv(te_file, sep=" "); testset = Testset.values
    if graph_is_uncertain is False and unobserved_conf is False:
        trainset = SynthDataset(trainset) if remove_sensitive is False else SynthDataset_r(trainset)
        testset = SynthDataset(testset) if remove_sensitive is False else SynthDataset_r(testset)
    elif unobserved_conf is True:
        trainset = SynthMDataset(trainset) if remove_sensitive is False else SynthMDataset_r(trainset)
        testset = SynthMDataset(testset) if remove_sensitive is False else SynthMDataset_r(testset)
    else:
        trainset = SynthUDataset(trainset) if remove_sensitive is False else SynthUDataset_r(trainset)
        testset = SynthUDataset(testset) if remove_sensitive is False else SynthUDataset_r(testset)        
    return trainset, testset

if __name__ == "__main__":
    fio = False
    remove_sensitive = False
    unconstr = False
    extended = False ## if fio = True; then should be False
    oracle = False

    assert len(sys.argv) > 14
    lambda_fair = float(sys.argv[1])
    T = int(sys.argv[2])
    mode = str(sys.argv[3])
    lr = float(sys.argv[4])
    mom = float(sys.argv[5])
    opt_m = str(sys.argv[6])
    batchsize = int(sys.argv[7])
    nn_type = int(sys.argv[8])
    synth_num = int(sys.argv[9])
    if int(sys.argv[10]) == 1:
        fio = True
    if int(sys.argv[11]) == 1:
        remove_sensitive = True
    if int(sys.argv[12]) == 1:
        unconstr = True
    if int(sys.argv[13]) == 1:
        extended = True
    if int(sys.argv[14]) == 1:
        oracle = False
    uncertain = str(sys.argv[15])
    

    if mode == "german":
        trainset, testset = LoadGermanData("../../data/german_preprocessed_tr.csv", "../../data/german_preprocessed_te.csv", remove_sensitive)
        Prop = German(trainset.data, remove_sensitive)
        Prop_test = German(testset.data, remove_sensitive)
        D_in = Prop.d ## input data dimension
        if nn_type == 1:
            H1 = 100; H2 = 50
        elif nn_type == 2:
            H1 = 1; H2 = 1 # logistic case
        else:
            H1 = 100; H2 = 50
        resplot=False        
    elif mode == "adult":
        trainset, testset = LoadAdultData("../../data/adult_preprocessed_tr.csv", "../../data/adult_preprocessed_te.csv", remove_sensitive)
        Prop = Adult(trainset.data, remove_sensitive)
        Prop_test = Adult(testset.data, remove_sensitive)
        D_in = Prop.d ## input data dimension
        if nn_type == 1:
            H1 = 100; H2 = 50
        elif nn_type == 2:
            H1 = 1; H2 = 1 # logistic case
        else:
            H1 = 100; H2 = 50
            
        resplot=False
    elif mode == "synth":
        trainset, testset = LoadSynthData("./synth_data/file" + str(synth_num) +"_TRAIN.txt", "./synth_data/file" + str(synth_num) + "_TEST.txt", remove_sensitive)
        Prop = Synth(trainset.data, remove_sensitive)
        Prop_test = Synth(testset.data, remove_sensitive)
        D_in = Prop.d ## input data dimension
        #H1 = 0; H2 = 0   ## logistic case
        if nn_type == 1:
            H1 = 100; H2 = 50
        elif nn_type == 2:
            H1 = 1; H2 = 1 # logistic case
        resplot=False
    elif mode == "synthmiles":
        trainset, testset = LoadSynthData("./synth_uconf/file" + str(synth_num) +"_MilesTRAIN.txt", "./synth_uconf/file" + str(synth_num) + "_MilesTEST.txt", remove_sensitive)
        Prop = Synth_Miles(trainset.data, remove_sensitive, extended=extended)
        Prop_test = Synth_Miles(testset.data, remove_sensitive, extended=extended)
        D_in = Prop.d ## input data dimension
        if nn_type == 1:
            H1 = 100; H2 = 50
        elif nn_type == 2:
            H1 = 1; H2 = 1 # logistic case
        resplot=False
    elif mode == "synthu":
        trainset, testset = LoadSynthData("./synth_uncertain2/file" + str(synth_num) +"_TRAIN.txt", "./synth_uncertain2/file" + str(synth_num) + "_TEST.txt", remove_sensitive, True)
        Prop = Synth_Uncertain(trainset.data, remove_sensitive)
        Prop_test = Synth_Uncertain(testset.data, remove_sensitive)
        D_in = Prop.d ## input data dimension
        if nn_type == 1:
            H1 = 100; H2 = 50
        elif nn_type == 2:
            H1 = 1; H2 = 1 # logistic case
        resplot=False

    theta_t = Optim(H1, H2, batchsize, T, lambda_fair, Prop, trainset, testset, resplot, np.random.randint(0, 2 ** 32 -1), lr, mom, remove_sensitive, opt_m, fio, synth_num, unconstr, oracle, uncertain)



    
    ## compute test error
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                             shuffle=False, num_workers=0)
    ## for neural networks, logistic
    if H1 != 0: 
        model = TwoLayerNet(D_in, H1, H2)
    else: ## ?
        model = LogisticRegression(D_in)
        
    d = len(list(model.parameters()))

    # set learned parameters
    l = 0
    for param in model.parameters():
        param.data = theta_t[l]
        l += 1
        
    if mode == "synth" or mode == "synthmiles" or mode == "synthu":
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                  shuffle=False, num_workers=0)
        total = 0; correct = 0        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            outputs = model(inputs) ## outputs.data has (# of data, # of classes)
            #if mode != "synth":
            if H1 != 0:
                labels = labels.long()
                _, predicted = torch.max(outputs.data, 1) ## max in column -> predicted class
            else:
                labels = labels.float()
                predicted = torch.distributions.binomial.Binomial(1, probs=outputs.data[:, 0]).sample()


            total += labels.size(0) ## # of data
            correct += (predicted == labels).sum().item()
        print("Training accuracy = " + str(100 * float(correct/total)))
        if remove_sensitive is False:
            piu, condmean_array, mean_, std_ = Prop.EvaluatePIUandConditionalMean(model)
            print("Training PIU = " + str(piu))
            print("num of train subgroups = " + str(len(condmean_array)))
            print("Training min in condmean = " + str(np.min(condmean_array)))
            print("Training max in condmean = " + str(np.max(condmean_array)))
            print("Training std in condmean = " + str(std_))
            print("Training mean in condmean = " + str(mean_))
            
        


    total = 0; correct = 0
    if remove_sensitive is False:
        ## estimated unfair effect
        if mode != "synthmiles":
            pse = Prop_test.Pse_indiv(model)

    ## test accuracies
    total = 0; correct = 0
    labels_np = np.array([]); predicted_np = np.array([])
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        
        outputs = model(inputs) ## outputs.data has (# of data, # of classes)
        if H1 != 0: ## NN, Logistic
            labels = labels.long()
            _, predicted = torch.max(outputs.data, 1) ## max in log prob. (column) -> predicted class
        else: 
            labels = labels.float()
            _, predicted = torch.max(torch.softmax(outputs.data, dim=1), 1) ## max in column -> predicted class
            

        labels_np = np.append(labels_np, labels.detach().numpy())
        predicted_np = np.append(predicted_np, predicted.detach().numpy())
        
        total += labels.size(0) ## # of data
        correct += (predicted == labels).sum().item()

    print("Test Accuracy = " + str(100 * float(correct/total)))
    if remove_sensitive is False:
        print("Note: lambda_fair = " + str(lambda_fair))
        if mode == "synth" or mode == "synthmiles" or mode == "synthu":
            piu, condmean_array, mean_, std_ = Prop_test.EvaluatePIUandConditionalMean(model)
            print("Test PIU = " + str(piu))
            print("num of test subgroups = " + str(len(condmean_array)))
            print("Test min in condmean = " + str(np.min(condmean_array)))
            print("Test max in condmean = " + str(np.max(condmean_array)))
            print("Test std in condmean = " + str(std_))
            print("Test mean in condmean = " + str(mean_))



