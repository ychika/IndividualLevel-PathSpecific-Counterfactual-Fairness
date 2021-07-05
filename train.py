
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import datetime

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2):
        # D_in is input dimension; D_out is output dimension.
        # H1 and H2 are hidden dimension;
        super(TwoLayerNet, self).__init__()
        if H1 == 1:
            D_out = 2
            self.linear1 = torch.nn.Linear(D_in, D_out)
            nn.init.xavier_normal_(self.linear1.weight)        
        else:
            D_out = 2 ### need two probabilities for cross-entropy losses
            self.linear1 = torch.nn.Linear(D_in, H1)
            self.linear2 = torch.nn.Linear(H1, H2)
            self.linear3 = torch.nn.Linear(H2, D_out)
            # randomly init weights
            nn.init.xavier_normal_(self.linear1.weight)        
            nn.init.xavier_normal_(self.linear2.weight)        
            nn.init.xavier_normal_(self.linear3.weight)
            
        
    def forward(self, x):
        x = x.float()
        if self.linear1.weight.size()[0] == 2: ## logistic case
            h1 = self.linear1(x) ## input * 2
            y_pred = F.log_softmax(h1, dim=1)            
        else:
            h1 = torch.sigmoid(self.linear1(x))
            h2 = torch.sigmoid(self.linear2(h1))
            y_pred = F.log_softmax(self.linear3(h2), dim=1)            
        return y_pred
    

class LogisticRegression(torch.nn.Module):
    def __init__(self, D_in):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(D_in, 1)
        
    def forward(self, x):
        x = x.float()        
        out = torch.sigmoid( self.linear(x) * 0.1 )
        return out

    
def Mylossfunc(model, loss, Prop, data, labels, indices, lambda_fair, remove_sensitive, fio, unconstr, oracle=False, uncertain="No"):
    if remove_sensitive is False and unconstr is False:
        if uncertain == "No":
            constr = Prop.Pse_ub(model, indices, fio=fio) if oracle is False else Prop.OraclePIU(model, indices)
        elif uncertain == "YesAB":
            constr = Prop.Pse_ub(model, indices, fio=fio, graph="A") + Prop.Pse_ub(model, indices, fio=fio, graph="B")
        elif uncertain == "YesA":
            constr = Prop.Pse_ub(model, indices, fio=fio, graph="A")
        elif uncertain == "YesB":
            constr = Prop.Pse_ub(model, indices, fio=fio, graph="B")
        else:
            constr = Prop.Pse_ub(model, indices, fio=fio, graph=uncertain)

        obj = loss(model(data), labels) + lambda_fair * constr
        return obj, constr
    else:
        obj = loss(model(data), labels)
        return obj       
            

def Optim(H1, H2, batchsize, T, lambda_fair, Prop, trainset, testset, resplot, seed, lr, mom, remove_sensitive=False, opt_m="SGD", fio=False, synth_num=0, unconstr=False, oracle=False, uncertain="No"):
    ## define model with randomly inited weights
    torch.random.manual_seed(seed)
    D_in = Prop.d ## input data dimension
    n = Prop.n; 

    model = TwoLayerNet(D_in, H1, H2) #
    criterion = nn.NLLLoss()

    ## for neural networks
    n_batches = n // batchsize if n % batchsize == 0 else n // batchsize + 1    

    d = len(list(model.parameters()))
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=0)
    trainloader_accheck = DataLoader(trainset, batch_size=batchsize, shuffle=False, num_workers=0)
    testloader = DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=0)
        
    if resplot:
        grad_n = list()

    if opt_m == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
    elif opt_m == "Adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif opt_m == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
    for t in range(T-1):
        trainiter = iter(trainloader)
        #scheduler.step(running_loss)
        running_loss = 0.0
        running_ub = 0.0
        ub_array = np.zeros(n_batches-1)

        if t % 100 == 0 and H1 == 0: ## only when model is logistic
            print("parameters")
            for params in model.parameters():
                print(params.data)

        for j in range(n_batches-1):
            ## sample minibatch
            indices = np.arange(j*batchsize, (j+1)*batchsize, dtype=np.int) if (j+1)*batchsize < n else np.arange(j*batchsize, n, dtype=np.int)
            data, labels = trainiter.next()
            # wrap them in Variable
            data, labels = Variable(data), Variable(labels)                
            
            ## compute loss
            if opt_m == "SGD" or opt_m == "Adadelta" or opt_m == "Adam":
                ## init gradients
                optimizer.zero_grad()

                labels = labels.long()
                torch.reshape(labels, (batchsize, 1))    
                if remove_sensitive is False and unconstr is False:
                    loss, ub = Mylossfunc(model, criterion, Prop, data, labels, indices, lambda_fair, remove_sensitive, fio, unconstr, oracle, uncertain)
                    ub_array[j] = ub
                else:
                    loss = Mylossfunc(model, criterion, Prop, data, labels, indices, lambda_fair, remove_sensitive, fio, unconstr, oracle, uncertain)
                    
                
                ## compute gradients -> parameter update
                loss.backward()
                optimizer.step()
                
            ## save norm of gradient 
            if resplot and j == 0: ## if the first batch
                theta_n = 0.0
                for params in model.parameters():
                    theta_n += params.grad.data.norm(2).numpy()
                grad_n.append( theta_n / d )

            
            ## print stats
            running_loss += loss.item()
            if remove_sensitive is False and unconstr is False:
                running_ub += ub
        #if remove_sensitive is False and unconstr is False:
        #    print('[%d] objective: %.3f (l*ub: %.3f (ub: %.3f))' % (t,  running_loss / n_batches, lambda_fair * running_ub / n_batches, running_ub / n_batches))
        #else:
        #    print('[%d] objective: %.3f)' % (t,  running_loss / n_batches))

        if H1 == 0:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize,
                                                      shuffle=False, num_workers=0)
            total = 0; correct = 0        
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                labels = labels.float()
                
                outputs = model(inputs) ## outputs.data has (# of data, # of classes)
                predicted = torch.distributions.binomial.Binomial(1, probs=outputs.data[:, 0]).sample()
                
                total += labels.size(0) ## # of data
                correct += (predicted == labels).sum().item()
            print("Training accuracy = " + str(100 * float(correct/total)))
        
    if H1 == 1:
        model_name = "Logistic"
    else:
        model_name = "NN"
    if fio:
        method_name = "FIO"
    elif lambda_fair == 0.0 and unconstr:
        method_name = "Unconstrained"
    elif lambda_fair == 0.0 and remove_sensitive:
        method_name = "Remove"        
    else:
        method_name = "Proposed"
    mode_name = Prop.__class__.__name__

    theta_data = list()
    for param in model.parameters():
        theta_data.append( param.data )
    return theta_data    
