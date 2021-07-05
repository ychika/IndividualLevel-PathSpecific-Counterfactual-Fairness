import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from torch.autograd import Variable

import torch.nn.functional as F
import sys

clip = 3.0
flag_clip = True

class German:
    def __init__(self, X, remove_sensitive):
        if remove_sensitive is False:        
            self.X = X[:, 1:] ## remove label Y        
            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]

            self.clf_srx, self.clf_sx, self.clf_x = self.PropensityFuncs()
            X_1 = self.X.copy(); X_0 = self.X.copy() # prepare different object!
            X_1[:, 0] = np.ones(self.n)
            X_0[:, 0] = np.zeros(self.n)
            self.X_1 = X_1
            self.X_0 = X_0
            w_prop1 = np.zeros(self.n)
            w_prop0 = np.zeros(self.n)
            for i in range(self.n):
                w_prop1[i], w_prop0[i] = self.Propensity(self.X[i, 1:]) # remove sensitive feature
                if flag_clip:
                    w_prop1[i] = np.min((w_prop1[i], clip))
                    w_prop0[i] = np.min((w_prop0[i], clip))
            self.w_prop1 = w_prop1
            self.w_prop0 = w_prop0
            
        else:
            self.X = X[:, 1:] ## only R and X; remove Y (already removed A, S)
            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]
            
    
    def PropensityFuncs(self):
        ## fit the P(A|S, R, X) model
        clf_srx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_srx.fit(self.X[:, [1, 2, 3, 4, 5, 6, 7, 8]], self.X[:, 0])        
        ## fit the P(A|S, X) model
        clf_sx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_sx.fit(self.X[:, [1, 2, 3, 6, 7, 8]], self.X[:, 0])
        ## fit the P(A|X) model
        clf_x = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_x.fit(self.X[:, [6, 7, 8]], self.X[:, 0])
        return clf_srx, clf_sx, clf_x
    
    def Propensity(self, x):
        w_prop1 = (1.0 / self.clf_x.predict_proba([x[[5,6,7]]])[0][1]) * (self.clf_sx.predict_proba([x[[0, 1, 2, 5, 6, 7]]])[0][1] / self.clf_sx.predict_proba([x[[0, 1, 2, 5, 6, 7]]])[0][0] ) * (self.clf_srx.predict_proba([x[[0, 1, 2, 3, 4, 5, 6, 7]]])[0][0] / self.clf_srx.predict_proba([x[[0, 1, 2, 3, 4, 5, 6, 7]]])[0][1])
        w_prop0 = 1.0 / self.clf_x.predict_proba([x[[5,6,7]]])[0][0]
        return w_prop1, w_prop0
        
    def Pse_ub(self, model, indices=None, fio=False):
        p1 = 0.0; p0 = 0.0
        if indices is None:
            n = self.n ## = total number of data samples
            n1 = np.where(self.X[:, 0] == 1)[0].size
            n0 = np.where(self.X[:, 0] == 0)[0].size                        
            for i in range(n):
                i_is_male = self.X[i, 0] # = 1 if a_i = 1
                i_is_female = (1 - self.X[i, 0]) # = 1 if a_i = 0
                p1 += i_is_male * self.w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
                p0 += i_is_female * self.w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]                
            
        # use data indices for stochastic gradient descent method
        else:
            n = np.shape(indices)[0] ## = minibatch size
            n1 = np.where(self.X[indices, 0] == 1)[0].size
            n0 = np.where(self.X[indices, 0] == 0)[0].size            
            for i in range(n):
                i_is_male = self.X[indices[i], 0] 
                i_is_female = (1 - self.X[indices[i], 0]) 
                p1 += i_is_male *  self.w_prop1[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_1[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
                p0 += i_is_female *  self.w_prop0[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_0[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
                
            
        p1 = p1 / n
        p0 = p0 / n            

        if fio is False:
            constr = 2 * ( p1 * (1.0 - p0) + (1.0 - p1) * p0 )
        else:
            if p1 > p0:
                constr = p1 - p0
            else:
                constr = p0 - p1
            
        return constr


    def Pse_indiv(self, model):
        n = self.n ## = total number of data samples
        pse = np.zeros(n)
        prob1 = np.zeros(n)
        prob0 = np.zeros(n)
         
        for i in range(n):
            i_is_male = self.X[i, 0]
            i_is_female = (1 - self.X[i, 0]) 
            prob1[i] = i_is_male * self.w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()
            prob0[i] = i_is_female * self.w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()
                
            pse[i] = prob1[i] - prob0[i]
            
        prob1_mean = np.min((np.mean(prob1), 1.0))
        prob0_mean = np.min((np.mean(prob0), 1.0))
            
        print("mean of prob1 and prob0")
        print(str( np.mean(prob1) ) + " -> " + str(prob1_mean))
        print(str( np.mean(prob0) ) + " -> " + str(prob0_mean))
        
        print("mean: " + str(prob1_mean - prob0_mean ))
        print("ub: " + str(2 * ( (1 - prob0_mean) * prob1_mean + prob0_mean * (1 - prob1_mean) )))            
                
        return np.array(pse)

    
class Adult:
    def __init__(self, X, remove_sensitive):
        if remove_sensitive is False:
            self.X = X[:, 1:] ## remove label Y        
            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]

            self.clf_rlmx, self.clf_mx, self.clf_x = self.PropensityFuncs()
            X_1 = self.X.copy(); X_0 = self.X.copy() # prepare different object!
            X_1[:, 0] = np.ones(self.n)
            X_0[:, 0] = np.zeros(self.n)
            self.X_1 = X_1
            self.X_0 = X_0
            w_prop1 = np.zeros(self.n)
            w_prop0 = np.zeros(self.n)
            #clip = 3.5        
            for i in range(self.n):
                w_prop1[i], w_prop0[i] = self.Propensity(self.X[i, 1:]) # remove sensitive feature
                if flag_clip:
                    w_prop1[i] = np.min((w_prop1[i], clip))
                    w_prop0[i] = np.min((w_prop0[i], clip))
            
            self.w_prop1 = w_prop1
            self.w_prop0 = w_prop0
        else:
            self.X = X[:, 1:] ## only X; remove Y (already removed A, M, L, R)
            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]
            
    
    def PropensityFuncs(self):
        ## fit the P(A|R, L, M, X) model
        clf_rlmx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_rlmx.fit(self.X[:, 1:], self.X[:, 0])    
        ## fit the P(A|M, X) model
        clf_mx = LogisticRegression(solver='lbfgs', max_iter=1000)
        mx_ind = [1, 6, 7, 8]
        clf_mx.fit(self.X[:, mx_ind], self.X[:, 0])
        ## fit the P(A|X) model
        clf_x = LogisticRegression(solver='lbfgs', max_iter=1000)
        x_ind = [6, 7, 8]
        clf_x.fit(self.X[:, x_ind], self.X[:, 0])
        return clf_rlmx, clf_mx, clf_x   

    def Propensity(self, x):
        mx_ind = [0, 5, 6, 7] 
        x_ind = [5, 6, 7]        
        w_prop1 = ( 1.0 / self.clf_x.predict_proba([x[x_ind]])[0][1] ) * ( self.clf_rlmx.predict_proba([x])[0][0] / self.clf_mx.predict_proba([x[mx_ind]])[0][0] )
        w_prop0 = 1.0 / self.clf_x.predict_proba([x[x_ind]])[0][0]  
        return w_prop1, w_prop0

    def Pse_ub(self, model, indices=None, fio=False):
        p1 = 0.0; p0 = 0.0
        if indices is None:
            n = self.n ## = total number of data samples
            n1 = np.where(self.X[:, 0] == 1)[0].size
            n0 = np.where(self.X[:, 0] == 0)[0].size                        
            for i in range(n):
                i_is_male = self.X[i, 0]
                i_is_female = (1 - self.X[i, 0])
                p1 += i_is_male * self.w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
                p0 += i_is_female * self.w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]                
            
        # use data indices for stochastic gradient descent method
        else:
            n = np.shape(indices)[0] ## = minibatch size
            n1 = np.where(self.X[indices, 0] == 1)[0].size
            n0 = np.where(self.X[indices, 0] == 0)[0].size            
            for i in range(n):
                i_is_male = self.X[indices[i], 0] 
                i_is_female = (1 - self.X[indices[i], 0]) 
                p1 += i_is_male *  self.w_prop1[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_1[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
                p0 += i_is_female *  self.w_prop0[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_0[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]                
            
        p1 = p1 / n
        p0 = p0 / n            

        if fio is False:
            constr = 2 * ( p1 * (1.0 - p0) + (1.0 - p1) * p0 )
        else:
            if p1 > p0:
                constr = p1 - p0
            else:
                constr = p0 - p1            
        return constr


    def Pse_indiv(self, model):
        n = self.n ## = total number of data samples
        pse = np.zeros(n)
        prob1 = np.zeros(n)
        prob0 = np.zeros(n)
         
        for i in range(n):
            i_is_male = self.X[i, 0]
            i_is_female = (1 - self.X[i, 0]) 
            prob1[i] = i_is_male * self.w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()
            prob0[i] = i_is_female * self.w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()
            pse[i] = prob1[i] - prob0[i]
            
        prob1_mean = np.min((np.mean(prob1), 1.0))
        prob0_mean = np.min((np.mean(prob0), 1.0))
            
        print("mean of prob1 and prob0")
        print(str( np.mean(prob1) ) + " -> " + str(prob1_mean))
        print(str( np.mean(prob0) ) + " -> " + str(prob0_mean))
        
        print("mean: " + str(prob1_mean - prob0_mean ))
        print("ub: " + str(2 * ( (1 - prob0_mean) * prob1_mean + prob0_mean * (1 - prob1_mean) )))            
                
        return np.array(pse)
    
    
class Synth:
    def __init__(self, X, remove_sensitive):
        if remove_sensitive is False:
            self.X = X[:, 0:4] ## A, M, D, Q remove label Y, and M0
            self.M0 = X[:, 5]
            self.D0 = X[:, 6]
            self.D1 = X[:, 7]

            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]
            X_1 = self.X.copy(); X_0 = self.X.copy() # prepare different object!
            X_1[:, 0] = np.ones(self.n)
            X_0[:, 0] = np.zeros(self.n)
            self.X_1 = X_1
            self.X_0 = X_0

            self.clf_mdx, self.clf_dx, self.clf_x = self.PropensityFuncs()
            w_prop1 = np.zeros(self.n); w_prop0 = np.zeros(self.n)
            for i in range(self.n):
                w_prop1[i], w_prop0[i] = self.Propensity(self.X[i, 1:]) # remove sensitive feature
            self.w_prop1 = w_prop1; self.w_prop0 = w_prop0 
            if flag_clip:
                for i in range(self.n):
                    self.w_prop1[i] = np.min((self.w_prop1[i], clip))
                    self.w_prop0[i] = np.min((self.w_prop0[i], clip))
        else:
            self.X = X[:, [0,2]] ## M, Q remove label Y, and M0            
            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]
        
        
    def PropensityFuncs(self):
        ## fit the P(A|M, D, X) model
        clf_mdx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_mdx.fit(self.X[:, [1, 2, 3]], self.X[:, 0])        
        ## fit the P(A|D, X) model
        clf_dx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_dx.fit(self.X[:, [2, 3]], self.X[:, 0])
        ## fit the P(A|X) model
        clf_x = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_x.fit(self.X[:, 3:], self.X[:, 0])
        return clf_mdx, clf_dx, clf_x
    
    def Propensity(self, x):
        w_prop1 = (1.0 /  self.clf_x.predict_proba([x[[2]]])[0][1]) * (self.clf_dx.predict_proba([x[[1, 2]]])[0][1] / self.clf_dx.predict_proba([x[[1, 2]]])[0][0]) * (self.clf_mdx.predict_proba([x[[0, 1, 2]]])[0][0] / self.clf_mdx.predict_proba([x[[0, 1, 2]]])[0][1])
        w_prop0 = 1.0 /  self.clf_x.predict_proba([x[[2]]])[0][0] 
        return w_prop1, w_prop0
    
    def Pse_ub(self, model, indices=None, fio=False):
        p1 = 0.0; p0 = 0.0
        #clip = 5.0
        if indices is None:
            n = self.n ## = total number of data samples
            n1 = np.where(self.X[:, 0] == 1)[0].size
            n0 = np.where(self.X[:, 0] == 0)[0].size
            for i in range(n):
                i_is_male = self.X[i, 0] # = 1 if a_i = 1
                i_is_female = (1 - self.X[i, 0]) # = 1 if a_i = 0
                p1 += i_is_male * self.w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) ) [0][1]
                p0 += i_is_female * self.w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]                    
        # use data indices for stochastic gradient descent method
        else:
            n = np.shape(indices)[0] ## = minibatch size
            n1 = np.where(self.X[indices, 0] == 1)[0].size
            n0 = np.where(self.X[indices, 0] == 0)[0].size
            for i in range(n):
                i_is_male = self.X[indices[i], 0] 
                i_is_female = (1 - self.X[indices[i], 0]) 
                p1 += i_is_male * self.w_prop1[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_1[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ))[0][1]
                p0 += i_is_female * self.w_prop0[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_0[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ))[0][1]                                        
        p1 = p1 / n
        p0 = p0 / n

        if fio is False:
            constr = 2 * ( p1 * (1.0 - p0) + (1.0 - p1) * p0 )
        else:
            if p1 > p0:
                constr = p1 - p0
            else:
                constr = p0 - p1
        return constr


    def Pse_indiv(self, model):
        n = self.n ## = total number of data samples
        prob1 = np.zeros(n)
        prob0 = np.zeros(n)
        pse = np.zeros(n)
        for i in range(n):
            i_is_male = self.X[i, 0] 
            i_is_female = (1 - self.X[i, 0])
            prob1[i] = i_is_male * self.w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()
            prob0[i] = i_is_female * self.w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()    
            pse[i] = prob1[i] - prob0[i]
            

        prob1_mean = np.min((np.mean(prob1), 1.0))
        prob0_mean = np.min((np.mean(prob0), 1.0))
            
        print("mean of prob1 and prob0")
        print(str( np.mean(prob1) ) + " -> " + str(prob1_mean))
        print(str( np.mean(prob0) ) + " -> " + str(prob0_mean))
        
        print("mean: " + str(prob1_mean - prob0_mean ))
        print("ub: " + str(2 * ( (1 - prob0_mean) * prob1_mean + prob0_mean * (1 - prob1_mean) )))
            
        return np.array(pse)

    def EvaluatePIUandConditionalMean(self, model):
        n = self.n ## = total number of data samples
        prob1 = np.zeros(n)
        prob0 = np.zeros(n)
        self.X_1D1M0 = np.vstack((self.X_1[:, 0], self.M0, self.D1, self.X[:, 3])).T
        self.X_0D0M0 = np.vstack((self.X_0[:, 0], self.M0, self.D0, self.X[:, 3])).T


        Y1 = np.zeros(n); Y0 = np.zeros(n)
        for i in range(n):
            prob1[i] = torch.exp( model( Variable(torch.Tensor(self.X_1D1M0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
            prob0[i] = torch.exp( model( Variable(torch.Tensor(self.X_0D0M0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
            if prob1[i] >= 0.5:
                Y1[i] = 1
            if prob0[i] >= 0.5:
                Y0[i] = 1

        PIU = np.mean(np.abs(Y1 - Y0))
        
        ## to compute conditional mean unfair effect...
        # enumerate combinations of feature values
        fc = np.unique(self.X, axis=0)
        num_of_fc = fc.shape[0]
        condmean_array = np.zeros(num_of_fc)
        # belonging_subgroup = np.zeros(self.n)
        num_of_members = np.zeros(num_of_fc)
        
        for i in range(self.n):
            for j in range(num_of_fc):
                if len(np.where(self.X[i, :] == fc[j, :])[0]) == self.d: # if i-th individual feature is j-th feature combination
                    # belonging_subgroup[i] = j ## i-th individual belongs to j-th subgroup
                    num_of_members[j] += 1 ## num of members in j-th subgroup increases
                    condmean_array[j] += Y1[i] - Y0[i] ##

        #print(num_of_members)

        condmean_array_ = np.zeros(num_of_fc)        
        for j in range(num_of_fc):
            condmean_array_[j] = (num_of_members[j] * condmean_array[j])
            
        ind_ = np.where(num_of_members > 1)[0]
        mean_ = np.sum(condmean_array_) / self.n
        
        std_ = np.sqrt( np.sum( np.array([ num_of_members[j] * np.square( condmean_array[j] - mean_) for j in range(num_of_fc)]) ) / ((self.n - 1) * self.n)  )

        return np.mean(np.abs(Y1 - Y0)), condmean_array, mean_, std_
      
    def OraclePIU(self, model, indices):
        n = np.shape(indices)[0] ## = minibatch size
        prob1 = torch.zeros(n)
        prob0 = torch.zeros(n)
        self.X_1D1M0 = np.vstack((self.X_1[:, 0], self.M0, self.D1, self.X[:, 3])).T
        self.X_0D0M0 = np.vstack((self.X_0[:, 0], self.M0, self.D0, self.X[:, 3])).T
        Y1 = torch.zeros(n); Y0 = torch.zeros(n)
        for i in range(n):
            prob1[i] = torch.exp( model( Variable(torch.Tensor(self.X_1D1M0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
            prob0[i] = torch.exp( model( Variable(torch.Tensor(self.X_0D0M0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
        PIU = torch.mean(torch.bernoulli(prob1) - torch.bernoulli(prob0))
        return PIU
    

class Synth_Miles:
    def __init__(self, X, remove_sensitive, extended=True):
        self.X = X[:, [0, 1, 2]] ## A, R, M
        self.M0 = X[:, 5] #M0

        self.n = np.shape(self.X)[0]
        self.d = np.shape(self.X)[1]

        X_1 = self.X.copy(); X_0 = self.X.copy() # prepare different object!
        X_1[:, 0] = np.ones(self.n)
        X_0[:, 0] = np.zeros(self.n)
        self.X_1 = X_1
        self.X_0 = X_0
        self.extended = extended
        
        if remove_sensitive is False:
            if self.extended is True:
                self.clf_m_a, self.clf_r_a = self.PropensityFuncs_extended()
                self.list_m_values = list(set(X[:, 2]))
                self.list_r_values = list(set(X[:, 1]))
            else:
                self.clf_mx, self.clf_x = self.PropensityFuncs()
                w_prop0 = np.zeros(self.n)
                w_prop1 = np.zeros(self.n)
                for i in range(self.n):
                    w_prop0[i], w_prop1[i] = self.Propensity(self.X[i, [1,2]]) ## remove sensitive feature A
                    w_prop0[i] = np.min((w_prop0[i], clip))
                    w_prop1[i] = np.min((w_prop1[i], clip))
                
                self.w_prop0 = w_prop0
                self.w_prop1 = w_prop1
        else:
            self.X = X[:, [0,1]]
            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]

        
    def PropensityFuncs(self):
        ## fit the P(A|M, R) model
        clf_mx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_mx.fit(self.X[:, [1, 2]], self.X[:, 0])
        ## fit the P(A|R) model
        clf_x = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_x.fit(self.X[:, 2].reshape(-1, 1), self.X[:, 0])
        return clf_mx, clf_x
    
    def Propensity(self, x):
        return 1.0 / self.clf_x.predict_proba([x[1:]])[0][0], 1.0 / self.clf_x.predict_proba([x[1:]])[0][1]


    def PropensityFuncs_extended(self):
        ## fit the P(M|A) model
        clf_m_a = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_m_a.fit(self.X[:, 0].reshape(-1, 1), self.X[:, 2].reshape(-1, 1))
        ## fit the P(R|A) model
        clf_r_a = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_r_a.fit(self.X[:, 0].reshape(-1, 1), self.X[:, 1].reshape(-1, 1))
        return clf_m_a, clf_r_a
    

    ## NDE A -> Y
    def Pse_ub(self, model, indices=None, fio=False):
        if self.extended is True:
            l1 = 0.0; l0 = 0.0; u1 = 0.0; u0 = 0.0
            sample_0 = np.array([0.0]).reshape(-1, 1)
            sample_1 = np.array([0.0]).reshape(-1, 1)
            for i_m in range(len(self.list_m_values)):
                p1_tmp = 0; p0_tmp = 0
                for i_r in range(len(self.list_r_values)):
                    p1_tmp += torch.exp( model( Variable(torch.Tensor([1, self.list_m_values[i_m], self.list_r_values[i_r]]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1] * self.clf_r_a.predict_proba(sample_1)[0][i_r]
                    p0_tmp += torch.exp( model( Variable(torch.Tensor([0, self.list_m_values[i_m], self.list_r_values[i_r]]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1] * self.clf_r_a.predict_proba(sample_0)[0][i_r]
                l1 += torch.max(torch.Tensor([0.0]), self.clf_m_a.predict_proba(sample_0)[0][i_m] - 1 + p1_tmp)
                u1 += torch.min(torch.Tensor([self.clf_m_a.predict_proba(sample_0)[0][i_m]]), p1_tmp)
                l0 += torch.max(torch.Tensor([0.0]), self.clf_m_a.predict_proba(sample_0)[0][i_m] - 1 + p0_tmp)
                u0 += torch.min(torch.Tensor([self.clf_m_a.predict_proba(sample_0)[0][i_m]]), p0_tmp)
            
            constr = (u1 * (1.0 - l0) + (1.0 - l1) * u0)
            constr = 2 * constr
            print("(l0, u0, l1, u1) = " + str(l0) + " " + str(u0) + " " + str(l1) + " " + str(u1))
            # sys.exit()

        else:
            p1 = 0.0; p0 = 0.0
            if indices is None:
                n = self.n ## = total number of data samples
                for i in range(n):
                    i_is_male = self.X[i, 0] 
                    i_is_female = (1 - self.X[i, 0])
                    p1 += i_is_male * self.w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1]
                    p0 += i_is_female * self.w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1]
            
            else:
                n = np.shape(indices)[0] ## = minibatch size
                for i in range(n):
                    i_is_male = self.X[indices[i], 0]
                    i_is_female = (1 - self.X[indices[i], 0])
                    p1 += i_is_male * self.w_prop1[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_1[indices[i], :]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1]
                    p0 += i_is_female * self.w_prop0[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_0[indices[i], :]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1]
                    
            p1 = p1 / n
            p0 = p0 / n
            ## check if marginal probabilities lies in [0.0, 1.0] (due to numerical error)
            p1 = torch.min(torch.Tensor([1.0]), p1)
            p0 = torch.min(torch.Tensor([1.0]), p0)
            

            constr = p1 * (1.0 - p0) + (1.0 - p1) * p0
            constr = 2 * constr
            
            
        return constr

    def EvaluatePIUandConditionalMean(self, model):
        n = self.n ## = total number of data samples
        prob1 = np.zeros(n)
        prob0 = np.zeros(n)
        self.X_1M0 = np.vstack((self.X_1[:, 0], self.M0, self.X[:, 1])).T
        self.X_0M0 = np.vstack((self.X_0[:, 0], self.M0, self.X[:, 1])).T

        Y1 = np.zeros(n); Y0 = np.zeros(n)        
        for i in range(n):
            prob1[i] = torch.exp( model( Variable(torch.Tensor(self.X_1M0[i, :]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1]
            prob0[i] = torch.exp( model( Variable(torch.Tensor(self.X_0M0[i, :]).reshape(-1, np.shape(self.X_1)[1])  ) ))[0][1]
            if prob1[i] >= 0.5:
                Y1[i] = 1
            if prob0[i] >= 0.5:
                Y0[i] = 1

        print("mean of prob1 and prob0")
        print(np.min(prob1))
        print(np.max(prob1))
        print(np.mean(prob1))
        print(np.mean(prob0))
        
        PIU = np.mean(np.abs(Y1 - Y0))
        
        ## to compute conditional mean unfair effect...
        # enumerate combinations of feature values
        fc = np.unique(self.X, axis=0)
        num_of_fc = fc.shape[0]
        condmean_array = np.zeros(num_of_fc)
        # belonging_subgroup = np.zeros(self.n)
        num_of_members = np.zeros(num_of_fc)
        
        for i in range(self.n):
            for j in range(num_of_fc):
                if len(np.where(self.X[i, :] == fc[j, :])[0]) == self.d: # if i-th individual feature is j-th feature combination
                    # belonging_subgroup[i] = j ## i-th individual belongs to j-th subgroup
                    num_of_members[j] += 1 ## num of members in j-th subgroup increases
                    condmean_array[j] += Y1[i] - Y0[i] ##

        print(num_of_members)
            
        #for j in range(num_of_fc):
        #    condmean_array[j] /= num_of_members[j]
        #ind_ = np.where(num_of_members > 1)[0]
        #return np.mean(np.abs(Y1 - Y0)), condmean_array[ind_]

        condmean_array_ = np.zeros(num_of_fc)        
        for j in range(num_of_fc):
            condmean_array_[j] = (num_of_members[j] * condmean_array[j])
            
        ind_ = np.where(num_of_members > 1)[0]
        mean_ = np.sum(condmean_array_) / self.n
        
        std_ = np.sqrt( np.sum( np.array([ num_of_members[j] * np.square( condmean_array[j] - mean_) for j in range(num_of_fc)]) ) / ((self.n - 1) * self.n)  )

        return np.mean(np.abs(Y1 - Y0)), condmean_array, mean_, std_




class Synth_Uncertain:
    def __init__(self, X, remove_sensitive):
        if remove_sensitive is False:
            self.X = X[:, 0:6] ## A, M, D, Q1, Q2, Q3 
            self.M0 = X[:, 7]
            self.D0 = X[:, 8]
            self.D1 = X[:, 9]
            self.Q30 = X[:, 10]

            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]
            X_1 = self.X.copy(); X_0 = self.X.copy() # prepare different object!
            X_1[:, 0] = np.ones(self.n)
            X_0[:, 0] = np.zeros(self.n)
            self.X_1 = X_1
            self.X_0 = X_0

            ### propensity weights for graph A
            wA_prop1 = np.zeros(self.n)
            wA_prop0 = np.zeros(self.n)
            self.clfA_mdx, self.clfA_dx, self.clfA_x = self.PropensityFuncsA()
            for i in range(self.n):
                wA_prop1[i], wA_prop0[i] = self.PropensityA(self.X[i, 1:]) # remove sensitive feature (M, D, Q1, Q2, Q3)  
            self.wA_prop1 = wA_prop1 
            self.wA_prop0 = wA_prop0 
            if flag_clip:
                for i in range(self.n):
                    self.wA_prop1[i] = np.min((self.wA_prop1[i], clip))
                    self.wA_prop0[i] = np.min((self.wA_prop0[i], clip))

           ### propensity weights for graph B
            wB_prop1 = np.zeros(self.n)
            wB_prop0 = np.zeros(self.n)
            self.clfB_mdx, self.clfB_dx, self.clfB_x = self.PropensityFuncsB() 
            for i in range(self.n):
                wB_prop1[i], wB_prop0[i] = self.PropensityB(self.X[i, 1:]) # remove sensitive feature (M, D, Q1, Q2, Q3)  
            self.wB_prop1 = wB_prop1 
            self.wB_prop0 = wB_prop0 
            if flag_clip:
                for i in range(self.n):
                    self.wB_prop1[i] = np.min((self.wB_prop1[i], clip))
                    self.wB_prop0[i] = np.min((self.wB_prop0[i], clip))
        else:
            self.X = X[:, [0,2,3]] ## M, Q1, Q2          
            self.n = np.shape(self.X)[0]
            self.d = np.shape(self.X)[1]
        
    def PropensityFuncsA(self): ### deleted clf_mx
        ## fit the P(A|M, D, (Q1, Q2, Q3)) model
        clf_mdx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_mdx.fit(self.X[:, [1, 2, 3, 4, 5]], self.X[:, 0]) ## (M, D, Q1, Q2, Q3), A
        ## fit the P(A|D, X) model
        clf_dx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_dx.fit(self.X[:, [2, 3, 4, 5]], self.X[:, 0]) ## (D, Q1, Q2, Q3), A
        ## fit the P(A|X) model
        clf_x = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_x.fit(self.X[:, [3, 4, 5]], self.X[:, 0]) ## (Q1, Q2, Q3), A
        return clf_mdx, clf_dx, clf_x
    
    def PropensityA(self, x):
        w_prop1 = (1.0 /  self.clfA_x.predict_proba([x[[2, 3, 4]]])[0][1]) * (self.clfA_dx.predict_proba([x[[1, 2, 3, 4]]])[0][1] / self.clfA_dx.predict_proba([x[[1, 2, 3, 4]]])[0][0]) * (self.clfA_mdx.predict_proba([x[[0, 1, 2, 3, 4]]])[0][0] / self.clfA_mdx.predict_proba([x[[0, 1, 2, 3, 4]]])[0][1])
        w_prop0 = 1.0 /  self.clfA_x.predict_proba([x[[2, 3, 4]]])[0][0] 
        return w_prop1, w_prop0
    
    def PropensityFuncsB(self):
        ## fit the P(A|M, D, (Q1, Q2, Q3)) model
        clf_mdx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_mdx.fit(self.X[:, [1, 2, 3, 4, 5]], self.X[:, 0]) ## (M, D, Q1, Q2, Q3), A
        ## fit the P(A|D, X) model
        clf_dx = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_dx.fit(self.X[:, [2, 3, 4]], self.X[:, 0]) ## (D, Q1, Q2, Q3), A
        ## fit the P(A|X) model
        clf_x = LogisticRegression(solver='lbfgs', max_iter=1000)
        clf_x.fit(self.X[:, [3, 4]], self.X[:, 0]) ## (Q1, Q2, Q3), A
        return clf_mdx, clf_dx, clf_x
    
    def PropensityB(self, x):
        w_prop1 = (1.0 /  self.clfB_x.predict_proba([x[[2, 3]]])[0][1]) * (self.clfB_dx.predict_proba([x[[1, 2, 3]]])[0][1] / self.clfB_dx.predict_proba([x[[1, 2, 3]]])[0][0]) * (self.clfB_mdx.predict_proba([x[[0, 1, 2, 3, 4]]])[0][0] / self.clfB_mdx.predict_proba([x[[0, 1, 2, 3, 4]]])[0][1])
        w_prop0 = 1.0 /  self.clfB_x.predict_proba([x[[2, 3]]])[0][0]
        return w_prop1, w_prop0       

    def Pse_ub(self, model, indices=None, fio=False, graph="B"): ###### self.w_prop1[i] --> (self.)w_prop1[indices[i]]; ipw_w1 = self.X[i, 0] --> i_is_male = self.X[indices[i], 0]
        p1 = 0.0; p0 = 0.0
        #clip = 5.0
        w_prop1 = self.wA_prop1 if graph == "A" else self.wB_prop1
        w_prop0 = self.wA_prop0 if graph == "A" else self.wB_prop0
        if indices is None:
            n = self.n ## = total number of data samples
            n1 = np.where(self.X[:, 0] == 1)[0].size
            n0 = np.where(self.X[:, 0] == 0)[0].size
            for i in range(n):
                i_is_male = self.X[i, 0] # = 1 if a_i = 1
                i_is_female = (1 - self.X[i, 0]) # = 1 if a_i = 0
                p1 += i_is_male * w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) ) [0][1]
                p0 += i_is_female * w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]                    
                 
        # use data indices for stochastic gradient descent method
        else:
            n = np.shape(indices)[0] ## = minibatch size
            n1 = np.where(self.X[indices, 0] == 1)[0].size
            n0 = np.where(self.X[indices, 0] == 0)[0].size
            for i in range(n):
                i_is_male = self.X[indices[i], 0] 
                i_is_female = (1 - self.X[indices[i], 0]) 
                p1 += i_is_male * w_prop1[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_1[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ))[0][1]
                p0 += i_is_female * w_prop0[indices[i]] * torch.exp( model( Variable(torch.Tensor(self.X_0[indices[i], :].reshape(-1, np.shape(self.X_1)[1]))  ) ))[0][1]                                        

        p1 = p1 / n
        p0 = p0 / n

        if fio is False:
            constr = 2 * ( p1 * (1.0 - p0) + (1.0 - p1) * p0 )
        else:
            if p1 > p0:
                constr = p1 - p0
            else:
                constr = p0 - p1
        return constr

    def Pse_indiv(self, model, graph="B"):
        n = self.n ## = total number of data samples
        prob1 = np.zeros(n)
        prob0 = np.zeros(n)
        pse = np.zeros(n)
        w_prop1 = self.wA_prop1 if graph == "A" else self.wB_prop1
        w_prop0 = self.wA_prop0 if graph == "A" else self.wB_prop0
        for i in range(n):
            i_is_male = self.X[i, 0] 
            i_is_female = (1 - self.X[i, 0]) 
            prob1[i] = i_is_male * w_prop1[i] * torch.exp( model( Variable(torch.Tensor(self.X_1[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()
            prob0[i] = i_is_female * w_prop0[i] * torch.exp( model( Variable(torch.Tensor(self.X_0[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1].detach().numpy()
            pse[i] = prob1[i] - prob0[i]
            

        prob1_mean = np.min((np.mean(prob1), 1.0))
        prob0_mean = np.min((np.mean(prob0), 1.0))
            
        print("mean of prob1 and prob0")
        print(str( np.mean(prob1) ) + " -> " + str(prob1_mean))
        print(str( np.mean(prob0) ) + " -> " + str(prob0_mean))
        
        print("mean: " + str(prob1_mean - prob0_mean ))
        print("ub: " + str(2 * ( (1 - prob0_mean) * prob1_mean + prob0_mean * (1 - prob1_mean) )))
            
        return np.array(pse)

    def EvaluatePIUandConditionalMean(self, model, H1=0):
        n = self.n ## = total number of data samples
        prob1 = np.zeros(n)
        prob0 = np.zeros(n)
        self.X_1D1M0Q30 = np.vstack((self.X_1[:, 0], self.M0, self.D1, self.X[:, 3], self.X[:, 4], self.Q30)).T
        self.X_0D0M0Q30 = np.vstack((self.X_0[:, 0], self.M0, self.D0, self.X[:, 3], self.X[:, 4], self.Q30)).T

        Y1 = np.zeros(n); Y0 = np.zeros(n)
        for i in range(n):
            prob1[i] = torch.exp( model( Variable(torch.Tensor(self.X_1D1M0Q30[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
            prob0[i] = torch.exp( model( Variable(torch.Tensor(self.X_0D0M0Q30[i, :].reshape(-1, np.shape(self.X_1)[1]))  ) ) )[0][1]
            if prob1[i] >= 0.5:
                Y1[i] = 1
            if prob0[i] >= 0.5:
                Y0[i] = 1

        PIU = np.mean(np.abs(Y1 - Y0))
        
        ## to compute conditional mean unfair effect...
        # enumerate combinations of feature values
        fc = np.unique(self.X, axis=0)
        num_of_fc = fc.shape[0]
        condmean_array = np.zeros(num_of_fc)
        # belonging_subgroup = np.zeros(self.n)
        num_of_members = np.zeros(num_of_fc)
        
        for i in range(self.n):
            for j in range(num_of_fc):
                if len(np.where(self.X[i, :] == fc[j, :])[0]) == self.d: # if i-th individual feature is j-th feature combination
                    # belonging_subgroup[i] = j ## i-th individual belongs to j-th subgroup
                    num_of_members[j] += 1 ## num of members in j-th subgroup increases
                    condmean_array[j] += Y1[i] - Y0[i] ##

        #print(num_of_members)

        condmean_array_ = np.zeros(num_of_fc)        
        for j in range(num_of_fc):
            condmean_array_[j] = (num_of_members[j] * condmean_array[j])
            
        ind_ = np.where(num_of_members > 1)[0]
        mean_ = np.sum(condmean_array_) / self.n
        
        std_ = np.sqrt( np.sum( np.array([ num_of_members[j] * np.square( condmean_array[j] - mean_) for j in range(num_of_fc)]) ) / ((self.n - 1) * self.n)  )

        return np.mean(np.abs(Y1 - Y0)), condmean_array, mean_, std_
      

