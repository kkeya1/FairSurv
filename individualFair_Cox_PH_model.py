# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 00:12:20 2019

@author: Kamrun Naher Keya
"""
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#%% dataset pre-processing
from performance_measures import c_index, brier_score, weighted_c_index, weighted_brier_score,log_partial_lik
from neural_models import negLogLikelihood, linearCoxPH_Regression
from fairness_measures import individual_fairness, group_fairness, intersect_fairness

from sksurv.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import brier_score_loss
from sksurv.metrics import concordance_index_censored
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc
#%% linear Cox PH model in PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

#%%
from compute_survival_function import predict_survival_function  
#%%
#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

RANDOM_STATE = 1
set_random_seed(RANDOM_STATE)

#%% hinge loss to compute fairness loss
def criterionHinge(target_fairness,prediction,X_distance, scale):
    zeroTerm = torch.tensor(0.0)
    model_fairness = individual_fairness_Train(prediction,X_distance, scale)
    return torch.max(zeroTerm, (model_fairness-target_fairness))
#%% individual fairness as reguralizer for training Cox model
def individual_fairness_Train(prediction,X_distance, scale):
    HazardFunction = torch.exp(prediction)
    
    norm = X_distance.norm(p=2, dim=1, keepdim=True)
    X_distance = X_distance.div(norm)
    
    N = len(prediction)
    R_beta = torch.tensor(0.0)  #initialization of individual fairnessd  
    zeroTerm = torch.tensor(0.0)
    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j<=i:
                continue
            else:
                distance = torch.sqrt(torch.sum((X_distance[i]-X_distance[j])**2))
                R_beta = R_beta + torch.max(zeroTerm,(torch.abs(HazardFunction[i]-HazardFunction[j]) - scale*distance))
    R_beta_avg = R_beta/(N*(N-1))
    return R_beta_avg

#%%Individual fairness measure:
def individual_fairness_scale(prediction,X, scale):
    HazardFunction = np.exp(prediction)
    N = len(prediction)
    R_beta = 0.0 #initialization of individual fairnessd 
    for i in range(len(prediction)):
        for j in range(len(prediction)):
            if j<=i:
                continue
            else:
                distance = np.sqrt(sum((X[i]-X[j])**2)) # euclidean distance
                R_beta = R_beta + max(0,(np.abs(HazardFunction[i]-HazardFunction[j])-scale*distance))  
    R_beta_avg = R_beta/(N*(N-1))
    return R_beta_avg
    
#%% FLC data:
from utilities import prepare_data
from utilities import check_arrays_survival
from flc_data_preprocess import flc_preprocess
#Survival Data
data_x, data_y, protect_attr = flc_preprocess()

# train-test split
data_X_train, data_X_test, data_y_train, data_y_test, S_train, S_test = train_test_split(data_x, data_y, protect_attr, test_size=0.2,stratify=data_y["death"], random_state=7)
data_X_train, data_X_dev, data_y_train, data_y_dev, S_train, S_dev = train_test_split(data_X_train, data_y_train, S_train, test_size=0.2,stratify=data_y_train["death"], random_state=7)

data_X_train, data_event_train, data_time_train = check_arrays_survival(data_X_train, data_y_train)

data_X_test, data_event_test, data_time_test = check_arrays_survival(data_X_test, data_y_test)
data_X_test, data_event_test, data_time_test, S_test = prepare_data(data_X_test, data_event_test, data_time_test, S_test)
#
intersectionalGroups = np.unique(S_train,axis=0) # all intersecting groups, i.e. black-women, white-man etc 
# data normalization: mean subtraction method to compute euclidean distance
scaler = StandardScaler()
scaler.fit(data_X_train)
data_X_train = scaler.transform(data_X_train)
data_X_test = scaler.transform(data_X_test)

#%%
# hyperparameters of the model
input_size = data_X_train.shape[1]
output_size = 1
learning_rate = 0.01  
num_epochs = 25
mini_batch = 128 

target_fairness = torch.tensor(0.0)
#lambda_list = torch.tensor([0.1, 1.0])

scale = torch.tensor(0.01)
lamda = torch.tensor(1.0) # chosen by grid search on dev set

data_X_test = Variable((torch.from_numpy(data_X_test)).float())

import sys
sys.stdout=open("individual_fair_cox_ph_output_scale.txt","w")

    
#%% intialize model and optimizar
# initialize cox PH model   
coxPH_model = linearCoxPH_Regression(input_size,output_size)
# Loss and optimizer
criterion = negLogLikelihood()
optimizer = optim.Adam(coxPH_model.parameters(),lr = learning_rate) # adam optimizer
#optimizer = optim.SGD(coxPH_model.parameters(),lr = learning_rate) # SGD optimizer

#%% training cox ph model    

  
for epoch in range(num_epochs):   
    for batch in range(0,np.int64(np.floor(len(data_X_train)/mini_batch))*mini_batch,mini_batch):       
        # backward propagation
        X_train_batch = data_X_train[batch:(batch+mini_batch)] # protected attributes in the mini-batch
        event_train_batch = data_event_train[batch:(batch+mini_batch)]
        time_train_batch = data_time_train[batch:(batch+mini_batch)]
        S_train_batch = S_train[batch:(batch+mini_batch)]
        
        # Sort Training Data for Accurate Likelihood
        X_train_batch, event_train_batch, time_train_batch, S_train_batch  = prepare_data(X_train_batch, event_train_batch, time_train_batch,S_train_batch)
        X_train_batch_for_distance = scaler.transform(X_train_batch)
        
        X_train_batch = Variable((torch.from_numpy(X_train_batch)).float())
        event_train_batch = Variable((torch.from_numpy(event_train_batch)).float()) 
        X_train_batch_for_distance = Variable((torch.from_numpy(X_train_batch_for_distance)).float()) 
        
        outputs = coxPH_model(X_train_batch)
        logLike_loss = criterion(outputs, event_train_batch) # loss between prediction and target
        
        # fairness constraint
        R_loss = criterionHinge(target_fairness,outputs,X_train_batch_for_distance, scale)
        
        total_loss = logLike_loss + lamda*R_loss
        
        optimizer.zero_grad() # zero the parameter gradients
        total_loss.backward()
        optimizer.step()
    #print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss.item()))

#%% Evaluate the model

#Measuring the Performance of Survival Models
# linear predictor for train data
with torch.no_grad():
    data_X_train = Variable((torch.from_numpy(data_X_train)).float())
    base_prediction = coxPH_model(data_X_train)
    base_prediction = (base_prediction.numpy()).reshape((-1,)) # beta \dot x

data_X_train = data_X_train.numpy()
# linear predictor for test/dev data
with torch.no_grad():
    model_prediction = coxPH_model(data_X_test)
    model_prediction = (model_prediction.numpy()).reshape((-1,)) # beta \dot x

skSurv_result_test = concordance_index_censored(data_event_test, data_time_test, model_prediction)
print(f"skSurv implemented C-index for test data: {skSurv_result_test[0]: .4f}")
    
eval_time = [int(np.percentile(data_time_train, 25)), int(np.percentile(data_time_train, 50)), int(np.percentile(data_time_train, 75))] 
tmp_br_score = np.zeros(len(eval_time)) 
    
#%%
survFunction_test = predict_survival_function(base_prediction, data_event_train, data_time_train, model_prediction)
for t in range(len(eval_time)):
    cif_test = np.zeros((len(data_X_test)))
    for i in range(len(data_X_test)):
        time_point = survFunction_test[i].x
        probs = survFunction_test[i].y
        index=np.where((time_point==eval_time[t]))[0][0]
        cif_test[i] = 1 - probs[index]
        
    tmp_br_score[t] = weighted_brier_score(data_time_train, data_event_train, cif_test, data_time_test, data_event_test, eval_time[t])

weighted_br_score = np.mean(tmp_br_score)
print(f"weighted brier score: {weighted_br_score: .4f}")
# =============================================================================
#%% Time-dependent Area under the ROC

survival_train=np.dtype([('event',data_event_train.dtype),('surv_time',data_time_train.dtype)])
survival_train=np.empty(len(data_event_train),dtype=survival_train)
survival_train['event']=data_event_train
survival_train['surv_time']=data_time_train

survival_test=np.dtype([('event',data_event_test.dtype),('surv_time',data_time_test.dtype)])
survival_test=np.empty(len(data_event_test),dtype=survival_test)
survival_test['event']=data_event_test
survival_test['surv_time']=data_time_test

event_times = np.arange(np.min(data_time_test), np.max(data_time_test)/2, 75)

test_auc, test_mean_auc = cumulative_dynamic_auc(survival_train, survival_test, model_prediction, event_times)

print(f"Time-dependent Area under the ROC: {test_mean_auc: .4f}")

plt.plot(event_times, test_auc, marker="o")
plt.axhline(test_mean_auc, linestyle="--")
plt.xlabel("Days from Enrollment")
plt.ylabel("Time-dependent Area under the ROC")
plt.grid(True)
plt.savefig('individualFair_auc.png',dpi = 600)

#%% log -partial likelihood
log_lik = log_partial_lik(model_prediction.reshape(-1,1), data_event_test.reshape(-1,1))
print(f"Log partial likelihood: {log_lik: .4f}")

# #%% fairness measures
data_X_test_for_distance = data_X_test.numpy()
data_X_test_for_distance = data_X_test_for_distance / np.linalg.norm(data_X_test_for_distance,axis=1,keepdims=1)

scale_measure = 0.01
R_beta_scale = individual_fairness_scale(model_prediction,data_X_test_for_distance, scale_measure)
print(f"average individual fairness metric with scale={scale_measure: .4f}: {R_beta_scale: .4f}")

#%% group fairness measures - age
S_age = S_test[:,0] # age is in the 1st column

group_fairness_age = group_fairness(model_prediction,S_age)
print(f"group fairness metric (for age): {group_fairness_age: .4f}")


#%% intersectional fairness measures
epsilon = intersect_fairness(model_prediction,S_test, intersectionalGroups)
print(f"intersectional fairness metric: {epsilon: .4f}")


#%% save the model
torch.save(coxPH_model.state_dict(), "trained-models/Individual_Fair_CoxPH")