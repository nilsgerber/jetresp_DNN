#!/usr/bin/python

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
import uproot
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import time
from pickle import dump,load
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve,roc_auc_score, auc

import hyperopt
from hyperopt import fmin, tpe, hp, space_eval,Trials

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
import random
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib as mpl
from scipy import optimize
import math
import time
import sys
from scipy.stats import ks_2samp
from sklearn.linear_model import Ridge
from sklearn.model_selection import validation_curve
from pickle import dump,load
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
scaler = load(open('scaler.pkl', 'rb'))
#model = load(open('model.pkl', 'rb'))

class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)    
    
    
class TrainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

    
class BinaryClassification(nn.Module):
    def __init__(self):
        super(BinaryClassification, self).__init__()
        # Number of input features is 3.
        self.layer_1 = nn.Linear(3, 128) 
        self.layer_2 = nn.Linear(128,1024)
        self.layer_3 = nn.Linear(1024, 8)
        #self.layer_4 = nn.Linear(4, 8)
        self.layer_out = nn.Linear(8, 1) 
        
        self.relu = nn.ReLU() #Leaky?
        #self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.0003)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(8)
        #self.batchnorm4 = nn.BatchNorm1d(8)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)   
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        #x = self.relu(self.layer_4(x))
        #x = self.batchnorm4(x)
        x = self.dropout(x)
        #x = self.sigmoid(self.layer_out(x))
        x = self.layer_out(x)
        
        return x
    
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    #y_pred_tag = torch.round(y_pred)

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc    

def get_Data(input_data_path = '/home/gerberni/data/', input_data_filename = 'SkimSusyTesting.root'): #gets the root data for further use, max defualt is Length = 586302
    file = uproot.open(input_data_path  + input_data_filename)
    tree = file["tJet"]
    data= tree.arrays(["GenJetPt", "GenJetEta", "JetResponse_FullSim"], library = "np") 
    df = pd.DataFrame(data=data, columns = ["GenJetPt", "GenJetEta", "JetResponse_FullSim"])
    df = df.drop(df[df.JetResponse_FullSim < 0].index)
    df = df.abs()
    df = df.to_numpy()
    #for i in range(0,df.shape[0]):
        #df[i][0] = df[i][0] / 2000
        #df[i][1] = df[i][1] / 2.5
    #df = df.iloc[:Length,:] only needed if smaller dataset is desired ,uses the first 'Length' entries
    return df

def get_Data_ttbar(input_data_path = './data/', input_data_filename = 'SkimTtbarTesting.root'): #gets the root data for further use, max defualt is Length = 586302
    file = uproot.open(input_data_path  + input_data_filename)
    tree = file["tJet"]
    data= tree.arrays(["GenJetPt", "GenJetEta", "JetResponse_FullSim"], library = "np") 
    df = pd.DataFrame(data=data, columns = ["GenJetPt", "GenJetEta", "JetResponse_FullSim"])
    df = df.drop(df[df.JetResponse_FullSim < 0].index)
    df = df.abs()
    df = df.to_numpy()
    #for i in range(0,df.shape[0]):
        #df[i][0] = df[i][0] / 2000
        #df[i][1] = df[i][1] / 2.5
    #df = df.iloc[:Length,:] only needed if smaller dataset is desired ,uses the first 'Length' entries
    return df

def get_Data_FS(input_data_path = './data/', input_data_filename = 'SkimSusyTesting.root'): #gets the root data for further use, max defualt is Length = 586302
    file = uproot.open(input_data_path  + input_data_filename)
    tree = file["tJet"]
    data= tree.arrays(["GenJetPt", "GenJetEta","JetResponse_FastSim"], library = "np") 
    df = pd.DataFrame(data=data, columns = ["GenJetPt", "GenJetEta","JetResponse_FastSim"])
    df = df.drop(df[df.JetResponse_FastSim < 0].index)
    df = df.abs()
    df = df.to_numpy()
    return df

def GetDNNData(input_data_path = '/home/gerberni/data/', input_data_filename = 'SkimSusyTraining.root'):
    file = uproot.open(input_data_path  + input_data_filename)
    tree = file["tJet"]
    data= tree.arrays(["GenJetPt", "GenJetEta", "JetResponse_FullSim"], library = "np") 
    df = pd.DataFrame(data=data)
    df = df.drop(df[df.JetResponse_FullSim < 0].index) #Drop JetPT < 0 (also drops jetresp < 0


    df_np = df.to_numpy()
    df_np_bg = np.copy(df_np)  
    for i in range(0,df_np.shape[0]):
       # df_np[i][2] = df_np[i][2] /  df_np[i][0]    #Preprocess Jetresponse
        df_np_bg[i][2] =  random.uniform(0.0, 3.0)  #generate Background
        #df_np_bg[i][2] = np.random.normal(1, 0.13) # dont do it

    df_np_s = np.c_[ df_np, np.ones(df_np.shape[0], dtype = int)]  #Signal data
    df_np_bg = np.c_[df_np_bg, np.zeros(df_np_bg.shape[0], dtype = int)] #bg data

    df_np_SandB = np.vstack((df_np_s, df_np_bg))
    df = pd.DataFrame(data=df_np_SandB, columns =["GenJetPt", "GenJetEta", "JetResponse", "Signal"]) #Back into the Dataframe
    df = df.abs() #Take absolute value of ETA
    return df

    

    
def Check_Whole(Array): #this is vectorized and much faster (~10^3) than single checks
    df = pd.DataFrame(data=Array, columns = ["GenJetPt", "GenJetEta", "RecJetPt_FullSim"])
    test_data = torch.FloatTensor(scaler.transform(df))
    with torch.no_grad():
        test_data = test_data.to(device)
        y_test_pred = model(test_data)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_test = y_test_pred.tolist()
    return y_test

def ArraySampleSubstep(df): #substep of ArraySample100
    df = df[:,:-1] 
    guesses = 3*np.random.random(size=(df.shape[0],1))
    df = np.append(df, guesses, axis=1)
    #print("df: ",df)
    y = np.array(Check_Whole(df))
    y = y/(1-y)
    #if you want: replace the 20 with max(y), might lead the last ~50(?) events being missampled, will be faster (~2 to 10 times) though
    sampling_y = 20*np.random.random(size=(df.shape[0],1)) #14?
    ok = ( sampling_y < y )
    ok = ok.ravel()
    df_y = df[ok]
    #print("y:",df_y)
    df_rest = df[~ok]
    #print("rest:",df_rest)
    return df_y,df_rest

def ArraySample100(): #generates a jetresponse for every Event in the data, doesnt return FullSim values
    model.eval()
    df = get_Data()  
    #df,df_bg = get_test()
    df_1, df = ArraySampleSubstep(df)
    df_end = df_1
    empty_runs = 0
    cut = 1000
    while True:
        #print("yeet")
        #print(df.shape)
        start = time.time()
        old = len(df)
        df_1, df = ArraySampleSubstep(df)
        if old == len(df):
            empty_runs += 0
        else:
            empty_runs = 0
        end = time.time()
        
        #print(df_1,df_1.shape)
        #print(df,df.shape)
        df_end = np.concatenate((df_end, df_1), axis=0)
        if df.shape[0] == 0 or empty_runs == cut:
            if empty_runs == cut:
                print("abbruch",df.shape[0])
                return df_end,df
            else:
                return df_end,0
        
def RatioPlottingResp(df_sig): #gets own background data, needs signal data (such as from ArraySample100
    #for i in range(len(df_bg)):
    #       df_bg[i][2] = df_bg[i][2] / df_bg[i][0]
    df_bg = get_Data()
    df_bg = df_bg[:, -1]
    df_sig = df_sig[:, -1]
    print(len(df_sig),len(df_sig)-len(df_bg))

    bins = np.linspace(0, 3, num=50)
    bincenters = np.multiply(0.5, bins[1:] + bins[:-1])
    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].set_title("Jetresp sampling NN vs FullSim - pt[15,3300] GeV, eta[0,2.5]")
    #axes[0].set_yscale("log")
    bincontent1, bins1, patches1 = axes[0].hist(df_bg , bins = bins, alpha = 0.5, label = 'FullSim')
    bincontent2, bins2, patches2 = axes[0].hist(df_sig, bins = bins,edgecolor='r', alpha = 0.5, label = 'NN Simulated',hatch='/',fill=False)


    mask = np.logical_and(bincontent1 != 0, bincontent2 != 0)
    ratio = np.divide(bincontent2, bincontent1, where=mask)

    error = ratio * np.sqrt(np.divide(1, bincontent1, where=mask) + np.divide(1, bincontent2, where=mask))

    axes[1].errorbar(bincenters, ratio, yerr=error, fmt='.', color='red',alpha = 0.7)
    axes[1].axhline(1, color='black')

    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel('Fullsim/DNN')
    axes[0].legend(loc='best')
    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])
    plt.savefig(Path + '/figures/Array100Full.png', bbox_inches='tight')
    plt.show()
    
def Sample1(pt, eta, model, xmin=0, xmax=3, ylow = 0, ymax = 20):
    while True:
        x = np.random.uniform(low=xmin, high=xmax)
       # y = Normal(x, mu=1, sig=0.13)
        y = np.random.uniform(low=ylow, high=ymax)
        if y < Check(x, pt, eta, model):
            #print("checks",y,Check(x, pt, eta))
            return x
    
def Check(jetresp,pt,eta):
    df_np = np.array([[pt,eta,jetresp]])
    df_C = pd.DataFrame(data=df_np, columns = ["GenJetPt", "GenJetEta", "RecJetPt_FullSim"])
    test_data = torch.FloatTensor(scaler.transform(df_C))
    with torch.no_grad():
        test_data = test_data.to(device)
        y_test_pred = model(test_data)
        y = y_test_pred.tolist()[0][0]
    return(y)

def RatioPlottingResp_FastSim(df_sig): #gets own background data, needs signal data (such as from ArraySample100
    #for i in range(len(df_bg)):
    #       df_bg[i][2] = df_bg[i][2] / df_bg[i][0]
    df_bg = get_Data()
    df_fs = get_Data_FS()[:, -1]
    df_bg = df_bg[:, -1]
    df_sig = df_sig[:, -1]
    print(len(df_sig),len(df_sig)-len(df_bg))

    bins = np.linspace(0, 3, num=50)
    bincenters = np.multiply(0.5, bins[1:] + bins[:-1])
    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].set_title("Jetresp sampling NN vs FullSim - pt[15,3300] GeV, eta[0,2.5]")
    axes[0].set_yscale("log")
    bincontent1, bins1, patches1 = axes[0].hist(df_bg , bins = bins, alpha = 0.5, label = 'FullSim')
    bincontent3, bins3, patches3 = axes[0].hist(df_fs , bins = bins, alpha = 0.5, label = 'FastSim')
    bincontent2, bins2, patches2 = axes[0].hist(df_sig, bins = bins,edgecolor='r', alpha = 0.5, label = 'NN Simulated',hatch='/',fill=False)



    mask = np.logical_and(bincontent1 != 0, bincontent2 != 0)
    ratio = np.divide(bincontent2, bincontent1, where=mask)
    mask2 = np.logical_and(bincontent1 != 0, bincontent3 != 0)
    ratio2 = np.divide(bincontent3, bincontent1, where=mask2)

    error = ratio * np.sqrt(np.divide(1, bincontent1, where=mask) + np.divide(1, bincontent2, where=mask))
    error2 = ratio2 * np.sqrt(np.divide(1, bincontent1, where=mask2) + np.divide(1, bincontent3, where=mask2))

    axes[1].errorbar(bincenters, ratio, yerr=error, fmt='.', color='red',alpha = 0.7, label = 'NN')
    #print(bincenters, ratio, error)
    #print(bincenters, ratio2, error2)
    axes[1].errorbar(bincenters, ratio2, yerr=error2, fmt='.', color='blue',alpha = 0.7, label = 'FS')
    axes[1].axhline(1, color='black')

    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel('Fullsim/DNN')
    axes[0].legend(loc='best')
    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])
    plt.savefig(Path + '/figures/Array100FScomp.png', bbox_inches='tight')
    plt.show()
    
def RatioPlottingResp_ttbar(df_sig): #gets own background data, needs signal data (such as from ArraySample100
    #for i in range(len(df_bg)):
    #       df_bg[i][2] = df_bg[i][2] / df_bg[i][0]
    df_bg = get_Data_ttbar()
    df_bg = df_bg[:, -1]
    df_sig = df_sig[:, -1]
    print(len(df_sig),len(df_sig)-len(df_bg))

    bins = np.linspace(0, 3, num=50)
    bincenters = np.multiply(0.5, bins[1:] + bins[:-1])
    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].set_title("Jetresp sampling NN vs FullSim - pt[15,3300] GeV, eta[0,2.5]")
    axes[0].set_yscale("log")
    bincontent1, bins1, patches1 = axes[0].hist(df_bg , bins = bins, alpha = 0.5, label = 'FullSim')
    bincontent2, bins2, patches2 = axes[0].hist(df_sig, bins = bins,edgecolor='r', alpha = 0.5, label = 'NN Simulated',hatch='/',fill=False)


    mask = np.logical_and(bincontent1 != 0, bincontent2 != 0)
    ratio = np.divide(bincontent2, bincontent1, where=mask)

    error = ratio * np.sqrt(np.divide(1, bincontent1, where=mask) + np.divide(1, bincontent2, where=mask))

    axes[1].errorbar(bincenters, ratio, yerr=error, fmt='.', color='red',alpha = 0.7)
    axes[1].axhline(1, color='black')

    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel('Fullsim/DNN')
    axes[0].legend(loc='best')
    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])
    plt.savefig(Path + '/figures/Array100Full_ttbar.png', bbox_inches='tight')
    plt.show()    
    
def ArraySample100_ttbar(): #generates a jetresponse for every Event in the data, doesnt return FullSim values
    model.eval()
    df = get_Data_ttbar()  
    #df,df_bg = get_test()
    df_1, df = ArraySampleSubstep(df)
    df_end = df_1
    empty_runs = 0
    cut = 1000
    while True:
        #print("yeet")
        #print("left to sample: ", df.shape[0])
        start = time.time()
        old = len(df)
        df_1, df = ArraySampleSubstep(df)
        if old == len(df):
            empty_runs += 0
        else:
            empty_runs = 0
        end = time.time()
        
        #print(df_1,df_1.shape)
        #print(df,df.shape)
        df_end = np.concatenate((df_end, df_1), axis=0)
        if df.shape[0] == 0 or empty_runs == cut:
            if empty_runs == cut:
                print("abbruch",df.shape[0])
                return df_end,df
            else:
                return df_end,0

            

def RatioPlottingResp_pt(df_sig): #gets own background data, needs signal data (such as from ArraySample100
    #for i in range(len(df_bg)):
    #       df_bg[i][2] = df_bg[i][2] / df_bg[i][0]
    df_bg = get_Data()
    df_bg = df_bg[:,0]*df_bg[:, -1]
    df_sig = df_sig[:,0]*df_sig[:, -1]
    print(len(df_sig),len(df_sig)-len(df_bg))

    bins = np.linspace(0, 3300, num=50)
    bincenters = np.multiply(0.5, bins[1:] + bins[:-1])
    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].set_title("Jetresp sampling pt diff NN vs FullSim - pt[15,3300] GeV, eta[0,2.5]")
    axes[0].set_yscale("log")
    bincontent1, bins1, patches1 = axes[0].hist(df_bg , bins = bins, alpha = 0.5, label = 'FullSim')
    bincontent2, bins2, patches2 = axes[0].hist(df_sig, bins = bins,edgecolor='r', alpha = 0.5, label = 'NN Simulated',hatch='/',fill=False)


    mask = np.logical_and(bincontent1 != 0, bincontent2 != 0)
    ratio = np.divide(bincontent2, bincontent1, where=mask)

    error = ratio * np.sqrt(np.divide(1, bincontent1, where=mask) + np.divide(1, bincontent2, where=mask))

    axes[1].errorbar(bincenters, ratio, yerr=error, fmt='.', color='red',alpha = 0.7)
    axes[1].axhline(1, color='black')

    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel('Fullsim/DNN')
    axes[0].legend(loc='best')
    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])
    plt.savefig(Path + '/figures/Array100Full_pt.png', bbox_inches='tight')
    plt.show()            
    
def RatioPlottingResp_FastSim_pt(df_sig): #gets own background data, needs signal data (such as from ArraySample100
    #for i in range(len(df_bg)):
    #       df_bg[i][2] = df_bg[i][2] / df_bg[i][0]
    df_bg = get_Data()
    df_fs = get_Data_FS()
    df_fs = df_fs[:,0]*df_fs[:, -1]
    df_bg = df_bg[:,0]*df_bg[:, -1]
    df_sig = df_sig[:,0]*df_sig[:, -1]
    print(len(df_sig),len(df_sig)-len(df_bg))

    bins = np.linspace(0, 3, num=50)
    bincenters = np.multiply(0.5, bins[1:] + bins[:-1])
    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].set_title("Jetresp sampling NN vs FullSim - pt[15,3300] GeV, eta[0,2.5]")
    axes[0].set_yscale("log")
    bincontent1, bins1, patches1 = axes[0].hist(df_bg , bins = bins, alpha = 0.5, label = 'FullSim')
    bincontent3, bins3, patches3 = axes[0].hist(df_fs , bins = bins, alpha = 0.5, label = 'FastSim')
    bincontent2, bins2, patches2 = axes[0].hist(df_sig, bins = bins,edgecolor='r', alpha = 0.5, label = 'NN Simulated',hatch='/',fill=False)



    mask = np.logical_and(bincontent1 != 0, bincontent2 != 0)
    ratio = np.divide(bincontent2, bincontent1, where=mask)
    mask2 = np.logical_and(bincontent1 != 0, bincontent3 != 0)
    ratio2 = np.divide(bincontent3, bincontent1, where=mask2)

    error = ratio * np.sqrt(np.divide(1, bincontent1, where=mask) + np.divide(1, bincontent2, where=mask))
    error2 = ratio2 * np.sqrt(np.divide(1, bincontent1, where=mask2) + np.divide(1, bincontent3, where=mask2))

    axes[1].errorbar(bincenters, ratio, yerr=error, fmt='.', color='red',alpha = 0.7, label = 'NN')
    #print(bincenters, ratio, error)
    #print(bincenters, ratio2, error2)
    axes[1].errorbar(bincenters, ratio2, yerr=error2, fmt='.', color='blue',alpha = 0.7, label = 'FS')
    axes[1].axhline(1, color='black')

    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel('Fullsim/DNN')
    axes[0].legend(loc='best')
    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])
    plt.savefig(Path + '/figures/Array100FScomp.png', bbox_inches='tight')
    plt.show()
    
def RatioPlottingResp_pt_ttbar(df_sig): #gets own background data, needs signal data (such as from ArraySample100
    #for i in range(len(df_bg)):
    #       df_bg[i][2] = df_bg[i][2] / df_bg[i][0]
    df_bg = get_Data_ttbar()
    df_bg = df_bg[:,0]*df_bg[:, -1]
    df_sig = df_sig[:,0]*df_sig[:, -1]
    print(len(df_sig),len(df_sig)-len(df_bg))

    bins = np.linspace(0, 3300, num=50)
    bincenters = np.multiply(0.5, bins[1:] + bins[:-1])
    f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    axes[0].set_title("Jetresp sampling pt diff NN vs FullSim on ttbar - pt[15,3300] GeV, eta[0,2.5]")
    axes[0].set_yscale("log")
    bincontent1, bins1, patches1 = axes[0].hist(df_bg , bins = bins, alpha = 0.5, label = 'FullSim')
    bincontent2, bins2, patches2 = axes[0].hist(df_sig, bins = bins,edgecolor='r', alpha = 0.5, label = 'NN Simulated',hatch='/',fill=False)


    mask = np.logical_and(bincontent1 != 0, bincontent2 != 0)
    ratio = np.divide(bincontent2, bincontent1, where=mask)

    error = ratio * np.sqrt(np.divide(1, bincontent1, where=mask) + np.divide(1, bincontent2, where=mask))

    axes[1].errorbar(bincenters, ratio, yerr=error, fmt='.', color='red',alpha = 0.7)
    axes[1].axhline(1, color='black')

    axes[1].set_ylim(0.5, 1.5)
    axes[1].set_ylabel('Fullsim/DNN')
    axes[0].legend(loc='best')
    for ax in axes:
        ax.set_xlim(bins[0], bins[-1])
    plt.savefig(Path + '/figures/Array100Full_pt_ttbar.png', bbox_inches='tight')
    plt.show()                        
            
            
Path = '/home/gerberni/Models/SUSY_Skimmed/'

train_loader = torch.load(Path + 'train_loader.pth')
val_loader = torch.load(Path + 'val_loader.pth')

model = load(open(Path + '5E_model.pkl', 'rb'))
scaler = load(open(Path + '5E_scaler.pkl', 'rb'))

loss_values = load(open(Path + 'loss_values.pkl', 'rb'))
val_loss = load(open(Path + 'val_loss.pkl', 'rb'))
acc_values = load(open(Path + 'acc_values.pkl', 'rb'))
val_values= load(open(Path + 'val_values.pkl', 'rb'))
val_values= load(open(Path + 'val_values.pkl', 'rb'))


s,Müll = ArraySample100()
print(s,Müll)
RatioPlottingResp(s)
RatioPlottingResp_pt(s)
RatioPlottingResp_FastSim(s)

s_ttbar, Müll2 = ArraySample100_ttbar()
RatioPlottingResp_ttbar(s_ttbar)
RatioPlottingResp_pt_ttbar(s_ttbar)


pt_low = {'1' :15,'2' :100,'3' :500}
pt_high ={'1' :100,'2' :500,'3' :3300}
eta_low = {'1' :0,'2' :0.5,'3' :1}
eta_high = {'1' :0.5,'2' :1,'3':2.5}


df_data = get_Data()
df_sample,temp = ArraySample100()
#print(df_sample,df_data)

for key in pt_low:
    for jey in eta_low:

        df_sig = pd.DataFrame(data=df_sample, columns = ["GenJetPt", "GenJetEta", "RecJetPt_FullSim"])
        df_bg = pd.DataFrame(data=df_data, columns = ["GenJetPt", "GenJetEta", "RecJetPt_FullSim"])
        df_sig = df_sig.drop(df_sig[df_sig.GenJetPt < pt_low[key]].index)
        df_sig = df_sig.drop(df_sig[df_sig.GenJetPt > pt_high[key]].index)
        df_sig = df_sig.drop(df_sig[df_sig.GenJetEta < eta_low[jey]].index)
        df_sig = df_sig.drop(df_sig[df_sig.GenJetEta > eta_high[jey]].index)
        df_sig = df_sig.to_numpy()
        df_sig = df_sig[:, -1]
        df_bg = df_bg.drop(df_bg[df_bg.GenJetPt < pt_low[key]].index)
        df_bg = df_bg.drop(df_bg[df_bg.GenJetPt > pt_high[key]].index)
        df_bg = df_bg.drop(df_bg[df_bg.GenJetEta < eta_low[jey]].index)
        df_bg = df_bg.drop(df_bg[df_bg.GenJetEta > eta_high[jey]].index)
        df_bg = df_bg.to_numpy()
        df_bg = df_bg[:,-1]

        print(len(df_sig),len(df_sig)-len(df_bg))


        bins = np.linspace(0, 3, num=50)
        bincenters = np.multiply(0.5, bins[1:] + bins[:-1])

        f, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
        axes[0].set_title("Jet response sampling NN vs FullSim - pt["+str(pt_low[key])+" GeV, "+str(pt_high[key])+" GeV], eta["+str(eta_low[jey])+","+str(eta_high[jey])+"]")

        bincontent1, _, _ = axes[0].hist(df_bg , bins = bins, alpha = 0.5, stacked = 1, label = 'FullSim')
        bincontent2, _, _ = axes[0].hist(df_sig, bins = bins,edgecolor='r', alpha = 0.5, stacked = 1, label = 'NN',hatch='/',fill=False)

        mask = np.logical_and(bincontent1 != 0, bincontent2 != 0)
        ratio = np.divide(bincontent2, bincontent1, where=mask)
        error = ratio * np.sqrt(np.divide(1, bincontent1, where=mask) + np.divide(1, bincontent2, where=mask))

        axes[1].errorbar(bincenters, ratio, yerr=error, fmt='.', color='red')
        axes[1].axhline(1, color='black')

        axes[1].set_ylim(0.5, 1.5)
        axes[1].set_ylabel('NN/Fullsim')
        axes[0].legend(loc='best')
        axes[0].set_ylabel('Jets')
        axes[0].set_yscale("log")
        axes[1].set_xlabel('Jet response R')
        for ax in axes:
            ax.set_xlim(bins[0], bins[-1])
        plt.savefig(Path + '/figures/jetresp NN vs FS - pt = '+str(pt_low[key])+', '+str(pt_high[key])+', eta = '+str(eta_low[jey])+', '+str(eta_high[jey])+' - log.png', bbox_inches='tight')
        plt.show()    
 





