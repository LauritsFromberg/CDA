from tkinter import N
import numpy as np
from tqdm import tqdm
from sklearn import linear_model
import sys
import ast
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre
from data import *


# Signals to include
signals = ["EDA","TEMP","BVP","HR"]
window_size = [20,20,20,20]
window_shift = [5,5,5,5]
fs = [4,4,32,1]

# initialise memory
X_train = np.zeros((65,109)) # number of phases (for 13 subjects) x number of features
y_train = np.zeros((65,32)) # number of phases (for 13 subjects) x number of features
X_test = np.zeros((5,109)) # number of phases (for one subject) x number of features
y_test = np.zeros((5,32)) # number of phases x number of questions

# initialise number of cross-validation iterations
#n = 10
n = 1

# make list of random subjects to exclude in each iteration (without replacement)
s = np.arange(len(sub_lst))
np.random.shuffle(s)


# leave-one-patient-out cross-validation
for v in range(n):
    CV_idx = [np.array(sub_lst)[s[v]]] # pick random subject to exclude for testing 
    train_idx = [np.array(sub_lst)[np.arange(len(sub_lst))!=s[v]]][0] # use all other subjects for training 
    c_train = -1 # counter 
    c_test = -1 # counter
    for j in range(5): # for every phase
        
        # training-set
        for i in train_idx: # for every member of training
            c_train += 1 # update counter
            # feature extraction
            X_temp = [] # initialise memory
            for k in signals:
                count = 0
                if k == "EDA":
                    X_slope_tonic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_tonic)
                    X_stat_tonic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_tonic)
                    X_extra_tonic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_tonic)
                    X_slope_phasic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_phasic)
                    X_stat_phasic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_phasic)
                    X_extra_phasic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_phasic)
                    count += 1
                else:
                    X_slope = [*pre.slope_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_extra = [*pre.extra_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
            # assign values
            X_train[c_train,:] = X_temp # features
            y_train[c_train,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets

            # standardise 

        # test_set
        for i in CV_idx: # for every member of testing
            c_test += 1 # update counter
            # feature extraction 
            X_temp = [] # initialise memory
            for k in signals:
                count = 0
                if k == "EDA":
                    X_slope_tonic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_tonic)
                    X_stat_tonic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_tonic)
                    X_extra_tonic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_tonic)
                    X_slope_phasic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_phasic)
                    X_stat_phasic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_phasic)
                    X_extra_phasic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_phasic)
                    count += 1
                else:
                    X_slope = [*pre.slope_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_extra = [*pre.extra_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
            # assign values
            X_test[c_test,:] = X_temp # features
            y_test[c_test,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets


    # models
    print(X_train,y_train)





