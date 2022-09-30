import sys
import pickle
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils_neurokit2 as pre

## compute feature importance

# train best model on all individals 

# load model parameters
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_param_neurokit2.pkl"
file = open(filename,"rb")
param = pickle.load(file)
file.close()

# load dataset
filename = "C:/Users/Bruger/Documents/CDA/CDA/data/dataset_neurokit2.pkl"
file = open(filename,"rb")
dataset = pickle.load(file)
file.close()
# load questions
filename = "C:/Users/Bruger/Documents/CDA/CDA/data/quest_neurokit2.pkl"
file = open(filename,"rb")
quest = pickle.load(file)
file.close()

# define model
model = RandomForestRegressor(random_state=50317,**param)

# subjects
sub_lst = ["S10","S11","S13","S14","S15","S16","S17","S2","S3","S4","S6","S7","S8"]
test_idx = ["S9"]

# emotional states (versions of study-protocol)
cond = ["base","stress","amusement","meditation_1","meditation_2"]

# Signals to include and each respective window-size, window-shift as well as sample-rate
signals = ["EDA","Tonic","Phasic","TEMP","BVP","HR"]
window_size = [40,40,40,40,40,40] 
window_shift = [25,25,25,25,25,25]
fs = [4,4,4,4,64,1]

# initialise burn-in period
t = 5

# initialise memory
y_train = np.zeros((65,24)) # number of phases (for 13 subjects) x number of questions
y_test = np.zeros((5,24)) # number of phases x number of questions

# counters
c_train = -1 
c_test = -1 

df_train = pd.DataFrame()
df_test = pd.DataFrame()

# for every phase
for j in range(5): 
        
    # train set
    for i in sub_lst: # for every member of training
        c_train += 1 # update counter

        # feature extraction
        X_time = pre.time_features(dataset[i][cond[j]],window_size,window_shift,fs)
        X_freq = pre.freq_features(dataset[i][cond[j]],fs)
        X_extra = pre.EDA_extra_features(dataset[i][cond[j]],fs)
        X = pd.concat([X_time,X_freq,X_extra],axis=1)
        # assign values
        df_train = pd.concat([df_train,X],axis=0)  # features 
        y_train[c_train,:] = [int(a) for a in quest[i][cond[j]]["PANAS"]] # targets

    # test set
    for i in test_idx: # for every member of testing
        c_test += 1 # update counter
        
        # feature extraction
        X_time = pre.time_features(dataset[i][cond[j]],window_size,window_shift,fs)
        X_freq = pre.freq_features(dataset[i][cond[j]],fs)
        X_extra = pre.EDA_extra_features(dataset[i][cond[j]],fs)
        X = pd.concat([X_time,X_freq,X_extra],axis=1)
        # assign values
        df_test = pd.concat([df_test,X],axis=0)  # features 
        y_test[c_test,:] = [int(a) for a in quest[i][cond[j]]["PANAS"]] # targets

# compute values for standardisation
X_train_mean = np.mean(df_train,axis=0)
X_train_std = np.std(df_train,axis=0)
# standardise train and test
X_train = pre.Standardise(df_train,X_train_mean,X_train_std).dropna(axis=1) # drop NaN as these columns corresponds to imaginary parts which are zero (due to the standardisation)
X_test = pre.Standardise(df_test,X_train_mean,X_train_std).dropna(axis=1)

print(X_train,X_test)

# train model
model.fit(X_train,y_train)

# assessment
y_pred = model.predict(X_test)
print("max:",max(norm(y_test[0]-y_pred[0],np.inf),norm(y_test[1]-y_pred[1],np.inf),norm(y_test[2]-y_pred[2],np.inf),norm(y_test[3]-y_pred[3],np.inf),norm(y_test[4]-y_pred[4],np.inf)))
print("mae:",1/5*(norm(y_test[0]-y_pred[0],1)+norm(y_test[1]-y_pred[1],1)+norm(y_test[2]-y_pred[2],1)+norm(y_test[3]-y_pred[3],1)+norm(y_test[4]-y_pred[4],1)))
print("mse:",1/5*(norm(y_test[0]-y_pred[0])**2+norm(y_test[1]-y_pred[1])**2+norm(y_test[2]-y_pred[2])**2+norm(y_test[3]-y_pred[3])**2+norm(y_test[4]-y_pred[4])**2))

# feature importance    
features = model.feature_importances_
print(features[np.where(features>np.std(features))])