import sys
import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre

## compute feature importance

# train best model on all individals 

# load model parameters
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_param_8cv.pkl"
file = open(filename,"rb")
param = pickle.load(file)
file.close()

# load dataset
filename = "C:/Users/Bruger/Documents/CDA/CDA/data/dataset.pkl"
file = open(filename,"rb")
dataset = pickle.load(file)
file.close()

# load questions
filename = "C:/Users/Bruger/Documents/CDA/CDA/data/quest.pkl"
file = open(filename,"rb")
quest = pickle.load(file)
file.close()

# define model
model = RandomForestRegressor(random_state=10220,**param)

# subjects
sub_lst = ["S10","S11","S13","S14","S15","S16","S17","S2","S3","S4","S6","S7","S8"]
test_idx = ["S9"]

# emotional states (versions of study-protocol)
cond = ["base","stress","amusement","meditation_1","meditation_2"]

# Signals to include
signals = ["EDA","TEMP","BVP","HR"]
window_size = [65,65,65,65]
window_shift = [13,13,13,13]
fs = [4,4,32,1]

# initialise burn-in period
t = 10

# initialise memory
X_train = np.zeros((65,112)) # number of phases (for 13 subjects) x number of features
y_train = np.zeros((65,32)) # number of phases (for 13 subjects) x number of questions
X_test = np.zeros((5,112)) # number of phases (for one subject) x number of features
y_test = np.zeros((5,32)) # number of phases x number of questions

# counters
c_train = -1 
c_test = -1 

# for every phase
for j in range(5):    
     # for every subject
    for i in sub_lst:
            c_train += 1 # update counter
            # feature extraction
            X_temp = [] # initialise memory
            for k in signals:
                count = 0
                if k == "EDA":
                    X_slope_tonic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_tonic)
                    X_stat_tonic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_tonic)
                    X_extra_tonic =  [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_tonic)
                    X_slope_phasic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_phasic)
                    X_stat_phasic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_phasic)
                    X_extra_phasic = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_phasic)
                    count += 1
                elif k == "BVP":
                    X_slope = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat[0:11]) # exclude entropy features for BVP
                    X_extra = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
                else:
                    X_slope = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat)
                    X_extra = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
            # assign values
            X_train[c_train,:] = X_temp # features without entropy for BVP
            y_train[c_train,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets

# test set
    for i in test_idx: # for every member of testing
        c_test += 1 # update counter
        # feature extraction 
        X_temp = [] # initialise memory
        for k in signals:
            count = 0
            if k == "EDA":
                X_slope_tonic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_slope_tonic)
                X_stat_tonic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_stat_tonic)
                X_extra_tonic =  [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_extra_tonic)
                X_slope_phasic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_slope_phasic)
                X_stat_phasic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_stat_phasic)
                X_extra_phasic =  [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_extra_phasic)
                count += 1
            elif k == "BVP":
                X_slope = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_slope)
                X_stat = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_stat[0:11]) # exclude entropy features for BVP
                X_extra = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_extra)
                count += 1
            else:
                X_slope = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_slope)
                X_stat = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_stat)
                X_extra = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                X_temp.extend(X_extra)
                count += 1
        # assign values
        X_test[c_test,:] = X_temp # features
        y_test[c_test,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets

# compute values for standardisation
X_train_mean = np.mean(X_train,axis=0)
X_train_std = np.std(X_train,axis=0)
# standardise train and test
X_train = pre.Standardise(X_train,X_train_mean,X_train_std)
X_test = pre.Standardise(X_test,X_train_mean,X_train_std)

# train model
model.fit(X_train,y_train)

# assessment
y_pred = model.predict(X_test)
print("max:",max(norm(y_test[0]-y_pred[0],np.inf),norm(y_test[1]-y_pred[1],np.inf),norm(y_test[2]-y_pred[2],np.inf),norm(y_test[3]-y_pred[3],np.inf),norm(y_test[4]-y_pred[4],np.inf)))
print("mae:",1/5*(norm(y_test[0]-y_pred[0],1)+norm(y_test[1]-y_pred[1],1)+norm(y_test[2]-y_pred[2],1)+norm(y_test[3]-y_pred[3],1)+norm(y_test[4]-y_pred[4],1)))
print("mse:",1/5*(norm(y_test[0]-y_pred[0])**2+norm(y_test[1]-y_pred[1])**2+norm(y_test[2]-y_pred[2])**2+norm(y_test[3]-y_pred[3])**2+norm(y_test[4]-y_pred[4])**2))

# feature importance    
features = model.feature_importances_
print(features[np.where(features>np.std(features)/(len(features)**0.5))])