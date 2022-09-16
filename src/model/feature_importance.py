import sys
import pickle
import numpy as np
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
model = RandomForestRegressor(**param)

# subjects
sub_lst = ["S10","S11","S13","S14","S15","S16","S17","S2","S3","S4","S6","S7","S8","S9"]

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
X_train = np.zeros((70,112)) # number of phases (for 14 subjects) x number of features
y_train = np.zeros((70,32)) # number of phases (for 14 subjects) x number of questions

# create training set for all subjects
c_train = -1 # counter 
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

# train model on all subjects
model.fit(X_train,y_train)

# feature importance
features = model.feature_importances_
print(features[np.where(features>np.std(features)/(len(features)**0.5))])
print(np.where(features>np.std(features)/(len(features)**0.5)))
print(dataset["S2"]["base"]["wrist"])