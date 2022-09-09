import numpy as np
from tqdm import tqdm
from sklearn import linear_model
import sys
import ast
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre
from data import *


# index for leave-one-patient-out cross-validation
CV_idx = ["S6"]

# Signals to include
signals = ["EDA","TEMP","BVP","HR"]
window_size = [20,20,20,20]
window_shift = [5,5,5,5]
fs = [4,4,32,1]

# initialise memory
X_test = np.zeros((5,109))
y_test = np.zeros((5,32))

# training-set
#for i in vers1 and vers2 if i not in CV_idx:


# test_set
for j in range(5): # for every phase
    for i in CV_idx: # for every member of testing
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
        X_test[j,:] = X_temp # features
        y_test[j,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets

print(X_test,y_test)






# cv

# models





