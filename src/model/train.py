import numpy as np
from tqdm import tqdm
from sklearn import linear_model
import sys
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre
from data import *

# index for leave-one-patient-out cross-validation
CV_idx = ["S6"]

# Signals to include
signals = ["EDA","TEMP","BVP"]
window_size = [20,20,20]
window_shift = [5,5,5]
fs = [4,4,32]


# initialise memory
#X_test = np.empty((5,470))
#X_temp = []
#y_test = np.empty((5,470))

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
    print(print(X_temp),len(X_temp))            
    #print(X_temp,[i for i, arr in enumerate(X_temp) if not np.isfinite(arr).all()],len(X_temp))
        


        # #X_test= np.vstack(X_test,np.hstack(X_temp)) # bag lav append til kolumn og en prædisigneret numpy array matrix ting efter hvert for loop, samme med ytest
        # #y_test = np.vstack(y_test,np.hstack(*quest[i][cond[j]].values())) # targets


        #     # TRANSPONER matrix sådan at det passer med sklearn notation!!
        # # find ud af hvordan du gør det for hver fase sådan at hver fase giver en enkel observation??? 
        # # kunne være rart at nå i dag sålede at du kan spørge om dette i morgen!!!!!!
        # # det kunne være rart at se om man kan bruge det på en sklearn model!!!








# cv

# models





