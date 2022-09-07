import numpy as np
from tqdm import tqdm
from sklearn import linear_model
import sys
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre
from data import *

# index for leave-one-patient-out cross-validation
CV_idx = []

# Signals to include
signals = ["EDA","TEMP","BVP"]
window_size = [20,20,20]
window_shift = [5,5,5]
fs = [4,4,32]


# initialise memory
X_test = np.empty((5,))
X_temp = []
y_test = np.empty((5,))

# training-set
#for i in vers1 and vers2 if i not in CV_idx:


# test_set
for j in tqdm(range(5)): # for every phase
    for i in CV_idx: # for every member of testing
        # feature extraction 
        for k in signals:
            count = 0
            if k == "EDA":
                X_slope_tonic = np.hstack(*pre.slope_features(np.hstack(*dataset[i][cond[j]]["wrist"]["EDA"]["Tonic"]),window_size[count],window_shift[count],fs[count]).values())
                X_stat_tonic = np.hstack(*pre.stat_features(np.hstack(*dataset[i][cond[j]]["wrist"]["EDA"]["Tonic"]),window_size[count],window_shift[count],fs[count]).values())
                X_extra_tonic =  np.hstack(*pre.extra_features(np.hstack(*dataset[i][cond[j]]["wrist"]["EDA"]["Tonic"]),window_size[count],window_shift[count],fs[count]).values())
                X_slope_phasic = np.hstack(*pre.slope_features(np.hstack(*dataset[i][cond[j]]["wrist"]["EDA"]["Phasic"]),window_size[count],window_shift[count],fs[count]).values())
                X_stat_phasic = np.hstack(*pre.stat_features(np.hstack(*dataset[i][cond[j]]["wrist"]["EDA"]["Phasic"]),window_size[count],window_shift[count],fs[count]).values())
                X_extra_phasic =  np.hstack(*pre.extra_features(np.hstack(*dataset[i][cond[j]]["wrist"]["EDA"]["Phasic"]),window_size[count],window_shift[count],fs[count]).values())
                X_temp.append(X_slope_tonic,X_stat_tonic,X_extra_tonic,X_slope_phasic,X_stat_tonic,X_extra_phasic)
                count += 1
            else:
                X_slope = np.hstack(*pre.slope_features(np.hstack(*dataset[i][cond[j]]["wrist"][k]),window_size[count],window_shift[count],fs[count]).values())
                X_stat = np.hstack(*pre.stat_features(np.hstack(*dataset[i][cond[j]]["wrist"][k]),window_size[count],window_shift[count],fs[count]).values())
                X_extra =  np.hstack(*pre.extra_features(np.hstack(*dataset[i][cond[j]]["wrist"][k]),window_size[count],window_shift[count],fs[count]).values())
                X_temp.append(X_slope,X_stat,X_extra)
                count += 1
        print(len(np.hstack(X_temp)))
        #X_test= np.vstack(X_test,np.hstack(X_temp)) # bag lav append til kolumn og en prædisigneret numpy array matrix ting efter hvert for loop, samme med ytest
        #y_test = np.vstack(y_test,np.hstack(*quest[i][cond[j]].values())) # targets


            # TRANSPONER matrix sådan at det passer med sklearn notation!!
        # find ud af hvordan du gør det for hver fase sådan at hver fase giver en enkel observation??? 
        # kunne være rart at nå i dag sålede at du kan spørge om dette i morgen!!!!!!
        # det kunne være rart at se om man kan bruge det på en sklearn model!!!


    






# cv

# models





