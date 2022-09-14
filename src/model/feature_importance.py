import sys
import pickle
import numpy as np
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre

## compute feature importance

# train best model on all individals 

# load model parameters
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_param.pkl"
file = open(filename,"rb")
param = pickle.load(file)
file.close()
print(param)

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
#model = 


# training


#     CV_idx = [np.array(sub_lst)[s[v]]] # pick random subject to exclude for testing 
#     train_idx = [np.array(sub_lst)[np.arange(len(sub_lst))!=s[v]]][0] # use all other subjects for training 
#     c_train = -1 # counter 
#     c_test = -1 # counter

#     # for every phase
#     for j in range(5): 
        
#         # train set
#         for i in train_idx: # for every member of training
#             c_train += 1 # update counter
#             # feature extraction
#             X_temp = [] # initialise memory
#             for k in signals:
#                 count = 0
#                 if k == "EDA":
#                     X_slope_tonic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_slope_tonic)
#                     X_stat_tonic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_stat_tonic)
#                     X_extra_tonic =  [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_extra_tonic)
#                     X_slope_phasic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_slope_phasic)
#                     X_stat_phasic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_stat_phasic)
#                     X_extra_phasic = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_extra_phasic)
#                     count += 1
#                 else:
#                     X_slope = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_slope)
#                     X_stat = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_slope)
#                     X_extra = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
#                     X_temp.extend(X_extra)
#                     count += 1
#             # assign values
#             X_train[c_train,:] = X_temp # features
#             y_train[c_train,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets


# feature importance
#print()