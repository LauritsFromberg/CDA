import os
import pickle
import neurokit2 as nk
import numpy as np
from utils import preprocessing_utils as pre

## load data

# empty dictionary
dataset_temp = {}
dataset = {}

# define path 
path = "C:/Users/Bruger/Documents/CDA/CDA/data/open/WESAD"

# initialise Butterworth filter (EDA)
N = 4 # order of filter
fs = 4 # critical frequency
Wn = 2 * 2.5 / fs # critical frequency of filter

# main for-loop
for folder,sub_folders,files in os.walk(path):
    for name in files:
        if name.endswith(".pkl"): # find files with .pkl extension
            with open(os.path.join(folder,name) ,"rb") as file:
                
                tmp = pickle._Unpickler(file) # unpickle
                tmp.encoding = "latin1" # encode to appropriate format
                dict = tmp.load() # load data
                dict = {"wrist":dict["signal"]["wrist"],"label": dict["label"]} # ignore RespiBAN data
                
                dict["wrist"]["EDA"] = pre.EMA(pre.Butter(dict["wrist"]["EDA"],N,Wn,fs),0.4) # pre-process EDA
                eda_temp = nk.eda_phasic(dict["wrist"]["EDA"],fs,method="cvxEDA") # decompose EDA
                dict["wrist"]["EDA"] = {"Tonic": eda_temp["EDA_Tonic"].to_numpy(),"Phasic":eda_temp["EDA_Phasic"].to_numpy()} 
                
                # divide into the 4 phases (baseline, stress, amusement,meditation)
                for phase in range(1,5):      
                        
                        idx_acc = np.array(np.where(np.array(dict["label"][::22]) == phase)) # find indices
                        ACC = np.array(dict["wrist"]["ACC"])[idx_acc[0]] # assign appropriate values
                        
                        idx_bvp = np.array(np.where(np.array(dict["label"][::11]) == phase)) # note 700/64 approx 11
                        BVP = np.array(dict["wrist"]["BVP"])[idx_bvp[0]] 
                        
                        idx_eda_temp = np.array(np.where(np.array(dict["label"][::int(700/4)]) == phase))
                        EDA_tonic = np.array(dict["wrist"]["EDA"]["Tonic"])[idx_eda_temp[0]]
                        EDA_phasic = np.array(dict["wrist"]["EDA"]["Phasic"])[idx_eda_temp[0]]
                        EDA = {"Tonic":EDA_tonic,"Phasic":EDA_phasic}
                        
                        TEMP = np.array(dict["wrist"]["TEMP"])[idx_eda_temp[0]]
                        
                        idx_label = np.array(np.where(np.array(dict["label"][::int(700/32)]) == phase)) 
                        label = np.array(dict["label"])[idx_label[0]]
                        
                        dataset_temp[str(phase)] = {"label":label,"wrist":{"ACC":ACC,"BVP":BVP,"EDA":EDA,"TEMP":TEMP}} # initialise dictionary 
                
                dataset[str(name).rsplit(".",1)[0]] = dataset_temp # add to nested dictionary


