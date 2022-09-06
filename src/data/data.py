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
                
                # load questions
                

                # divide into the 5 phases (baseline, stress, amusement, meditation x 2)
                for phase in range(1,5):      

                        if phase == 4: # divide meditation into 2 phases

                            # change separation margin based on special cases
                            if str(name).rsplit(".",1)[0] == "S3" or str(name).rsplit(".",1)[0] == "S6":
                                sep = [160000,330000,20000,3750000]   # ACC, BVP, EDA/TEMP, label
                            elif str(name).rsplit(".",1)[0] == "S2" or str(name).rsplit(".",1)[0] == "S16":
                                sep = [150000,290000,18000,3250000]
                            else: 
                                sep = [130000,290000,18000,3250000]
                            
                            # phase 4

                            idx_acc = np.array(np.where(np.array(dict["label"][::22]) == phase)) # find indices
                            ACC = np.hstack(dict["wrist"]["ACC"])[idx_acc[idx_acc<sep[0]]] # assign appropriate values
                            
                            idx_bvp = np.array(np.where(np.array(dict["label"][::11]) == phase)) # note: 700/64 approx 11
                            BVP = np.hstack(dict["wrist"]["BVP"])[idx_bvp[idx_bvp<sep[1]]]
                            
                            idx_eda_temp = np.array(np.where(np.array(dict["label"][::int(700/4)]) == phase))
                            EDA_tonic = np.hstack(dict["wrist"]["EDA"]["Tonic"])[idx_eda_temp[idx_eda_temp<sep[2]]]
                            EDA_phasic = np.hstack(dict["wrist"]["EDA"]["Phasic"])[idx_eda_temp[idx_eda_temp<sep[2]]]
                            EDA = {"Tonic":EDA_tonic,"Phasic":EDA_phasic}
                            
                            TEMP = np.hstack(dict["wrist"]["TEMP"])[idx_eda_temp[idx_eda_temp<sep[2]]]
                        
                            idx_label = np.array(np.where(np.array(dict["label"]) == phase))
                            label = np.hstack(dict["label"])[idx_label[idx_label<sep[3]]]
                            
                            dataset_temp[str(4)] = {"label":label,"wrist":{"ACC":ACC,"BVP":BVP,"EDA":EDA,"TEMP":TEMP}} # initialise dictionary 

                            # phase 5     
                            
                            ACC2 = np.hstack(dict["wrist"]["ACC"])[idx_acc[idx_acc>sep[0]]] 
                            BVP2 = np.hstack(dict["wrist"]["BVP"])[idx_bvp[idx_bvp>sep[1]]]
                            EDA_tonic2 = np.hstack(dict["wrist"]["EDA"]["Tonic"])[idx_eda_temp[idx_eda_temp>sep[2]]]
                            EDA_phasic2 = np.hstack(dict["wrist"]["EDA"]["Phasic"])[idx_eda_temp[idx_eda_temp>sep[2]]]
                            EDA2 = {"Tonic":EDA_tonic2,"Phasic":EDA_phasic2}
                            TEMP2 = np.hstack(dict["wrist"]["TEMP"])[idx_eda_temp[idx_eda_temp>sep[2]]]  
                            label2 = np.hstack(dict["label"])[idx_label[idx_label>sep[3]]]
                            
                            dataset_temp[str(5)] = {"label":label2,"wrist":{"ACC":ACC2,"BVP":BVP2,"EDA":EDA2,"TEMP":TEMP2}} # initialise dictionary 
                    
                        else:

                            idx_acc = np.array(np.where(np.array(dict["label"][::22]) == phase))
                            ACC = np.hstack(dict["wrist"]["ACC"])[idx_acc]
                            
                            idx_bvp = np.array(np.where(np.array(dict["label"][::11]) == phase))
                    
                            BVP = np.hstack(dict["wrist"]["BVP"])[idx_bvp]  
                            
                            idx_eda_temp = np.array(np.where(np.array(dict["label"][::int(700/4)]) == phase))
                            EDA_tonic = np.hstack(dict["wrist"]["EDA"]["Tonic"])[idx_eda_temp]
                            EDA_phasic = np.hstack(dict["wrist"]["EDA"]["Phasic"])[idx_eda_temp]
                            EDA = {"Tonic":EDA_tonic,"Phasic":EDA_phasic}
                            
                            TEMP = np.hstack(dict["wrist"]["TEMP"])[idx_eda_temp]
                            
                            idx_label = np.array(np.where(np.array(dict["label"]) == phase))
                            label = np.hstack(dict["label"])[idx_label]
                            
                            dataset_temp[str(phase)] = {"label":label,"wrist":{"ACC":ACC,"BVP":BVP,"EDA":EDA,"TEMP":TEMP}} # initialise dictionary 
                    
                dataset[str(name).rsplit(".",1)[0]] = dataset_temp # add to nested dictionary


print(dataset)