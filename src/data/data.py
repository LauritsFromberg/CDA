import os
import pickle
import neurokit2 as nk
import numpy as np
import scipy
from scipy import signal
from utils import preprocessing_utils as pre
import pandas as pd

## load and pre-process data

# empty dictionaries
dataset_temp = {}
dataset = {}
quest_temp = {}
quest = {}
HR_dict = {}


# subjects in each version
sub_lst = ["S10","S11","S13","S14","S15","S16","S17","S2","S3","S4","S6","S7","S8","S9"]
vers1 =  ["S4","S7","S8","S10","S13","S15","S17"] # remove S5 due to NaN
vers2 = ["S2","S3","S6","S9","S11","S14","S16"]

# emotional states (versions of study-protocol)
cond = ["base","stress","amusement","meditation_1","meditation_2"]
cond1 = ["base","amusement","meditation_1","stress","meditation_2"]
cond2 = ["base","stress","meditation_1","amusement","meditation_2"]

# define path 
path = "C:/Users/Bruger/Documents/CDA/CDA/data/open/WESAD"

# initialise Butterworth filter (EDA)
N = 4 # order of filter
fs = 4 # critical frequency
Wn = 2 * 2.5 / fs # critical frequency of filter

# counter 
c = -1

# load HR biosignals
for folder,sub_folders,files in os.walk(path):
    for name in files:
        if name.endswith("HR.csv"):
            c += 1
            df = pd.read_csv(os.path.join(folder,name),sep=";",skiprows=1)
            HR_dict[sub_lst[c]] = {"HR": df.values.tolist()}  # load into dictonary

# main for-loop
for folder,sub_folders,files in os.walk(path):
    for name in files:
        sub = str(name).rsplit(".",1)[0]
        # load questions
        if name.endswith("_quest.csv"): # find desired csv files
            df = pd.read_csv(os.path.join(folder,name),sep=";") # load into dataframe
            df_val = df.values.tolist() # get values
            quest_temp = {}
            quest[sub.rsplit("_",1)[0]] = {}
            count = 0
            if sub.rsplit("_",1)[0] in vers1: # version 1 of study-protocol
                for i in cond1:
                    # special case
                    #if i == "stress":
                    #    quest_temp[i] = {"PANAS": df_val[4+count][1:],"STAI": df_val[10+count][1:7],"DIM": df_val[16+count][1:3],"SSSQ":df_val[22][1:6]}
                    #    count += 1
                    #else:   
                    quest_temp[i] = {"PANAS": df_val[4+count][1:25],"STAI": df_val[10+count][1:7],"DIM": df_val[16+count][1:3]}
                    count += 1     
                quest[sub.rsplit("_",1)[0]] = quest_temp # add to nested dictionary
            else: # version 2 of study-protocol
                for i in cond2:
                    #if i == "stress": 
                    #    quest_temp[i] = {"PANAS": df_val[4+count][1:],"STAI": df_val[10+count][1:7],"DIM": df_val[16+count][1:3],"SSSQ":df_val[22][1:6]}
                    #    count += 1
                    #else:   
                    quest_temp[i] = {"PANAS": df_val[4+count][1:25],"STAI": df_val[10+count][1:7],"DIM": df_val[16+count][1:3]}
                    count += 1
                quest[sub.rsplit("_",1)[0]] = quest_temp

        # load biosignals and labels     
        elif name.endswith(".pkl"): # find files with .pkl extension
            dataset_temp = {}
            dataset[sub] = {}
            with open(os.path.join(folder,name) ,"rb") as file:
                
                tmp = pickle._Unpickler(file) # unpickle
                tmp.encoding = "latin1" # encode to appropriate format
                dict = tmp.load() # load data
                dict = {"wrist":dict["signal"]["wrist"],"label": dict["label"]} # ignore RespiBAN data
                dict["wrist"]["HR"] = np.array(HR_dict[str(name).rsplit(".",1)[0]]["HR"]) # add HR

            dict["wrist"]["EDA"] = pre.EMA(pre.Butter(dict["wrist"]["EDA"],N,Wn,fs),0.4) # pre-process EDA
            eda_temp = nk.eda_phasic(dict["wrist"]["EDA"],fs,method="cvxEDA") # decompose EDA
            dict["wrist"]["EDA"] = {"Tonic": eda_temp["EDA_Tonic"].to_numpy(),"Phasic":eda_temp["EDA_Phasic"].to_numpy()} 

            dict["wrist"]["BVP"] = scipy.signal.resample(dict["wrist"]["BVP"],int(len(dict["wrist"]["BVP"])*70/64)) # resample BVP to approx. 70Hz
                # divide into the 5 phases (baseline, stress, amusement, meditation x 2)
            for phase in range(1,5):      

                if phase == 4: # divide meditation into 2 phases

                    # change separation margins based on special cases
                    if sub == "S3" or sub == "S6":
                        sep = [5000,330000,20000,3750000]   # ACC, BVP, EDA/TEMP, label
                    elif sub == "S2" or sub == "S16":
                        sep = [4500,290000,18000,3250000]
                    else: 
                        sep = [4200,290000,18000,3250000]
                            
                    # phase 4

                    #idx_acc = np.array(np.where(np.array(dict["label"][::22]) == phase)) # find indices
                    #ACC = np.array(dict["wrist"]["ACC"])[idx_acc[idx_acc<sep[0]]] # assign appropriate values
                            
                    idx_hr = np.array(np.where(np.array(dict["label"][::700]) == phase))
                    HR = np.array(dict["wrist"]["HR"])[idx_hr[idx_hr<sep[0]]]

                    idx_bvp = np.array(np.where(np.array(dict["label"][::10]) == phase)) 
                    BVP = np.array(dict["wrist"]["BVP"])[idx_bvp[idx_bvp<sep[1]]]
                            
                    idx_eda_temp = np.array(np.where(np.array(dict["label"][::int(700/4)]) == phase))
                    EDA_tonic = np.array(dict["wrist"]["EDA"]["Tonic"])[idx_eda_temp[idx_eda_temp<sep[2]]]
                    EDA_phasic = np.array(dict["wrist"]["EDA"]["Phasic"])[idx_eda_temp[idx_eda_temp<sep[2]]]
                    EDA = {"Tonic":EDA_tonic,"Phasic":EDA_phasic}
                            
                    TEMP = np.array(dict["wrist"]["TEMP"])[idx_eda_temp[idx_eda_temp<sep[2]]]
                        
                    #idx_label = np.array(np.where(np.array(dict["label"]) == phase))
                    #label = np.array(dict["label"])[idx_label[idx_label<sep[3]]]
                            
                    dataset_temp[cond[3]] = {"wrist":{"BVP":BVP,"HR":HR,"EDA":EDA,"TEMP":TEMP}}
                    #dataset_temp[cond[3]] = {"label":label,"wrist":{"ACC":ACC,"BVP":BVP,"EDA":EDA,"TEMP":TEMP}} # initialise dictionary 

                    # phase 5     
                            
                    #ACC2 = np.array(dict["wrist"]["ACC"])[idx_acc[idx_acc>sep[0]]] 
                    HR2 = np.array(dict["wrist"]["HR"])[idx_hr[idx_hr>sep[0]]]
                    BVP2 = np.array(dict["wrist"]["BVP"])[idx_bvp[idx_bvp>sep[1]]]
                    EDA_tonic2 = np.array(dict["wrist"]["EDA"]["Tonic"])[idx_eda_temp[idx_eda_temp>sep[2]]]
                    EDA_phasic2 = np.array(dict["wrist"]["EDA"]["Phasic"])[idx_eda_temp[idx_eda_temp>sep[2]]]
                    EDA2 = {"Tonic":EDA_tonic2,"Phasic":EDA_phasic2}
                    TEMP2 = np.array(dict["wrist"]["TEMP"])[idx_eda_temp[idx_eda_temp>sep[2]]]  
                    #label2 = np.array(dict["label"])[idx_label[idx_label>sep[3]]]
                            
                    dataset_temp[cond[4]] = {"wrist":{"BVP":BVP2,"HR":HR2,"EDA":EDA2,"TEMP":TEMP2}}
                    #dataset_temp[cond[4]] = {"label":label2,"wrist":{"ACC":ACC2,"BVP":BVP2,"EDA":EDA2,"TEMP":TEMP2}} 
                    
                else:

                    #idx_acc = np.array(np.where(np.array(dict["label"][::22]) == phase))
                    #ACC = np.array(dict["wrist"]["ACC"])[idx_acc]

                    idx_hr = np.array(np.where(np.array(dict["label"][::700]) == phase))
                    HR = np.array(dict["wrist"]["HR"])[idx_hr]   
                    
                    idx_bvp = np.array(np.where(np.array(dict["label"][::10]) == phase))
                    BVP = np.array(dict["wrist"]["BVP"])[idx_bvp]  
                            
                    idx_eda_temp = np.array(np.where(np.array(dict["label"][::int(700/4)]) == phase))
                    EDA_tonic = np.array(dict["wrist"]["EDA"]["Tonic"])[idx_eda_temp]
                    EDA_phasic = np.array(dict["wrist"]["EDA"]["Phasic"])[idx_eda_temp]
                    EDA = {"Tonic":EDA_tonic,"Phasic":EDA_phasic}
                            
                    TEMP = np.array(dict["wrist"]["TEMP"])[idx_eda_temp]
                            
                    #idx_label = np.array(np.where(np.array(dict["label"]) == phase))
                    #label = np.array(dict["label"])[idx_label]
                            
                    dataset_temp[cond[phase-1]] = {"wrist":{"BVP":BVP,"HR":HR,"EDA":EDA,"TEMP":TEMP}}
                    #dataset_temp[cond[phase-1]] = {"label":label,"wrist":{"ACC":ACC, "BVP":BVP,"EDA":EDA,"TEMP":TEMP}} # initialise dictionary 

        dataset[sub] = dataset_temp # add to nested dictionary

#print(quest["S2"]["base"],quest["S3"]["base"],quest["S4"]["base"])

#print(type(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset["S3"]["base"]["wrist"]["TEMP"])),10,4))),np.hstack(pre.burn_in(np.hstack(np.hstack(dataset["S3"]["base"]["wrist"]["TEMP"])),10,4)))
#print(type(pre.burn_in(np.hstack(np.hstack(dataset["S3"]["base"]["wrist"]["TEMP"])),10,4)),pre.burn_in(np.hstack(np.hstack(dataset["S3"]["base"]["wrist"]["TEMP"])),10,4))
