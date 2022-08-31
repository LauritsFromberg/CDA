import os
import pickle
import csv

# define path 
path = "C:/Users/Bruger/Documents/CDA/CDA/data/open/WESAD"

# empty dictionary
dataset = {} 

for folder,sub_folders,files in os.walk(path):
    for name in files:
        if name.endswith(".pkl"): # find files with .pkl extension
            with open(os.path.join(folder,name) ,"rb") as file:
                tmp = pickle._Unpickler(file) # unpickle
                tmp.encoding = "latin1" # encode to appropriate format
                dict = tmp.load() # load data
                dict = {"wrist":dict["signal"]["wrist"],"label": dict["label"]} # ignore RespiBAN data
                dataset[str(name).rsplit(".",1)[0]] = dict # add to nested dictionary



