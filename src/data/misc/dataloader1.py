import os
import pickle
from torch.utils.data import Dataset

class dataset(Dataset):
    def __init__(self,path):

        dataset = {} # empty dictionary

        for folder,sub_folders,files in os.walk(path):
            for name in files:
                if name.endswith(".pkl"): # find files with .pkl extension
                    with open(os.path.join(folder,name) ,"rb") as file:
                        tmp = pickle._Unpickler(file) # unpickle
                        tmp.encoding = "latin1" # encode to appropriate format
                        dict = tmp.load() # load data
                        dict = {"wrist":dict["signal"]["wrist"],"label": dict["label"]} # ignore RespiBAN data
                        dataset[str(name).rsplit(".",1)[0]] = dict # add to nested dictionary

        self.dataset = dataset

    def __getitem__(self,index):
        return (self.dataset[index])

    def __len__(self):
        return len(self.dataset)

    

