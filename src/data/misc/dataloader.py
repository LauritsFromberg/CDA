import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self,path,special_file_name,skip_r):
  
        # create empty dataframe
        df = pd.DataFrame()

        # counter 
        count = 0

        # load all time-series into a single dataframe
        for folder,sub_folders,files in os.walk(path): # iterate through path
            for special_file in files:
                if special_file == special_file_name: # find special file
                    count += 1 # update counter
                    file_path = os.path.join(folder,special_file) # join path
                    df = pd.concat([df,pd.read_csv(file_path,header=None,names=[count],skiprows=skip_r)],axis=1) # add onto existing dataframe


        self.df = df
        self.count = count

    def __getitem__(self,index):

        df = self.df # dataframe

        # pre-processing
        scaler = StandardScaler() # standardiser
        series = df.values.reshape(len(df),self.count) # create appropriate shape numpy-arrary 
        scaler = scaler.fit(series) # fit scaler 
        df = pd.DataFrame(scaler.transform(series)) # transform data            

        return (df.iloc[index])

    def __len__(self):
        return len(self.df)

    
