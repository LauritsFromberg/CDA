import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.cm import cool

## load data

# define path 
path = "C:/Users/Bruger/Documents/CDA/CDA/data/open/WESAD"

# empty dictionaries
dataset = {} 
HR_dict = {}

# subjects
sub_lst = ["S10","S11","S13","S14","S15","S16","S17","S2","S3","S4","S6","S7","S8","S9"]

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
        if name.endswith(".pkl"): # find files with .pkl extension
            with open(os.path.join(folder,name) ,"rb") as file:
                tmp = pickle._Unpickler(file) # unpickle
                tmp.encoding = "latin1" # encode to appropriate format
                dict = tmp.load() # load data
                dict = {"wrist":dict["signal"]["wrist"],"label": dict["label"]} # ignore RespiBAN data
                dict["wrist"]["HR"] = np.array(HR_dict[str(name).rsplit(".",1)[0]]["HR"]) # add HR
                dataset[str(name).rsplit(".",1)[0]] = dict # add to nested dictionary

# create data for supervised learning
BVP = dataset["S4"]["wrist"]["BVP"] 
HR = dataset["S4"]["wrist"]["HR"]
EDA = dataset["S4"]["wrist"]["EDA"]
TEMP = dataset["S4"]["wrist"]["TEMP"]
y = dataset["S4"]["label"] # labels

## visualise

# legend 
legend_elements = [Line2D([0],[0],marker=".",markerfacecolor="black",markersize=15,color="w",label="Transient"),
                   Line2D([0],[0],marker=".",markerfacecolor="green",markersize=15,color="w",label="Baseline"),
                   Line2D([0],[0],marker=".",markerfacecolor="red",markersize=15,color="w",label="Stress"),
                   Line2D([0],[0],marker=".",markerfacecolor="blue",markersize=15,color="w",label="Amusement"),
                   Line2D([0],[0],marker=".",markerfacecolor="orange",markersize=15,color="w",label="Meditation")]

# function for plotting according to labels
def pltcolour(lst):
    cols = []
    for l in lst:
        if l == 0:
            cols.append("black")
        if l == 1:
            cols.append("green")
        if l == 2:
            cols.append("red")
        if l == 3:
            cols.append("blue")
        if l == 4:
            cols.append("orange")
    return cols

# initialise plot
fs = 1 # lowest (HR)
y = y[::int(700/fs)] # match labels with biosignal
BVP = BVP[::int(64/fs)] # match BVP 
EDA = EDA[::int(4/fs)] # match EDA 
TEMP = TEMP[::int(4/fs)] # match TEMP
s = 5000 # number of seconds
n = fs*s # number of point specified by the inverse sampling rate
col = pltcolour(y) # colours
t = np.linspace(0,s,n) # time steps 

# plot
fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1,sharex=True)
fig.suptitle("Empatica E4 - Biosignals")
ax1.scatter(t,EDA[0:n],s=1,c=col[0:n]) 
ax2.scatter(t,BVP[0:n],s=1,c=col[0:n])
ax3.scatter(t,TEMP[0:n],s=1,c=col[0:n])
ax4.scatter(t,HR[0:n],s=1,c=col[0:n])
ax1.set(ylabel = "EDA [$\mu S$]")
ax2.set(ylabel = "BVP [N/A]")
ax3.set(ylabel = "TEMP [$^\circ C$]")   
ax4.set(xlabel = "Time [Seconds]", ylabel = "HR [BPM]")                                                                                                                                  
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax1.legend(handles=legend_elements,loc="center left", prop={'size': 8})
ax2.legend(handles=legend_elements,loc="center left", prop={'size': 8})
ax3.legend(handles=legend_elements,loc="center left", prop={'size': 8})
ax4.legend(handles=legend_elements,loc="center left", prop={'size': 8})
plt.show()

# extra
plt.plot(BVP)
plt.show()




# hyper-parameters

# # model
# with warnings.catch_warnings(): # disable all convergence warnings from elastic net
#    warnings.simplefilter("ignore")

#    model = sklearn.linear_model.ElasticNet().fit(X,y) # fit elastic net model

