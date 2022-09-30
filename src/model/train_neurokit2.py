import sys
import time
import warnings
import numpy as np
import pickle
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils_neurokit2 as pre

## train models

# time the script
startTime = time.time()

# load dataset
filename = "C:/Users/Bruger/Documents/CDA/CDA/data/dataset_neurokit2.pkl"
file = open(filename,"rb")
dataset = pickle.load(file)
file.close()

# load questions
filename = "C:/Users/Bruger/Documents/CDA/CDA/data/quest_neurokit2.pkl"
file = open(filename,"rb")
quest = pickle.load(file)
file.close()

# subjects
sub_lst = ["S10","S11","S13","S14","S15","S16","S17","S2","S3","S4","S6","S7","S8"] # exclude S9 for final assessment in feature importance

# emotional states
cond = ["base","stress","amusement","meditation_1","meditation_2"]

# initialise memory for the random states used in the cross-validation
states = []

# Signals to include and each respective window-size, window-shift as well as sample-rate
signals = ["EDA","Tonic","Phasic","TEMP","BVP","HR"]
window_size = [40,40,40,40,40,40] 
window_shift = [25,25,25,25,25,25]
fs = [4,4,4,4,64,1]

# initialise memory
y_train = np.zeros((60,24)) # number of phases (for 12 subjects) x number of questions
y_test = np.zeros((5,24)) # number of phases x number of questions

# dictonaries to save best models
en = {}
nn = {}
dt = {}
rf = {}
gb = {}

# list for test errors
test_en_mse = []
test_nn_mse = []
test_dt_mse = []
test_rf_mse = []
test_gb_mse = []

# parameters for cross-validation
param_grid_en = {"alpha": list(np.arange(0.25,1.75,0.75)), "l1_ratio": list(np.arange(0.1,0.7,0.1))}
param_grid_nn = {"n_neighbors": range(6,15)}
param_grid_dt = {"max_depth":[2,3,5,8] ,"min_samples_split": range(2,5),"min_samples_leaf":range(1,4), "max_features": range(50,101,25)}
param_grid_rf = {"n_estimators":range(90,111,10),"max_depth":[2,3,5,8],"min_samples_split": range(2,5),"min_samples_leaf":range(1,4),"max_features": range(50,101,25)}
param_grid_gb = {"estimator__n_estimators":range(90,111,10),"estimator__max_depth":[2,3,5,8],"estimator__min_samples_split": range(2,5),"estimator__min_samples_leaf":range(1,4),"estimator__max_features": range(50,101,25)}   

# initialise number of cross-validation iterations
n = 10

# make list of random subjects to exclude in each iteration (without replacement)
s = np.arange(len(sub_lst))
np.random.shuffle(s)

# leave-one-patient-out cross-validation
for v in range(n):

    # pick random state
    state = np.random.randint(0,100000) 
    states.append(state)
  
     # models
    model_en = linear_model.ElasticNet(tol=0.01,max_iter=10000,selection="random",random_state=state)
    model_nn = KNeighborsRegressor(weights="distance",p=2)
    model_dt = DecisionTreeRegressor(random_state=state)
    model_rf = RandomForestRegressor(random_state=state) 
    model_gb = GradientBoostingRegressor(random_state=state)

    CV_idx = [np.array(sub_lst)[s[v]]] # pick random subject to exclude for testing 
    train_idx = [np.array(sub_lst)[np.arange(len(sub_lst))!=s[v]]][0] # use all other subjects for training 
    c_train = -1 # counter 
    c_test = -1 # counter 

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    # for every phase
    for j in range(5): 
        
        # train set
        for i in train_idx: # for every member of training
            c_train += 1 # update counter

            # feature extraction
            X_time = pre.time_features(dataset[i][cond[j]],window_size,window_shift,fs)
            X_freq = pre.freq_features(dataset[i][cond[j]],fs)
            X_extra = pre.EDA_extra_features(dataset[i][cond[j]],fs)
            X = pd.concat([X_time,X_freq,X_extra],axis=1)
            # assign values
            df_train = pd.concat([df_train,X],axis=0)  # features 
            y_train[c_train,:] = [int(a) for a in quest[i][cond[j]]["PANAS"]] # targets

        # test set
        for i in CV_idx: # for every member of testing
            c_test += 1 # update counter
        
            # feature extraction
            X_time = pre.time_features(dataset[i][cond[j]],window_size,window_shift,fs)
            X_freq = pre.freq_features(dataset[i][cond[j]],fs)
            X_extra = pre.EDA_extra_features(dataset[i][cond[j]],fs)
            X = pd.concat([X_time,X_freq,X_extra],axis=1)
            # assign values
            df_test = pd.concat([df_test,X],axis=0)  # features 
            y_test[c_test,:] = [int(a) for a in quest[i][cond[j]]["PANAS"]] # targets

    # compute values for standardisation
    X_train_mean = np.mean(df_train,axis=0)
    X_train_std = np.std(df_train,axis=0)
    # standardise train and test
    X_train = pre.Standardise(df_train,X_train_mean,X_train_std).dropna(axis=1) # drop NaN as these columns corresponds to imaginary parts which are zero (due to the standardisation)
    X_test = pre.Standardise(df_test,X_train_mean,X_train_std).dropna(axis=1)

    # grid search
    grid_en = GridSearchCV(estimator=model_en,param_grid=param_grid_en,cv=5,n_jobs=-1)
    with warnings.catch_warnings(): # disable convergence warnings
        warnings.simplefilter("ignore")
        grid_en.fit(X_train,y_train)
    grid_nn = GridSearchCV(estimator=model_nn,param_grid=param_grid_nn,cv=5,n_jobs=-1)
    grid_nn.fit(X_train,y_train)
    grid_dt = GridSearchCV(estimator=model_dt,param_grid=param_grid_dt,cv=5,n_jobs=-1)
    grid_dt.fit(X_train,y_train)
    grid_rf = GridSearchCV(estimator=model_rf,param_grid=param_grid_rf,cv=5,n_jobs=-1)
    grid_rf.fit(X_train,y_train)
    grid_gb = GridSearchCV(estimator=MultiOutputRegressor(model_gb,n_jobs=-1),param_grid=param_grid_gb,cv=5,n_jobs=-1)
    grid_gb.fit(X_train,y_train)

    # save best model for testing 
    en[v] = {"best estimator":grid_en.best_estimator_, "best parameters": grid_en.best_params_}
    nn[v] = {"best estimator":grid_nn.best_estimator_, "best parameters": grid_nn.best_params_}
    dt[v] = {"best estimator":grid_dt.best_estimator_, "best parameters": grid_dt.best_params_}
    rf[v] = {"best estimator":grid_rf.best_estimator_, "best parameters": grid_rf.best_params_}
    gb[v] = {"best estimator":grid_gb.best_estimator_, "best parameters": grid_gb.best_params_}

    # testing
    test_en_mse.append(mean_squared_error(y_test,en[v]["best estimator"].predict(X_test)))
    test_nn_mse.append(mean_squared_error(y_test,nn[v]["best estimator"].predict(X_test)))
    test_dt_mse.append(mean_squared_error(y_test,dt[v]["best estimator"].predict(X_test)))
    test_rf_mse.append(mean_squared_error(y_test,rf[v]["best estimator"].predict(X_test)))
    test_gb_mse.append(mean_squared_error(y_test,gb[v]["best estimator"].predict(X_test)))
    print(v)

# compare generalisation errors
gen_en = np.mean(np.array(test_en_mse))
gen_nn = np.mean(np.array(test_nn_mse))
gen_dt = np.mean(np.array(test_dt_mse))
gen_rf = np.mean(np.array(test_rf_mse))
gen_gb = np.mean(np.array(test_gb_mse))

gen_err = [gen_en,gen_nn,gen_dt,gen_rf,gen_gb]

# find best overall model
best_method = np.argmin(np.array(gen_err))
if best_method == 0:
    best_model_best_method = np.argmin(np.array(test_en_mse))
    best = en[best_model_best_method]["best estimator"]
    best_param = en[best_model_best_method]["best parameters"]
elif best_method == 1:
    best_model_best_method = np.argmin(np.array(test_nn_mse))
    best = nn[best_model_best_method]["best estimator"]
    best_param = nn[best_model_best_method]["best parameters"]
elif best_method == 2:
    best_model_best_method = np.argmin(np.array(test_dt_mse))
    best = dt[best_model_best_method]["best estimator"]
    best_param = dt[best_model_best_method]["best parameters"]
elif best_method == 3:
    best_model_best_method = np.argmin(np.array(test_rf_mse))
    best = rf[best_model_best_method]["best estimator"]
    best_param = rf[best_model_best_method]["best parameters"]
else: 
    best_model_best_method = np.argmin(np.array(test_gb_mse))
    best = gb[best_model_best_method]["best estimator"]
    best_param = gb[best_model_best_method]["best parameters"]

# save model 
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_model_neurokit2.pkl"
pickle.dump(best,open(filename,"wb"))

# save model parameters
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_param_neurokit2.pkl"
pickle.dump(best_param,open(filename ,"wb"))

# save extra information
extra = ["best method:",best_method,"best model best method:",best_model_best_method,"generalisation errors mse:",gen_err,"states",states]
with open("C:/Users/Bruger/Documents/CDA/CDA/models/extra_inf_neurokit2.txt","w") as f:
    for line in extra:
        f.write(f"{line}\n")

# time 
executionTime = (time.time() - startTime)
print("execution time in seconds:" + str(executionTime))
