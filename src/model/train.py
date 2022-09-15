import sys
import time
import warnings
import numpy as np
import pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre
from data import *

## train models

# time 
startTime = time.time()

# Signals to include
signals = ["EDA","TEMP","BVP","HR"]
window_size = [65,65,65,65]
window_shift = [13,13,13,13]
fs = [4,4,32,1]

# initialise burn-in period
t = 10

# initialise memory
X_train = np.zeros((65,112)) # number of phases (for 13 subjects) x number of features
y_train = np.zeros((65,32)) # number of phases (for 13 subjects) x number of questions
X_test = np.zeros((5,112)) # number of phases (for one subject) x number of features
y_test = np.zeros((5,32)) # number of phases x number of questions

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

    # for every phase
    for j in range(5): 
        
        # train set
        for i in train_idx: # for every member of training
            c_train += 1 # update counter
            # feature extraction
            X_temp = [] # initialise memory
            for k in signals:
                count = 0
                if k == "EDA":
                    X_slope_tonic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_tonic)
                    X_stat_tonic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_tonic)
                    X_extra_tonic =  [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_tonic)
                    X_slope_phasic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_phasic)
                    X_stat_phasic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_phasic)
                    X_extra_phasic = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_phasic)
                    count += 1
                else:
                    X_slope = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat)
                    X_extra = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
            # assign values
            X_train[c_train,:] = X_temp[np.arange(len(X_temp))!=[83,84,85]] # features without entropy for BVP
            y_train[c_train,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets

        # test set
        for i in CV_idx: # for every member of testing
            c_test += 1 # update counter
            # feature extraction 
            X_temp = [] # initialise memory
            for k in signals:
                count = 0
                if k == "EDA":
                    X_slope_tonic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_tonic)
                    X_stat_tonic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_tonic)
                    X_extra_tonic =  [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_tonic)
                    X_slope_phasic = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_phasic)
                    X_stat_phasic = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_phasic)
                    X_extra_phasic =  [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_phasic)
                    count += 1
                else:
                    X_slope = [*pre.slope_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat)
                    X_extra = [*pre.extra_features(np.hstack(pre.burn_in(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),t,fs[count])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
            # assign values
            X_test[c_test,:] = X_temp[np.arange(len(X_temp))!=[83,84,85]] # features
            y_test[c_test,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets

    #compute values for standardisation
    X_train_mean = np.mean(X_train,axis=0)
    X_train_std = np.std(X_train,axis=0)
    # standardise train and test
    X_train = pre.Standardise(X_train,X_train_mean,X_train_std)
    X_test = pre.Standardise(X_test,X_train_mean,X_train_std)

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
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_model.pkl"
pickle.dump(best,open(filename,"wb"))

# save model parameters
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_param.pkl"
pickle.dump(best_param,open(filename ,"wb"))

# save extra information
extra = ["best method:",best_method,"best model best method:",best_model_best_method,"generalisation errors mse:",gen_err]
with open("C:/Users/Bruger/Documents/CDA/CDA/models/extra_inf.txt","w") as f:
    for line in extra:
        f.write(f"{line}\n")

# time 
executionTime = (time.time() - startTime)
print("execution time in seconds:" + str(executionTime))
