import numpy as np
import pickle
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import sys
import warnings
sys.path.insert(0, 'C:/Users/Bruger/Documents/CDA/CDA/src/data')
from utils import preprocessing_utils as pre
from data import *

with warnings.catch_warnings(): # disable the convergence warnings from elastic net
    warnings.simplefilter("ignore")

# Signals to include
signals = ["EDA","TEMP","BVP","HR"]
window_size = [50,50,50,50]
window_shift = [13,13,13,13]
fs = [4,4,32,1]

# initialise memory
X_train = np.zeros((65,82)) # number of phases (for 13 subjects) x number of features
y_train = np.zeros((65,32)) # number of phases (for 13 subjects) x number of features
X_test = np.zeros((5,82)) # number of phases (for one subject) x number of features
y_test = np.zeros((5,32)) # number of phases x number of questions

# dictonaries to save best models
en = {}
nn = {}
dt = {}
rf = {}

# list for test errors
test_en = []
test_nn = []
test_dt = []
test_rf = []

# parameters for cross-validation
param_grid_en = {"alpha": np.arange(0.25,3,0.25), "l1_ratio": np.arange(0,1,0.1)}
param_grid_nn = {"n_neighbors": range(3,10)}
param_grid_dt = {"max_depth":range(5,10) ,"min_samples_split": range(2,5),"min_samples_leaf":range(2,5), "max_features": range(25,100,25)}
param_grid_rf = {"n_estimators":range(90,111,5),"max_depth":range(5,10) ,"min_samples_split": range(2,5),"min_samples_leaf":range(2,5), "max_features": range(25,100,25)}
    
 # models
model_en = linear_model.ElasticNet()
model_nn = KNeighborsRegressor(weights="distance",p=2)
model_dt = DecisionTreeRegressor()
model_rf = RandomForestRegressor()

# initialise number of cross-validation iterations
n = 2

# make list of random subjects to exclude in each iteration (without replacement)
s = np.arange(len(sub_lst))
np.random.shuffle(s)

# leave-one-patient-out cross-validation
for v in range(n):

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
                    X_slope_tonic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_tonic)
                    X_stat_tonic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_tonic)
                    X_extra_tonic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_tonic)
                    X_slope_phasic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_phasic)
                    X_stat_phasic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_phasic)
                    X_extra_phasic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_phasic)
                    count += 1
                else:
                    X_slope = [*pre.slope_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_extra = [*pre.extra_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
            # assign values
            X_train[c_train,:] = X_temp # features
            y_train[c_train,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets

        # test set
        for i in CV_idx: # for every member of testing
            c_test += 1 # update counter
            # feature extraction 
            X_temp = [] # initialise memory
            for k in signals:
                count = 0
                if k == "EDA":
                    X_slope_tonic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_tonic)
                    X_stat_tonic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_tonic)
                    X_extra_tonic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Tonic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_tonic)
                    X_slope_phasic = [*pre.slope_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope_phasic)
                    X_stat_phasic = [*pre.stat_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_stat_phasic)
                    X_extra_phasic =  [*pre.extra_features(np.hstack(dataset[i][cond[j]]["wrist"][k]["Phasic"]),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra_phasic)
                    count += 1
                else:
                    X_slope = [*pre.slope_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_stat = [*pre.stat_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_slope)
                    X_extra = [*pre.extra_features(np.hstack(np.hstack(dataset[i][cond[j]]["wrist"][k])),window_size[count],window_shift[count],fs[count]).values()]
                    X_temp.extend(X_extra)
                    count += 1
            # assign values
            X_test[c_test,:] = X_temp # features
            y_test[c_test,:] = np.hstack([[int(a) for a in lst] for lst in [*quest[i][cond[j]].values()]]) # targets
                 
    # compute values for standardisation
    X_train_mean = np.mean(X_train,axis=0)
    X_train_std = np.std(X_train,axis=0)
    y_train_mean = np.mean(y_train,axis=0)
    y_train_std = np.std(y_train,axis=0)

    # standardise train and test
    X_train = pre.Standardise(X_train,X_train_mean,X_train_std)
    y_train = pre.Standardise(y_train,y_train_mean,y_train_std)
    X_test = pre.Standardise(X_test,X_train_mean,X_train_std)
    y_test = pre.Standardise(y_test,y_train_mean,y_train_std)

    # grid search
    grid_en = GridSearchCV(estimator=model_en,param_grid=param_grid_en,cv=5)
    grid_en.fit(X_train,y_train)
    print(1)
    grid_nn = GridSearchCV(estimator=model_nn,param_grid=param_grid_nn,cv=5)
    grid_nn.fit(X_train,y_train)
    print(2)
    grid_dt = GridSearchCV(estimator=model_dt,param_grid=param_grid_dt,cv=5)
    grid_dt.fit(X_train,y_train)
    print(3)
    grid_rf = GridSearchCV(estimator=model_rf,param_grid=param_grid_rf,cv=5)
    grid_rf.fit(X_train,y_train)
    print(4)

    # save best model for testing 
    en[v] = {"best estimator":grid_en.best_estimator, "best parameters": grid_en.best_params_}
    nn[v] = {"best estimator":grid_nn.best_estimator, "best parameters": grid_nn.best_params_}
    dt[v] = {"best estimator":grid_dt.best_estimator, "best parameters": grid_dt.best_params_}
    rf[v] = {"best estimator":grid_rf.best_estimator, "best parameters": grid_rf.best_params_}

    # testing
    test_en.append(mean_squared_error(y_test,en[v]["best estimator"].predict(X_test)))
    test_nn.aopend(mean_squared_error(y_test,nn[v]["best estimator"].predict(X_test)))
    test_dt.append(mean_squared_error(y_test,dt[v]["best estimator"].predict(X_test)))
    test_rf.append(mean_squared_error(y_test,rf[v]["best estimator"].predict(X_test)))

# compare generalisation errors
gen_en = np.mean(np.array(test_en))
gen_nn = np.mean(np.array(test_nn))
gen_dt = np.mean(np.array(test_dt))
gen_rf = np.mean(np.array(test_rf))
gen_err = [gen_en,gen_nn,gen_dt,gen_rf]

# find best overall model
best_method = np.argmin(np.array(gen_err))
if best_method == 0:
    best_model_best_method = np.argmin(np.array(test_en))
    best = en[best_model]["best estimator"]
    best_param = en[best_model]["best parameters"]
elif best_method == 1:
    best_model_best_method = np.argmin(np.array(test_nn))
    best = nn[best_model]["best estimator"]
    best_param = nn[best_model]["best parameters"]
elif best_method == 2:
    best_model_best_method = np.argmin(np.array(test_dt))
    best = dt[best_model]["best estimator"]
    best_param = dt[best_model]["best parameters"]
else: 
    best_model_best_method = np.argmin(np.array(test_rf))
    best = rf[best_model]["best estimator"]
    best_param = rf[best_model]["best parameters"]

# save model
filename = "C:/Users/Bruger/Documents/CDA/CDA/models/best_model.pkl"
pickle.dump(best,open(filename,"wb"))

# save extra information
extra = ["best method:",best_method,"best model best method",best_model_best_method,"generalisation errors",gen_err]
with open("C:/Users/Bruger/Documents/CDA/CDA/models/extra_inf.txt","w") as f:
    for line in extra:
        f.write(f"{line}\n")

print("best method",best_method,"best model best method",best_model_best_method,"best parameters", best_param,"generalisation error", gen_err)