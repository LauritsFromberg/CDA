import numpy as np 
from scipy import signal
from scipy import stats as stats
from scipy import integrate as inte

## functions for pre-processing

# helper function for Butterworth filter
def extract_coefficients(p, degree):
    n = degree + 1
    sample_x = [ x for x in range(n) ]
    sample_y = [ p(x) for x in sample_x ]
    A = [ [ 0 for _ in range(n) ] for _ in range(n) ]
    for line in range(n):   
        for column in range(n):
            A[line][column] = sample_x[line] ** column
    c = np.linalg.solve(A, sample_y)
    return c[::-1]

def Standardise(X,mean,std):
    X = (X-mean)/std
    return X

def burn_in(X,burn,fs):
    n_exclude = burn*fs
    X = X[n_exclude:]
    return X

def EMA(X,a):
    S = np.zeros(len(X))
    S[0] = X[0]
    for i in range(len(X)-1):
        S[i+1] = a*X[i+1]+(1-a)*S[i]
    return S

def Butter(X,order,cut_off,fs):
    sos = signal.butter(N=order,Wn=cut_off,btype="lowpass",output="sos",fs=fs)
    filtered = signal.sosfilt(sos,X)
    return filtered

def Butter_self(X,order,cut_off,fs):

    # compute pre-warping frequency
    delta = np.tan((np.pi*cut_off)/fs)

    # compute coefficients
    a = np.zeros(order+1) # initialise memory
    a[0] = 1 # first coefficient is one
    g = np.pi/(2*order) # constant
    for i in range(1,order+1):
        a[i] = a[i-1] * (np.cos((i-1)*g))/(np.sin((i*g))) # recursive formula
    a[order] = 1 # last coefficient is one

    # multiply through with delta to simplify into suitable form
    for i in range(order+1):
        a[i] = a[i]*(delta**order)

    # compute Butterworth polynomial
    H = lambda x: sum(a[i] * ((x/delta)**i) for i in list(range(order+1))) # normalise by frequency pre-warping 
    coef_butt = extract_coefficients(H, order) # extract coefficient for later use

    # transform into z-domain 
    top = lambda z: (delta**order) * ((z+1)**order) # compute enumerator polynomial
    bot = lambda z: sum(coef_butt[i]*((z-1)**(order-i))*((z+1)**i)  for i in list(range(order+1))) # compute denominator polynomial

    # coefficients for difference equation
    top_coef = extract_coefficients(top, order)
    bot_coef = extract_coefficients(bot, order)

    # normalise by first bottom coefficient to enable filtering 
    top_coef = top_coef/bot_coef[0]
    bot_coef = bot_coef/bot_coef[0]
    
    # filter
    Y = np.zeros(len(X)) # initialise memory
    for i in range(len(X)):
        Y[i] =  sum(-bot_coef[j]*Y[i-j] for j in list(range(1,order+1))) + sum(top_coef[j]*X[i-j] for j in list(range(order+1))) 

    return Y

def slope_features(X,window_size,window_shift,fs):

    # initialise memory
    slope = []

    # calculate number of points from specified frequency
    window_size = window_size * fs 
    window_shift = window_shift * fs 

    # find indices 
    indices = np.arange(len(X))

    # compute all windows 
    windows = [indices[i:i+window_size] for i in range(0,len(X),window_size-window_shift)]

    # compute slope for each window and append to list
    for i in range(len(windows)):
        y2 = np.max(X[windows[i]])
        y1 = np.min(X[windows[i]])
        slope.append((y2-y1)/window_size)
    
    return {"max": max(slope), "min": min(slope),"avg": sum(slope)/len(slope)}


#slope_feat = slope_features(np.arange(40),5,1,1)
#print(slope_feat)

def stat_features(X,window_size,window_shift,fs):
    
    # initialise memory
    mean = []
    median = []
    std = []
    #skewness = []
    #kurtosis = []
    min_ = []
    max_ = []
    entropy = []

    # calculate number of points from specified frequency
    window_size = window_size * fs 
    window_shift = window_shift * fs 

    # find indices 
    indices = np.arange(len(X))

    # compute all windows 
    windows = [indices[i:i+window_size] for i in range(0,len(X),window_size-window_shift)]

    # compute statistics for each window and append to list
    for i in range(len(windows)):
        temp = X[windows[i]]
        mean.append(np.mean(temp))
        median.append(np.median(temp))
        std.append(np.std(temp))
        min_.append(np.min(temp))
        max_.append(np.max(temp)) 
        entropy.append(stats.entropy(temp))
        
    return {"max mean": max(mean), "min mean": min(mean), "avg mean": sum(mean)/len(mean),
            "max median": max(median), "min median": min(median), "avg median": sum(median)/len(median),
            "max std": max(std), "min std": min(std), "avg std": sum(std)/len(std),
            "avg min": sum(min_)/len(min_),"avg max": sum(max_)/len(max_), "min entropy": min(entropy),
            "max entropy": max(entropy), "avg entropy": sum(entropy)/len(entropy)}

def extra_features(X,window_size,window_shift,fs):

    # initialise memory
    avg_gradient = []
    abs_int = []

    # calculate number of points from specified frequency
    window_size = window_size * fs 
    window_shift = window_shift * fs 

    # find indices 
    indices = np.arange(len(X))

    # compute all windows 
    windows = [indices[i:i+window_size] for i in range(0,len(X),window_size-window_shift)]

    # compute features for each window and append to list
    for i in range(len(windows)):
        temp = X[windows[i]]
        avg_gradient.append(np.mean(np.gradient(temp))) # note: numpy uses "complex" finite difference method
        abs_int.append(abs(inte.simpson(temp,windows[i]))) # absolute integral

    return {"max avg_gradient": max(avg_gradient),"min avg_gradient": min(avg_gradient), "avg avg_gradient": sum(avg_gradient)/len(avg_gradient),
            "max absolute integral": max(abs_int), "min absolute integral": min(abs_int), "avg absolute integral": sum(abs_int)/len(abs_int)}


