import numpy as np 
from scipy import signal
from scipy.stats import stats
from scipy.fft import rfft
import pandas as pd
import antropy as ant
import neurokit2 as nk

## functions for pre-processing

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

def slope_features(X,window_size,window_shift,fs):

    # initialise memory
    slope = []

    # calculate number of points from specified sample-rate
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

    return {"max": max(slope),"min": min(slope),"avg": sum(slope)/len(slope),"overall":(np.max(X)-np.min(X))/len(X)}
    

def time_features(signal, window_size, window_shift, fs):
    
    # counter 
    count = -1

    df = pd.DataFrame()

    for key, item in signal.items():
        
        count += 1 # update counter

    
        if key == "Phasic":
            df.at[0,"{}_time_abs_int".format(key)] = np.trapz(item)
            mobility, complexity = ant.hjorth_params(item)
            df.at[0,"{}_hjorth_mobility".format(key)] = mobility
            df.at[0,"{}_hjorth_complexity".format(key)] = complexity
        elif key != "EDA":
            df.at[0,"{}_time_mean".format(key)] = item.mean()
            df.at[0,"{}_time_median".format(key)] = np.median(item)
            df.at[0,"{}_time_std".format(key)] = item.std()
            df.at[0,"{}_time_min".format(key)] = item.min()
            df.at[0,"{}_time_max".format(key)] = item.max()
            df.at[0,"{}_time_abs_int".format(key)] = np.trapz(item)
            grad = np.gradient(item)
            df.at[0,"{}_time_avg_grad".format(key)] = grad.mean()
            df.at[0,"{}_time_avg_neg_grad".format(key)] = grad[grad<0].mean()
            slope_dict = slope_features(item,window_size[count],window_shift[count],fs[count])
            df.at[0,"{}_max_slope".format(key)] = slope_dict["max"]
            df.at[0,"{}_min_slope".format(key)] = slope_dict["min"]
            df.at[0,"{}_avg_slope".format(key)] = slope_dict["avg"]
            df.at[0,"{}_overall_slope".format(key)] = slope_dict["overall"]
            if key == "HR":
                df.at[0,"{}_time_kurtosis".format(key)] = stats.kurtosis(item)
                df.at[0,"{}_time_skewness".format(key)] = stats.skew(item)

    return df

def freq_features(signal, fs):

    # counter 
    count = -1

    df = pd.DataFrame()

    for key, item in signal.items():

        count += 1 # update counter

        if key == "BVP" or key == "Tonic" or key == "Phasic":          

            # transform to frequency domain using the Fourier transform
            freq_item = rfft(item)

            df.at[0,"{}_freq_mean".format(key)] = freq_item.mean()
            df.at[0,"{}_freq_median".format(key)] = np.median(freq_item)
            df.at[0,"{}_freq_std".format(key)] = freq_item.std()
            df.at[0,"{}_freq_min".format(key)] = freq_item.min()
            df.at[0,"{}_freq_max".format(key)] = freq_item.max()
            if key != "BVP":
                df.at[0,"{}_freq_kurtosis".format(key)] = stats.kurtosis(freq_item)
                df.at[0,"{}_freq_skewness".format(key)] = stats.skew(freq_item)
            df.at[0,"{}_freq_sma".format(key)] = freq_item.sum()
            df.at[0,"{}_freq_iqr".format(key)] = stats.iqr(freq_item)
            spectral_ent = ant.spectral_entropy(item,sf=fs[count],method="welch", normalize=True)
            df.at[0,"{}_freq_spectral_entropy".format(key)] = spectral_ent

    # divide into real and imaginary parts respectively

    col_names = df.columns
    real_col_names = [x + "_real" for x in col_names]
    im_col_names = [x + "_im" for x in col_names]
    real_val = df.values.real
    im_val = df.values.imag

    df = pd.concat([pd.DataFrame(real_val,columns = real_col_names),pd.DataFrame(im_val, columns = im_col_names)], axis = 1)

    return df 

def EDA_extra_features(signal, fs):

    df = pd.DataFrame()

    # counter
    count = -1

    for key, item in signal.items():

        # update counter 
        count += 1 

        if key == "EDA":
            eda_symp = nk.eda_sympathetic(item, fs[count], method = "ghiasi", show = False)
            df.at[0,"EDA_symp"] = eda_symp["EDA_Symp"]
            df.at[0,"EDA_sympN"] = eda_symp["EDA_SympN"]
        elif key == "Phasic":
            n_peaks  = nk.eda_peaks(item, fs[count], amplitude_min = 0.03)
            df.at[0,"EDA_phasic_n_peaks"] = n_peaks[1]["SCR_Peaks"].size
            df.at[0,"EDA_phasic_avg_rise"] = np.nanmean(n_peaks[1]["SCR_RiseTime"])
            df.at[0,"EDA_phasic_avg_recovery"] = np.nanmean(n_peaks[1]["SCR_RecoveryTime"])
            
    return df 