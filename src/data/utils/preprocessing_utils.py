import numpy as np 
from scipy import signal

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
    X = X-mean
    X = X/std
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


t = np.linspace(0, 1, 1000, False)  # 1 second
sig = np.sin(2*np.pi*10*t) + np.sin(2*np.pi*20*t)

b,a = signal.butter(N=2, Wn=2, fs=1000)
filtered = signal.lfilter(b,a,sig)
filtered1 = Butter_self(sig,2,2,1000)


#print(sum(filtered-filtered1))
