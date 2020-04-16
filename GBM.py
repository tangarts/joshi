import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal as rnorm
from math import sqrt, exp

seed = 5
N = 2.**9

def brownian(seed, N):
    np.random.seed(seed)
    dt = 1.0/N
    b = rnorm(0, 1, int(N))*sqrt(dt) # brownian increment
    W = np.cumsum(b) # Brownian path
    return W, b

b = brownian(seed, N)[1]

W = brownian(seed, N)[0]
W = np.insert(W, 0, 0)

## increments
#plt.rcParams['figure.figsize'] = (10,8)
#xb = np.linspace(1, len(b), len(b))
##plt.plot(xb, b)
#plt.title('Brownian Increments')
#
## motion
#xw = np.linspace(1, len(W), len(W))
##plt.plot(xw, W)
#plt.title('Brownian Motion')

spot0 = 100; mu = 0.1; sigma = 0.2; T = 1; N = 2**9; T = 1

def GBM(spot0, mu, sigma, W, T, N):
    t = np.linspace(0, 1, N+1)
    S = []
    S.append(spot0)
    for i in range(1, int(N+1)):
        drift = (mu - 0.5*sigma**2)*t[i]
        diffusion = sigma*W[i-1]
        S_temp = spot0*exp(drift + diffusion)
        S.append(S_temp)
    return S, t

def EM(spot0, mu, sigma, b, T, N):
    dt = 1/N
    w = [spot0]
    for j in range(0, N):
        w_ = w[j]*(1 + mu*dt + sigma*rnorm(0,1)*sqrt(1/N))
        w.append(w_)
    return w, dt

