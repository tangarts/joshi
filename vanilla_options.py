# -*- coding: utf-8 -*-

from numpy.random import normal as rnorm
import numpy as np
from scipy.stats import norm
from math import exp, log, sqrt # python functions faster than numpy for scalar 

"""
For the black scholes model we have assumptions on the underying assets:
    - (riskless rate) The rate of return on the riskless asset is constant
    - (random walk) Log returns of a stock price is an infintesimal random walk
    with drift. ie geometric Brownian motion with constant volatility and drift.
    - The stock doesn't pay dividend.

Assumptions on the market:
    - No-arbitrage
    - Possible to buy and sell any amount of stock
    - No transaction fees
"""
#seed = 1
#np.random.seed(seed)
# Put-Call parity: C - P = exp(-rT)(F - K) = S - exp(-rT)K
class Param:
    def __init__(self, S0, K, sigma, r, d, T):
        self.S0     =  S0     #  spot price
        self.K      =  K      #  strike price
        self.sigma  =  sigma  #  volatility
        self.r      =  r      #  risk free rate
        self.d      =  d      #  dividend rate
        self.T      =  T      #  time-to-maturity

# Call option can be decomposed into difference of asset minus cash digital
# option
# TODO : digital-call, digital-put, zero-coupon bond
# variance and standard error of prices S_T for MC and EM
#

class VanillaOption(Param):
    def __init__(self, S0, K, sigma, r, d, T):
        super().__init__(S0, K, sigma, r, d, T)

    # We use the Black76 formula- substututing strike price for future price.
    def F(self):
        return exp(self.r*self.T)*self.S0

    def d1(self):
        return (log(self.F()/self.K) +\
                0.5*self.sigma**2*self.T)/(self.sigma*sqrt(self.T))
    def d2(self):
        return self.d1() - self.sigma*sqrt(self.T)

    def c(self):
        return exp(-self.r*self.T)*(self.F()*norm.cdf(self.d1()) -\
                self.K*norm.cdf(self.d2()))

    def p(self):
        return exp(-self.r*self.T)*(self.K*norm.cdf(-self.d2()) -\
                self.F()*norm.cdf(-self.d1()))

class Numerical(Param):
    def __init__(self, S0, K, sigma, r, d, T, M):
       super().__init__(S0, K, sigma, r, d, T)
       self.M = M # Number of simulations/paths

    def payoff(self, S_t):
        # call option
        return exp(-self.r*self.T)*np.mean(np.maximum(S_t - self.K, 0))
        # put option
        #return exp(-self.r*self.T)*np.mean(np.maximum(self.K - S_t, 0))

    def GBM(self):
        Z = rnorm(0, 1, self.M)
        drift = (self.r - self.d - 0.5*self.sigma**2)*self.T
        diffusion =  self.sigma*sqrt(self.T)*Z
        return self.S0*np.exp(drift + diffusion)

    def EM(self, N):
        dt = self.T / N # timestep
        S = np.zeros((self.M, N))
        Z = rnorm(0, 1, (self.M, N))
        for i in range(self.M):
            S[i][0] = self.S0
            for j in range(1, N):
                S[i][j] = S[i][j-1]*(1 + self.r*dt +\
                        self.sigma*sqrt(dt)*Z[i][j-1])
        return S[:,-1]
