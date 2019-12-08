#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numpy.random import normal as rnorm
import numpy as np
from scipy.stats import norm
from math import exp, log, sqrt # python functions faster than numpy for scalar 
from param import Param

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
# Call option can be decomposed into difference of asset minus cash digital
# option
#
#
# TODO:
#
# price of:
#     digital-call,exp(-r*T)*norm(d2)
#     digital-put, exp(-r*T)*norm(-d2)
#     zero-coupon bond, at time T = exp(-rT) (discount factor?) 
#
# variance and standard error of prices S_T for MC and EM


class BlackScholes(Param):
    def __init__(self, S0, K, sigma, r, delta, T):
        super().__init__(S0, K, sigma, r, delta, T)

        self.d1 = (log(self.S0/self.K) + (self.r - self.delta + 0.5*self.sigma**2)*self.T)/(self.sigma*sqrt(self.T))

        self.d2 = self.d1 - self.sigma*sqrt(self.T)

    def Bond(self):
        return exp(-self.r*self.T)

    def F(self, X, mu):
        return X*exp(-mu*self.T)


    def call(self):
        return self.F(self.S0, self.delta)*norm.cdf(self.d1) -\
               self.F(self.K, self.r)*norm.cdf(self.d2)

    def put(self):
        return self.F(self.K, self.r)*norm.cdf(-self.d2) -\
                self.F(self.S0, self.delta)*norm.cdf(-self.d1)

    def digitalCall(self): return exp(-self.r*self.T)*norm.cdf(self.d2)

    def digitalPut(self): return exp(-self.r*self.T)*norm.cdf(-self.d2)
    
    def Delta(self, op):
        if op == 'p':
            return -exp(-self.delta*self.T)*norm.cdf(-self.d1)
        elif op == 'c':
            return exp(-self.delta*self.T)*norm.cdf(self.d1)

    def Gamma(self): return exp(-self.delta*self.T)*norm.pdf(self.d1) / (self.S0*self.sigma*sqrt(self.T))
    def Vega(self): return exp(-self.delta*self.T)*self.S0*norm.pdf(self.d1)*sqrt(self.T)

    def Theta(self, op_type):
        if op_type == 'c':
            return self.delta*self.S0*exp(-self.delta*self.T)*norm.cdf(self.d1) \
                    - (exp(-self.r*self.T)*self.K*norm.pdf(self.d2)*self.sigma) / (2*sqrt(self.T)) \
                    - self.r*self.K*exp(-self.r*self.T)*norm.cdf(self.d2)
        elif op_type == 'p':
            return - self.delta*self.S0*exp(-self.delta*self.T)*norm.cdf(-self.d1) \
                    - (exp(-self.delta*self.T)*self.S0*norm.pdf(self.d1)*self.sigma) / 2*sqrt(self.T) \
                    + self.r*self.K*exp(-self.r*self.T)*norm.cdf(-self.d2)

    def Rho(self, op_type):
        if op_type == 'c':
            return self.K*self.T*exp(-self.r*self.T)*norm.cdf(self.d2)
        elif op_type == 'p':
            return -self.K*self.T*exp(-self.r*self.T)*norm.cdf(-self.d2)


    def perpetual(self,option):
        if option == 'c':
            h1 = 0.5 - (self.r - self.delta)/self.sigma**2 + sqrt(((self.r - self.delta)/self.sigma**2 - 0.5)**2 + 2*self.r/self.sigma**2)
            H = self.K*(h1 / (h1-1))
            payoff = (H - self.K)* (self.S0/H)**h1
            return self.K/(h1 - 1) * (self.S0/self.K * (h1 - 1) / h1)**h1

        elif option == 'p':
            h2 = 0.5 - (self.r - self.delta)/self.sigma**2 - sqrt(((self.r - self.delta)/self.sigma**2 - 0.5)**2 + 2*self.r/self.sigma**2)
            H = self.K*(h2 / (h2-1))
            payoff = (self.K - H)* (self.S0/H)**h2
            return self.K/(1 - h2) * (self.S0/self.K * (h2 - 1) / h2)**h2


class Numerical(Param):
    def __init__(self, S0, K, sigma, r, d, T, M):
       super().__init__(S0, K, sigma, r, d, T)
       self.M = M # Number of simulations/paths

    def payoff(self, S_t, option):
        if option == 'c':
            return exp(-self.r*self.T)*np.mean(np.maximum(S_t - self.K, 0))
        elif option == 'p':
            return exp(-self.r*self.T)*np.mean(np.maximum(self.K - S_t, 0))

    def GBM(self):
        Z = rnorm(0, 1, self.M)
        drift = (self.r - self.delta - 0.5*self.sigma**2)*self.T
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
