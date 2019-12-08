#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
class Param():
    def __init__(self, S0, K, sigma, r, d, T):
        self.S0     =  S0     #  spot price
        self.K      =  K      #  strike price
        self.sigma  =  sigma  #  volatility
        self.r      =  r      #  risk free rate
        self.delta      =  d      #  dividend rate
        self.T      =  T      #  time-to-maturity
        
    
    def print_parameters(self):
        """print parameters"""
        print("---------------------------------------------") 
        print("---------------------------------------------")
        print("Parameters of Option Pricer:")
        print("---------------------------------------------")
        print("Underlying Asset Price = ", self.S0)
        print("Strike Price = ", self.K)
        print("Volatility = ", self.sigma)
        print("Risk-Free Rate = ", self.r)
        print("Dividend Rate = ", self.delta)
        print("Time to Maturity (years) = ", self.T)
        print("---------------------------------------------")
        print("---------------------------------------------")


