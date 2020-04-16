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
# Put-Call parity: C - P = exp(-rT)(F - K) = S - exp(-rT)K
class Param():

    DEFAULT_BINOMIAL_TREE_NUM_STEPS = 25
    DEFAULT_MONTE_CARLO_NUM_STEPS = 50
    DEFAULT_MONTE_CARLO_NUM_PATHS = 100

    def __init__(self, spot0, K, sigma, r, d, T, opt_type, exer_type):
        self.spot0     =  spot0     #  spot price
        self.K      =  K      #  strike price
        self.sigma  =  sigma  #  volatility
        self.r      =  r      #  risk free rate
        self.delta      =  d      #  dividend rate
        self.T      =  T      #  time-to-maturity
        self.opt_type = opt_type or OptionType.CALL 
        self.exer_type = exer_type or OptionExerciseType.EUROPEAN

       
   
    def print_parameters(self):
        """print parameters"""
        print("---------------------------------------------") 
        print("---------------------------------------------")
        print("Parameters of Option Pricer:")
        print("---------------------------------------------")
        print("Underlying Asset Price = ", self.spot0)
        print("Strike Price = ", self.K)
        print("Volatility = ", self.sigma)
        print("Risk-Free Rate = ", self.r)
        print("Dividend Rate = ", self.delta)
        print("Time to Maturity (years) = ", self.T)
        print("---------------------------------------------")
        print("---------------------------------------------")

    def enum(**enums): return type('Enum', (), enums)

    OptionType = enum(CALL='call', PUT='put')
    OptionExerciseType = enum(EUROPEAN='european', AMERICAN='american')

    OptionModel = enum(BLACK_SCHOLES='black_scholes', BINOMIAL_TREE='binomial_tree', MONTE_CARLO='monte_carlo')
    OptionMeasure = enum(VALUE='value', DELTA='delta', THETA='theta', RHO='rho', VEGA='vega', GAMMA='gamma')

