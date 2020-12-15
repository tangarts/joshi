# -*- coding: utf-8 -*-

from numpy.random import normal as rnorm
import numpy as np
from scipy.stats import norm
from math import exp, log, sqrt # python functions faster than numpy for scalar 

# Put-Call parity: C - P = exp(-rT)(F - K) = S - exp(-rT)K
# Call option can be decomposed into difference of asset minus cash digital
# option

#
# TODO:
#

class Param():
    """
    Parameter class shared across models
    """
    DEFAULT_BINOMIAL_TREE_NUM_STEPS = 25
    DEFAULT_MONTE_CARLO_NUM_STEPS = 50
    DEFAULT_MONTE_CARLO_NUM_PATHS = 100

    def enum(**enums): return type('Enum', (), enums)
    
    OptionType = enum(CALL='call', PUT='put')
    OptionExerciseType = enum(EUROPEAN='european', AMERICAN='american')

    Model = enum(BLACK_SCHOLES='black_scholes', BINOMIAL_TREE='binomial_tree', MONTE_CARLO='monte_carlo')

    OptionMeasure = enum(VALUE='value', DELTA='delta', THETA='theta', RHO='rho', VEGA='vega', GAMMA='gamma')

    def __init__(self, spot0, K, sigma, r, d, T, opt_type=None, exer_type=None):
        """

        args:
           spot0: int
                spot price
           K: int
                strike price
           sigma: float
                volatility
           r: float
                risk-free rate
           d: float
                dividend rate
           T: float 
                time to maturity
           opt_type: Enum
                option type {CALL, PUT}
           exer_type: Enum
                exercise type {AMERICAN, EUROPEAN}
            
        """
        self.spot0  =  spot0    
        self.K      =  K      
        self.sigma  =  sigma  
        self.r      =  r     
        self.delta  =  d      
        self.T      =  T      
        self.opt_type = opt_type or self.OptionType.CALL 
        self.exer_type = exer_type or self.OptionExerciseType.EUROPEAN


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


    
"""
For the Black-Scholes model we have assumptions on the underlying assets:
    - (riskless rate) The rate of return on the riskless asset is constant
    - (random walk) Log returns of a stock price is an infinitesimal random walk
    with drift. i.e. Geometric Brownian motion with constant volatility and drift.
    - The stock doesn't pay dividend.

Assumptions on the market:
    - No-arbitrage
    - Possible to buy and sell any amount of stock
    - No transaction fees
"""


class BlackScholes(Param):

    def __init__(self, spot0, K, sigma, r, delta, T, opt_type=None, exer_type=None):
        super().__init__(spot0, K, sigma, r, delta, T, opt_type, exer_type)

        assert self.exer_type == self.OptionExerciseType.EUROPEAN, \
                "Black-Scholes does not support early exercise"

    def bond(self): return exp(-self.r*self.T)
    
    def forward(self): return self.spot0*exp(-self.delta*self.T)

    def d1(self):
        return (log(self.spot0/self.K)\
                + (self.r - self.delta + 0.5*self.sigma**2)
                * self.T)\
                / self.sigma*sqrt(self.T)

    def d2(self): return self.d1() - self.sigma*sqrt(self.T)

    def vanilla_price(self):
        if self.opt_type == self.OptionType.CALL:
            return self.spot0*exp(-self.delta*self.T)*norm.cdf(self.d1())\
                    - self.K*exp(-self.r*self.T)*norm.cdf(self.d2())

        elif self.opt_type == self.OptionType.PUT:
            return self.K*exp(-self.r*self.T)*norm.cdf(-self.d2())\
                    - self.spot0*exp(-self.delta*self.T)*norm.cdf(-self.d1())

    def digital_price(self):
        if self.opt_type == self.OptionType.CALL:
            return exp(-self.r*self.T)*norm.cdf(self.d2())
        else:
            return exp(-self.r*self.T)*norm.cdf(-self.d2())
    
    def get_greeks(self):
        the_greeks = {
                'delta': self.Delta(),
                'gamma': self.Gamma(),
                'theta': self.Theta(),
                'vega':  self.Vega(),
                'rho':   self.Rho()
                }
        return the_greeks
    
    def Delta(self):
        if self.opt_type == 'put':
            return -exp(-self.delta*self.T)*norm.cdf(-self.d1())
        elif self.opt_type == 'call':
            return exp(-self.delta*self.T)*norm.cdf(self.d1())

    def Gamma(self):
        return exp(-self.delta*self.T)*norm.pdf(self.d1()) / (self.spot0*self.sigma*sqrt(self.T))

    def Vega(self):
        return exp(-self.delta*self.T)*self.spot0*norm.pdf(self.d1())*sqrt(self.T)

    def Theta(self):
        if self.opt_type == 'call':
            return self.delta*self.spot0*exp(-self.delta*self.T)*norm.cdf(self.d1()) \
                    - (exp(-self.r*self.T)*self.K*norm.pdf(self.d2())*self.sigma) / (2*sqrt(self.T)) \
                    - self.r*self.K*exp(-self.r*self.T)*norm.cdf(self.d2())

        elif self.opt_type == 'put':
            return -self.delta*self.spot0*exp(-self.delta*self.T)*norm.cdf(-self.d1()) \
                    - (exp(-self.delta*self.T)*self.spot0*norm.pdf(self.d1())*self.sigma) / 2*sqrt(self.T) \
                    + self.r*self.K*exp(-self.r*self.T)*norm.cdf(-self.d2())

    def Rho(self):
        if self.opt_type == 'call':
            return self.K*self.T*exp(-self.r*self.T)*norm.cdf(self.d2())
        elif self.opt_type == 'put':
            return -self.K*self.T*exp(-self.r*self.T)*norm.cdf(-self.d2())


    def perpetual(self):
        if self.opt_type == 'call':
            h1 = 0.5 - (self.r - self.delta)/self.sigma**2\
                     + sqrt(((self.r - self.delta)/self.sigma**2 - 0.5)**2\
                     + 2*self.r/self.sigma**2)

            H = self.K*(h1 / (h1-1))
            payoff = (H - self.K)* (self.spot0/H)**h1
            return self.K/(h1 - 1) * (self.spot0/self.K * (h1 - 1) / h1)**h1, H

        elif self.opt_type == 'put':
            h2 = 0.5 - (self.r - self.delta)/self.sigma**2\
                     - sqrt(((self.r - self.delta)/self.sigma**2 - 0.5)**2\
                     + 2*self.r/self.sigma**2)

            H = self.K*(h2 / (h2-1))
            payoff = (self.K - H)* (self.spot0/H)**h2
            return self.K/(1 - h2) * (self.spot0/self.K * (h2 - 1) / h2)**h2, H

    @staticmethod
    def gap_option(S, K1, K2, sigma, r, delta, T):
        """ compute gap option """

        d1 = log(S*exp(-delta*T)/(K2*exp(-r*T))) + 0.5*sigma**2*T \
                / sigma*sqrt(T)
        d2 = d1 - sigma*sqrt(T)

        return S*exp(-delta*T)*norm.cdf(d1) - K1*exp(-r*T)*norm.cdf(d2)


    @staticmethod
    def exchange_option(S, K, sig_s, sig_k, delta_s, delta_k, rho, T):

        """
        With a call we give up cash to acquire stock. The dividend yield on
        cash is the interest rate. This if we set delta_s = delta, delta_k = r,
        and sigma_k = 0, the formula reduces to standard Black-Scholes formula
        for a call

        With a put, we give up stock to acquire cash. Thus setting delta_s = r,
        delta_k = delta and sigma_s = 0, the formula reduces to the
        Black-Scholes formula for a put option.
        """

        sigma = sqrt(sig_s**2 + sig_k**2 - 2*rho*sig_s*sig_k)

        d1 = log(S*exp(-delta_s*T)/(K*exp(-delta_k*T))) + 0.5*sigma**2*T \
                / sigma*sqrt(T)

        d2 = d1 - sigma*sqrt(T)

        return S*exp(-delta_s*T)*norm.cdf(d1) - K*exp(-delta_k*T)*norm.cdf(d2)


class Numerical(Param):
    """
    Numerical methods 
    """

    def __init__(self, spot0, K, sigma, r, d, T, opt_type=None, exer_type=None, M=None, N=None):
        """
        args:
            Params
        """
        super().__init__(spot0, K, sigma, r, d, T, opt_type, exer_type)
        self.M = M or self.DEFAULT_MONTE_CARLO_NUM_PATHS # Number of simulations/paths
        self.N = N or self.DEFAULT_MONTE_CARLO_NUM_STEPS


    def payoff(self, S_t):
        if self.opt_type == 'call':
            return exp(-self.r*self.T)*np.mean(np.maximum(S_t - self.K, 0))
        elif self.opt_type == 'put':
            return exp(-self.r*self.T)*np.mean(np.maximum(self.K - S_t, 0))

    def GBM(self):
        Z = rnorm(0, 1, self.M)
        drift = (self.r - self.delta - 0.5*self.sigma**2)*self.T
        diffusion =  self.sigma*sqrt(self.T)*Z
        return self.spot0*np.exp(drift + diffusion)


    def generate_gbm_path(self):
        dt = self.T / self.N
        t = np.linspace(dt, self.T, self.N)
        dW = rnorm(0, dt**0.5, self.N) # (self.M, self.N))
        W = np.cumsum(dW) #, axis=1) if self.M > 1 else np.cumsum(dW)
        drift = (self.r - self.delta - 0.5*self.sigma**2)*t
        diffusion = self.sigma*W
        return self.spot0*np.exp(drift + diffusion)

    def EM(self):
        dt = self.T / self.N # timestep
        S = np.zeros((self.M, self.N))
        W = rnorm(0, dt**0.5, (self.M, self.N))
        mu = (self.r - self.delta)
        for i in range(self.M):
            S[i, 0] = self.spot0
            for j in range(1, self.N):
                S[i, j] = S[i, j-1] * (1 + mu*dt + self.sigma*W[i, j-1])
        return S[:,-1]

    def value(self): return self.payoff(self.GBM())

    def error(self, steps):
        exact = BlackScholes(self.spot0, self.K, self.sigma, self.r,\
                self.delta, self.T, self.opt_type, self.exer_type).vanilla_price()
        simulated_price = [self.value() for self.M in steps]
        error = abs(simulated_price - exact)
        return error

class Binomial(Param):

    def __init__(self, spot0, K, sigma, r, delta, T, opt_type=None, exer_type=None, N=None):
        super().__init__(spot0, K, sigma, r, delta, T, opt_type, exer_type)
        self.N = N or self.DEFAULT_BINOMIAL_TREE_NUM_STEPS

    @property
    def h(self): return self.T / self.N 
    @property
    def u(self): return exp((self.r-self.delta)*self.h + self.sigma*sqrt(self.h))
    @property
    def d(self): return exp((self.r-self.delta)*self.h - self.sigma*sqrt(self.h))
    @property
    def q(self): return (exp((self.r-self.delta)*self.h) - self.d) / (self.u - self.d)

    # true probability
    def p(self, alpha): return (exp(alpha*self.h) - self.d) / (self.u - self.d)

    def stock(self):
        # make stock price tree
        stock = np.zeros([self.N + 1, self.N + 1])
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock[j, i] = self.spot0 * (self.u ** (i - j)) * (self.d ** j)
        return stock
    
    def european(self):
        # Generate option prices recursively
        stock = self.stock()
        option = np.zeros([self.N + 1, self.N + 1])

        if self.opt_type == 'call':
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (stock[:, self.N] - self.K))
            for i in range(self.N - 1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)\
                            * (self.q*option[j, i + 1]\
                            + (1-self.q)*option[j + 1, i + 1])

        elif self.opt_type == 'put':
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (self.K - stock[:, self.N]) ) # put payoff
            for i in range(self.N - 1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)\
                            * (self.q*option[j, i + 1]\
                            + (1-self.q)*option[j + 1, i + 1])

        return option

    def american(self):

        stock = self.stock()
        option = np.zeros([self.N + 1, self.N + 1])

        if self.opt_type == 'call':
            payoff = np.maximum( np.triu(stock - self.K), 0 )
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (stock[:, self.N] - self.K))
            for i in range(self.N -1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)\
                            * (self.q*max(option[j, i + 1], payoff[j, i + 1])\
                            + (1-self.q)*max(option[j + 1, i + 1], payoff[j + 1, i + 1]))

        elif self.opt_type == 'put':
            payoff = np.maximum( np.triu(self.K - stock), 0 )
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (self.K - stock[:, self.N]))
            for i in range(self.N -1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)\
                            * (self.q*max(option[j, i + 1], payoff[j, i + 1])\
                            + (1-self.q)*max(option[j + 1, i + 1], payoff[j + 1, i + 1])) 

        return option

    def replicate_portfolio(self):
        option = self.european()
        stock = self.stock()
        delta = np.zeros([self.N + 1, self.N + 1])
        B = np.zeros([self.N + 1, self.N + 1])
        for i in range(self.N - 1, -1, -1):
            for j in range(0, i + 1):
                delta[j, i] = exp(-self.delta*self.h)\
                        * (option[j , i + 1] - option[j + 1, i + 1])\
                        / ( (self.u - self.d)*stock[j,i] )
                B[j, i] = exp(-self.r*self.h)*(option[j , i + 1]\
                        * self.d - option[j + 1, i + 1] * self.u) / (self.u - self.d)
        return delta

class Hedge:

    def stop_loss():
        pass

    def delta_hedge():
        pass

