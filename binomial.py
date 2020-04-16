

#!/usr/bin/env python3

import numpy as np
from math import sqrt, exp
from param import Param


class Binomial(Param):
    def __init__(self, spot0, K, sigma, r, delta, T, N):
        super().__init__(spot0, K, sigma, r, delta, T)
        self. N = N

        self.h = T / N
        self.u = exp((r-delta)*self.h + sigma*sqrt(self.h))
        self.d = exp((r-delta)*self.h - sigma*sqrt(self.h))
        self.q = (exp((r-delta)*self.h) - self.d) / (self.u - self.d)

    # true probability
    def p(self, alpha):
        return (exp(alpha*self.h) - self.d) / (self.u - self.d)

    def stock(self):
        # make stock price tree
        stock = np.zeros([self.N + 1, self.N + 1])
        for i in range(self.N + 1):
            for j in range(i + 1):
                stock[j, i] = self.spot0 * (self.u ** (i - j)) * (self.d ** j)
        return stock
    
    def european(self, o_type):
        # Generate option prices recursively
        stock = self.stock()
        option = np.zeros([self.N + 1, self.N + 1])

        if o_type == 'c':
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (stock[:, self.N] - self.K))
            for i in range(self.N - 1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)* \
                    (self.q*option[j, i + 1] + (1-self.q)*option[j + 1, i + 1])

        elif o_type == 'p':
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (self.K - stock[:, self.N]) ) # put payoff
            for i in range(self.N - 1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)* \
                    (self.q*option[j, i + 1] + (1-self.q)*option[j + 1, i + 1])

        return option

    def american(self, o_type):
        stock = self.stock()
        option = np.zeros([self.N + 1, self.N + 1])
        if o_type == 'c':
            payoff = np.maximum( np.triu(stock - self.K), 0 )
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (stock[:, self.N] - self.K))
            for i in range(self.N -1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)* \
                            (self.q*max(option[j, i + 1], payoff[j, i + 1]) + (1-self.q)*max(option[j + 1, i + 1], payoff[j + 1, i + 1]))     

        elif o_type == 'p':
            payoff = np.maximum( np.triu(self.K - stock), 0 )
            option[:, self.N] = np.maximum(np.zeros(self.N + 1), (self.K - stock[:, self.N]))
            for i in range(self.N -1, -1, -1):
                for j in range(0, i + 1):
                    option[j, i] = exp(-self.r*self.h)* \
                            (self.q*max(option[j, i + 1], payoff[j, i + 1]) + (1-self.q)*max(option[j + 1, i + 1], payoff[j + 1, i + 1])) 

        return option

    def replicate_portfolio(self, op):
        option = self.european(op)
        stock = self.stock()
        delta = np.zeros([self.N + 1, self.N + 1])
        B = np.zeros([self.N + 1, self.N + 1])
        for i in range(self.N - 1, -1, -1):
            for j in range(0, i + 1):
                delta[j, i] = exp(-self.delta*self.h)*(option[j , i + 1] - option[j + 1, i + 1]) / ( (self.u - self.d)*stock[j,i] )
                B[j, i] = exp(-self.r*self.h)*(option[j , i + 1]*self.d - option[j + 1, i + 1]*self.u) / (self.u - self.d)
        return delta

