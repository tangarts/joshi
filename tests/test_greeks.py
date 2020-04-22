#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import sys
import os
sys.path.append(os.getcwd() + '/..')

from joshi.model import BlackScholes


EPSILON = 1e-5
TOL = 1e-4

class TestGreeks(unittest.TestCase):
    def test_delta(self):
        """ 
        Approximate delta calulated via finite difference
        delta equal to spot derivative of call option
        """
        option = BlackScholes(spot0=100, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')
        option_plus_epsilon = BlackScholes(spot0=100+EPSILON, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')

        approximate_delta = ( option_plus_epsilon.vanilla_price() - option.vanilla_price() ) / EPSILON
        
        assert option.Delta() - approximate_delta < TOL
        print("test_delta: PASS")

    def test_gamma(self):
        """
        Approximate gamma calulated via finite difference
        gamma equal to second spot derivative of call option 
        """
        option = BlackScholes(spot0=100, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')
        option_plus_epsilon = BlackScholes(spot0=100+EPSILON, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')

        option_minus_epsilon = BlackScholes(spot0=100-EPSILON, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')
        approximate_gamma = ( option_plus_epsilon.vanilla_price() -\
                2*option.vanilla_price() + option_minus_epsilon.vanilla_price() ) / EPSILON**2

        assert option.Gamma() - approximate_gamma < TOL
        print("test_gamma: PASS")

    def test_vega(self):
        """ 
        Approximate vega calulated via finite difference
        vega equal to volatility derivative of call option
        """
        option = BlackScholes(spot0=100, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')
        option_plus_epsilon = BlackScholes(spot0=100, K=100, \
                                sigma=0.3+EPSILON, r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')

        approximate_vega = ( option_plus_epsilon.vanilla_price() - option.vanilla_price() ) / EPSILON

        assert option.Vega() - approximate_vega < TOL
        print("test_vega: PASS")

    def test_rho(self):
        """ 
        Approximate rho calulated via finite difference
        rho equal to risk-free rate derivative of call option
        """
        option = BlackScholes(spot0=100, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')
        option_plus_epsilon = BlackScholes(spot0=100, K=100, \
                                sigma=0.3, r=0.05+EPSILON, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')
        approximate_rho = ( option_plus_epsilon.vanilla_price() - option.vanilla_price() ) / EPSILON
        assert option.Rho() - approximate_rho < TOL
        print("test_rho: PASS")

    def test_theta(self):
        """ 
        Approximate theta calulated via finite difference
        theta equal to negative time derivative of call option
        """
        option = BlackScholes(spot0=100, K=100, sigma=0.3, \
                                r=0.05, delta=0.01, T=1,\
                                opt_type='call', exer_type='european')
        option_plus_epsilon = BlackScholes(spot0=100, K=100, \
                                sigma=0.3, r=0.05, delta=0.01, T=1+EPSILON,\
                                opt_type='call', exer_type='european')
        approximate_theta = ( - option_plus_epsilon.vanilla_price() + option.vanilla_price() ) / EPSILON
        assert option.Theta() - approximate_theta < TOL
        print("test_theta: PASS")

if __name__ == '__main__':
    unittest.main()
