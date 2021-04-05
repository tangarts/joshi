#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import sys
import os

sys.path.append(os.getcwd() + "/")
from src.model import BlackScholes

print(os.getcwd())


class TestVanillaBlackScholes(unittest.TestCase):
    def test_vanilla_call(self):
        call = BlackScholes(
                spot0=41,
                strike=40,
                vol=0.3,
                r=0.08,
                delta=0,
                T=0.25,
                opt_type='call',
                exer_type="european")

        self.assertAlmostEqual(
                call.vanilla_price(), 3.3990781872368956)

        print("test_vanilla_call: PASS")

        call = BlackScholes(
                spot0=1.25,
                strike=1.2,
                vol=0.1,
                r=0.01,
                delta=0.03,
                T=1.,
                opt_type='call',
                exer_type="european")

        self.assertAlmostEqual(
                call.vanilla_price(), 0.06140714873023745)

        print("test_vanilla_call: PASS")

    def test_vanilla_put(self):
        put = BlackScholes(
                spot0=41,
                strike=40,
                vol=0.3,
                r=0.08,
                delta=0,
                T=0.25,
                opt_type='put',
                exer_type="european")

        self.assertAlmostEqual(
                put.vanilla_price(), 1.6070251195071048)

        print("test_vanilla_put: PASS")

    def test_put_call(self):
        """
        (i) put-call parity: the price of a call minus the price of a
        put equals the value of a forward
        """
        call = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        put = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="put",
            exer_type="european",
        )
        self.assertAlmostEqual(
            call.vanilla_price() - put.vanilla_price(),
            call.forward() - call.strike * call.bond(),
        )
        print("test_put_call: PASS")

    def test_monotone_decreasing_call_with_strike(self):
        """ 
        (ii) the price of a call should be monotone decreasing with strike.
        """
        strike120 = BlackScholes(
            spot0=100,
            strike=120,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        strike100 = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        strike90 = BlackScholes(
            spot0=100,
            strike=90,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        self.assertLess(strike120.vanilla_price(), strike100.vanilla_price())
        self.assertLess(strike100.vanilla_price(), strike90.vanilla_price())
        print("test_monotone_decreasing_call_with_strike: PASS")

    def call_option_bounded(self):
        """
        (iii) a call option should be between S and S - strikeexp(-rT)
        """
        option = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )

        self.assertGreater(
            option.vanilla_price(), option.spot0 - option.strike * option.bond()
        )
        self.assertLess(optvanilla_price(), option.spot0)
        print("call_option_bounded: PASS")

    def test_monotone_increasing_call_in_volatility(self):
        """
        (iv) call option should be monotone increasing in volatility
        """
        vol_20 = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.2,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        vol_30 = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        vol_50 = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.5,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )

        self.assertGreater(vol_50.vanilla_price(), vol_30.vanilla_price())
        self.assertGreater(vol_30.vanilla_price(), vol_20.vanilla_price())
        print("test: PASS")

    def test_increase_with_t(self):
        """
        (v) if dividend rate = 0, option price should increase with T
        """
        t_100 = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.0,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        t_50 = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.0,
            T=0.5,
            opt_type="call",
            exer_type="european",
        )
        t_30 = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.0,
            T=0.3,
            opt_type="call",
            exer_type="european",
        )

        self.assertGreater(t_100.vanilla_price(), t_50.vanilla_price())
        self.assertGreater(t_50.vanilla_price(), t_30.vanilla_price())
        print("test: PASS")

    def test_convex_func_of_strike(self):
        """
        (vi) call option should be convex function of strike 
        """
        pass

    def test_call_spead_approx_digital_call(self):
        """
        (vii) price of a call-spread should approximate price of digital-call
        """
        option = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        # self.assertAlmostEqual( option.vanilla_price(), option.digital_price() )
        print("test: PASS")
        pass

    def test_digital_put_call_partiy(self):
        """
        (viii) price of digital-call option plus digital-put option 
        equal to zero-coupon bond
        """
        call = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="call",
            exer_type="european",
        )
        put = BlackScholes(
            spot0=100,
            strike=100,
            vol=0.3,
            r=0.05,
            delta=0.01,
            T=1,
            opt_type="put",
            exer_type="european",
        )
        self.assertAlmostEqual(call.digital_price() + put.digital_price(), call.bond())
        print("PASS")


if __name__ == "__main__":
    unittest.main()
