import unittest
import numpy as np
from scipy.stats import norm

import os
import sys
from pickletools import optimize
sys.path.append(os.path.abspath('../'))

from maxest.estimate import product_integral, prod_int_div_cdf

class TestIntegralMethods(unittest.TestCase):

    def test_productIntegral(self):
        # wolfram alpha
        # Exp[Integrate[Ln[x**2+1], x, -2, 10]]
        f = lambda x: x**2+1
        int_v = product_integral(f, -2, 10)
        true_v = 1.8094583901794209633267270746934836578344553822673503 * 10**13
        self.assertTrue(np.isclose(int_v,true_v,rtol=1e-3))

        # wolfram alpha
        # Exp[Integrate[Ln[CDF[NormalDistribution[x/2, 2*x**2], 2]], x, -2, 2]]
        # In python prevent that standard deviation equals zero
        f = lambda x: norm.cdf(2, loc=x/2, scale=max(0.001, 2*(x**2)))
        int_v = product_integral(f, -2, 2)
        true_v = 0.432756
        self.assertTrue(np.isclose(int_v,true_v,rtol=1e-3), msg='{} != {}'.format(int_v, true_v))

    def test_prod_int_div_cdf(self):
        class tmp:
            def predict(self, x):
                log = 20*np.abs(np.sin(x) / (x + 10))
                scale = (x/2)**2
                return log, scale
            def ess(self, x):
                return 1.0

        evalp = 2
        gps = tmp()

        def tmpf(y):
            loc, scale = gps.predict(y)
            return norm.cdf(evalp, loc=loc, scale=scale)

        #Exp[Integrate[Ln[CDF[NormalDistribution[20*abs(sin(x) / (x + 10)), (x/2)**2], 2]], x, -5, 8]]
        pi_hat = product_integral(tmpf, -5, 8)
        pi_true = 0.00159266
        self.assertTrue(np.isclose(pi_hat, pi_true, rtol=1e-3), msg='{} != {}'.format(pi_hat, pi_true))

        #Exp[Integrate[Ln[CDF[NormalDistribution[20*abs(sin(x) / (x + 10)), (x/2)**2], 2]], x, -5, 8]]
        # / CDF[NormalDistribution[1, 0.5], 2]
        cdf = 0.97725 # CDF[NormalDistribution[1, 0.5], 2]
        cmean = 1.0
        cstddev = 0.5
        v_hat = prod_int_div_cdf(evalp, cmean, cstddev, gps, -5, 8)
        v_true = pi_true/cdf
        self.assertTrue(np.isclose(v_hat, v_true, rtol=1e-3), msg='{} != {}'.format(v_hat, v_true))


if __name__ == '__main__':
    unittest.main()