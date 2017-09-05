# from __future__ import division, print_function, absolute_import

# import numpy as np
# from numpy.testing import (assert_almost_equal, assert_array_almost_equal,
#                            assert_equal, assert_,
#                            assert_allclose, assert_warns)
# from pytest import raises as assert_raises

# from labugr.filters.iir_filters import (buttap, besselap, cheb1ap, cheb1ap,
# 										ellipap)

# class TestPrototypeType(object):

#     def test_output_type(self):
#         # Prototypes should consistently output arrays, not lists
#         # https://github.com/scipy/scipy/pull/441
#         for func in (buttap,
#                      besselap,
#                      lambda N: cheb1ap(N, 1),
#                      lambda N: cheb2ap(N, 20),
#                      lambda N: ellipap(N, 1, 20)):
#             for N in range(7):
#                 z, p, k = func(N)
#                 assert_(isinstance(z, np.ndarray))
#                 assert_(isinstance(p, np.ndarray))
