from __future__ import division, print_function, absolute_import

import warnings

import numpy as np
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
                           assert_)
from pytest import raises as assert_raises

from labugr.systems.lti_conversion import (ss2tf, tf2ss, abcd_normalize)

class TestSS2TF:

    def check_matrix_shapes(self, p, q, r):
        ss2tf(np.zeros((p, p)),
              np.zeros((p, q)),
              np.zeros((r, p)),
              np.zeros((r, q)), 0)

    def test_shapes(self):
        # Each tuple holds:
        #   number of states, number of inputs, number of outputs
        for p, q, r in [(3, 3, 3), (1, 3, 3), (1, 1, 1)]:
            self.check_matrix_shapes(p, q, r)

    def test_basic(self):
        # Test a round trip through tf2ss and ss2tf.
        b = np.array([1.0, 3.0, 5.0])
        a = np.array([1.0, 2.0, 3.0])

        A, B, C, D = tf2ss(b, a)
        assert_allclose(A, [[-2, -3], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[1, 2]], rtol=1e-13)
        assert_allclose(D, [[1]], rtol=1e-14)

        bb, aa = ss2tf(A, B, C, D)
        assert_allclose(bb[0], b, rtol=1e-13)
        assert_allclose(aa, a, rtol=1e-13)

    def test_zero_order_round_trip(self):
        # See gh-5760
        tf = (2, 1)
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[0]], rtol=1e-13)
        assert_allclose(B, [[0]], rtol=1e-13)
        assert_allclose(C, [[0]], rtol=1e-13)
        assert_allclose(D, [[2]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[2, 0]], rtol=1e-13)
        assert_allclose(den, [1, 0], rtol=1e-13)

        tf = ([[5], [2]], 1)
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[0]], rtol=1e-13)
        assert_allclose(B, [[0]], rtol=1e-13)
        assert_allclose(C, [[0], [0]], rtol=1e-13)
        assert_allclose(D, [[5], [2]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[5, 0], [2, 0]], rtol=1e-13)
        assert_allclose(den, [1, 0], rtol=1e-13)

    def test_simo_round_trip(self):
        # See gh-5753
        tf = ([[1, 2], [1, 1]], [1, 2])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-2]], rtol=1e-13)
        assert_allclose(B, [[1]], rtol=1e-13)
        assert_allclose(C, [[0], [-1]], rtol=1e-13)
        assert_allclose(D, [[1], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[1, 2], [1, 1]], rtol=1e-13)
        assert_allclose(den, [1, 2], rtol=1e-13)

        tf = ([[1, 0, 1], [1, 1, 1]], [1, 1, 1])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-1, -1], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[-1, 0], [0, 0]], rtol=1e-13)
        assert_allclose(D, [[1], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[1, 0, 1], [1, 1, 1]], rtol=1e-13)
        assert_allclose(den, [1, 1, 1], rtol=1e-13)

        tf = ([[1, 2, 3], [1, 2, 3]], [1, 2, 3, 4])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-2, -3, -4], [1, 0, 0], [0, 1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0], [0]], rtol=1e-13)
        assert_allclose(C, [[1, 2, 3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(D, [[0], [0]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, 2, 3], [0, 1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 2, 3, 4], rtol=1e-13)

        tf = ([1, [2, 3]], [1, 6])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6]], rtol=1e-31)
        assert_allclose(B, [[1]], rtol=1e-31)
        assert_allclose(C, [[1], [-9]], rtol=1e-31)
        assert_allclose(D, [[0], [2]], rtol=1e-31)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1], [2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6], rtol=1e-13)

        tf = ([[1, -3], [1, 2, 3]], [1, 6, 5])
        A, B, C, D = tf2ss(*tf)
        assert_allclose(A, [[-6, -5], [1, 0]], rtol=1e-13)
        assert_allclose(B, [[1], [0]], rtol=1e-13)
        assert_allclose(C, [[1, -3], [-4, -2]], rtol=1e-13)
        assert_allclose(D, [[0], [1]], rtol=1e-13)

        num, den = ss2tf(A, B, C, D)
        assert_allclose(num, [[0, 1, -3], [1, 2, 3]], rtol=1e-13)
        assert_allclose(den, [1, 6, 5], rtol=1e-13)

    def test_multioutput(self):
        # Regression test for gh-2669.

        # 4 states
        A = np.array([[-1.0, 0.0, 1.0, 0.0],
                      [-1.0, 0.0, 2.0, 0.0],
                      [-4.0, 0.0, 3.0, 0.0],
                      [-8.0, 8.0, 0.0, 4.0]])

        # 1 input
        B = np.array([[0.3],
                      [0.0],
                      [7.0],
                      [0.0]])

        # 3 outputs
        C = np.array([[0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [8.0, 8.0, 0.0, 0.0]])

        D = np.array([[0.0],
                      [0.0],
                      [1.0]])

        # Get the transfer functions for all the outputs in one call.
        b_all, a = ss2tf(A, B, C, D)

        # Get the transfer functions for each output separately.
        b0, a0 = ss2tf(A, B, C[0], D[0])
        b1, a1 = ss2tf(A, B, C[1], D[1])
        b2, a2 = ss2tf(A, B, C[2], D[2])

        # Check that we got the same results.
        assert_allclose(a0, a, rtol=1e-13)
        assert_allclose(a1, a, rtol=1e-13)
        assert_allclose(a2, a, rtol=1e-13)
        assert_allclose(b_all, np.vstack((b0, b1, b2)), rtol=1e-13, atol=1e-14)

class Test_abcd_normalize(object):
    def setup_method(self):
        self.A = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.B = np.array([[-1.0], [5.0]])
        self.C = np.array([[4.0, 5.0]])
        self.D = np.array([[2.5]])

    def test_no_matrix_fails(self):
        assert_raises(ValueError, abcd_normalize)

    def test_A_nosquare_fails(self):
        assert_raises(ValueError, abcd_normalize, [1, -1],
                      self.B, self.C, self.D)

    def test_AB_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_AC_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      [[4.0], [5.0]], self.D)

    def test_CD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, self.B,
                      self.C, [2.5, 0])

    def test_BD_mismatch_fails(self):
        assert_raises(ValueError, abcd_normalize, self.A, [-1, 5],
                      self.C, self.D)

    def test_normalized_matrices_unchanged(self):
        A, B, C, D = abcd_normalize(self.A, self.B, self.C, self.D)
        assert_equal(A, self.A)
        assert_equal(B, self.B)
        assert_equal(C, self.C)
        assert_equal(D, self.D)

    def test_shapes(self):
        A, B, C, D = abcd_normalize(self.A, self.B, [1, 0], 0)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape[0], C.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(B.shape[1], D.shape[1])

    def test_zero_dimension_is_not_none1(self):
        B_ = np.zeros((2, 0))
        D_ = np.zeros((0, 0))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, D=D_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(D, D_)
        assert_equal(C.shape[0], D_.shape[0])
        assert_equal(C.shape[1], self.A.shape[0])

    def test_zero_dimension_is_not_none2(self):
        B_ = np.zeros((2, 0))
        C_ = np.zeros((0, 2))
        A, B, C, D = abcd_normalize(A=self.A, B=B_, C=C_)
        assert_equal(A, self.A)
        assert_equal(B, B_)
        assert_equal(C, C_)
        assert_equal(D.shape[0], C_.shape[0])
        assert_equal(D.shape[1], B_.shape[1])

    def test_missing_A(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))

    def test_missing_B(self):
        A, B, C, D = abcd_normalize(A=self.A, C=self.C, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))

    def test_missing_C(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, D=self.D)
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_D(self):
        A, B, C, D = abcd_normalize(A=self.A, B=self.B, C=self.C)
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_AB(self):
        A, B, C, D = abcd_normalize(C=self.C, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(A.shape, (self.C.shape[1], self.C.shape[1]))
        assert_equal(B.shape, (self.C.shape[1], self.D.shape[1]))

    def test_missing_AC(self):
        A, B, C, D = abcd_normalize(B=self.B, D=self.D)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(C.shape, (self.D.shape[0], self.B.shape[0]))

    def test_missing_AD(self):
        A, B, C, D = abcd_normalize(B=self.B, C=self.C)
        assert_equal(A.shape[0], A.shape[1])
        assert_equal(A.shape[0], B.shape[0])
        assert_equal(D.shape[0], C.shape[0])
        assert_equal(D.shape[1], B.shape[1])
        assert_equal(A.shape, (self.B.shape[0], self.B.shape[0]))
        assert_equal(D.shape, (self.C.shape[0], self.B.shape[1]))

    def test_missing_BC(self):
        A, B, C, D = abcd_normalize(A=self.A, D=self.D)
        assert_equal(B.shape[0], A.shape[0])
        assert_equal(B.shape[1], D.shape[1])
        assert_equal(C.shape[0], D.shape[0])
        assert_equal(C.shape[1], A.shape[0])
        assert_equal(B.shape, (self.A.shape[0], self.D.shape[1]))
        assert_equal(C.shape, (self.D.shape[0], self.A.shape[0]))

    def test_missing_ABC_fails(self):
        assert_raises(ValueError, abcd_normalize, D=self.D)

    def test_missing_BD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, C=self.C)

    def test_missing_CD_fails(self):
        assert_raises(ValueError, abcd_normalize, A=self.A, B=self.B)

