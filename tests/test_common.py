import dataclasses
import unittest
import numpy as np
from gpbr.direct.heat_equation.common import MFSData
from gpbr.direct.heat_equation.common import MFSConfig, CollocationData2D, MFSPolinomials2D, Point2D

class TestMFSConfig(unittest.TestCase):
        def setUp(self):
            self.N = np.int32(10)
            self.T = np.float64(1.0)
            self.n_coll = np.int64(5)
            self.n_source = np.int64(5)
            self.f1 = lambda p: np.float64(p.x + p.y)
            self.f2 = lambda p: np.float64(p.x - p.y)
            self.eta1 = np.float64(0.5)
            self.eta2 = np.float64(0.25)

        def test_mfsconfig_initialization(self):
            config = MFSConfig(N=self.N, T=self.T, n_coll=self.n_coll, n_source=self.n_source, f1=self.f1, f2=self.f2, eta1=self.eta1, eta2=self.eta2)
            self.assertEqual(config.N, self.N)
            self.assertEqual(config.T, self.T)
            self.assertEqual(config.n_coll, self.n_coll)
            self.assertEqual(config.n_source, self.n_source)
            self.assertEqual(config.eta1, self.eta1)
            self.assertEqual(config.eta2, self.eta2)
            self.assertEqual(config.f1(Point2D(1, 1)), self.f1(Point2D(1, 1)))
            self.assertEqual(config.f2(Point2D(1, 1)), self.f2(Point2D(1, 1)))

        def test_mfsconfig_immutable(self):
            config = MFSConfig(N=self.N, T=self.T, n_coll=self.n_coll, n_source=self.n_source, f1=self.f1, f2=self.f2, eta1=self.eta1, eta2=self.eta2)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                config.N = np.int32(20)

class TestMFSData(unittest.TestCase):
    def setUp(self):
        self.config = MFSConfig(
            N=np.int32(10),
            T=np.float64(1.0),
            n_coll=np.int64(5),
            n_source=np.int64(5),
            f1=lambda p: np.float64(p.x + p.y),
            f2=lambda p: np.float64(p.x - p.y),
            eta1=np.float64(0.5),
            eta2=np.float64(0.25)
        )
        self.h = np.float64(0.1)
        self.tn = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.Beta = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        self.nu = np.float64(1.0)
        self.collocation = CollocationData2D(None,None)
        self.source_collocation = CollocationData2D(None,None)
        self.polynomials = MFSPolinomials2D(None,None,None)

    def test_mfsdata_initialization(self):
        data = MFSData(
            config=self.config,
            h=self.h,
            tn=self.tn,
            Beta=self.Beta,
            nu=self.nu,
            collocation=self.collocation,
            source_collocation=self.source_collocation,
            polynomials=self.polynomials
        )
        self.assertEqual(data.config, self.config)
        self.assertEqual(data.h, self.h)
        np.testing.assert_array_equal(data.tn, self.tn)
        np.testing.assert_array_equal(data.Beta, self.Beta)
        self.assertEqual(data.nu, self.nu)
        self.assertEqual(data.collocation, self.collocation)
        self.assertEqual(data.source_collocation, self.source_collocation)
        self.assertEqual(data.polynomials, self.polynomials)

    def test_mfsdata_properties(self):
        data = MFSData(
            config=self.config,
            h=self.h,
            tn=self.tn,
            Beta=self.Beta,
            nu=self.nu,
            collocation=self.collocation,
            source_collocation=self.source_collocation,
            polynomials=self.polynomials
        )
        self.assertEqual(data.N, self.config.N)
        self.assertEqual(data.T, self.config.T)
        self.assertEqual(data.M, self.config.n_coll)
        self.assertEqual(data.eta1, self.config.eta1)
        self.assertEqual(data.eta2, self.config.eta2)
        self.assertEqual(data.f1(Point2D(1, 1)), self.config.f1(Point2D(1, 1)))
        self.assertEqual(data.f2(Point2D(1, 1)), self.config.f2(Point2D(1, 1)))

    def test_mfsdata_immutable(self):
        data = MFSData(
            config=self.config,
            h=self.h,
            tn=self.tn,
            Beta=self.Beta,
            nu=self.nu,
            collocation=self.collocation,
            source_collocation=self.source_collocation,
            polynomials=self.polynomials
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            data.h = np.float64(0.2)

if __name__ == '__main__':
    unittest.main()