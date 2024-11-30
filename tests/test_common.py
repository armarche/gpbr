import dataclasses
import unittest
import numpy as np
from gpbr.direct.heat_equation.common import MFSData

class TestMFSData(unittest.TestCase):
    def setUp(self):
        self.N = np.int32(10)
        self.T = np.float64(1.0)
        self.tn = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        self.M = np.int64(5)
        self.Beta = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
        self.nu = np.float64(1.0)

    def test_mfsdata_initialization(self):
        data = MFSData(N=self.N, T=self.T, tn=self.tn, M=self.M, Beta=self.Beta, nu=self.nu)
        self.assertEqual(data.N, self.N)
        self.assertEqual(data.T, self.T)
        np.testing.assert_array_equal(data.tn, self.tn)
        self.assertEqual(data.M, self.M)
        np.testing.assert_array_equal(data.Beta, self.Beta)
        self.assertEqual(data.nu, self.nu)

    def test_mfsdata_immutable(self):
        data = MFSData(N=self.N, T=self.T, tn=self.tn, M=self.M, Beta=self.Beta, nu=self.nu)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            data.N = np.int32(20)

if __name__ == '__main__':
    unittest.main()