import unittest
import pickle
import numpy as np

from gpbr.direct.heat_equation.common import (
    MFSConfig, MFSConfig2D, MFSConfig3D, MFSData, Dimension
)
from gpbr.direct.common.boundary import Point2D, Point3D
from gpbr.direct.common.collocation import CollocationData2D, CollocationData3D
from gpbr.direct.heat_equation.polynomial import MFSPolinomials2D, MFSPolinomials3D

def f1_const(p):
    return 1.0

def f2_const(p):
    return 2.0

class TestPickling(unittest.TestCase):
    def test_mfsconfig2d_picklable(self):
        config = MFSConfig2D(
            N=np.int32(10),
            T=np.float64(1.0),
            n_coll=np.int64(20),
            n_source=np.int64(20),
            f1=f1_const,
            f2=f2_const,
            eta1=np.float64(0.5),
            eta2=np.float64(1.5)
        )
        s = pickle.dumps(config)
        loaded = pickle.loads(s)
        self.assertEqual(config, loaded)
        self.assertEqual(loaded.dim, Dimension.TWO_D)

    def test_mfsconfig3d_picklable(self):
        config = MFSConfig3D(
            N=np.int32(10),
            T=np.float64(1.0),
            n_coll=np.int64(20),
            n_source=np.int64(20),
            f1=f1_const,
            f2=f2_const,
            eta1=np.float64(0.5),
            eta2=np.float64(1.5),
            n_coll_theta=np.int64(5),
            n_coll_phi=np.int64(5),
            n_source_theta=np.int64(5),
            n_source_phi=np.int64(5)
        )
        s = pickle.dumps(config)
        loaded = pickle.loads(s)
        self.assertEqual(config, loaded)
        self.assertEqual(loaded.dim, Dimension.THREE_D)

    def test_mfsdata_picklable(self):
        config = MFSConfig2D(
            N=np.int32(10),
            T=np.float64(1.0),
            n_coll=np.int64(20),
            n_source=np.int64(20),
            f1=f1_const,
            f2=f2_const,
            eta1=np.float64(0.5),
            eta2=np.float64(1.5)
        )
        collocation = CollocationData2D(20, np.zeros((2, 20)))
        source_collocation = CollocationData2D(20, np.zeros((2, 20)))
        polynomials = MFSPolinomials2D(None, None, None)
        data = MFSData(
            config=config,
            h=np.float64(0.1),
            tn=np.linspace(0, 1, 10),
            Beta=np.ones(10),
            nu=np.float64(1.0),
            collocation=collocation,
            source_collocation=source_collocation,
            polynomials=polynomials
        )
        s = pickle.dumps(data)
        loaded = pickle.loads(s)
        self.assertEqual(data.config, loaded.config)
        self.assertTrue(np.allclose(data.tn, loaded.tn))
        self.assertTrue(np.allclose(data.Beta, loaded.Beta))
        self.assertEqual(data.nu, loaded.nu)

if __name__ == "__main__":
    unittest.main()