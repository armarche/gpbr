import unittest
import numpy as np
import copy
import deap.gp as gp

from gpbr.gp.operators import cxSwapPeriodic


def build_pset():
    pset = gp.PrimitiveSet("main", 1)
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.sin, 1)
    pset.renameArguments(ARG0='s')
    return pset


class TestCxSwapPeriodic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pset = build_pset()

    def test_no_periodic_subtrees_returns_same_individuals(self):
        # Build two simple different trees
        ind1 = gp.PrimitiveTree.from_string("add(sin(s), s)", self.pset)
        ind2 = gp.PrimitiveTree.from_string("add(cos(s), s)", self.pset)

        # Create subtree_values with no periodic entries (arrays where first != last)
        ind1.subtree_values = [np.array([0.0, 1.0]) for _ in range(len(ind1))]
        ind2.subtree_values = [np.array([0.1, 0.2]) for _ in range(len(ind2))]

        # Keep copies to compare after calling operator
        before1 = copy.deepcopy(ind1)
        before2 = copy.deepcopy(ind2)

        out1, out2 = cxSwapPeriodic(ind1, ind2)

        # No periodic subtree in either: individuals should remain unchanged
        self.assertEqual(list(before1), list(out1))
        self.assertEqual(list(before2), list(out2))

    def test_swaps_periodic_subtrees_between_individuals(self):
        # Trees with an inner primitive (sin / cos) that we'll mark as periodic
        ind1 = gp.PrimitiveTree.from_string("add(sin(s), s)", self.pset)
        ind2 = gp.PrimitiveTree.from_string("add(cos(s), s)", self.pset)

        # Find the index of the sin/cos node (non-root)
        idx1 = next(i for i, node in enumerate(ind1) if node.name == 'sin')
        idx2 = next(i for i, node in enumerate(ind2) if node.name == 'cos')

        # Build subtree_values lists: mark only the chosen subtree as periodic by
        # using arrays with equal first and last element; other entries are non-periodic.
        sv1 = [np.array([0.0, 1.0]) for _ in range(len(ind1))]
        sv2 = [np.array([0.1, 0.2]) for _ in range(len(ind2))]

        sv1[idx1] = np.array([5.0, 0.0, 5.0])  # periodic (first == last)
        sv2[idx2] = np.array([7.0, 3.0, 7.0])  # periodic

        ind1.subtree_values = sv1
        ind2.subtree_values = sv2

        # Record the subtrees that should be exchanged
        subtree1 = ind1[ind1.searchSubtree(idx1)]
        subtree2 = ind2[ind2.searchSubtree(idx2)]

        # Call operator
        out1, out2 = cxSwapPeriodic(ind1, ind2)

        # After swap, the subtree at idx1 in out1 should equal original subtree2
        new_subtree1 = out1[out1.searchSubtree(idx1)]
        new_subtree2 = out2[out2.searchSubtree(idx2)]

        self.assertEqual(list(new_subtree1), list(subtree2))
        self.assertEqual(list(new_subtree2), list(subtree1))


if __name__ == '__main__':
    unittest.main()
