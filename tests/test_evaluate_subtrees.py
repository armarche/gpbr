import numpy as np
import unittest
import deap.gp as gp

from gpbr.gp.funcs import sqrtabs, expplusone
from gpbr.gp.evaluators import evaluate_subtrees


def build_pset():
    pset = gp.PrimitiveSet("main", 1)
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(sqrtabs, 1)
    pset.addPrimitive(expplusone, 1)
    pset.addEphemeralConstant('rand', (np.random.rand, 1)[0])
    pset.renameArguments(ARG0='s')
    return pset


class TestEvaluateSubtrees(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Seed RNG before building the primitive set so ephemeral constants are deterministic
        np.random.seed(0)
        cls.pset = build_pset()
        cls.test_set = np.random.rand(100,100)
        cls.examples = [
            "s",
            "1.0",
            "add(s,s)",
            "sqrtabs(s)",
            "add(add(sin(s), cos(s)), multiply(s, 0.1))",
            "expplusone(add(cos(sin(add(0.5041008352677898, s))), sin(multiply(sin(0.9508309438650218), expplusone(s)))))",
        ]

    def test_examples_match_deap_compile(self):
        for expr in self.examples:
            with self.subTest(expr=expr):
                tree = gp.PrimitiveTree.from_string(expr, self.pset)
                root_val, subtree_vals = evaluate_subtrees(tree, self.pset, s=self.test_set)

                for i in range(len(tree)):
                    node = tree[i]
                    sl = tree.searchSubtree(i)
                    subtree = gp.PrimitiveTree(tree[sl])

                    compiled_fn = gp.compile(subtree, self.pset)

                    compiled_val = compiled_fn(self.test_set)
                    eval_val = subtree_vals[i]

                    a = np.asarray(compiled_val)
                    b = np.asarray(eval_val)

                    self.assertEqual(a.shape, b.shape, f"shape mismatch for node {i} ({node.name}): {a.shape} != {b.shape}")
                    np.testing.assert_allclose(a, b, equal_nan=True)
