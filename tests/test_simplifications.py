import unittest
import numpy as np
from deap import gp, base
from gpbr.gp.evaluators import evaluate_subtrees
from gpbr.gp.simplyfications import simplify_constant_subtrees

class TestSimplifyConstantSubtrees(unittest.TestCase):

    def setUp(self):
        # Reset the PrimitiveSet for each test if needed, or define it once globally
        self.pset = gp.PrimitiveSet("MAIN", 1)
        self.pset.addPrimitive(np.add, 2, "add")
        self.pset.addPrimitive(np.subtract, 2, "sub")
        self.pset.addPrimitive(np.multiply, 2, "mul")
        self.pset.addTerminal(1.0, name="T1")
        self.pset.addTerminal(2.0, name="T2")
        self.pset.addTerminal(3.0, name="T3")
        self.pset.addTerminal(0.5, name="T05")
        self.test_set = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

        self.toolbox = base.Toolbox()
        self.toolbox.register('compile', gp.compile, pset=self.pset)
        self.toolbox.register('eval', evaluate_subtrees, pset=self.pset, arg_shape=self.test_set.shape, ARG0=self.test_set)

    def test_no_simplification_needed(self):
        # Create an individual that should not be simplified
        # e.g., add(1, 2)
        individual = gp.PrimitiveTree([self.pset.mapping["add"], self.pset.mapping["ARG0"], self.pset.mapping["T2"]])
        individual.subtrees_values = self.toolbox.eval(individual)
        
        simplified_ind, changed = simplify_constant_subtrees(individual, tol=1e-8)
        self.assertFalse(changed)
        self.assertEqual(str(simplified_ind), str(individual))

    def test_simple_constant_subtree(self):
        # Create an individual with a constant subtree: mul(T1, T2) -> 2.0
        # The tree would be add(ARG0, mul(T1, T2))
        individual = gp.PrimitiveTree([
            self.pset.mapping["add"],
            self.pset.mapping["ARG0"],
            self.pset.mapping["mul"],
            self.pset.mapping["T1"],
            self.pset.mapping["T2"],
        ])
        individual.subtrees_values = self.toolbox.eval(individual)
        
        simplified_ind, changed = simplify_constant_subtrees(individual, tol=1e-8)
        self.assertTrue(changed)
        # Expected simplified tree: add(ARG0, 2.0)
        # Note: DEAP may represent the simplified terminal differently, so we check the value
        self.assertEqual(len(simplified_ind), 3)
        self.assertEqual(simplified_ind[0].name, "add")
        self.assertEqual(simplified_ind[1].name, "ARG0")
        self.assertIsInstance(simplified_ind[2], gp.Terminal)
        self.assertAlmostEqual(simplified_ind[2].value, 2.0)
        self.assertIsNone(simplified_ind.subtrees_values) # Should be reset

    def test_multiple_constant_subtrees(self):
        # Create an individual with multiple constant subtrees: add(mul(T1,T2), sub(T3,T1))
        individual = gp.PrimitiveTree([
            self.pset.mapping["add"],
            self.pset.mapping["mul"],
            self.pset.mapping["T1"],
            self.pset.mapping["T2"],
            self.pset.mapping["sub"],
            self.pset.mapping["T3"],
            self.pset.mapping["T1"]
        ])
        individual.subtrees_values = self.toolbox.eval(individual)
        
        simplified_ind, changed = simplify_constant_subtrees(individual, tol=1e-8)
        self.assertTrue(changed)
        # The entire tree should be simplified to a single constant: 4.0
        self.assertEqual(len(simplified_ind), 1)
        self.assertIsInstance(simplified_ind[0], gp.Terminal)
        self.assertAlmostEqual(simplified_ind[0].value, 4.0)
        self.assertIsNone(simplified_ind.subtrees_values)

    def test_nested_constant_subtrees(self):
        # Create an individual with nested constant subtrees: add(add(T1, T2), T3)
        individual = gp.PrimitiveTree([
            self.pset.mapping["add"],
            self.pset.mapping["add"],
            self.pset.mapping["T1"],
            self.pset.mapping["T2"],
            self.pset.mapping["T3"]
        ])
        individual.subtrees_values = self.toolbox.eval(individual)
        
        simplified_ind, changed = simplify_constant_subtrees(individual, tol=1e-8)
        self.assertTrue(changed)
        # The entire tree should be simplified to a single constant: 6.0
        self.assertEqual(len(simplified_ind), 1)
        self.assertIsInstance(simplified_ind[0], gp.Terminal)
        self.assertAlmostEqual(simplified_ind[0].value, 6.0)
        self.assertIsNone(simplified_ind.subtrees_values)

    def test_non_finite_values(self):
        # Create an individual with non-finite values (e.g., NaN)
        # We'll create a situation that produces NaN, e.g. by dividing by zero if we had division
        # For this test, we'll manually insert a NaN into the subtrees_values
        individual = gp.PrimitiveTree([self.pset.mapping["add"], self.pset.mapping["T1"], self.pset.mapping["T2"]])
        individual.subtrees_values = self.toolbox.eval(individual)
        individual.subtrees_values[0] = np.array([np.nan]) # Mock evaluation result with NaN
        
        simplified_ind, changed = simplify_constant_subtrees(individual, tol=1e-8)
        self.assertFalse(changed)
        self.assertEqual(str(simplified_ind), str(individual))

    def test_values_above_tolerance(self):
        # Create an individual with values just above tolerance
        individual = gp.PrimitiveTree([self.pset.mapping["add"], self.pset.mapping["ARG0"], self.pset.mapping["T1"]])
        individual.subtrees_values = self.toolbox.eval(individual)
        # Manually make the root node's values have a std dev > tol
        individual.subtrees_values[0] = np.array([1.0, 1.0 + 2e-8, 2.0, 3.0, 4.0])
        
        simplified_ind, changed = simplify_constant_subtrees(individual, tol=1e-8)
        self.assertFalse(changed)
        self.assertEqual(str(simplified_ind), str(individual))

    def test_values_below_tolerance(self):
        # Create an individual with values just below tolerance
        individual = gp.PrimitiveTree([self.pset.mapping["add"], self.pset.mapping["T1"], self.pset.mapping["T05"]]) # 1.0 + 0.5
        individual.subtrees_values = self.toolbox.eval(individual)
        # Manually make the root node's values have a std dev < tol
        base_val = 1.5
        values = np.full_like(self.test_set, base_val, dtype=float)
        values[0] += 5e-9
        individual.subtrees_values[0] = values

        simplified_ind, changed = simplify_constant_subtrees(individual, tol=1e-8)
        self.assertTrue(changed)
        # Expected simplified tree: 1.5 (mean of [1.5, 1.5 + 5e-9])
        self.assertEqual(len(simplified_ind), 1)
        self.assertIsInstance(simplified_ind[0], gp.Terminal)
        self.assertAlmostEqual(simplified_ind[0].value, np.mean(values))
        self.assertIsNone(simplified_ind.subtrees_values)

if __name__ == '__main__':
    unittest.main()
