import random
import numpy as np

from deap import gp
from .common import is_2pi_periodic

def cxLinearCombination(ind1, ind2, pset):

    for p in ['multiply', 'add']:
        assert p in pset.mapping, "A '" + p + "' function is required in order to perform semantic crossover"
    index1 = random.randint(0, len(ind1) - 1)
    index2 = random.randint(0, len(ind2) - 1)

    subtree1_slice = ind1.searchSubtree(index1)
    subtree2_slice = ind2.searchSubtree(index2)
    r = random.random()

    tr = gp.PrimitiveTree([])
    tr.insert(0, pset.mapping['multiply'])
    tr.append(gp.Terminal(r, False, object))
    tr.extend(ind1[subtree1_slice])
    tr.insert(0, pset.mapping['add'])
    tr.append(pset.mapping['multiply'])
    tr.append(gp.Terminal(1.0-r, False, object))
    tr.extend(ind2[subtree2_slice])

    ind1[subtree1_slice] = tr
    ind2[subtree2_slice] = tr
    return ind1, ind2

def cxSwapPeriodic(ind1, ind2):
    """Crossover that exchanges only 2pi-periodic subtrees between individuals."""
    # Find periodic subtrees in both individuals
    periodic_indexes1 = [i for i, vals in enumerate(ind1.subtree_values) if is_2pi_periodic(vals) and i != 0]
    periodic_indexes2 = [i for i, vals in enumerate(ind2.subtree_values) if is_2pi_periodic(vals) and i != 0]
    
    if not periodic_indexes1 or not periodic_indexes2:
        return ind1, ind2  # No periodic subtrees to exchange
    
    # Select random periodic subtree from each individual
    idx1 = np.random.choice(periodic_indexes1)
    idx2 = np.random.choice(periodic_indexes2)
    
    # Get the subtrees
    subtree1 = ind1[ind1.searchSubtree(idx1)]
    subtree2 = ind2[ind2.searchSubtree(idx2)]
    
    # Swap the subtrees
    ind1[ind1.searchSubtree(idx1)] = subtree2
    ind2[ind2.searchSubtree(idx2)] = subtree1
    
    return ind1, ind2
