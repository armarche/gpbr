import random

from deap import gp

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


