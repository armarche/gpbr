from deap import gp
import numpy as np

def simplify_constant_subtrees(ind, tol=1e-8):
    """
    Traverse tree bottom-up, replace constant subtrees with a single constant node.
    Modifies `ind` in-place and returns it.
    """
    simplified = False

    new_subtree = []
    i=0
    while i < len(ind):
        slice_ = ind.searchSubtree(i)
        node = ind[i]
        subtree_values = ind.subtrees_values[i]

        if not isinstance(node, gp.Terminal) and np.all(np.isfinite(subtree_values)) and np.std(subtree_values) < tol:
            # We select object return type to align with DEAP's expectations for Terminal nodes
            new_subtree.append(gp.Terminal(np.mean(subtree_values), False, object))
            i = slice_.stop
            simplified = True
        else:
            new_subtree.append(ind[i])
            i+=1
    ind[ind.searchSubtree(0)] = new_subtree
    if simplified:
        ind.subtrees_values =None
    return ind, simplified