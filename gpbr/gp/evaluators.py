from deap.gp import PrimitiveTree, PrimitiveSet, Primitive, Terminal
import numpy as np

from deap.gp import PrimitiveTree, PrimitiveSet, Primitive, Terminal
import numpy as np

def evaluate_subtrees(tree: PrimitiveTree, pset: PrimitiveSet, arg_shape:tuple,  **kwargs):
    """
    Evaluate the GP tree and compute results for every subtree using a stack-based approach,
    inspired by the __str__ method in PrimitiveTree.
    
    :param tree: deap.gp.PrimitiveTree, the tree to evaluate.
    :param pset: deap.gp.PrimitiveSet, the primitive set used to define functions and terminals.
    :param arg_shape: tuple, the shape of input arguments.
    :param args: Variable arguments representing the inputs for ARG0, ARG1, etc.
    :return: subtree_values is a list where subtree_values[i] is the result of the subtree rooted at tree[i].
    """
    subtree_values = np.empty((len(tree), *arg_shape), dtype=np.float64)

    stack = []
    
    for i in range(len(tree)):
        stack.append((i, []))
        
        while stack and len(stack[-1][1]) == tree[stack[-1][0]].arity:
            idx, child_vals = stack.pop()
            node: Primitive | Terminal = tree[idx]
            
            if node.arity == 0:  # Terminal
                if node.name.startswith('ARG'):
                    val = np.array(kwargs[node.value], dtype=np.float64)
                else:
                    if isinstance(node.value, str):
                        val = np.full(arg_shape, pset.context[node.value], dtype=np.float64)
                    else:
                        val = np.full(arg_shape, node.value, dtype=np.float64)
            else:  # Primitive
                func = pset.context[node.name]
                val = np.asanyarray(func(*child_vals), dtype=np.float64)
            
            subtree_values[idx] = val
            
            if stack:
                stack[-1][1].append(val)

    return subtree_values