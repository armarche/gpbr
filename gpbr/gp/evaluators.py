from deap.gp import PrimitiveTree, PrimitiveSet, Primitive, Terminal

def evaluate_subtrees(tree: PrimitiveTree, pset: PrimitiveSet, **kwargs):
    """
    Evaluate the GP tree and compute results for every subtree using a stack-based approach,
    inspired by the __str__ method in PrimitiveTree.
    
    :param tree: deap.gp.PrimitiveTree, the tree to evaluate.
    :param pset: deap.gp.PrimitiveSet, the primitive set used to define functions and terminals.
    :param args: Variable arguments representing the inputs for ARG0, ARG1, etc.
    :return: A tuple (root_value, subtree_values), where root_value is the result of the entire tree,
             and subtree_values is a list where subtree_values[i] is the result of the subtree rooted at tree[i].
    """
    subtree_values = [None] * len(tree)
    stack = []
    
    for i in range(len(tree)):
        stack.append((i, []))
        
        while stack and len(stack[-1][1]) == tree[stack[-1][0]].arity:
            idx, child_vals = stack.pop()
            node: Primitive | Terminal = tree[idx]
            
            if node.arity == 0:  # Terminal
                if node.name.startswith('ARG'):
                    val = kwargs[node.value]
                else:
                    val = node.value
            else:  # Primitive
                func = pset.context[node.name]
                val = func(*child_vals)
            
            subtree_values[idx] = val
            
            if stack:
                stack[-1][1].append(val)
    
    return subtree_values