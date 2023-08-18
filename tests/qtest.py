from typing import Any, Iterable, Tuple

class Variable:
    def __init__(self, value):
        self.value = value
        self.grad = None

class Scalar:
    def __init__(self, value):
        self.value = value
        self.parents = []
        self.grad_fn = None
    
    def backward(self):
        if self.grad_fn is not None:
            self.grad_fn(self.value)
    
    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        gradients = {}
        visited = set()
        
        def backward_pass(node, d_input):
            if node in visited:
                return
            visited.add(node)
            
            if node.grad_fn is not None:
                grads = node.grad_fn(d_input)
                for parent, grad in zip(node.parents, grads):
                    backward_pass(parent, grad)
            
            if isinstance(node, Variable):
                if node not in gradients:
                    gradients[node] = d_input
                else:
                    gradients[node] += d_input
        
        backward_pass(self, d_output)
        return gradients.items()

# Example usage
def add(x, y):
    z = Scalar(x.value + y.value)
    z.parents = [x, y]
    z.grad_fn = lambda d_output: [d_output, d_output]
    return z

x = Variable(2)
y = Variable(3)
z = add(x, y)

# Forward pass
result = z.value
print("Forward result:", result)

# Backward pass
z.chain_rule(1.0)
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)
