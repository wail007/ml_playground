import numpy as np

def newton(f, x, y, e=0.0001, verbose=False):
    cost_old = float("inf")
    cost_new = float("inf")
    
    while True:
        f.w = f.w - np.dot(np.linalg.pinv(f.hessian(x, y)), f.gradient(x, y))

        cost_new = f.cost(x, y)
        if verbose:
            print("cost: %f" % cost_new)
        if abs(cost_new - cost_old) < e:
            break

        cost_old = cost_new
         