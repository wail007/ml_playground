import numpy as np

class Newton(object):
    def __init__(self, e=1e-4, alpha=1.0, verbose=False):
        self.e       = e
        self.alpha   = alpha
        self.verbose = verbose

    def solve(self, f, *args):
        cost_old = float("inf")
        cost_new = float("inf")
        
        while True:
            f.w = f.w - self.alpha * np.dot(np.linalg.pinv(f.hessian(*args)), f.gradient(*args))

            cost_new = f.cost(*args)
            if self.verbose:
                print("cost: %f" % cost_new)
            if abs(cost_new - cost_old) < self.e:
                break

            cost_old = cost_new


class GradientDescent(object):
    def __init__(self, e=1e-4, alpha=1.0, verbose=False):
        self.e       = e
        self.alpha   = alpha
        self.verbose = verbose

    def solve(self, f, *args):
        cost_old = float("inf")
        cost_new = float("inf")
        
        while True:
            f.w = f.w - self.alpha * f.gradient(*args)

            cost_new = f.cost(*args)
            if self.verbose:
                print("cost: %f" % cost_new)
            if abs(cost_new - cost_old) < self.e:
                break

            cost_old = cost_new


class ValidationSet(object):
    def __init__(self, solver, frac=0.7, shuffle=False, e=1e-4, verbose=False):
        self.solver = solver
        self.frac   = frac
        self.shuffle= shuffle
        self.e      = e
        self.verbose= verbose

    def solve(self, *arg):
        pass