import numpy as np
import numpy.polynomial.legendre as leg


def is_in_domain(X, domain):
    return ((np.min(X) < domain[0]) or (np.max(X) > domain[1]))


def scaled_x(X, domain):
    return ((2.*X-domain[0]-domain[1])/(domain[1]-domain[0]))


def norm(n_states):
    return (2.*np.array(n_states)+1)/2.


def coeff(n_states):
    return np.eye(len(n_states))*norm(n_states)


def leg_x(X, domain, n_states):
     return (leg.legval(scaled_x(X, domain), coeff(n_states)))
    

def rollaxis_(X):
    return np.rollaxis(X, 0, len(X.shape))


def discretize(X, domain=[-1,1], n_states=np.arange(2)):
        return rollaxis_(leg_x(X, domain, n_states))
    

def legendre_basis(X, domain=[-1,1], n_states=2):
    if not is_in_domain(X, domain):
        if isinstance(n_states, int):
            return discretize(X, domain, np.arange(n_states))
        else:
            return discretize(X, domain, n_states)
    else:
        raise RuntimeError("X must be within the specified domain")


def P(x):
    x = 4 * x - 1
    polys = np.array((np.ones_like(x), x, (3.*x**2 - 1.) / 2.))
    tmp = (2. * np.arange(3)[:, None, None] + 1.) / 2. * polys
    return np.rollaxis(tmp, 0, 3)


def run_test1():
    n_states = 3
    X = np.array([[0.25, 0.1],
                  [0.5, 0.25]])
    domain = [0., 0.5]
    assert(np.allclose(legendre_basis(X, domain, n_states), P(X)))


def run_test2():
    n_states = 2
    X = np.array([-1, 1])
    domain=[0, 1]
    legendre_basis(X, domain, n_states)


if __name__ == "__main__":
    run_test1()
    run_test2()
