import numpy as np

def is_in_domain(X, domain):
    return ((np.min(X) < domain[0]) or (np.max(X) > domain[1]))


def makeStates(domain, n_states):
    return np.linspace(domain[0], domain[1], n_states)


def discretize_nomax(X, states):
    return (1-(abs(X[..., None]-states))/(states[1]-states[0]))


def discretize(X, domain, n_states):
    return np.maximum(
        discretize_nomax(X = np.clip(X, domain[0], domain[1]),
                            states=makeStates(domain, n_states)), 
        0)


def primitive_basis(X, domain=[-1,1], n_states=2):
    if not is_in_domain(X, domain):
            return discretize(X, domain, n_states)
    else:
        raise RuntimeError("X must be within the specified domain")
