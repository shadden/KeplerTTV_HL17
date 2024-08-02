import numpy as np
from celmech.miscellaneous import critical_relative_AMD

def get_samples(n):
    """
    Generate a sample from joint distribution
    of n uniform random variables between
    0 and 1 subject to the constraint that
    their sum is equal to 1.
    """
    x = np.random.uniform(0,1,size=n-1)
    xs = np.sort(x)
    y = np.zeros(n)
    y[0] = xs[0]
    y[1:n-1] = xs[1:] - xs[:-1]
    y[n-1] = 1 - xs[-1]
    return y

def get_critical_AMD(pvars):
    AMD_crit = np.inf
    for i in range(1,pvars.N-1):
        p1,p2 = pvars.particles[i],pvars.particles[i+1]
        AMD_crit=np.min((AMD_crit,p2.Lambda * critical_relative_AMD(p1.a/p2.a,p1.m/p2.m)))
    return AMD_crit