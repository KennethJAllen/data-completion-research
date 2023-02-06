import numpy as np

def random_mask(m,n,k):
    #creates a random mxn mask with exactly k true entries
    perm = np.random.choice(m*n, k, replace=False)
    Omega_linear = np.zeros(m*n)
    Omega_linear[perm] = 1
    Omega = np.reshape(Omega_linear,(m,n))
    Omega = Omega==1 #turns Omega into a logical matrix
    return Omega