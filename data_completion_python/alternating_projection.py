import numpy as np
from numpy import linalg as LA
from data_completion_python import utils

def alternating_projection(M, Omega, r):
    """
    Alternating projection matrix completion algorithm based on work by Lai, Varghese.
    M: initial completion guess
    Omega: unknown entries mask
    r: rank of completion
    """

    N = 500 # number of itterations
    X = M.copy()

    for _ in range(N):
        U, S, Vh = LA.svd(X) # singular value decompostion of X
        Sigma = np.zeros(np.shape(M))
        np.fill_diagonal(Sigma,S) # diagonal matrix of singular values
        X = U[:,0:r] @ Sigma[0:r,0:r] @ Vh[0:r,:] # rank r projection
        X[Omega] = M[Omega] # known_entries #projection onto plane of completions
    return X # completion

def alternating_projection_example():
    """An example of the alternating projection matrix completion algorithm
    using sample parameters."""
    m = 40
    n = 30 #matrix size m x n
    r = 5 #rank of completion
    A = np.random.rand(m, r)
    B = np.random.rand(r, n) #random factor matrices
    M = A @ B #random rank r matrix

    known_ratio = 0.75 #ratio of known to unknown entries
    num_known = round(known_ratio * m * n) #number of known entries
    Omega = utils.random_mask(m, n, num_known) #random mask of unknown entries

    M0 = M.copy()
    M0[~Omega] = 0 #sets unknown entries equal to zero

    X = alternating_projection(M0, Omega, r)
    err = LA.norm(X-M, 2)
    print("The spectral norm error between the original and completion is", err)


if __name__ == "__main__":
    alternating_projection_example()
