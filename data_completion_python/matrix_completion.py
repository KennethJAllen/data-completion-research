import numpy as np
from numpy import linalg as LA
from data_completion_python import utils

def alternating_projection(M: np.ndarray, mask: np.ndarray, r: int, n_iters: int = 50) -> np.ndarray:
    """
    Alternating projection matrix completion algorithm based on work by Lai, Varghese.
    Inputs:
        M: initial completion guess.
        mask: unknown entries mask.
        r: rank of completion.
        n_iters: number of iterations
    Output:
        X: Approximately rank r completion of M.
    """

    X = M.copy()

    for _ in range(n_iters):
        U, S, Vh = LA.svd(X) # singular value decompostion of X
        Sigma = np.zeros(np.shape(M))
        np.fill_diagonal(Sigma,S) # diagonal matrix of singular values
        X = U[:,:r] @ Sigma[:r,:r] @ Vh[:r,:] # rank r projection
        X[mask] = M[mask] # known_entries #projection onto plane of completions
    return X # completion

def alternating_projection_example(m: int = 40, n: int = 30, r: int = 5) -> None:
    """An example of the alternating projection matrix completion algorithm
    using sample parameters. m x n matrix, rank r completiojn."""
    A = np.random.rand(m, r)
    B = np.random.rand(r, n) # random factor matrices
    M = A @ B # random rank r matrix

    known_ratio = 0.75 # ratio of known to unknown entries
    num_known = round(known_ratio * m * n) # number of known entries
    mask = utils.random_mask(m, n, num_known) # random mask of known entries

    M0 = M.copy()
    M0[~mask] = 0 # sets unknown entries equal to zero

    X = alternating_projection(M0, mask, r)
    err = LA.norm(X-M, 2)
    print(f"The spectral norm error between the original and completion is {err}.")


if __name__ == "__main__":
    alternating_projection_example()
