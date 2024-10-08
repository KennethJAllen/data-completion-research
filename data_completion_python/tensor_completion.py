import numpy as np
from numpy import linalg as LA

def unfold(T: np.ndarray, i: int) -> np.ndarray:
    """
    Inputs: Tensor T, mode i.
    Output: the mode-i unfolding of tensor T.
    i = 1, 2, or 3
    """
    return np.reshape(np.moveaxis(T, i-1, 0), (T.shape[i-1], -1), order='F')

def fold(X: np.ndarray, tensor_shape: tuple[int], i: int) -> np.ndarray:
    """
    Folds the mode-i unfolding of T back into T.
    Input: np.ndarray X, the mode i unfolding of T, tensor_shape, the shape of T.
    i = 1, 2, or 3.
    fold is the inverse of unfold in the sense that fold(unfold(T,i), np.shape(T), i) = T
    """
    # move element of tuple tensor_shape in position i-1 to front
    ts = list(tensor_shape) # converts tuple tensor_shape to list
    ts.insert(0, ts.pop(i-1)) # moves element in position i-1 to front
    shift_shape = tuple(ts) # converts back to tuple

    return np.moveaxis(np.reshape(X, shift_shape, order='F'), 0, i-1)

def tensor_completion(T_Omega: np.ndarray, r: int) -> np.ndarray:
    """
    written by K. Allen under Dr. Ming-Jun Lai's supervision
    from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

    given an m x n x p partially known tensor T_Omega, where T_Omega has the block tensor structure:
    T_Omega[:,:,:r] = [[A, B],[C, G]]
    T_Omega[:,:,r:] = [[D, F],[E, H]]
    A, B, C, D are known
    E, F, G, H are unknonwn
    completes T_Omega into a multilinear rank (r,r,r) tensor T
    if a multilinear rank (r,r,r) completion exists, it is unique

    A = T_Omega[:r,:r,:r] is an r x r x r multilinear rank (r,r,r) fully known subtensor
    B = T_Omega[:r,r:,:r] is the corresponding fully known r x (n-r) x r subtensor
    C = T_Omega[r:,:r,:r] is the corresponding fully known (m-r) x r x r subtensor
    D = T_Omega[:r,:r,r:] is the corresponding fully known r x r x (p-r) subtensor
    """

    T = T_Omega.copy()
    (m, n, p) = np.shape(T)

    A = T[:r, :r, :r] # assumes has multilinear rank (r,r,r)
    A1 = unfold(A, 1) # mode-1 unfolding of A
    A2 = unfold(A, 2) # mode-2 unfolding of A

    # assumes that the first r x r sub-matrix of the mode-1 and mode-2 unfoldings of A are nonsingular
    # this can be improved to search for a nonsingualr r x r submatrix of A1, A2
    AJ = A1[:r, :r]
    AI = A2[:r, :r]
    r1 = LA.matrix_rank(AI)
    r2 = LA.matrix_rank(AJ)
    if [r1, r2] != [r, r]:
        raise ValueError('''The top-left r x r submatrix of the mode-1 and mode-2 unfoldings
        of the top-left r x r x r subtensor of T_Omega must be full rank''')

    B = T[:r, r:, :r]
    C = T[r:, :r, :r]
    D = T[:r, :r, r:]

    B1 = unfold(B, 1)
    C1 = unfold(C, 1)
    CJ = C1[:(m-r), :r]
    D1 = unfold(D, 1)
    G1 = CJ @ LA.solve(AJ, B1) # completes G
    E1 = CJ @ LA.solve(AJ, D1) # completes E
    G = fold(G1,(m-r,n-r,r), 1) # folds mode-1 unfolding of G
    E = fold(E1,(m-r,r,p-r), 1) # folds mode-1 unfolding of E

    B2 = unfold(B, 2)
    BI = B2[:(n-r), :r]
    D2 = unfold(D, 2)
    E2 = unfold(E, 2)
    F2 = BI @ LA.solve(AI, D2) # comples F
    H2 = BI @ LA.solve(AI, E2) # completes H
    F = fold(F2, (r, n-r, p-r), 2) # folds mode-2 unfolding of F
    H = fold(H2, (m-r, n-r, p-r), 2) # folds mode-2 unfolding of H

    # assembling completion
    T[r:, r:, :r] = G
    T[r:, :r, r:] = E
    T[:r, r:, r:] = F
    T[r:, r:, r:] = H
    return T

def forget_EFGH(T: np.ndarray, r: int) -> np.ndarray:
    """
    if T has the tensor block structure:
    T[:, :, :r] = [[A,B],[C,G]]
    T[:, :, r:] = [[D,F],[E,H]]
    replaces entries in positions E, F, G, and H with zeros.
    """
    T0 = T.copy()
    T0[r:, r:, :r] = 0 # sets G to zero
    T0[r:, :r, r:] = 0 # sets E to zero
    T0[:r, r:, r:] = 0 # sets F to zero
    T0[r:, r:, r:] = 0 # sets H to zero
    return T0

def rand_rank_r_tensor(dims: tuple[int], r: int):
    """Input: dims = (m, n, p), rank r.
    Output: random m x n x p tensor T of rank R"""
    m, n, p = dims
    T = np.zeros([m, n, p])
    for _ in range(r): # generates a random rank r order three tensor
        a = np.random.rand(m)
        b = np.random.rand(n)
        c = np.random.rand(p)
        A = np.tensordot(a, b, 0)
        X = np.tensordot(A, c, 0) # tensor product of a, b, and c
        T = T + X # T is the sum of r random rank one tensors, so has rank r with probability one
    return T

def tensor_completion_example(dims: tuple[int] = (20, 19, 18), r: int = 8) -> None:
    """An example of the tensor completion algorithm using sample parameter.
    Default: dimensions dims = (20, 19, 18). rank r = 8."""

    T_true = rand_rank_r_tensor(dims, r)
    T_Omega = forget_EFGH(T_true, r) # initial guess

    T = tensor_completion(T_Omega, r) # completes T_Omega into multilinear rank (r,r,r) tensor T
    err = LA.norm(T-T_true) # error between completion and true soluition
    print(f'The error between the completion and the true solution is {err}')

if __name__ == "__main__":
    tensor_completion_example()
