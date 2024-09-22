import numpy as np
from numpy import linalg as LA

def maxvol(X: np.ndarray, I_initial: np.ndarray) -> np.ndarray:
    """
    Algorithm based on paper How to Find a Good Submatrix written by K. Allen.
    Finds a close to dominant r x r submatrix of m x r matrix X.

    A = X[I,:] is the resulting close to dominant submatrix.
    abs(det(A)) is close to maximum over all choices of submatrixes
    """
    I = I_initial.copy()
    tolerance = 1e-8

    r = len(I)
    r2 = X.shape[1]
    if r != r2: # checks that the size of the initial guess matches the size of the matrix
        raise ValueError("The size of the initial guess does not match the width of the matrix")

    A = X[I,:] # initial submatrix

    if LA.cond(A) > 1e12: # initial submatrix must be nonsingular
        raise ValueError("Initial submatrix is close to singular")

    N = 1000 # maximum number of itterations
    for k in range(N):
        Yh = LA.solve(A.T,X.T)
        Y = Yh.T # Y =  XA^{-1}
        Ya = np.abs(Y) # entry-wise absolute value of Y
        y = np.amax(Ya) # largest element
        if y < 1 + tolerance: # if A is within the acceptable tolerance of a dominant submatrix
            break
        elif k == N-1:
            raise ValueError("maxvol did not converge in", N, "steps")

        position = np.where(Ya == y) # indices of maximum element in Ya
        i = position[0][0]
        j = position[1][0] # (i,j) are the coordinates of y in Ya
        I[j] = i # replaces jth row of A with the ith row of X
        A = X[I,:]

    return I

def alternating_maxvol(X: np.ndarray, I_initial: np.ndarray, J_initial: np.ndarray) -> tuple[np.ndarray]:
    """
    Algorithm based on paper How to Find a Good Submatrix modified for two-directional search
    written by K. Allen
    alternating version of the one directional maxvol algorithm
    finds a close to dominant r x r submatrix of m x n matrix X

    A: X[I,:][:,J] is the resulting close to dominant submatrix
    abs(det(A)): is close to maximum over all choices of submatrixes
    """
    I = I_initial.copy()
    J = J_initial.copy()
    epsilon = 1e-8 # tolerance

    rank = len(I)
    if rank != len(J): # checks that initial submatrix is r x r
        raise ValueError("initial submatrix is not r x r")

    A = X[I,:][:,J] # initial submatrix
    initial_rank = np.linalg.matrix_rank(A)
    if initial_rank != rank: # initial submatrix should be nonsingular
        print(f"Warning: Submatrix in positions I and J is not invertible. Rank {initial_rank} when expecting rank {rank}.")

    row_dom = False # indicates if near dominant in rows
    column_dom = False # indicates if near dominant in columns

    N = 1000 # maximum number of itterations
    for k in range(N):
        Yh = LA.solve(A.T,X[:,J].T)
        Y = Yh.T # Y =  X[:,J]A^{-1}
        Ya = np.abs(Y) # entry-wise absolute value of Y
        y = np.amax(Ya) # largest element of Y in modulus
        if y > 1+epsilon: # if A is not within the acceptable tolerance of a dominant submatrix in columns
            position_y = np.where(Ya == y) # indices of maximum element in Ya
            i = position_y[0][0]
            j = position_y[1][0] # (i,j) are the coordinates of y in Ya
            I[j] = i # replaces jth row of A with the ith row of X[:,J]
            A = X[I,:][:,J]
            column_dom = False # not near dominant in columns
        elif row_dom: # if near dominant in both rows and columns
            break
        else:
            column_dom = True # indicates that A is near dominant in columns

        Z = LA.solve(A,X[I,:]) #Z = A^{-1}X[I,:]
        Za = np.abs(Z) # entry-wise absolute value of Z
        z = np.amax(Za) # largest element of Z in modulus
        if z > 1+epsilon: # if A is not within the acceptable tolerance of a dominant submatrix in rows
            position_z = np.where(Za == z) # indices of maximum element in Za
            p = position_z[0][0]
            q = position_z[1][0] # (p,q) are the coordinates of z in Za
            J[p] = q # replace pth column of A with qth row of X[I,:]
            A = X[I,:][:,J]
            row_dom = False # not near dominant in rows
        elif column_dom: # if near dominant in both rows and columns
            break
        else:
            row_dom = True # indicates that A is near dominant in rows

        if k == N-1:
            raise ValueError(f"alt_maxvol did not converge in {N} steps")
    return I, J

def cur_decomposition(X: np.ndarray, I: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Returns the CUR decomposition of X 
    with respect to the submatrix in given row indices I
    and column indices J."""
    rank = len(I)
    if rank != len(J):
        raise ValueError("Length of I and J must be equal.")
    C = X[:,J]
    A = X[I,:][:,J]
    R = X[I,:]
    submatrix_rank = np.linalg.matrix_rank(A)
    if submatrix_rank != rank:
        print(f"Warning: Submatrix in positions I and J is not invertible. Rank {submatrix_rank} when expecting rank {rank}.")
    U = np.linalg.pinv(A)
    return C @ U @ R

# Examples

def maxvol_example(m: int = 100, r: int = 10) -> None:
    """Example for how to use the maxvol function.
    Matrix size m x r, submatrix is of size r x r"""

    X = np.random.rand(m,r)
    I_initial = np.random.choice(m, r, replace=False) # indices of the initial submatrix in X
    A_initial = X[I_initial,:] # initial submatrix guess
    initial_vol = np.abs(LA.det(A_initial))
    print(f"The volume of the initial {r} by {r} submatrix in the {m} by {r} matrix is {round(initial_vol, 6)}.")

    I = maxvol(X,I_initial)
    A = X[I,:]
    maximum_volume = np.abs(LA.det(A))
    print(f"The volume of the domainant {r} by {r} submatrix is {round(maximum_volume, 6)}.")

def alternating_maxvol_example(m: int = 400, n: int = 500, r: int = 10) -> None:
    """Example for how to use the maxvol function.
    Matrix size m x r, submatrix is of size r x r."""

    X = np.random.rand(m,n)
    I_initial = np.random.choice(m, r, replace=False) # row indices of the initial submatrix in X
    J_initial = np.random.choice(n, r, replace=False) # column indices of the initial submatrix in X
    A_initial = X[I_initial,:][:,J_initial] # initial submatrix guess
    initial_vol = np.abs(LA.det(A_initial))
    print(f"The volume of the initial {r} by {r} submatrix in the {m} by {n} matrix is {round(initial_vol, 6)}.")

    I, J = alternating_maxvol(X,I_initial,J_initial)
    A = X[I,:][:,J]
    maximum_volume = np.abs(LA.det(A))
    print(f"The volume of the domainant {r} by {r} submatrix is {round(maximum_volume, 6)}.")

if __name__ == "__main__":
    maxvol_example()
    alternating_maxvol_example()
