import numpy as np
from numpy import linalg as LA

def maxvol(X, I_initial):
    """
    Algorithm based on paper How to Find a Good Submatrix
    written by K. Allen
    finds a close to dominant r x r submatrix of m x r matrix X

    A = X[I,:] is the resulting close to dominant submatrix
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

if __name__ == "__main__":
    # example
    m = 100
    r = 10 # matrix size m x r, submatrix is of size r x r

    X = np.random.rand(m,r)
    I_initial = np.random.choice(m, r, replace=False) # indices of the initial submatrix in X
    A_initial = X[I_initial,:] # initial submatrix guess
    print("volume of initial submatrix is", np.abs(LA.det(A_initial)))

    I = maxvol(X,I_initial)
    A = X[I,:]
    print("volume of dominant submatrix is", np.abs(LA.det(A)))
