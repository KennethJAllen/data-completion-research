import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

def maxvol(X,I):
    #Algorithm based on paper How to Find a Good Submatrix
    #written by K. Allen
    #finds a close to dominant r x r submatrix of m x r matrix X

    #A = X(I,:) is the resulting close to dominant submatrix
    #abs(det(A)) is close to maximum over all choices of submatrixes

    epsilon = 1e-8 #tolerance
    A = X[I,:] #initial submatrix

    if LA.cond(A) > 1e12: #initial submatrix must be nonsingular
        print("Initial submatrix is close to singular")
        exit()
    
    N = 1000 #maximum number of itterations
    for k in range(N):
        Yh = LA.solve(A.T,X.T)
        Y = Yh.T #Y =  XA^{-1}
        Ya = np.abs(Y) #entry-wise absolute value of Y
        y = np.amax(Ya) #largest element
        if y<1+epsilon: #if A is within the acceptable tolerance of a dominant submatrix
            break
        elif k==N-1:
            print("maxvol did not converge in", N, "steps")
            exit()
        position = np.where(Ya == y) # Get the indices of maximum element in numpy array
        i = position[0][0]
        j = position[1][0] #(i,j) are the coordinates of y in Ya
        I[j] = i #replaces jth row of A with the ith row of X
        A = X[I,:]
    return I

#example
m = 100
r = 10 #matrix size m x r, submatrix is of size r x r
X = np.random.rand(m,r)

I = np.random.choice(m, r, replace=False) #indices of the initial submatrix in X
A_initial = X[I,:]
print("volume of initial submatrix is", np.abs(LA.det(A_initial)))

I = maxvol(X,I)
A = X[I,:]
print("volume of maxvol submatrix is", np.abs(LA.det(A)))