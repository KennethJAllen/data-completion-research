import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt

def alt_maxvol(X,I_initial,J_initial):
    #Algorithm based on paper How to Find a Good Submatrix modified for two-directional search
    #written by K. Allen
    #alternating version of the one directional maxvol algorithm
    #finds a close to dominant r x r submatrix of m x n matrix X

    #A = X[I,:][:,J] is the resulting close to dominant submatrix
    #abs(det(A)) is close to maximum over all choices of submatrixes
    I = I_initial.copy()
    J = J_initial.copy()
    epsilon = 1e-8 #tolerance

    r = len(I)
    r2 = len(J)
    if r != r2: #checks that initial submatrix is r x r
        print("initial submatrix is not r x r")
        exit()

    A = X[I,:][:,J] #initial submatrix
    if LA.cond(A) > 1e12: #initial submatrix must be nonsingular
        print("Initial submatrix is close to singular")
        exit()

    row_dom = False #indicates if near dominant in rows
    column_dom = False #indicates if near dominant in columns

    N = 1000 #maximum number of itterations
    for k in range(N):
        Yh = LA.solve(A.T,X[:,J].T)
        Y = Yh.T #Y =  X[:,J]A^{-1}
        Ya = np.abs(Y) #entry-wise absolute value of Y
        y = np.amax(Ya) #largest element of Y in modulus
        if y > 1+epsilon: #if A is not within the acceptable tolerance of a dominant submatrix in columns
            position_y = np.where(Ya == y) #indices of maximum element in Ya
            i = position_y[0][0]
            j = position_y[1][0] #(i,j) are the coordinates of y in Ya
            I[j] = i #replaces jth row of A with the ith row of X[:,J]
            A = X[I,:][:,J]
            column_dom = False #not near dominant in columns
        elif row_dom == True: #if near dominant in both rows and columns
            break
        else:
            column_dom = True #indicates that A is near dominant in columns

        Z = LA.solve(A,X[I,:]) #Z = A^{-1}X[I,:]
        Za = np.abs(Z) #entry-wise absolute value of Z
        z = np.amax(Za) #largest element of Z in modulus
        if z > 1+epsilon: #if A is not within the acceptable tolerance of a dominant submatrix in rows
            position_z = np.where(Za == z) #indices of maximum element in Za
            p = position_z[0][0]
            q = position_z[1][0] #(p,q) are the coordinates of z in Za
            J[p] = q #replace pth column of A with qth row of X[I,:]
            A = X[I,:][:,J]
            row_dom = False #not near dominant in rows
        elif column_dom == True: #if near dominant in both rows and columns
            break
        else:
            row_dom = True #indicates that A is near dominant in rows

        if k==N-1:
            print("alt_maxvol did not converge in", N, "steps")
            exit()
    return I,J

#example
m = 400
n = 500
r = 10 #matrix size m x r, submatrix is of size r x r
X = np.random.rand(m,n)
I_initial = np.random.choice(m, r, replace=False) #row indices of the initial submatrix in X
J_initial = np.random.choice(n, r, replace=False) #column indices of the initial submatrix in X
A_initial = X[I_initial,:][:,J_initial] #initial submatrix guess
print("volume of initial submatrix is", np.abs(LA.det(A_initial)))

I, J = alt_maxvol(X,I_initial,J_initial)
A = X[I,:][:,J]
print("volume of dominant submatrix is", np.abs(LA.det(A)))