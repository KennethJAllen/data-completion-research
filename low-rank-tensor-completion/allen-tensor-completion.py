import numpy as np
from numpy import linalg as LA

def unfold(T, i):
    #returns the mode-i unfolding of T
    #i = 1, 2, or 3
    return np.reshape(np.moveaxis(T, i-1, 0), (T.shape[i-1], -1), order='F')

def fold(X, tensor_shape, i):
    #refolds the mode-i unfolding of T back into T
    #i = 1, 2, or 3
    #fold(unfold(T,i),np.shape(T),i) = T

    #move element of tuple tensor_shape in position i-1 to front
    ts = list(tensor_shape) #converts tuple tensor_shape to list
    ts.insert(0,ts.pop(i-1)) #moves element in position i-1 to front
    shift_shape = tuple(ts) #converts back to tuple

    return np.moveaxis(np.reshape(X, shift_shape, order='F'), 0, i-1)

def Allen_Tensor_Completion(T_Omega, r):
    #written by K. Allen under Dr. Ming-Jun Lai's supervision
    #from K. Allen's dissertation A Geometric Approach to Low-Rank Matrix and Tensor Completion

    #given an m x n x p partially known tensor T_Omega, where T_Omega has the block tensor structure:
    #T_Omega[:,:,0:r] = [[A, B],[C, G]]
    #T_Omega[:,:,r:] = [[D, F],[E, H]]
    #A, B, C, D are known
    #E, F, G, H are unknonwn
    #completes T_Omega into a multilinear rank (r,r,r) tensor T
    #if a multilinear rank (r,r,r) completion exists, it is unique

    #A = T_Omega[0:r,0:r,0:r] is an r x r x r multilinear rank (r,r,r) fully known subtensor
    #B = T_Omega[0:r,r:,0:r] is the corresponding fully known r x (n-r) x r subtensor
    #C = T_Omega[r:,0:r,0:r] is the corresponding fully known (m-r) x r x r subtensor
    #D = T_Omega[0:r,0:r,r:] is the corresponding fully known r x r x (p-r) subtensor

    T = T_Omega.copy()
    (m,n,p) = np.shape(T)

    A = T[0:r,0:r,0:r] #assumes has multilinear rank (r,r,r)
    print(np.shape(A))
    A1 = unfold(A,1) #mode-1 unfolding of A
    A2 = unfold(A,2) #mode-2 unfolding of A

    #assumes that the first r x r sub-matrix of the mode-1 and mode-2 unfoldings of A are nonsingular
    #this can be improved to search for a nonsingualr r x r submatrix of A1, A2
    AJ = A1[0:r,0:r]
    AI = A2[0:r,0:r]
    r1 = LA.matrix_rank(AI)
    r2 = LA.matrix_rank(AJ)
    if [r1,r2] != [r,r]:
        print('''The top-left r x r submatrix of the mode-1 and mode-2 unfoldings
        of the top-left r x r x r subtensor of T_Omega must be full rank''')
        exit()

    B = T[0:r,r:,0:r]
    C = T[r:,0:r,0:r]
    D = T[0:r,0:r,r:]

    B1 = unfold(B,1)
    C1 = unfold(C,1)
    CJ = C1[0:(m-r),0:r]
    D1 = unfold(D,1)
    G1 = CJ @ LA.solve(AJ,B1) #completes G
    E1 = CJ @ LA.solve(AJ,D1) #completes E
    G = fold(G1,(m-r,n-r,r),1) #folds mode-1 unfolding of G
    E = fold(E1,(m-r,r,p-r),1) #folds mode-1 unfolding of E

    B2 = unfold(B,2)
    BI = B2[0:(n-r),0:r]
    D2 = unfold(D,2)
    E2 = unfold(E,2)
    F2 = BI @ LA.solve(AI,D2) #comples F
    H2 = BI @ LA.solve(AI,E2) #completes H
    F = fold(F2,(r,n-r,p-r),2) #folds mode-2 unfolding of F
    H = fold(H2,(m-r,n-r,p-r),2) #folds mode-2 unfolding of H

    #assembling completion
    T[r:,r:,0:r] = G
    T[r:,0:r,r:] = E
    T[0:r,r:,r:] = F
    T[r:,r:,r:] = H
    return T

def forget_EFGH(T,r):
    #if T has the tensor block structure:
    #T[:,:,0:r] = [[A,B],[C,G]]
    #T[:,:,r:] = [[D,F],[E,H]]
    #replaces entries in positions E, F, G, and H with zeros
    T0 = T.copy()
    T0[r:,r:,0:r] = 0 #sets G to zero
    T0[r:,0:r,r:] = 0 #sets E to zero
    T0[0:r,r:,r:] = 0 #sets F to zero
    T0[r:,r:,r:] = 0 #sets H to zero
    return T0

def rand_rank_r_tensor(m, n, p, r):
    #input: tensor dimensions m x n x p, rank r
    T = np.zeros([m,n,p])
    for i in range(r): #generates a random rank r order three tensor
        a = np.random.rand(m)
        b = np.random.rand(n)
        c = np.random.rand(p)
        A = np.tensordot(a,b,0)
        X = np.tensordot(A,c,0) #tensor product of a, b, and c
        T = T + X #T is the sum of r random rank one tensors, so has rank r with probability one
    return T

#example
m = 20
n = 19
p = 18 #m x n x p tensor
r = 8 #rank of the tensor

T_true = rand_rank_r_tensor(m, n, p, r)
T_Omega = forget_EFGH(T_true,r) #initial guess

T = Allen_Tensor_Completion(T_Omega, r) #completes T_Omega into multilinear rank (r,r,r) tensor T
err = LA.norm(T-T_true) #error between completion and true soluition
print('The error between the completion and the true solution is', err)